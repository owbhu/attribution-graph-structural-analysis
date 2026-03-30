"""
Graph Generator
===============

Generate attribution graphs from language models using circuit-tracer.
Supports Gemma-2-2B and Llama-3.2-1B via TransformerLens and nnsight backends.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


@dataclass
class AttributionNode:
    """A node in an attribution graph (feature or token)."""
    node_id: str
    layer: int
    feature_idx: int
    activation: float
    node_type: str  # 'feature', 'token_embed', 'logit'
    label: Optional[str] = None  # human-readable label if available
    is_interpretable: Optional[bool] = None


@dataclass
class AttributionEdge:
    """A directed edge in an attribution graph."""
    source_id: str
    target_id: str
    weight: float  # attribution score


@dataclass
class AttributionGraph:
    """Complete attribution graph for a single prompt."""
    prompt: str
    model_name: str
    target_token: str
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    generation_time: float = 0.0

    def to_dict(self):
        return {
            'prompt': self.prompt,
            'model_name': self.model_name,
            'target_token': self.target_token,
            'nodes': [asdict(n) for n in self.nodes],
            'edges': [asdict(e) for e in self.edges],
            'metadata': self.metadata,
            'generation_time': self.generation_time,
        }

    @classmethod
    def from_dict(cls, d):
        graph = cls(
            prompt=d['prompt'],
            model_name=d['model_name'],
            target_token=d['target_token'],
            metadata=d.get('metadata', {}),
            generation_time=d.get('generation_time', 0.0),
        )
        graph.nodes = [AttributionNode(**n) for n in d['nodes']]
        graph.edges = [AttributionEdge(**e) for e in d['edges']]
        return graph

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class CircuitTracerGenerator:
    """
    Generate attribution graphs using Anthropic's circuit-tracer library.
    """

    TRANSCODER_MAP = {
        "google/gemma-2-2b": "gemma",
        "meta-llama/Llama-3.2-1B": "llama",
    }

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        device: str = "auto",
        backend: str = "transformerlens",
        dtype: Optional[str] = "bfloat16",
    ):
        self.model_name = model_name
        self.backend = backend
        self.dtype = getattr(torch, dtype) if dtype else torch.bfloat16

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.transcoder_set = self.TRANSCODER_MAP.get(model_name)
        if self.transcoder_set is None:
            raise ValueError(
                f"No transcoder set known for '{model_name}'. "
                f"Supported models: {list(self.TRANSCODER_MAP.keys())}."
            )

        self.replacement_model = None
        self._debug_printed = False  # only print debug info once

    def load_model(self):
        """Load the model and create the replacement model for circuit tracing."""
        try:
            from circuit_tracer import ReplacementModel
        except ImportError:
            raise ImportError(
                "circuit-tracer is not installed. Install with:\n"
                "  pip install git+https://github.com/decoderesearch/circuit-tracer.git"
            )

        print(f"Loading {self.model_name} with transcoder_set='{self.transcoder_set}' "
              f"on {self.device} ({self.backend} backend)...")

        self.replacement_model = ReplacementModel.from_pretrained(
            self.model_name,
            self.transcoder_set,
            dtype=self.dtype,
            backend=self.backend,
        )

        print("Model loaded successfully.")

    def generate_graph(
        self,
        prompt: str,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
        max_nodes: int = 200,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> AttributionGraph:
        if self.replacement_model is None:
            self.load_model()

        start_time = time.time()

        try:
            from circuit_tracer import attribute
            from circuit_tracer.graph import prune_graph
        except ImportError:
            raise ImportError(
                "circuit-tracer is not installed. Install with:\n"
                "  pip install git+https://github.com/decoderesearch/circuit-tracer.git"
            )

        raw_graph = attribute(
            prompt=prompt,
            model=self.replacement_model,
            max_feature_nodes=max_nodes,
            batch_size=batch_size,
            verbose=verbose,
        )

        prune_result = prune_graph(
            raw_graph,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        attr_graph = self._convert_graph(prompt, raw_graph, prune_result)
        attr_graph.generation_time = time.time() - start_time

        return attr_graph

    def _convert_graph(
        self,
        prompt: str,
        raw_graph,
        prune_result,
    ) -> AttributionGraph:
        """
        Convert circuit-tracer Graph + PruneResult into our AttributionGraph format.

        This method first introspects the Graph object to understand its structure,
        then safely converts it regardless of the exact adjacency matrix layout.
        """
        graph = AttributionGraph(
            prompt=prompt,
            model_name=self.model_name,
            target_token='',
            metadata={},
        )

        adj = raw_graph.adjacency_matrix
        total_nodes = adj.shape[0]
        node_mask = prune_result.node_mask
        edge_mask = prune_result.edge_mask

        # ====== DEBUG: Print Graph structure (first time only) ======
        if not self._debug_printed:
            self._debug_printed = True
            print("\n" + "=" * 60)
            print("DEBUG: Introspecting circuit-tracer Graph object")
            print("=" * 60)

            # List all attributes
            for attr_name in sorted(dir(raw_graph)):
                if attr_name.startswith('_'):
                    continue
                try:
                    val = getattr(raw_graph, attr_name)
                    if callable(val):
                        continue
                    if isinstance(val, torch.Tensor):
                        print(f"  {attr_name}: Tensor shape={val.shape} dtype={val.dtype}")
                    elif isinstance(val, (list, tuple)):
                        print(f"  {attr_name}: {type(val).__name__} len={len(val)}")
                        if len(val) > 0:
                            print(f"    [0] = {val[0]}")
                    elif isinstance(val, (int, float, str, bool)):
                        print(f"  {attr_name}: {val}")
                    else:
                        print(f"  {attr_name}: {type(val).__name__}")
                except Exception as e:
                    print(f"  {attr_name}: <error reading: {e}>")

            print(f"\n  adjacency_matrix shape: {adj.shape}")
            print(f"  node_mask shape: {node_mask.shape}, sum={node_mask.sum().item()}")
            if edge_mask is not None:
                print(f"  edge_mask shape: {edge_mask.shape}, sum={edge_mask.sum().item()}")
            else:
                print(f"  edge_mask: None")
            print(f"  active_features shape: {raw_graph.active_features.shape}")
            if hasattr(raw_graph, 'selected_features') and raw_graph.selected_features is not None:
                print(f"  selected_features shape: {raw_graph.selected_features.shape}")
            print(f"  n_pos: {raw_graph.n_pos if hasattr(raw_graph, 'n_pos') else 'N/A'}")
            print(f"  logit_targets: {len(raw_graph.logit_targets)}")
            if hasattr(raw_graph, 'cfg') and raw_graph.cfg is not None:
                print(f"  cfg.n_layers: {raw_graph.cfg.n_layers if hasattr(raw_graph.cfg, 'n_layers') else 'N/A'}")
            print("=" * 60 + "\n")

        # ====== SAFE CONVERSION ======
        # Strategy: iterate over ALL indices in [0, total_nodes) and classify
        # each node based on the Graph's own structure, rather than assuming
        # a specific layout.

        n_active = raw_graph.active_features.shape[0]
        has_selected = (hasattr(raw_graph, 'selected_features')
                        and raw_graph.selected_features is not None)
        n_selected = len(raw_graph.selected_features) if has_selected else n_active
        n_pos = raw_graph.n_pos if hasattr(raw_graph, 'n_pos') else 0
        n_logits = len(raw_graph.logit_targets)
        n_layers = raw_graph.cfg.n_layers if (hasattr(raw_graph, 'cfg') and hasattr(raw_graph.cfg, 'n_layers')) else 26

        # According to circuit-tracer docs, the adjacency matrix layout is:
        # [active_features, error_nodes, embed_nodes, logit_nodes]
        # where error_nodes = n_layers * n_pos, embed_nodes = n_pos, logit_nodes = n_logits
        #
        # BUT the actual adj size may differ. Let's figure out the actual layout
        # by computing expected sizes and seeing what fits.

        n_error = n_layers * n_pos
        n_embed = n_pos

        # Try different possible layouts and see which one matches total_nodes
        layouts = {
            "selected+error+embed+logit": n_selected + n_error + n_embed + n_logits,
            "selected+embed+logit": n_selected + n_embed + n_logits,
            "selected+logit": n_selected + n_logits,
            "selected_only": n_selected,
            "active+error+embed+logit": n_active + n_error + n_embed + n_logits,
        }

        matched_layout = None
        for name, size in layouts.items():
            if size == total_nodes:
                matched_layout = name
                break

        if not self._debug_printed or True:  # always print layout match
            print(f"  Adjacency matrix: {total_nodes} nodes")
            print(f"  Layout candidates: {layouts}")
            print(f"  Matched layout: {matched_layout}")

        node_id_map = {}

        if matched_layout == "selected_only":
            # Adjacency matrix only has the selected feature nodes
            # No error/embed/logit nodes in the matrix
            n_feat = n_selected
            feat_offset = 0

            for i in range(n_feat):
                if i < len(node_mask) and not node_mask[i]:
                    continue

                if has_selected:
                    orig_idx = int(raw_graph.selected_features[i].item())
                    feat = raw_graph.active_features[orig_idx]
                else:
                    feat = raw_graph.active_features[i]

                layer = int(feat[0].item())
                pos = int(feat[1].item())
                feature_idx = int(feat[2].item())

                if hasattr(raw_graph, 'activation_values') and i < len(raw_graph.activation_values):
                    activation = float(raw_graph.activation_values[i].item())
                else:
                    activation = 0.0

                nid = f"feat_L{layer}_P{pos}_F{feature_idx}"
                node_id_map[i] = nid
                graph.nodes.append(AttributionNode(
                    node_id=nid, layer=layer, feature_idx=feature_idx,
                    activation=activation, node_type="feature",
                ))

        elif matched_layout and "selected" in matched_layout:
            # Layout starts with selected features, then possibly error, embed, logit
            feat_offset = 0
            n_feat = n_selected

            # Feature nodes
            for i in range(n_feat):
                if i < len(node_mask) and not node_mask[i]:
                    continue

                if has_selected:
                    orig_idx = int(raw_graph.selected_features[i].item())
                    feat = raw_graph.active_features[orig_idx]
                else:
                    feat = raw_graph.active_features[i]

                layer = int(feat[0].item())
                pos = int(feat[1].item())
                feature_idx = int(feat[2].item())

                if hasattr(raw_graph, 'activation_values') and i < len(raw_graph.activation_values):
                    activation = float(raw_graph.activation_values[i].item())
                else:
                    activation = 0.0

                nid = f"feat_L{layer}_P{pos}_F{feature_idx}"
                node_id_map[i] = nid
                graph.nodes.append(AttributionNode(
                    node_id=nid, layer=layer, feature_idx=feature_idx,
                    activation=activation, node_type="feature",
                ))

            offset = n_feat

            # Error nodes (if present in layout)
            if "error" in matched_layout:
                for idx in range(n_error):
                    adj_idx = offset + idx
                    if adj_idx < len(node_mask) and node_mask[adj_idx]:
                        layer_num = idx // n_pos
                        pos_num = idx % n_pos
                        nid = f"error_L{layer_num}_P{pos_num}"
                        node_id_map[adj_idx] = nid
                        graph.nodes.append(AttributionNode(
                            node_id=nid, layer=layer_num, feature_idx=pos_num,
                            activation=0.0, node_type="error",
                        ))
                offset += n_error

            # Embed nodes (if present)
            if "embed" in matched_layout:
                for p in range(n_embed):
                    adj_idx = offset + p
                    if adj_idx < len(node_mask) and node_mask[adj_idx]:
                        nid = f"embed_P{p}"
                        node_id_map[adj_idx] = nid
                        graph.nodes.append(AttributionNode(
                            node_id=nid, layer=0, feature_idx=p,
                            activation=0.0, node_type="token_embed",
                        ))
                offset += n_embed

            # Logit nodes (if present)
            if "logit" in matched_layout:
                for k in range(n_logits):
                    adj_idx = offset + k
                    if adj_idx < len(node_mask) and node_mask[adj_idx]:
                        target = raw_graph.logit_targets[k]
                        token_id = int(target.token_id) if hasattr(target, 'token_id') else k
                        nid = f"logit_{k}"
                        node_id_map[adj_idx] = nid
                        graph.nodes.append(AttributionNode(
                            node_id=nid, layer=999, feature_idx=token_id,
                            activation=float(raw_graph.logit_probabilities[k].item()) if hasattr(raw_graph, 'logit_probabilities') and k < len(raw_graph.logit_probabilities) else 0.0,
                            node_type="logit",
                        ))

        else:
            # Unknown layout — treat ALL nodes as generic feature-like nodes
            print(f"  WARNING: No known layout matched (total_nodes={total_nodes}). "
                  f"Treating all as feature nodes.")
            for i in range(total_nodes):
                if i < len(node_mask) and not node_mask[i]:
                    continue

                # Try to get feature info if available
                if i < n_active:
                    feat = raw_graph.active_features[i]
                    layer = int(feat[0].item())
                    pos = int(feat[1].item())
                    feature_idx = int(feat[2].item())
                else:
                    layer = -1
                    pos = 0
                    feature_idx = i

                activation = 0.0
                if hasattr(raw_graph, 'activation_values') and i < len(raw_graph.activation_values):
                    activation = float(raw_graph.activation_values[i].item())

                nid = f"node_{i}_L{layer}_F{feature_idx}"
                node_id_map[i] = nid
                graph.nodes.append(AttributionNode(
                    node_id=nid, layer=layer, feature_idx=feature_idx,
                    activation=activation, node_type="feature",
                ))

        # ====== Build edges (safe — only use indices in node_id_map) ======
        kept_indices = sorted(node_id_map.keys())

        for target_idx in kept_indices:
            for source_idx in kept_indices:
                if target_idx == source_idx:
                    continue
                # Bounds check
                if target_idx >= total_nodes or source_idx >= total_nodes:
                    continue
                weight = float(adj[target_idx, source_idx].item())
                if abs(weight) > 1e-8:
                    source_id = node_id_map[source_idx]
                    target_id = node_id_map[target_idx]
                    graph.edges.append(AttributionEdge(
                        source_id=source_id,
                        target_id=target_id,
                        weight=weight,
                    ))

        # Store target token info
        if raw_graph.logit_targets:
            target = raw_graph.logit_targets[0]
            if hasattr(target, 'token_str'):
                graph.target_token = target.token_str
            elif hasattr(target, 'token_id'):
                graph.target_token = str(target.token_id)

        graph.metadata['layout'] = matched_layout or 'unknown'
        graph.metadata['total_adj_nodes'] = total_nodes
        graph.metadata['n_active_features'] = n_active
        graph.metadata['n_selected_features'] = n_selected

        return graph

    def generate_batch(
        self,
        prompts: list[str],
        output_dir: str = "data/raw",
        **kwargs,
    ) -> list[AttributionGraph]:
        """Generate attribution graphs for a batch of prompts."""
        os.makedirs(output_dir, exist_ok=True)
        graphs = []

        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Generating graph for: {prompt[:60]}...")
            try:
                graph = self.generate_graph(prompt, **kwargs)
                filename = f"graph_{i:04d}.json"
                graph.save(os.path.join(output_dir, filename))
                graphs.append(graph)
                print(f"  -> {len(graph.nodes)} nodes, {len(graph.edges)} edges "
                      f"({graph.generation_time:.1f}s)")
            except Exception as e:
                import traceback
                print(f"  -> FAILED: {e}")
                traceback.print_exc()

        print(f"\nGenerated {len(graphs)}/{len(prompts)} graphs successfully.")
        return graphs


class SyntheticGraphGenerator:
    """
    Generate synthetic attribution graphs for development and testing.

    Creates graphs with known structural properties so we can validate
    our metrics pipeline before running expensive model inference.
    """

    @staticmethod
    def generate_clean_tree(
        depth: int = 5,
        branching_factor: int = 3,
        noise: float = 0.05,
    ) -> AttributionGraph:
        graph = AttributionGraph(
            prompt="[synthetic-clean-tree]",
            model_name="synthetic",
            target_token="[synthetic]",
            metadata={"synthetic": True, "type": "clean_tree", "label": 1.0},
        )

        node_count = 0
        layers = [[]]

        root = AttributionNode(
            node_id=f"node_{node_count}",
            layer=0, feature_idx=0,
            activation=np.random.uniform(0.8, 1.0),
            node_type="logit",
        )
        graph.nodes.append(root)
        layers[0].append(root)
        node_count += 1

        for d in range(1, depth):
            layers.append([])
            for parent in layers[d - 1]:
                n_children = max(1, branching_factor + np.random.randint(-1, 2))
                for _ in range(n_children):
                    child = AttributionNode(
                        node_id=f"node_{node_count}",
                        layer=d, feature_idx=node_count,
                        activation=np.random.uniform(0.1, 0.8) / d,
                        node_type="feature" if d < depth - 1 else "token_embed",
                    )
                    graph.nodes.append(child)
                    layers[d].append(child)

                    weight = np.random.uniform(0.3, 1.0) + np.random.normal(0, noise)
                    graph.edges.append(AttributionEdge(
                        source_id=child.node_id,
                        target_id=parent.node_id,
                        weight=max(0.01, weight),
                    ))
                    node_count += 1

        return graph

    @staticmethod
    def generate_tangled_graph(
        n_nodes: int = 50,
        edge_density: float = 0.15,
        n_layers: int = 6,
    ) -> AttributionGraph:
        graph = AttributionGraph(
            prompt="[synthetic-tangled]",
            model_name="synthetic",
            target_token="[synthetic]",
            metadata={"synthetic": True, "type": "tangled", "label": 0.0},
        )

        for i in range(n_nodes):
            layer = np.random.randint(0, n_layers)
            node = AttributionNode(
                node_id=f"node_{i}",
                layer=layer, feature_idx=i,
                activation=np.random.uniform(0.1, 1.0),
                node_type="feature",
            )
            graph.nodes.append(node)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and np.random.random() < edge_density:
                    graph.edges.append(AttributionEdge(
                        source_id=f"node_{i}",
                        target_id=f"node_{j}",
                        weight=np.random.uniform(0.01, 0.5),
                    ))

        return graph

    @staticmethod
    def generate_mixed_graph(
        n_clean_clusters: int = 3,
        cluster_size: int = 8,
        n_tangled_nodes: int = 15,
        cross_cluster_density: float = 0.08,
    ) -> AttributionGraph:
        graph = AttributionGraph(
            prompt="[synthetic-mixed]",
            model_name="synthetic",
            target_token="[synthetic]",
            metadata={"synthetic": True, "type": "mixed", "label": 0.5},
        )

        node_count = 0

        for c in range(n_clean_clusters):
            cluster_root_id = f"node_{node_count}"
            cluster_root = AttributionNode(
                node_id=cluster_root_id,
                layer=0, feature_idx=node_count,
                activation=np.random.uniform(0.7, 1.0),
                node_type="feature",
            )
            graph.nodes.append(cluster_root)
            node_count += 1

            prev_layer = [cluster_root_id]
            for depth in range(1, 3):
                curr_layer = []
                for parent_id in prev_layer:
                    for _ in range(np.random.randint(2, 4)):
                        nid = f"node_{node_count}"
                        graph.nodes.append(AttributionNode(
                            node_id=nid,
                            layer=depth + c * 3,
                            feature_idx=node_count,
                            activation=np.random.uniform(0.2, 0.6),
                            node_type="feature",
                        ))
                        graph.edges.append(AttributionEdge(
                            source_id=nid,
                            target_id=parent_id,
                            weight=np.random.uniform(0.4, 0.9),
                        ))
                        curr_layer.append(nid)
                        node_count += 1
                prev_layer = curr_layer

        tangled_ids = []
        for _ in range(n_tangled_nodes):
            nid = f"node_{node_count}"
            graph.nodes.append(AttributionNode(
                node_id=nid,
                layer=np.random.randint(0, 6),
                feature_idx=node_count,
                activation=np.random.uniform(0.1, 0.8),
                node_type="feature",
            ))
            tangled_ids.append(nid)
            node_count += 1

        for i, src in enumerate(tangled_ids):
            for j, tgt in enumerate(tangled_ids):
                if i != j and np.random.random() < 0.25:
                    graph.edges.append(AttributionEdge(
                        source_id=src,
                        target_id=tgt,
                        weight=np.random.uniform(0.05, 0.4),
                    ))

        all_ids = [n.node_id for n in graph.nodes]
        for src in all_ids:
            for tgt in tangled_ids:
                if src != tgt and np.random.random() < cross_cluster_density:
                    graph.edges.append(AttributionEdge(
                        source_id=src,
                        target_id=tgt,
                        weight=np.random.uniform(0.01, 0.15),
                    ))

        return graph

    def generate_dataset(
        self,
        n_clean: int = 100,
        n_tangled: int = 100,
        n_mixed: int = 50,
        output_dir: str = "data/raw/synthetic",
    ) -> list[AttributionGraph]:
        """Generate a labeled synthetic dataset for development."""
        os.makedirs(output_dir, exist_ok=True)
        graphs = []
        idx = 0

        for i in range(n_clean):
            g = self.generate_clean_tree()
            g.save(os.path.join(output_dir, f"graph_{idx:04d}.json"))
            graphs.append(g)
            idx += 1

        for i in range(n_tangled):
            g = self.generate_tangled_graph()
            g.save(os.path.join(output_dir, f"graph_{idx:04d}.json"))
            graphs.append(g)
            idx += 1

        for i in range(n_mixed):
            g = self.generate_mixed_graph()
            g.save(os.path.join(output_dir, f"graph_{idx:04d}.json"))
            graphs.append(g)
            idx += 1

        print(f"Generated {len(graphs)} synthetic graphs -> {output_dir}")
        return graphs
