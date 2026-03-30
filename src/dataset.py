"""
Dataset
=======

PyTorch Geometric dataset class for attribution graphs.
Converts our AttributionGraph objects into PyG Data objects
suitable for GNN training.
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import random_split

try:
    from torch_geometric.data import Data, InMemoryDataset
except ImportError:
    raise ImportError(
        "torch-geometric is required. Install with:\n"
        "pip install torch-geometric"
    )

from .graph_generator import AttributionGraph
from .structural_metrics import compute_all_metrics, attribution_to_networkx


class AttributionGraphDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for attribution graphs.

    Each graph is converted to a PyG Data object with:
    - x: Node feature matrix [num_nodes, num_node_features]
    - edge_index: Edge connectivity [2, num_edges]
    - edge_attr: Edge weights [num_edges, 1]
    - y: Graph-level interpretability label [1]
    - structural_features: Precomputed structural metrics [num_structural_features]
    """

    def __init__(
        self,
        root: str,
        raw_dir: str = "data/raw",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.raw_source_dir = raw_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        source = Path(self.raw_source_dir)
        if source.exists():
            return sorted([f.name for f in source.glob("graph_*.json")])
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        """Convert raw JSON attribution graphs to PyG Data objects."""
        data_list = []
        source = Path(self.raw_source_dir)

        json_files = sorted(source.glob("graph_*.json"))
        print(f"Processing {len(json_files)} attribution graphs...")

        for i, json_path in enumerate(json_files):
            try:
                graph = AttributionGraph.load(str(json_path))
                data = attribution_graph_to_pyg(graph, graph_id=f"graph_{i:04d}")
                if data is not None and data.num_nodes > 0:
                    data_list.append(data)
            except Exception as e:
                print(f"  Skipping {json_path.name}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(json_files)}")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {len(data_list)} graphs to {self.processed_paths[0]}")


def attribution_graph_to_pyg(
    graph: AttributionGraph,
    graph_id: str = "",
) -> Optional[Data]:
    """
    Convert a single AttributionGraph to a PyG Data object.

    Node features:
    - layer (normalized)
    - activation
    - in_degree (normalized)
    - out_degree (normalized)
    - node_type one-hot (feature, token_embed, logit)

    Edge features:
    - weight
    """
    if len(graph.nodes) == 0:
        return None

    # Build node ID -> index mapping
    node_ids = [n.node_id for n in graph.nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # Build edge index
    sources = []
    targets = []
    edge_weights = []
    for edge in graph.edges:
        if edge.source_id in id_to_idx and edge.target_id in id_to_idx:
            sources.append(id_to_idx[edge.source_id])
            targets.append(id_to_idx[edge.target_id])
            edge_weights.append(edge.weight)

    if len(sources) == 0:
        # No valid edges — create self-loops as fallback
        sources = list(range(n_nodes))
        targets = list(range(n_nodes))
        edge_weights = [0.0] * n_nodes

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(-1)

    # Build node features
    node_type_map = {'token_embed': 0, 'feature': 1, 'logit': 2}

    # Compute degrees
    in_degrees = np.zeros(n_nodes)
    out_degrees = np.zeros(n_nodes)
    for s, t in zip(sources, targets):
        out_degrees[s] += 1
        in_degrees[t] += 1

    max_degree = max(np.max(in_degrees), np.max(out_degrees), 1)

    features = []
    for i, node in enumerate(graph.nodes):
        # Normalize layer by max layer
        max_layer = max(n.layer for n in graph.nodes) or 1
        norm_layer = node.layer / max_layer

        # Node type one-hot
        type_idx = node_type_map.get(node.node_type, 1)
        type_onehot = [0.0, 0.0, 0.0]
        type_onehot[type_idx] = 1.0

        feat = [
            norm_layer,
            node.activation,
            in_degrees[i] / max_degree,
            out_degrees[i] / max_degree,
        ] + type_onehot

        features.append(feat)

    x = torch.tensor(features, dtype=torch.float)

    # Graph-level label
    label = graph.metadata.get('label', None)
    y = torch.tensor([label], dtype=torch.float) if label is not None else None

    # Compute structural features as additional graph-level features
    metrics = compute_all_metrics(graph, graph_id=graph_id)
    structural_features = torch.tensor(
        metrics.to_feature_vector(), dtype=torch.float
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        structural_features=structural_features,
    )

    return data


def create_splits(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Split dataset into train/val/test."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def load_graphs_from_dir(directory: str) -> list[AttributionGraph]:
    """Load all attribution graphs from a directory."""
    graphs = []
    json_files = sorted(Path(directory).glob("graph_*.json"))
    for path in json_files:
        try:
            graphs.append(AttributionGraph.load(str(path)))
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    print(f"Loaded {len(graphs)} graphs from {directory}")
    return graphs
