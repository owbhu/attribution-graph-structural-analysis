"""
Structural Metrics
==================

Compute graph-theoretic structural metrics on attribution graphs.
These metrics form the core of our data mining stage — we hypothesize
that the structural properties of an attribution graph predict whether
it will yield a faithful mechanistic interpretation.

Metrics computed:
- Basic stats (nodes, edges, density)
- Degree distribution (in/out/total)
- Clustering coefficient
- Cycle density
- Path length statistics
- Modularity (community structure)
- Spectral properties (eigenvalues of adjacency/Laplacian)
- Layer-wise connectivity patterns
- Hierarchical depth ratio
- Cross-layer edge ratio
- Feature type homophily
"""

import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .graph_generator import AttributionGraph


@dataclass
class StructuralMetrics:
    """All computed structural metrics for a single attribution graph."""
    # Identity
    prompt: str = ""
    graph_id: str = ""

    # Basic stats
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0

    # Degree statistics
    mean_in_degree: float = 0.0
    std_in_degree: float = 0.0
    max_in_degree: int = 0
    mean_out_degree: float = 0.0
    std_out_degree: float = 0.0
    max_out_degree: int = 0
    degree_assortativity: float = 0.0

    # Clustering
    avg_clustering: float = 0.0
    transitivity: float = 0.0

    # Cycles
    n_simple_cycles_sampled: int = 0  # from sampled cycle detection
    cycle_density: float = 0.0  # cycles per edge

    # Paths
    avg_shortest_path: float = 0.0  # on largest weakly connected component
    diameter: int = 0
    n_weakly_connected_components: int = 0
    largest_wcc_fraction: float = 0.0  # fraction of nodes in largest WCC

    # Modularity
    modularity: float = 0.0
    n_communities: int = 0

    # Spectral
    spectral_gap: float = 0.0  # difference between first two eigenvalues
    algebraic_connectivity: float = 0.0  # second-smallest eigenvalue of Laplacian
    spectral_radius: float = 0.0

    # Layer structure
    n_layers: int = 0
    cross_layer_edge_ratio: float = 0.0  # edges connecting non-adjacent layers
    backward_edge_ratio: float = 0.0  # edges going from later to earlier layers
    layer_entropy: float = 0.0  # entropy of node distribution across layers

    # Hierarchy
    dag_depth: float = 0.0  # longest path in DAG approximation
    tree_likeness: float = 0.0  # ratio: (n_nodes - 1) / n_edges (1.0 = perfect tree)

    # Activation statistics
    mean_activation: float = 0.0
    std_activation: float = 0.0
    mean_edge_weight: float = 0.0
    std_edge_weight: float = 0.0
    weight_entropy: float = 0.0

    # Label (if available)
    interpretability_label: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a flat feature vector for ML (excludes identity and label)."""
        d = self.to_dict()
        exclude = {'prompt', 'graph_id', 'interpretability_label'}
        return np.array([
            float(v) if v is not None else 0.0
            for k, v in d.items()
            if k not in exclude
        ])

    @staticmethod
    def feature_names() -> list[str]:
        """Names of features in the feature vector."""
        d = StructuralMetrics().to_dict()
        exclude = {'prompt', 'graph_id', 'interpretability_label'}
        return [k for k in d.keys() if k not in exclude]


def attribution_to_networkx(graph: AttributionGraph) -> nx.DiGraph:
    """Convert an AttributionGraph to a NetworkX DiGraph."""
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            layer=node.layer,
            feature_idx=node.feature_idx,
            activation=node.activation,
            node_type=node.node_type,
            label=node.label,
        )

    for edge in graph.edges:
        if edge.source_id in G and edge.target_id in G:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.weight,
            )

    return G


def compute_basic_stats(G: nx.DiGraph) -> dict:
    """Compute basic graph statistics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    return {'n_nodes': n, 'n_edges': m, 'density': density}


def compute_degree_stats(G: nx.DiGraph) -> dict:
    """Compute degree distribution statistics."""
    if G.number_of_nodes() == 0:
        return {
            'mean_in_degree': 0, 'std_in_degree': 0, 'max_in_degree': 0,
            'mean_out_degree': 0, 'std_out_degree': 0, 'max_out_degree': 0,
            'degree_assortativity': 0,
        }

    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])

    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except (nx.NetworkXError, ZeroDivisionError):
        assortativity = 0.0

    return {
        'mean_in_degree': float(np.mean(in_degrees)),
        'std_in_degree': float(np.std(in_degrees)),
        'max_in_degree': int(np.max(in_degrees)) if len(in_degrees) > 0 else 0,
        'mean_out_degree': float(np.mean(out_degrees)),
        'std_out_degree': float(np.std(out_degrees)),
        'max_out_degree': int(np.max(out_degrees)) if len(out_degrees) > 0 else 0,
        'degree_assortativity': float(assortativity) if not np.isnan(assortativity) else 0.0,
    }


def compute_clustering(G: nx.DiGraph) -> dict:
    """Compute clustering coefficients."""
    G_undirected = G.to_undirected()
    avg_clustering = nx.average_clustering(G_undirected) if G.number_of_nodes() > 2 else 0.0
    transitivity = nx.transitivity(G_undirected)
    return {
        'avg_clustering': float(avg_clustering),
        'transitivity': float(transitivity),
    }


def compute_cycle_stats(G: nx.DiGraph, max_cycles: int = 1000) -> dict:
    """
    Estimate cycle density by sampling simple cycles.
    Full cycle enumeration is NP-hard, so we sample up to max_cycles.
    """
    n_cycles = 0
    try:
        for cycle in nx.simple_cycles(G):
            n_cycles += 1
            if n_cycles >= max_cycles:
                break
    except Exception:
        pass

    m = G.number_of_edges()
    cycle_density = n_cycles / m if m > 0 else 0.0

    return {
        'n_simple_cycles_sampled': n_cycles,
        'cycle_density': float(cycle_density),
    }


def compute_path_stats(G: nx.DiGraph) -> dict:
    """Compute path length statistics on the largest weakly connected component."""
    if G.number_of_nodes() == 0:
        return {
            'avg_shortest_path': 0, 'diameter': 0,
            'n_weakly_connected_components': 0, 'largest_wcc_fraction': 0,
        }

    wccs = list(nx.weakly_connected_components(G))
    n_wcc = len(wccs)
    largest_wcc = max(wccs, key=len)
    largest_wcc_fraction = len(largest_wcc) / G.number_of_nodes()

    subgraph = G.subgraph(largest_wcc)
    G_und = subgraph.to_undirected()

    try:
        avg_path = nx.average_shortest_path_length(G_und)
        diameter = nx.diameter(G_und)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        avg_path = 0.0
        diameter = 0

    return {
        'avg_shortest_path': float(avg_path),
        'diameter': int(diameter),
        'n_weakly_connected_components': n_wcc,
        'largest_wcc_fraction': float(largest_wcc_fraction),
    }


def compute_modularity(G: nx.DiGraph) -> dict:
    """Compute modularity using greedy community detection."""
    if G.number_of_nodes() < 3:
        return {'modularity': 0.0, 'n_communities': 1}

    G_und = G.to_undirected()
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G_und))
        modularity = nx.algorithms.community.modularity(G_und, communities)
        return {
            'modularity': float(modularity),
            'n_communities': len(communities),
        }
    except Exception:
        return {'modularity': 0.0, 'n_communities': 1}


def compute_spectral_properties(G: nx.DiGraph) -> dict:
    """Compute spectral properties of the adjacency and Laplacian matrices."""
    if G.number_of_nodes() < 3:
        return {
            'spectral_gap': 0.0,
            'algebraic_connectivity': 0.0,
            'spectral_radius': 0.0,
        }

    G_und = G.to_undirected()

    try:
        A = nx.adjacency_matrix(G_und, dtype=float)
        n = A.shape[0]
        k = min(5, n - 2)

        if k < 1:
            return {
                'spectral_gap': 0.0,
                'algebraic_connectivity': 0.0,
                'spectral_radius': 0.0,
            }

        # Adjacency eigenvalues (largest)
        eigenvalues_A = eigsh(A.astype(float), k=k, which='LM', return_eigenvectors=False)
        eigenvalues_A = np.sort(eigenvalues_A)[::-1]
        spectral_radius = float(eigenvalues_A[0])
        spectral_gap = float(eigenvalues_A[0] - eigenvalues_A[1]) if len(eigenvalues_A) > 1 else 0.0

        # Laplacian eigenvalues (smallest)
        L = nx.laplacian_matrix(G_und, dtype=float).astype(float)
        eigenvalues_L = eigsh(L, k=min(3, n - 2), which='SM', return_eigenvectors=False)
        eigenvalues_L = np.sort(eigenvalues_L)
        algebraic_connectivity = float(eigenvalues_L[1]) if len(eigenvalues_L) > 1 else 0.0

    except Exception:
        spectral_gap = 0.0
        algebraic_connectivity = 0.0
        spectral_radius = 0.0

    return {
        'spectral_gap': spectral_gap,
        'algebraic_connectivity': max(0.0, algebraic_connectivity),
        'spectral_radius': spectral_radius,
    }


def compute_layer_structure(G: nx.DiGraph) -> dict:
    """Analyze layer-wise connectivity patterns."""
    if G.number_of_nodes() == 0:
        return {
            'n_layers': 0, 'cross_layer_edge_ratio': 0,
            'backward_edge_ratio': 0, 'layer_entropy': 0,
        }

    layers = nx.get_node_attributes(G, 'layer')
    if not layers:
        return {
            'n_layers': 0, 'cross_layer_edge_ratio': 0,
            'backward_edge_ratio': 0, 'layer_entropy': 0,
        }

    unique_layers = set(layers.values())
    n_layers = len(unique_layers)

    # Count edge types
    n_cross = 0
    n_backward = 0
    n_total = G.number_of_edges()

    for u, v in G.edges():
        if u in layers and v in layers:
            layer_diff = abs(layers[v] - layers[u])
            if layer_diff > 1:
                n_cross += 1
            if layers[u] > layers[v]:
                n_backward += 1

    cross_ratio = n_cross / n_total if n_total > 0 else 0.0
    backward_ratio = n_backward / n_total if n_total > 0 else 0.0

    # Layer entropy (how evenly are nodes distributed across layers?)
    layer_counts = np.array([
        sum(1 for v in layers.values() if v == l)
        for l in unique_layers
    ], dtype=float)
    layer_probs = layer_counts / layer_counts.sum()
    layer_entropy = float(-np.sum(layer_probs * np.log(layer_probs + 1e-10)))

    return {
        'n_layers': n_layers,
        'cross_layer_edge_ratio': float(cross_ratio),
        'backward_edge_ratio': float(backward_ratio),
        'layer_entropy': float(layer_entropy),
    }


def compute_hierarchy_stats(G: nx.DiGraph) -> dict:
    """Compute hierarchy and tree-likeness metrics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n <= 1 or m == 0:
        return {'dag_depth': 0, 'tree_likeness': 0}

    # Tree-likeness: a perfect tree has exactly n-1 edges
    tree_likeness = (n - 1) / m if m > 0 else 0.0

    # DAG depth: longest path in the graph (if DAG) or approximate
    try:
        dag_depth = nx.dag_longest_path_length(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        # Graph has cycles — approximate via longest shortest path from sources
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not sources:
            sources = list(G.nodes())[:5]
        max_depth = 0
        for s in sources[:10]:  # limit for performance
            lengths = nx.single_source_shortest_path_length(G, s)
            if lengths:
                max_depth = max(max_depth, max(lengths.values()))
        dag_depth = max_depth

    return {
        'dag_depth': float(dag_depth),
        'tree_likeness': float(min(1.0, tree_likeness)),
    }


def compute_activation_stats(graph: AttributionGraph) -> dict:
    """Compute statistics over node activations and edge weights."""
    activations = np.array([n.activation for n in graph.nodes]) if graph.nodes else np.array([0.0])
    weights = np.array([e.weight for e in graph.edges]) if graph.edges else np.array([0.0])

    # Weight entropy
    if len(weights) > 0 and weights.sum() > 0:
        w_probs = np.abs(weights) / np.abs(weights).sum()
        weight_entropy = float(-np.sum(w_probs * np.log(w_probs + 1e-10)))
    else:
        weight_entropy = 0.0

    return {
        'mean_activation': float(np.mean(activations)),
        'std_activation': float(np.std(activations)),
        'mean_edge_weight': float(np.mean(weights)),
        'std_edge_weight': float(np.std(weights)),
        'weight_entropy': float(weight_entropy),
    }


def compute_all_metrics(
    graph: AttributionGraph,
    graph_id: str = "",
    max_cycles: int = 1000,
) -> StructuralMetrics:
    """
    Compute all structural metrics for a single attribution graph.

    This is the main entry point for the data mining stage.
    """
    G = attribution_to_networkx(graph)

    metrics = StructuralMetrics(
        prompt=graph.prompt[:100],
        graph_id=graph_id,
    )

    # Compute all metric groups
    results = {}
    results.update(compute_basic_stats(G))
    results.update(compute_degree_stats(G))
    results.update(compute_clustering(G))
    results.update(compute_cycle_stats(G, max_cycles=max_cycles))
    results.update(compute_path_stats(G))
    results.update(compute_modularity(G))
    results.update(compute_spectral_properties(G))
    results.update(compute_layer_structure(G))
    results.update(compute_hierarchy_stats(G))
    results.update(compute_activation_stats(graph))

    # Set label if available
    if 'label' in graph.metadata:
        results['interpretability_label'] = float(graph.metadata['label'])

    # Update metrics dataclass
    for key, value in results.items():
        if hasattr(metrics, key):
            setattr(metrics, key, value)

    return metrics


def compute_metrics_batch(
    graphs: list[AttributionGraph],
) -> list[StructuralMetrics]:
    """Compute structural metrics for a batch of graphs."""
    all_metrics = []
    for i, graph in enumerate(graphs):
        metrics = compute_all_metrics(graph, graph_id=f"graph_{i:04d}")
        all_metrics.append(metrics)
        if (i + 1) % 50 == 0:
            print(f"  Computed metrics for {i+1}/{len(graphs)} graphs")
    return all_metrics
