"""
Model
=====

GNN architectures for predicting interpretability from attribution graph structure.

Two model variants:
1. StructuralMLP: Baseline MLP on hand-crafted structural features
2. AttributionGNN: Graph Neural Network that learns directly from graph topology

The GNN uses message-passing to propagate local structural signatures.
The key insight: a node surrounded by contradictory-context neighbors
(high cross-layer connectivity, high local cycle density) is likely in
superposition. The GNN learns to detect these patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv,
        GATConv,
        SAGEConv,
        global_mean_pool,
        global_max_pool,
        global_add_pool,
    )
    from torch_geometric.data import Data, Batch
except ImportError:
    raise ImportError("torch-geometric required. Install with: pip install torch-geometric")


class StructuralMLP(nn.Module):
    """
    Baseline: MLP on precomputed structural metrics.
    Takes the structural feature vector and predicts interpretability score.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: list[int] = [128, 64, 32],
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, structural_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            structural_features: [batch_size, num_structural_features]
        Returns:
            predictions: [batch_size, 1] interpretability scores
        """
        return self.network(structural_features)


class AttributionGNN(nn.Module):
    """
    Graph Neural Network for predicting interpretability from attribution graphs.

    Architecture:
    1. Node-level GNN layers (message passing)
    2. Graph-level pooling (mean + max)
    3. Optional fusion with structural features
    4. MLP head for prediction

    The message-passing layers learn to detect local structural patterns
    that correlate with interpretability success or failure.
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gnn_type: str = "GAT",  # 'GCN', 'GAT', 'SAGE'
        num_heads: int = 4,
        structural_feature_dim: int = 0,  # 0 = no structural feature fusion
        dropout: float = 0.3,
        pool_type: str = "mean_max",  # 'mean', 'max', 'add', 'mean_max'
    ):
        super().__init__()

        self.num_gnn_layers = num_gnn_layers
        self.pool_type = pool_type
        self.dropout = dropout

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            if gnn_type == "GAT":
                # GAT concatenates heads, so output is hidden_dim
                head_dim = hidden_dim // num_heads
                self.gnn_layers.append(
                    GATConv(in_dim, head_dim, heads=num_heads, dropout=dropout)
                )
            elif gnn_type == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.gnn_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling output dimension
        if pool_type == "mean_max":
            pool_dim = hidden_dim * 2
        else:
            pool_dim = hidden_dim

        # Optional structural feature fusion
        self.use_structural = structural_feature_dim > 0
        if self.use_structural:
            self.structural_encoder = nn.Sequential(
                nn.Linear(structural_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            mlp_input_dim = pool_dim + hidden_dim
        else:
            mlp_input_dim = pool_dim

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor = None,
        structural_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes_total, node_feature_dim]
            edge_index: Edge indices [2, num_edges_total]
            batch: Batch assignment [num_nodes_total]
            edge_attr: Edge weights [num_edges_total, 1] (optional)
            structural_features: Graph-level structural metrics [batch_size, structural_dim]

        Returns:
            predictions: [batch_size, 1]
        """
        # Message passing
        h = x
        for i in range(self.num_gnn_layers):
            h = self.gnn_layers[i](h, edge_index)
            h = self.gnn_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph-level pooling
        if self.pool_type == "mean":
            graph_embed = global_mean_pool(h, batch)
        elif self.pool_type == "max":
            graph_embed = global_max_pool(h, batch)
        elif self.pool_type == "add":
            graph_embed = global_add_pool(h, batch)
        elif self.pool_type == "mean_max":
            graph_embed = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
            ], dim=-1)

        # Fuse with structural features if available
        if self.use_structural and structural_features is not None:
            struct_embed = self.structural_encoder(structural_features)
            graph_embed = torch.cat([graph_embed, struct_embed], dim=-1)

        # Predict
        out = self.prediction_head(graph_embed)
        return out

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get node-level embeddings (for analyzing which nodes
        the model thinks are in superposition).
        """
        h = x
        for i in range(self.num_gnn_layers):
            h = self.gnn_layers[i](h, edge_index)
            h = self.gnn_norms[i](h)
            h = F.relu(h)
        return h


class NodeSuperpositionDetector(nn.Module):
    """
    Node-level classifier that predicts whether individual features
    in an attribution graph are in superposition.

    Uses the same GNN backbone but adds a node-level prediction head
    instead of graph-level pooling.
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gnn_type: str = "GAT",
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout

        # GNN layers (same as AttributionGNN)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            if gnn_type == "GAT":
                head_dim = hidden_dim // num_heads
                self.gnn_layers.append(
                    GATConv(in_dim, head_dim, heads=num_heads, dropout=dropout)
                )
            elif gnn_type == "GCN":
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == "SAGE":
                self.gnn_layers.append(SAGEConv(in_dim, hidden_dim))

            self.gnn_norms.append(nn.BatchNorm1d(hidden_dim))

        # Node-level prediction head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            node_predictions: [num_nodes, 1] superposition scores per node
        """
        h = x
        for i in range(self.num_gnn_layers):
            h = self.gnn_layers[i](h, edge_index)
            h = self.gnn_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.node_head(h)
