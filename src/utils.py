"""
Utilities
=========

Visualization, logging, and helper functions.
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from .graph_generator import AttributionGraph
from .structural_metrics import StructuralMetrics, attribution_to_networkx


# ──────────────────────────────────────────────────────────────
# Attribution Graph Visualization
# ──────────────────────────────────────────────────────────────

def visualize_attribution_graph(
    graph: AttributionGraph,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    node_size_scale: float = 300,
    edge_width_scale: float = 2.0,
):
    """
    Visualize an attribution graph with nodes colored by layer
    and sized by activation magnitude.
    """
    G = attribution_to_networkx(graph)
    if G.number_of_nodes() == 0:
        print("Empty graph, nothing to visualize.")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Layout: hierarchical by layer
    layers = nx.get_node_attributes(G, 'layer')
    if layers:
        pos = _hierarchical_layout(G, layers)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors by layer
    layer_values = [layers.get(n, 0) for n in G.nodes()]
    max_layer = max(layer_values) if layer_values else 1

    # Node sizes by activation
    activations = [abs(G.nodes[n].get('activation', 0.5)) for n in G.nodes()]
    node_sizes = [a * node_size_scale + 50 for a in activations]

    # Edge widths by weight
    edge_weights = [abs(G.edges[e].get('weight', 0.1)) for e in G.edges()]
    edge_widths = [w * edge_width_scale + 0.1 for w in edge_weights]

    # Draw
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=layer_values,
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        vmin=0, vmax=max_layer,
        alpha=0.8,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.4,
        edge_color='gray',
        arrows=True,
        arrowsize=10,
    )

    plt.colorbar(nodes, ax=ax, label='Layer', shrink=0.8)
    ax.set_title(title or f"Attribution Graph ({G.number_of_nodes()} nodes)")
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.close()


def _hierarchical_layout(G: nx.DiGraph, layers: dict) -> dict:
    """Position nodes in a hierarchical layout based on layer attribute."""
    layer_groups = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    pos = {}
    for layer, nodes in sorted(layer_groups.items()):
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - n / 2) * 1.5
            y = -layer * 2  # layers go top to bottom
            pos[node] = (x, y)

    return pos


# ──────────────────────────────────────────────────────────────
# Metrics Visualization
# ──────────────────────────────────────────────────────────────

def plot_metric_distributions(
    metrics_list: list[StructuralMetrics],
    save_dir: str = "results/figures",
    key_metrics: Optional[list[str]] = None,
):
    """
    Plot distributions of structural metrics, colored by interpretability label.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to dataframe
    df = pd.DataFrame([m.to_dict() for m in metrics_list])

    if key_metrics is None:
        key_metrics = [
            'density', 'avg_clustering', 'cycle_density', 'modularity',
            'tree_likeness', 'spectral_gap', 'cross_layer_edge_ratio',
            'backward_edge_ratio', 'degree_assortativity', 'weight_entropy',
        ]

    # Filter to metrics that exist in the dataframe
    key_metrics = [m for m in key_metrics if m in df.columns]

    n_metrics = len(key_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    has_labels = 'interpretability_label' in df.columns and df['interpretability_label'].notna().any()

    for i, metric in enumerate(key_metrics):
        ax = axes[i]
        if has_labels:
            # Color by label
            for label_val, color, name in [
                (1.0, '#2ecc71', 'Interpretable'),
                (0.0, '#e74c3c', 'Uninterpretable'),
                (0.5, '#f39c12', 'Mixed'),
            ]:
                subset = df[df['interpretability_label'] == label_val]
                if len(subset) > 0:
                    ax.hist(subset[metric].dropna(), bins=20, alpha=0.6,
                            color=color, label=name, density=True)
            ax.legend(fontsize=8)
        else:
            ax.hist(df[metric].dropna(), bins=20, alpha=0.7, color='#3498db', density=True)

        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')

    # Hide unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Structural Metric Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metric distributions to {save_dir}/metric_distributions.png")


def plot_correlation_matrix(
    metrics_list: list[StructuralMetrics],
    save_dir: str = "results/figures",
):
    """Plot correlation matrix of structural metrics."""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {'n_nodes', 'n_edges'}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap='RdBu_r', center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        annot=False,
    )
    ax.set_title('Structural Metrics Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix to {save_dir}/correlation_matrix.png")


def plot_metric_vs_interpretability(
    metrics_list: list[StructuralMetrics],
    save_dir: str = "results/figures",
    top_n: int = 6,
):
    """
    Plot the top-N most correlated metrics against interpretability label.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame([m.to_dict() for m in metrics_list])

    if 'interpretability_label' not in df.columns or df['interpretability_label'].isna().all():
        print("No interpretability labels available for scatter plots.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {'interpretability_label', 'n_nodes', 'n_edges'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Find most correlated features
    correlations = df[feature_cols].corrwith(df['interpretability_label']).abs()
    top_features = correlations.nlargest(top_n).index.tolist()

    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        ax.scatter(
            df[feat], df['interpretability_label'],
            alpha=0.5, s=20, c='#3498db',
        )
        # Add trend line
        z = np.polyfit(df[feat].dropna(), df['interpretability_label'].dropna(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.7)

        corr_val = correlations[feat]
        ax.set_title(f'{feat}\n(|r| = {corr_val:.3f})', fontsize=10)
        ax.set_xlabel(feat)
        ax.set_ylabel('Interpretability')

    for i in range(top_n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Top Correlated Metrics vs Interpretability', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_vs_interpretability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plots to {save_dir}/metric_vs_interpretability.png")


# ──────────────────────────────────────────────────────────────
# Training Visualization
# ──────────────────────────────────────────────────────────────

def plot_training_history(
    history_path: str = "results/metrics/training_history.json",
    save_dir: str = "results/figures",
):
    """Plot training curves from saved history."""
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, [h['train_loss'] for h in history], label='Train', alpha=0.8)
    axes[0].plot(epochs, [h['val_loss'] for h in history], label='Val', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    # MAE
    axes[1].plot(epochs, [h['train_mae'] for h in history], label='Train', alpha=0.8)
    axes[1].plot(epochs, [h['val_mae'] for h in history], label='Val', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()

    # R2
    axes[2].plot(epochs, [h['train_r2'] for h in history], label='Train', alpha=0.8)
    axes[2].plot(epochs, [h['val_r2'] for h in history], label='Val', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('R2 Score')
    axes[2].set_title('R-Squared')
    axes[2].legend()

    plt.suptitle('Training History', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {save_dir}/training_history.png")


# ──────────────────────────────────────────────────────────────
# Summary / Reporting
# ──────────────────────────────────────────────────────────────

def generate_metrics_summary(
    metrics_list: list[StructuralMetrics],
) -> pd.DataFrame:
    """Generate a summary statistics table for all structural metrics."""
    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe().T
    return summary


def compare_graph_types(
    metrics_list: list[StructuralMetrics],
) -> pd.DataFrame:
    """
    Compare structural metrics across graph types (if labels exist).
    Returns a table showing mean values per graph type.
    """
    df = pd.DataFrame([m.to_dict() for m in metrics_list])

    if 'interpretability_label' not in df.columns:
        return df.describe().T

    # Map labels to readable names
    label_map = {1.0: 'Interpretable', 0.0: 'Uninterpretable', 0.5: 'Mixed'}
    df['type'] = df['interpretability_label'].map(label_map).fillna('Unknown')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'interpretability_label']

    comparison = df.groupby('type')[numeric_cols].mean().T
    return comparison
