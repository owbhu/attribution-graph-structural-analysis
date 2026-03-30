# Structural Signatures of Interpretability Failure

**Mining Attribution Graphs to Detect Superposition in Neural Networks**

## Overview

This project applies graph mining and graph neural networks to Anthropic's [attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — computational graphs that reveal how language models process information internally.

Attribution graphs are a core tool in mechanistic interpretability, but they only yield satisfying interpretations ~25% of the time. We investigate whether the **structural properties** of an attribution graph can predict whether it will be interpretable, and whether structural signatures can detect features in [superposition](https://transformer-circuits.pub/2022/toy_model/index.html).

### Key Question

> Do interpretable attribution graphs have measurably different structural properties than uninterpretable ones? If so, can a GNN learn to detect these patterns automatically?

## Project Structure

```
├── src/
│   ├── graph_generator.py        # Generate attribution graphs via circuit-tracer
│   ├── structural_metrics.py     # 30+ graph-theoretic structural metrics
│   ├── dataset.py                # PyG dataset for attribution graphs
│   ├── model.py                  # StructuralMLP baseline + AttributionGNN
│   ├── train.py                  # Training pipeline with early stopping
│   └── utils.py                  # Visualization and analysis utilities
├── notebooks/
│   ├── 01_graph_generation.ipynb      # Generate synthetic + real attribution graphs
│   ├── 02_structural_analysis.ipynb   # Compute metrics and analyze distributions
│   ├── 03_data_mining.ipynb           # Homophily, community detection, superposition scoring
│   └── 04_gnn_training.ipynb          # Train and compare MLP vs GNN models
├── data/
│   ├── raw/                      # Raw attribution graph JSON files
│   ├── processed/                # PyG-processed datasets
│   └── prompts/                  # Curated prompt sets by task category
├── configs/
│   └── default.yaml              # Experiment configuration
└── results/
    ├── figures/                  # All generated plots
    └── metrics/                  # Training logs and metrics CSVs
```

## Methods

### Data Mining Stage

We compute 30+ structural metrics on each attribution graph:
- **Degree statistics** — in/out degree, assortativity
- **Clustering** — clustering coefficient, transitivity
- **Cycle density** — sampled simple cycles per edge
- **Path structure** — avg shortest path, diameter, connected components
- **Modularity** — greedy community detection
- **Spectral properties** — spectral gap, algebraic connectivity
- **Layer structure** — cross-layer edge ratio, backward edges, layer entropy
- **Hierarchy** — tree-likeness, DAG depth
- **Homophily** — layer homophily, activation homophily

We then analyze which metrics discriminate interpretable from uninterpretable circuits.

### Machine Learning Stage

Three models compared:
1. **StructuralMLP** — Baseline MLP on hand-crafted structural feature vectors
2. **AttributionGNN** — GAT-based GNN learning directly from graph topology
3. **AttributionGNN + Structural Fusion** — GNN with structural feature injection

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/attribution-graph-structural-analysis.git
cd attribution-graph-structural-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_graph_generation.ipynb
```

### Requirements

- Python 3.10+
- PyTorch 2.1+
- PyTorch Geometric 2.4+
- circuit-tracer (Anthropic's open-source library)
- ~8GB RAM for Gemma-2-2B (runs on Apple M-series via MPS)

## Motivation

Anthropic's circuit tracing research ([March 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)) represents a major advance in mechanistic interpretability. However, the authors note that attribution graphs only provide satisfying insight for about a quarter of prompts tested. The primary suspected cause is **superposition** — when features are encoded as non-orthogonal directions in activation space, linear attribution methods give misleading results.

This project asks: **can we detect when attribution graphs are likely to fail by analyzing their structure?** If structural signatures of superposition exist, they could serve as a triage tool — helping researchers focus on regions of the model where current methods work, and developing targeted approaches for regions where they don't.

## References

- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — Anthropic, 2025
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) — Anthropic, 2025
- [Open-sourcing circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing) — Anthropic, 2025
- [The Urgency of Interpretability](https://www.darioamodei.com/post/the-urgency-of-interpretability) — Dario Amodei, 2025
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Anthropic, 2022

## License

MIT
