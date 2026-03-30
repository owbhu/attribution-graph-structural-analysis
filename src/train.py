"""
Training Pipeline
=================

Training loop, evaluation, and experiment logging for both
the StructuralMLP baseline and AttributionGNN models.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PyGDataLoader

from .model import StructuralMLP, AttributionGNN


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    model_type: str = "gnn"  # 'mlp' or 'gnn'
    gnn_type: str = "GAT"
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.3
    pool_type: str = "mean_max"
    use_structural_fusion: bool = True

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15  # early stopping patience
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    # Logging
    log_dir: str = "results/metrics"
    save_best: bool = True

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    train_r2: float
    val_r2: float
    lr: float
    time_seconds: float


class Trainer:
    """
    Training pipeline for interpretability prediction models.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
        self.history: list[EpochMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def create_model(
        self,
        node_feature_dim: int = 7,
        structural_feature_dim: int = 30,
    ) -> nn.Module:
        """Create the model based on config."""
        if self.config.model_type == "mlp":
            model = StructuralMLP(
                input_dim=structural_feature_dim,
                hidden_dims=[128, 64, 32],
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "gnn":
            struct_dim = structural_feature_dim if self.config.use_structural_fusion else 0
            model = AttributionGNN(
                node_feature_dim=node_feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_gnn_layers=self.config.num_gnn_layers,
                gnn_type=self.config.gnn_type,
                num_heads=self.config.num_heads,
                structural_feature_dim=struct_dim,
                dropout=self.config.dropout,
                pool_type=self.config.pool_type,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        return model.to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        loader: PyGDataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> dict:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            # PyG concatenates graph-level attrs into 1D; reshape to (batch, feat_dim)
            struct_feat = self._reshape_structural(batch)

            if self.config.model_type == "mlp":
                preds = model(struct_feat)
            else:
                preds = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    structural_features=(
                        struct_feat
                        if self.config.use_structural_fusion
                        else None
                    ),
                )

            targets = batch.y.view(-1, 1)
            loss = criterion(preds, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return {
            'loss': total_loss / len(loader.dataset),
            'mae': float(np.mean(np.abs(all_preds - all_targets))),
            'r2': float(r2_score(all_targets, all_preds)),
        }

    def _reshape_structural(self, batch):
        """Reshape structural_features from PyG's flat concatenation to (batch_size, feat_dim)."""
        sf = batch.structural_features
        if sf.dim() == 1:
            # PyG concatenated all graph-level features into one flat tensor
            n_graphs = batch.num_graphs
            feat_dim = sf.shape[0] // n_graphs
            sf = sf.view(n_graphs, feat_dim)
        return sf

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: PyGDataLoader,
        criterion: nn.Module,
    ) -> dict:
        """Evaluate model on a dataset."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in loader:
            batch = batch.to(self.device)

            struct_feat = self._reshape_structural(batch)

            if self.config.model_type == "mlp":
                preds = model(struct_feat)
            else:
                preds = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    structural_features=(
                        struct_feat
                        if self.config.use_structural_fusion
                        else None
                    ),
                )

            targets = batch.y.view(-1, 1)
            loss = criterion(preds, targets)

            total_loss += loss.item() * batch.num_graphs
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return {
            'loss': total_loss / len(loader.dataset),
            'mae': float(np.mean(np.abs(all_preds - all_targets))),
            'r2': float(r2_score(all_targets, all_preds)),
        }

    def train(
        self,
        model: nn.Module,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
    ) -> nn.Module:
        """Full training loop with early stopping."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )
        criterion = nn.MSELoss()

        os.makedirs(self.config.log_dir, exist_ok=True)
        best_model_path = os.path.join(self.config.log_dir, "best_model.pt")

        print(f"Training on {self.device}")
        print(f"Model: {self.config.model_type}, Epochs: {self.config.num_epochs}")
        print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}")
        print("-" * 70)

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)

            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_mae=train_metrics['mae'],
                val_mae=val_metrics['mae'],
                train_r2=train_metrics['r2'],
                val_r2=val_metrics['r2'],
                lr=current_lr,
                time_seconds=elapsed,
            )
            self.history.append(epoch_metrics)

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"Val R2: {val_metrics['r2']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"{elapsed:.1f}s"
                )

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                if self.config.save_best:
                    torch.save(model.state_dict(), best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Load best model
        if self.config.save_best and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            print(f"Loaded best model (val_loss={self.best_val_loss:.4f})")

        # Save training history
        self.save_history()

        return model

    def save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.config.log_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump([asdict(m) for m in self.history], f, indent=2)
        print(f"Training history saved to {path}")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)
