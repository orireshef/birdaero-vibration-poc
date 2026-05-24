"""Training loop for MeshGraphNet."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from vibration_poc.dataset.config import DatasetConfig
from vibration_poc.dataset.dataloader import get_dataloaders
from vibration_poc.model.meshgraphnet import MeshGraphNet

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TrainingConfig(BaseModel):
    epochs: int = Field(default=5, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_decay: float = Field(default=0.9999991, gt=0, le=1)
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=8, gt=0)
    batch_size: int = Field(default=1, gt=0)
    num_workers: int = Field(default=0, ge=0)
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = Field(default=100, gt=0)


def train_epoch(
    model: MeshGraphNet,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        graph = {k: v.to(device) for k, v in batch.items()}
        pred = model(graph)
        loss = F.mse_loss(pred, graph["y"])
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def train(config: TrainingConfig, dataset_config: DatasetConfig) -> Path:
    """Full training loop. Returns path to best checkpoint."""
    device = torch.device(config.device)
    model = MeshGraphNet(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    loaders = get_dataloaders(dataset_config, batch_size=config.batch_size)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = config.checkpoint_dir / "best.pt"

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        scheduler.step()
        if epoch % config.log_interval == 0:
            print(f"epoch={epoch} loss={train_loss:.6f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_path)

    return best_path
