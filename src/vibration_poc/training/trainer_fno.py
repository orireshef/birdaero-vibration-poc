"""Training loop for FNO3d on grid data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from vibration_poc.dataset.config import DatasetConfig, GridConfig, GridNormStats
from vibration_poc.dataset.grid_dataloader import get_grid_dataloaders
from vibration_poc.model.fno import FNO3d
from vibration_poc.physics import (
    PhysicsConfig,
    compute_grid_bc_penalty,
    compute_grid_smoothness_loss,
    compute_masked_mse,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class FNOTrainingConfig(BaseModel):
    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=4, gt=0)
    batch_size: int = Field(default=8, gt=0)
    num_workers: int = Field(default=0, ge=0)
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints/fno")
    log_interval: int = Field(default=1, gt=0)
    grid: GridConfig = GridConfig()
    physics: PhysicsConfig = PhysicsConfig()


def _make_norm_tensors(
    stats: GridNormStats, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create [1, C, 1, 1, 1] normalization tensors for batched grid data."""

    def _t(vals: list[float]) -> Tensor:
        return torch.tensor(vals, dtype=torch.float32, device=device).reshape(1, -1, 1, 1, 1)

    return (
        _t(stats.channel_mean),
        _t(stats.channel_std),
        _t(stats.target_mean),
        _t(stats.target_std),
    )


def train_epoch_fno(
    model: FNO3d,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    physics: PhysicsConfig | None = None,
    norm: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> float:
    """Train one epoch on grid data. Returns mean loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        grid_input = batch["grid_input"].to(device)
        grid_target = batch["grid_target"].to(device)
        mask = batch["occupancy_mask"].to(device)

        if norm is not None:
            ch_mean, ch_std, tgt_mean, tgt_std = norm
            grid_input = (grid_input - ch_mean) / ch_std
            grid_target = (grid_target - tgt_mean) / tgt_std

        pred = model(grid_input)
        loss = compute_masked_mse(pred, grid_target, mask)

        if physics is not None:
            if physics.smoothness_weight > 0:
                loss = loss + physics.smoothness_weight * compute_grid_smoothness_loss(pred, mask)

            if physics.bc_loss_weight > 0:
                if norm is not None:
                    raw_input = batch["grid_input"].to(device)
                    node_type_grid = raw_input[:, 6:7, :, :, :]
                else:
                    node_type_grid = grid_input[:, 6:7, :, :, :]
                bc_mask = (node_type_grid > 0.5).float()
                loss = loss + physics.bc_loss_weight * compute_grid_bc_penalty(pred, bc_mask)

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def train_fno(config: FNOTrainingConfig, dataset_config: DatasetConfig) -> Path:
    """Full FNO training loop. Returns path to best checkpoint."""
    device = torch.device(config.device)
    model = FNO3d(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        modes=config.grid.modes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    loaders = get_grid_dataloaders(
        dataset_config,
        config.grid,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    stats_path = dataset_config.processed_dir / "grid" / "grid_norm_stats.json"
    norm: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    if stats_path.exists():
        with open(stats_path) as f:
            grid_stats = GridNormStats(**json.load(f))
        norm = _make_norm_tensors(grid_stats, device)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = config.checkpoint_dir / "best_fno.pt"

    for epoch in range(config.epochs):
        train_loss = train_epoch_fno(
            model,
            loaders["train"],
            optimizer,
            device,
            physics=config.physics,
            norm=norm,
        )
        scheduler.step()
        if epoch % config.log_interval == 0:
            print(f"epoch={epoch} loss={train_loss:.6f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_path)

    return best_path
