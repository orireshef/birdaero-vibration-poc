"""DataLoader for grid .pt files (FNO training)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vibration_poc.dataset.config import DatasetConfig, GridConfig


class GridDataset(Dataset[dict[str, Tensor]]):
    """Loads cached grid .pt files from disk."""

    def __init__(self, processed_dir: Path, split: str) -> None:
        grid_dir = processed_dir / "grid" / split
        self._files = sorted(grid_dir.glob("*.pt"))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self._files):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self._files)}")
        return torch.load(self._files[idx], weights_only=True)  # type: ignore[no-any-return]


def get_grid_dataloaders(
    config: DatasetConfig,
    grid_config: GridConfig,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict[str, DataLoader[dict[str, Tensor]]]:
    """Create batched DataLoaders for grid data."""
    loaders: dict[str, DataLoader[dict[str, Tensor]]] = {}
    for split in config.splits:
        ds = GridDataset(config.processed_dir, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
        )
    return loaders
