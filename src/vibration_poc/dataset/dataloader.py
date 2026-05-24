"""DataLoader wrapper for cached graph .pt files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vibration_poc.dataset.config import DatasetConfig


class GraphDataset(Dataset[dict[str, Tensor]]):
    """Loads cached .pt graph dicts from disk."""

    def __init__(self, processed_dir: Path, split: str) -> None:
        self._files = sorted((processed_dir / split).glob("*.pt"))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self._files):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self._files)}")
        return torch.load(self._files[idx], weights_only=True)  # type: ignore[no-any-return]


def _graph_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate for single-graph batches — returns the graph dict without added batch dim."""
    return batch[0]


def get_dataloaders(
    config: DatasetConfig,
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, DataLoader[dict[str, Tensor]]]:
    """Create train/val/test DataLoaders from cached processed data."""
    loaders: dict[str, DataLoader[dict[str, Tensor]]] = {}
    for split in config.splits:
        ds = GraphDataset(config.processed_dir, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
            collate_fn=_graph_collate,
        )
    return loaders
