"""Tests for dataloader.py — TDD RED phase."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_pt(path: Path, idx: int) -> None:
    """Write a tiny graph dict .pt file."""
    N, E = 4, 12
    g: dict[str, torch.Tensor] = {
        "x": torch.randn(N, 4) + idx,
        "edge_index": torch.zeros(2, E, dtype=torch.long),
        "edge_attr": torch.randn(E, 4),
        "y": torch.randn(N, 3),
        "target_stress": torch.randn(N, 1),
        "mesh_pos": torch.randn(N, 3),
    }
    torch.save(g, path)


def _make_split_cache(processed_dir: Path, split: str, n: int = 3) -> None:
    split_dir = processed_dir / split
    split_dir.mkdir(parents=True)
    for i in range(n):
        _make_graph_pt(split_dir / f"{i:06d}.pt", i)


# ---------------------------------------------------------------------------
# T8: GraphDataset
# ---------------------------------------------------------------------------


class TestGraphDataset:
    def test_len(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=5)
        ds = GraphDataset(tmp_path, "train")
        assert len(ds) == 5

    def test_getitem_returns_dict(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=3)
        ds = GraphDataset(tmp_path, "train")
        item = ds[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=3)
        ds = GraphDataset(tmp_path, "train")
        item = ds[0]
        for key in ("x", "edge_index", "edge_attr", "y", "target_stress", "mesh_pos"):
            assert key in item, f"missing key {key}"

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=3)
        ds = GraphDataset(tmp_path, "train")
        item = ds[0]
        assert item["x"].shape == (4, 4)
        assert item["edge_index"].shape == (2, 12)
        assert item["y"].shape == (4, 3)

    def test_getitem_returns_tensors(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=3)
        ds = GraphDataset(tmp_path, "train")
        item = ds[0]
        for k, v in item.items():
            assert isinstance(v, torch.Tensor), f"{k} not Tensor"

    def test_different_indices_different_data(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=3)
        ds = GraphDataset(tmp_path, "train")
        # x values differ (we add idx offset in _make_graph_pt)
        item0 = ds[0]
        item2 = ds[2]
        assert not torch.equal(item0["x"], item2["x"])

    def test_empty_split_raises(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=0)
        ds = GraphDataset(tmp_path, "train")
        assert len(ds) == 0

    def test_index_out_of_range(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.dataloader import GraphDataset

        _make_split_cache(tmp_path, "train", n=2)
        ds = GraphDataset(tmp_path, "train")
        with pytest.raises(IndexError):
            _ = ds[99]


# ---------------------------------------------------------------------------
# T8: get_dataloaders
# ---------------------------------------------------------------------------


class TestGetDataloaders:
    def test_returns_dict_of_dataloaders(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.dataloader import get_dataloaders

        cfg = DatasetConfig(
            processed_dir=tmp_path,
            splits=["train", "valid"],
        )
        for split in ["train", "valid"]:
            _make_split_cache(tmp_path, split, n=2)

        loaders = get_dataloaders(cfg, batch_size=1)
        assert isinstance(loaders, dict)
        for split in ["train", "valid"]:
            assert split in loaders
            assert isinstance(loaders[split], DataLoader)

    def test_batch_size_one(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.dataloader import get_dataloaders

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train"])
        _make_split_cache(tmp_path, "train", n=3)
        loaders = get_dataloaders(cfg, batch_size=1)
        batch = next(iter(loaders["train"]))
        # batch_size=1: x should be [1, N, 4] or [N, 4] depending on collate
        # default collate stacks → [1, 4, 4]
        assert batch["x"].shape[0] == 1

    def test_all_splits_covered(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.dataloader import get_dataloaders

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train", "valid", "test"])
        for split in ["train", "valid", "test"]:
            _make_split_cache(tmp_path, split, n=2)

        loaders = get_dataloaders(cfg)
        assert set(loaders.keys()) == {"train", "valid", "test"}

    def test_num_workers_param(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.dataloader import get_dataloaders

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train"])
        _make_split_cache(tmp_path, "train", n=2)
        loaders = get_dataloaders(cfg, num_workers=0)
        assert loaders["train"].num_workers == 0
