"""Tests for grid interpolation and grid data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vibration_poc.dataset.grid_interpolation import GridToMeshInterpolator, MeshToGridInterpolator


def _make_mesh_positions(n: int = 8) -> np.ndarray:
    """Create simple 3D mesh positions in a cube."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (n, 3)).astype(np.float32)


class TestMeshToGridInterpolator:
    def test_output_shape(self) -> None:
        pos = _make_mesh_positions(8)
        interp = MeshToGridInterpolator(pos, resolution=(4, 4, 4))
        values = np.random.randn(8, 3).astype(np.float32)
        grid = interp.interpolate(values)
        assert grid.shape == (3, 4, 4, 4)

    def test_occupancy_mask_shape(self) -> None:
        pos = _make_mesh_positions(8)
        interp = MeshToGridInterpolator(pos, resolution=(4, 4, 4))
        mask = interp.occupancy_mask()
        assert mask.shape == (1, 4, 4, 4)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_grid_bounds_shape(self) -> None:
        pos = _make_mesh_positions(8)
        interp = MeshToGridInterpolator(pos, resolution=(4, 4, 4))
        bounds = interp.grid_bounds
        assert bounds.shape == (6,)

    def test_single_channel(self) -> None:
        pos = _make_mesh_positions(8)
        interp = MeshToGridInterpolator(pos, resolution=(3, 3, 3))
        values = np.ones((8, 1), dtype=np.float32)
        grid = interp.interpolate(values)
        assert grid.shape == (1, 3, 3, 3)


class TestGridToMeshInterpolator:
    def test_output_shape(self) -> None:
        bounds = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        interp = GridToMeshInterpolator(bounds, resolution=(4, 4, 4))
        grid = np.random.randn(3, 4, 4, 4).astype(np.float32)
        points = np.random.uniform(0.1, 0.9, (10, 3)).astype(np.float32)
        result = interp.interpolate(grid, points)
        assert result.shape == (10, 3)


class TestRoundTrip:
    def test_round_trip_approximate(self) -> None:
        rng = np.random.default_rng(42)
        pos = rng.uniform(0.1, 0.9, (20, 3)).astype(np.float32)
        values = rng.standard_normal((20, 3)).astype(np.float32)

        m2g = MeshToGridInterpolator(pos, resolution=(16, 16, 16), padding_ratio=0.1)
        grid = m2g.interpolate(values)

        g2m = GridToMeshInterpolator(m2g.grid_bounds, resolution=(16, 16, 16))
        recovered = g2m.interpolate(grid, pos)

        rel_error = np.linalg.norm(recovered - values) / (np.linalg.norm(values) + 1e-8)
        assert rel_error < 1.0


def _make_grid_pt(path: Path, resolution: tuple[int, int, int] = (3, 3, 3)) -> None:
    """Write a tiny grid .pt file."""
    gx, gy, gz = resolution
    torch.save(
        {
            "grid_input": torch.randn(7, gx, gy, gz),
            "grid_target": torch.randn(3, gx, gy, gz),
            "occupancy_mask": torch.ones(1, gx, gy, gz),
            "mesh_pos": torch.randn(4, 3),
            "node_type": torch.zeros(4),
            "grid_bounds": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        },
        path,
    )


class TestGridDataset:
    def test_len(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.grid_dataloader import GridDataset

        grid_dir = tmp_path / "grid" / "train"
        grid_dir.mkdir(parents=True)
        for i in range(3):
            _make_grid_pt(grid_dir / f"{i:06d}.pt")
        ds = GridDataset(tmp_path, "train")
        assert len(ds) == 3

    def test_getitem_keys(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.grid_dataloader import GridDataset

        grid_dir = tmp_path / "grid" / "train"
        grid_dir.mkdir(parents=True)
        _make_grid_pt(grid_dir / "000000.pt")
        ds = GridDataset(tmp_path, "train")
        item = ds[0]
        for key in (
            "grid_input",
            "grid_target",
            "occupancy_mask",
            "mesh_pos",
            "node_type",
            "grid_bounds",
        ):
            assert key in item

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.grid_dataloader import GridDataset

        grid_dir = tmp_path / "grid" / "train"
        grid_dir.mkdir(parents=True)
        _make_grid_pt(grid_dir / "000000.pt", resolution=(4, 6, 2))
        ds = GridDataset(tmp_path, "train")
        item = ds[0]
        assert item["grid_input"].shape == (7, 4, 6, 2)
        assert item["grid_target"].shape == (3, 4, 6, 2)

    def test_index_out_of_range(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.grid_dataloader import GridDataset

        grid_dir = tmp_path / "grid" / "train"
        grid_dir.mkdir(parents=True)
        _make_grid_pt(grid_dir / "000000.pt")
        ds = GridDataset(tmp_path, "train")
        with pytest.raises(IndexError):
            _ = ds[99]


class TestPreprocessGrid:
    def test_graph_to_grid(self) -> None:
        from vibration_poc.dataset.config import GridConfig
        from vibration_poc.dataset.preprocess_grid import _graph_to_grid

        n_nodes = 20
        rng = np.random.default_rng(42)
        mesh_pos = torch.from_numpy(rng.uniform(0, 1, (n_nodes, 3)).astype(np.float32))
        x = torch.cat([mesh_pos, torch.zeros(n_nodes, 1)], dim=1)
        y = torch.randn(n_nodes, 3)
        graph = {"mesh_pos": mesh_pos, "x": x, "y": y}
        gc = GridConfig(resolution=(4, 4, 4))
        result = _graph_to_grid(graph, gc)
        assert result["grid_input"].shape == (7, 4, 4, 4)
        assert result["grid_target"].shape == (3, 4, 4, 4)
        assert result["occupancy_mask"].shape == (1, 4, 4, 4)
        assert result["grid_bounds"].shape == (6,)

    def test_compute_grid_norm_stats(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess_grid import _compute_grid_norm_stats

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for i in range(3):
            _make_grid_pt(grid_dir / f"{i:06d}.pt")
        stats = _compute_grid_norm_stats(grid_dir)
        assert len(stats.channel_mean) == 7
        assert len(stats.channel_std) == 7
        assert len(stats.target_mean) == 3
        assert len(stats.target_std) == 3

    def test_compute_grid_norm_stats_no_files(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess_grid import _compute_grid_norm_stats

        with pytest.raises(FileNotFoundError):
            _compute_grid_norm_stats(tmp_path)

    def test_preprocess_grid_dataset(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig, GridConfig
        from vibration_poc.dataset.preprocess_grid import preprocess_grid_dataset

        rng = np.random.default_rng(42)
        for split in ["train", "valid"]:
            split_dir = tmp_path / split
            split_dir.mkdir()
            for i in range(2):
                n = 15
                mesh_pos = torch.from_numpy(rng.uniform(0, 1, (n, 3)).astype(np.float32))
                x = torch.cat([mesh_pos, torch.zeros(n, 1)], dim=1)
                y = torch.randn(n, 3)
                torch.save({"mesh_pos": mesh_pos, "x": x, "y": y}, split_dir / f"{i:06d}.pt")

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train", "valid"])
        gc = GridConfig(resolution=(4, 4, 4))
        preprocess_grid_dataset(cfg, gc)
        assert (tmp_path / "grid" / "train").exists()
        assert len(list((tmp_path / "grid" / "train").glob("*.pt"))) == 2
        assert (tmp_path / "grid" / "grid_norm_stats.json").exists()


class TestGridDataloaders:
    def test_returns_dict(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig, GridConfig
        from vibration_poc.dataset.grid_dataloader import get_grid_dataloaders

        for split in ["train", "valid"]:
            grid_dir = tmp_path / "grid" / split
            grid_dir.mkdir(parents=True)
            for i in range(3):
                _make_grid_pt(grid_dir / f"{i:06d}.pt")

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train", "valid"])
        gc = GridConfig(resolution=(3, 3, 3))
        loaders = get_grid_dataloaders(cfg, gc, batch_size=2)
        assert "train" in loaders
        assert "valid" in loaders

    def test_batch_shape(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig, GridConfig
        from vibration_poc.dataset.grid_dataloader import get_grid_dataloaders

        grid_dir = tmp_path / "grid" / "train"
        grid_dir.mkdir(parents=True)
        for i in range(4):
            _make_grid_pt(grid_dir / f"{i:06d}.pt", resolution=(4, 6, 2))

        cfg = DatasetConfig(processed_dir=tmp_path, splits=["train"])
        gc = GridConfig(resolution=(4, 6, 2))
        loaders = get_grid_dataloaders(cfg, gc, batch_size=2)
        batch = next(iter(loaders["train"]))
        assert batch["grid_input"].shape == (2, 7, 4, 6, 2)
