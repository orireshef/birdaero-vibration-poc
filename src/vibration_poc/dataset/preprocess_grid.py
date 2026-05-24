"""Convert graph .pt files to grid .pt files for FNO training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from vibration_poc.dataset.config import DatasetConfig, GridConfig, GridNormStats
from vibration_poc.dataset.grid_interpolation import MeshToGridInterpolator

logger = logging.getLogger(__name__)


def _graph_to_grid(
    graph: dict[str, torch.Tensor],
    grid_config: GridConfig,
) -> dict[str, torch.Tensor]:
    """Convert a single graph dict to a grid dict."""
    mesh_pos = graph["mesh_pos"].numpy()
    world_pos = graph["x"][:, :3].numpy()
    node_type = graph["x"][:, 3:4].numpy()
    target = graph["y"].numpy()

    interp = MeshToGridInterpolator(
        mesh_pos,
        grid_config.resolution,
        grid_config.padding_ratio,
    )

    input_fields = np.concatenate([world_pos, mesh_pos, node_type], axis=1)
    grid_input = interp.interpolate(input_fields)
    grid_target = interp.interpolate(target)
    occ_mask = interp.occupancy_mask()

    return {
        "grid_input": torch.from_numpy(grid_input),
        "grid_target": torch.from_numpy(grid_target),
        "occupancy_mask": torch.from_numpy(occ_mask),
        "mesh_pos": graph["mesh_pos"],
        "node_type": graph["x"][:, 3],
        "grid_bounds": torch.from_numpy(interp.grid_bounds),
    }


def _compute_grid_norm_stats(
    grid_dir: Path,
) -> GridNormStats:
    """Compute per-channel mean/std from saved grid .pt files."""
    files = sorted(grid_dir.glob("*.pt"))
    if not files:
        msg = f"No .pt files in {grid_dir}"
        raise FileNotFoundError(msg)

    n = 0
    input_sum: np.ndarray | None = None
    input_sq_sum: np.ndarray | None = None
    target_sum: np.ndarray | None = None
    target_sq_sum: np.ndarray | None = None

    for f in files:
        g = torch.load(f, weights_only=True)
        gi = g["grid_input"].numpy()
        gt = g["grid_target"].numpy()
        mask = g["occupancy_mask"].numpy()

        count = float(mask.sum())
        if count < 1:
            continue

        if input_sum is None:
            c_in = gi.shape[0]
            c_out = gt.shape[0]
            input_sum = np.zeros(c_in)
            input_sq_sum = np.zeros(c_in)
            target_sum = np.zeros(c_out)
            target_sq_sum = np.zeros(c_out)

        assert input_sum is not None
        assert input_sq_sum is not None
        assert target_sum is not None
        assert target_sq_sum is not None
        for c in range(gi.shape[0]):
            vals = gi[c][mask[0] > 0.5]
            input_sum[c] += vals.sum()
            input_sq_sum[c] += (vals**2).sum()
        for c in range(gt.shape[0]):
            vals = gt[c][mask[0] > 0.5]
            target_sum[c] += vals.sum()
            target_sq_sum[c] += (vals**2).sum()
        n += int(count)

    if input_sum is None or input_sq_sum is None or target_sum is None or target_sq_sum is None:
        msg = "No valid samples found for normalization"
        raise ValueError(msg)

    ch_mean: list[float] = (input_sum / n).tolist()
    ch_std: list[float] = np.sqrt(input_sq_sum / n - (input_sum / n) ** 2).clip(1e-8).tolist()
    tgt_mean: list[float] = (target_sum / n).tolist()
    tgt_std: list[float] = np.sqrt(target_sq_sum / n - (target_sum / n) ** 2).clip(1e-8).tolist()

    return GridNormStats(
        channel_mean=ch_mean,
        channel_std=ch_std,
        target_mean=tgt_mean,
        target_std=tgt_std,
    )


def preprocess_grid_dataset(
    dataset_config: DatasetConfig,
    grid_config: GridConfig,
) -> None:
    """Convert preprocessed graph .pt files to grid .pt files."""
    for split in dataset_config.splits:
        src_dir = dataset_config.processed_dir / split
        dst_dir = dataset_config.processed_dir / "grid" / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_files = sorted(src_dir.glob("*.pt"))
        logger.info("Converting %d graphs to grids for split=%s", len(src_files), split)

        for i, src_file in enumerate(src_files):
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                continue
            graph = torch.load(src_file, weights_only=True)
            grid = _graph_to_grid(graph, grid_config)
            torch.save(grid, dst_file)
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d done", i + 1, len(src_files))

        logger.info("Split %s: %d grid files", split, len(list(dst_dir.glob("*.pt"))))

    train_grid_dir = dataset_config.processed_dir / "grid" / "train"
    if train_grid_dir.exists() and any(train_grid_dir.glob("*.pt")):
        stats = _compute_grid_norm_stats(train_grid_dir)
        stats_path = dataset_config.processed_dir / "grid" / "grid_norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats.model_dump(), f, indent=2)
        logger.info("Saved grid norm stats to %s", stats_path)
