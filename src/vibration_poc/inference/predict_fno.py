"""Autoregressive rollout for FNO inference on grid data."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from vibration_poc.dataset.config import GridConfig, GridNormStats
from vibration_poc.dataset.grid_interpolation import GridToMeshInterpolator, MeshToGridInterpolator
from vibration_poc.model.fno import FNO3d
from vibration_poc.physics import get_boundary_mask


def rollout_fno(
    model: FNO3d,
    initial_graph: dict[str, Tensor],
    num_steps: int,
    grid_norm_stats: GridNormStats,
    grid_config: GridConfig,
    device: torch.device | None = None,
    bc_node_types: list[int] | None = None,
) -> list[dict[str, Tensor]]:
    """Run autoregressive FNO rollout. Returns same format as GNN rollout."""
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    mesh_pos = initial_graph["mesh_pos"].numpy()
    world_pos = initial_graph["x"][:, :3].numpy().copy()
    node_type = initial_graph["x"][:, 3:4].numpy()
    raw_node_type = initial_graph["x"][:, 3].to(device)

    mesh_to_grid = MeshToGridInterpolator(
        mesh_pos,
        grid_config.resolution,
        grid_config.padding_ratio,
    )
    grid_to_mesh = GridToMeshInterpolator(
        mesh_to_grid.grid_bounds,
        grid_config.resolution,
    )
    ch_mean = np.array(grid_norm_stats.channel_mean, dtype=np.float32).reshape(-1, 1, 1, 1)
    ch_std = np.array(grid_norm_stats.channel_std, dtype=np.float32).reshape(-1, 1, 1, 1)

    results: list[dict[str, Tensor]] = []

    with torch.no_grad():
        for _ in range(num_steps):
            input_fields = np.concatenate([world_pos, mesh_pos, node_type], axis=1)
            grid_input = mesh_to_grid.interpolate(input_fields)

            grid_input_norm = (grid_input - ch_mean) / ch_std
            grid_input_t = torch.from_numpy(grid_input_norm).unsqueeze(0).to(device)

            grid_pred = model(grid_input_t).squeeze(0).cpu().numpy()

            tgt_mean = np.array(grid_norm_stats.target_mean, dtype=np.float32).reshape(-1, 1, 1, 1)
            tgt_std = np.array(grid_norm_stats.target_std, dtype=np.float32).reshape(-1, 1, 1, 1)
            grid_pred_denorm = grid_pred * tgt_std + tgt_mean

            displacement = grid_to_mesh.interpolate(grid_pred_denorm, mesh_pos)
            disp_tensor = torch.from_numpy(displacement).to(device)

            if bc_node_types is not None:
                bc_mask = get_boundary_mask(raw_node_type, bc_node_types)
                disp_tensor[bc_mask] = 0.0

            results.append(
                {
                    "world_pos": torch.from_numpy(world_pos.copy()).to(device),
                    "predicted_displacement": disp_tensor,
                    "mesh_pos": torch.from_numpy(mesh_pos).to(device),
                }
            )

            world_pos = world_pos + displacement

    return results
