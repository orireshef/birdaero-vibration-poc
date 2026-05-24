"""Autoregressive rollout for MeshGraphNet inference."""

from __future__ import annotations

import torch
from torch import Tensor

from vibration_poc.dataset.config import NormStats
from vibration_poc.model.meshgraphnet import MeshGraphNet


def rollout(
    model: MeshGraphNet,
    initial_graph: dict[str, Tensor],
    num_steps: int,
    norm_stats: NormStats,
    device: torch.device | None = None,
) -> list[dict[str, Tensor]]:
    """Run autoregressive rollout for num_steps.

    Returns list of dicts with keys: world_pos, predicted_displacement, mesh_pos.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Static graph topology
    edge_index = initial_graph["edge_index"].to(device)
    edge_attr = initial_graph["edge_attr"].to(device)
    mesh_pos = initial_graph["mesh_pos"].to(device)

    # Initial state
    world_pos = initial_graph["x"][:, :3].to(device)
    node_type = initial_graph["x"][:, 3:].to(device)

    # Normalization tensors
    node_mean = torch.tensor(norm_stats.node_mean, dtype=torch.float32, device=device)
    node_std = torch.tensor(norm_stats.node_std, dtype=torch.float32, device=device)
    edge_mean = torch.tensor(norm_stats.edge_mean, dtype=torch.float32, device=device)
    edge_std = torch.tensor(norm_stats.edge_std, dtype=torch.float32, device=device)

    results: list[dict[str, Tensor]] = []

    with torch.no_grad():
        for _ in range(num_steps):
            x = torch.cat([world_pos, node_type], dim=1)

            # Normalize
            x_norm = (x - node_mean) / node_std
            e_norm = (edge_attr - edge_mean) / edge_std

            graph: dict[str, Tensor] = {
                "x": x_norm,
                "edge_index": edge_index,
                "edge_attr": e_norm,
            }

            predicted_displacement: Tensor = model(graph)

            results.append(
                {
                    "world_pos": world_pos.clone(),
                    "predicted_displacement": predicted_displacement,
                    "mesh_pos": mesh_pos.clone(),
                }
            )

            world_pos = world_pos + predicted_displacement

    return results
