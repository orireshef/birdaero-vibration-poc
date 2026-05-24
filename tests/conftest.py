"""Shared test fixtures."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor


@pytest.fixture
def tiny_graph() -> dict[str, Tensor]:
    """Minimal graph dict for unit tests (N=4 nodes, E=12 edges)."""
    N, E = 4, 12
    return {
        "x": torch.randn(N, 4),
        "edge_index": torch.randint(0, N, (2, E)),
        "edge_attr": torch.randn(E, 4),
        "y": torch.randn(N, 3),
        "target_stress": torch.randn(N, 1),
        "mesh_pos": torch.randn(N, 3),
    }


@pytest.fixture
def tiny_trajectory(tiny_graph: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    """5-step trajectory with coherent world_pos progression."""
    steps: list[dict[str, Tensor]] = []
    world_pos = tiny_graph["x"][:, :3].clone()
    for _t in range(5):
        displacement = torch.randn_like(world_pos) * 0.01
        new_world_pos = world_pos + displacement
        node_type = tiny_graph["x"][:, 3:]
        g: dict[str, Tensor] = {
            "x": torch.cat([world_pos, node_type], dim=1),
            "edge_index": tiny_graph["edge_index"],
            "edge_attr": tiny_graph["edge_attr"],
            "y": displacement,
            "target_stress": tiny_graph["target_stress"],
            "mesh_pos": tiny_graph["mesh_pos"],
        }
        steps.append(g)
        world_pos = new_world_pos
    return steps
