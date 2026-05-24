"""Tests for physics constraint utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from vibration_poc.physics import (
    PhysicsConfig,
    compute_bc_penalty,
    compute_smoothness_loss,
    get_boundary_mask,
)


class TestPhysicsConfig:
    def test_defaults_all_zero(self) -> None:
        cfg = PhysicsConfig()
        assert cfg.bc_loss_weight == 0.0
        assert cfg.stress_loss_weight == 0.0
        assert cfg.smoothness_weight == 0.0

    def test_custom_values(self) -> None:
        cfg = PhysicsConfig(bc_loss_weight=1.0, smoothness_weight=0.5)
        assert cfg.bc_loss_weight == 1.0
        assert cfg.smoothness_weight == 0.5


class TestBoundaryMask:
    def test_basic_mask(self) -> None:
        node_type = torch.tensor([0.0, 1.0, 0.0, 2.0])
        mask = get_boundary_mask(node_type, bc_types=[1, 2])
        assert mask.tolist() == [False, True, False, True]

    def test_no_boundary_nodes(self) -> None:
        node_type = torch.tensor([0.0, 0.0, 0.0])
        mask = get_boundary_mask(node_type, bc_types=[1])
        assert not mask.any()

    def test_all_boundary_nodes(self) -> None:
        node_type = torch.tensor([1.0, 1.0])
        mask = get_boundary_mask(node_type, bc_types=[1])
        assert mask.all()

    def test_2d_input(self) -> None:
        node_type = torch.tensor([[0.0], [1.0], [0.0]])
        mask = get_boundary_mask(node_type, bc_types=[1])
        assert mask.shape == (3,)
        assert mask.tolist() == [False, True, False]


class TestBCPenalty:
    def test_zero_when_no_boundary(self) -> None:
        pred = torch.randn(4, 3)
        mask = torch.zeros(4, dtype=torch.bool)
        loss = compute_bc_penalty(pred, mask)
        assert loss.item() == 0.0

    def test_nonzero_when_boundary_has_displacement(self) -> None:
        pred = torch.ones(4, 3)
        mask = torch.tensor([False, False, True, True])
        loss = compute_bc_penalty(pred, mask)
        assert loss.item() > 0.0

    def test_zero_when_boundary_displacement_is_zero(self) -> None:
        pred = torch.zeros(4, 3)
        mask = torch.tensor([True, True, False, False])
        loss = compute_bc_penalty(pred, mask)
        assert loss.item() == 0.0


class TestSmoothnessLoss:
    def test_zero_for_uniform_displacement(self) -> None:
        pred = torch.ones(4, 3)  # all same
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        loss = compute_smoothness_loss(pred, edge_index)
        assert loss.item() == 0.0

    def test_nonzero_for_varying_displacement(self) -> None:
        pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])
        loss = compute_smoothness_loss(pred, edge_index)
        assert loss.item() > 0.0

    def test_uses_edge_index(self, tiny_graph: dict[str, Tensor]) -> None:
        pred = torch.randn(4, 3)
        loss = compute_smoothness_loss(pred, tiny_graph["edge_index"])
        assert loss.shape == ()  # scalar
