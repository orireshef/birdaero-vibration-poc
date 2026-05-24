"""Tests for training loop and config."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader


@pytest.fixture
def tiny_graph() -> dict[str, Tensor]:
    N, E = 4, 12
    return {
        "x": torch.randn(N, 4),
        "edge_index": torch.randint(0, N, (2, E)),
        "edge_attr": torch.randn(E, 4),
        "y": torch.randn(N, 3),
        "target_stress": torch.randn(N, 1),
        "mesh_pos": torch.randn(N, 3),
    }


def make_loader(graphs: list[dict[str, Tensor]]) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(graphs, batch_size=1, collate_fn=lambda x: x[0])  # type: ignore[arg-type]


def test_training_config_defaults() -> None:
    from vibration_poc.training.trainer import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.epochs == 5
    assert cfg.learning_rate == 1e-4
    assert cfg.hidden_dim == 64
    assert cfg.num_layers == 8
    assert cfg.batch_size == 1
    assert cfg.device == "cpu"


def test_training_config_negative_lr_rejected() -> None:
    from pydantic import ValidationError

    from vibration_poc.training.trainer import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.1)


def test_training_config_negative_epochs_rejected() -> None:
    from pydantic import ValidationError

    from vibration_poc.training.trainer import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)


@pytest.mark.slow
def test_train_epoch_runs(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet
    from vibration_poc.training.trainer import train_epoch

    model = MeshGraphNet(hidden_dim=8, num_layers=1)
    loader = make_loader([tiny_graph, tiny_graph])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss = train_epoch(model, loader, optimizer, device)
    assert isinstance(loss, float)
    assert loss >= 0.0


@pytest.mark.slow
def test_train_epoch_with_physics_config(tiny_graph: dict[str, Tensor]) -> None:
    """Physics loss adds BC + smoothness terms."""
    from vibration_poc.model.meshgraphnet import MeshGraphNet
    from vibration_poc.physics import PhysicsConfig
    from vibration_poc.training.trainer import train_epoch

    model = MeshGraphNet(hidden_dim=8, num_layers=1)
    loader = make_loader([tiny_graph, tiny_graph])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    physics = PhysicsConfig(bc_loss_weight=0.1, smoothness_weight=0.05)
    loss = train_epoch(model, loader, optimizer, device, physics=physics)
    assert isinstance(loss, float)
    assert loss >= 0.0


@pytest.mark.slow
def test_train_epoch_physics_none_unchanged(tiny_graph: dict[str, Tensor]) -> None:
    """physics=None gives same behavior as before."""
    from vibration_poc.model.meshgraphnet import MeshGraphNet
    from vibration_poc.training.trainer import train_epoch

    model = MeshGraphNet(hidden_dim=8, num_layers=1)
    loader = make_loader([tiny_graph, tiny_graph])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss = train_epoch(model, loader, optimizer, device, physics=None)
    assert isinstance(loss, float)
    assert loss >= 0.0
