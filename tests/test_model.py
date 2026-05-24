"""Tests for MeshGraphNet model components."""

from __future__ import annotations

import torch
from torch import Tensor


def test_mlp_output_shape() -> None:
    from vibration_poc.model.meshgraphnet import MLP

    mlp = MLP([4, 16, 8])
    x = torch.randn(10, 4)
    out = mlp(x)
    assert out.shape == (10, 8)


def test_mlp_with_layer_norm() -> None:
    from vibration_poc.model.meshgraphnet import MLP

    mlp = MLP([4, 16, 8], layer_norm=True)
    x = torch.randn(10, 4)
    out = mlp(x)
    assert out.shape == (10, 8)


def test_mlp_without_layer_norm() -> None:
    from vibration_poc.model.meshgraphnet import MLP

    mlp = MLP([4, 16, 8], layer_norm=False)
    x = torch.randn(10, 4)
    out = mlp(x)
    assert out.shape == (10, 8)


def test_edge_block_output_shape(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import EdgeBlock

    hidden_dim = 16
    # Encode inputs to hidden_dim first
    x = torch.randn(4, hidden_dim)
    edge_attr = torch.randn(12, hidden_dim)
    edge_index = tiny_graph["edge_index"]

    block = EdgeBlock(hidden_dim)
    out = block(x, edge_index, edge_attr)
    assert out.shape == edge_attr.shape


def test_node_block_output_shape(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import NodeBlock

    hidden_dim = 16
    x = torch.randn(4, hidden_dim)
    edge_attr = torch.randn(12, hidden_dim)
    edge_index = tiny_graph["edge_index"]

    block = NodeBlock(hidden_dim)
    out = block(x, edge_index, edge_attr)
    assert out.shape == x.shape


def test_meshgraphnet_forward(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=4,
        output_dim=3,
        hidden_dim=16,
        num_layers=2,
    )
    out = model(tiny_graph)
    assert out.shape == (4, 3)


def test_meshgraphnet_gradient_flow(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2)
    out = model(tiny_graph)
    loss = out.mean()
    loss.backward()
    # Check at least one param has grad
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_meshgraphnet_different_configs(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    for hidden_dim, num_layers in [(8, 1), (32, 4), (64, 6)]:
        model = MeshGraphNet(hidden_dim=hidden_dim, num_layers=num_layers)
        out = model(tiny_graph)
        assert out.shape == (4, 3), f"Failed for hidden_dim={hidden_dim}, num_layers={num_layers}"


def test_stress_head_disabled_by_default(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2)
    assert model.stress_decoder is None


def test_stress_head_forward_with_stress(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2, predict_stress=True)
    disp, stress = model.forward_with_stress(tiny_graph)
    assert disp.shape == (4, 3)
    assert stress.shape == (4, 1)


def test_stress_head_gradient_flow(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2, predict_stress=True)
    disp, stress = model.forward_with_stress(tiny_graph)
    loss = disp.mean() + stress.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_forward_unchanged_with_stress_head(tiny_graph: dict[str, Tensor]) -> None:
    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2, predict_stress=True)
    out = model(tiny_graph)
    assert out.shape == (4, 3)


def test_forward_with_stress_raises_without_head(tiny_graph: dict[str, Tensor]) -> None:
    import pytest

    from vibration_poc.model.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(hidden_dim=16, num_layers=2, predict_stress=False)
    with pytest.raises(RuntimeError):
        model.forward_with_stress(tiny_graph)
