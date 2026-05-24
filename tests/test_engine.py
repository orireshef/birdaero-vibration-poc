"""Tests for evaluation engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from vibration_poc.dataset.config import NormStats
from vibration_poc.engine import DesignMetrics, evaluate_design, load_model
from vibration_poc.model.meshgraphnet import MeshGraphNet


@pytest.fixture
def norm_stats() -> NormStats:
    return NormStats(
        node_mean=[0.0, 0.0, 0.0, 0.0],
        node_std=[1.0, 1.0, 1.0, 1.0],
        edge_mean=[0.0, 0.0, 0.0, 0.0],
        edge_std=[1.0, 1.0, 1.0, 1.0],
    )


@pytest.fixture
def tiny_model() -> MeshGraphNet:
    return MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=4,
        output_dim=3,
        hidden_dim=16,
        num_layers=2,
    )


class TestEvaluateDesign:
    def test_returns_metrics_and_results(
        self,
        tiny_graph: dict[str, Tensor],
        tiny_model: MeshGraphNet,
        norm_stats: NormStats,
    ) -> None:
        metrics, results = evaluate_design(tiny_graph, tiny_model, norm_stats, num_steps=5)
        assert isinstance(metrics, DesignMetrics)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_length(
        self,
        tiny_graph: dict[str, Tensor],
        tiny_model: MeshGraphNet,
        norm_stats: NormStats,
    ) -> None:
        num_steps = 7
        metrics, results = evaluate_design(tiny_graph, tiny_model, norm_stats, num_steps=num_steps)
        assert len(results) == num_steps
        assert metrics.num_steps == num_steps

    def test_with_bc(
        self,
        tiny_graph_with_bc: dict[str, Tensor],
        tiny_model: MeshGraphNet,
        norm_stats: NormStats,
    ) -> None:
        metrics, results = evaluate_design(
            tiny_graph_with_bc, tiny_model, norm_stats, num_steps=3, bc_node_types=[1]
        )
        assert isinstance(metrics, DesignMetrics)
        assert len(results) == 3


class TestDesignMetrics:
    def test_fields(self) -> None:
        m = DesignMetrics(
            dominant_frequencies=[(1.0, 0.5), (2.0, 0.3)],
            max_displacement=1.23,
            mean_displacement=0.45,
            num_steps=10,
        )
        assert isinstance(m.dominant_frequencies, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in m.dominant_frequencies)
        assert isinstance(m.max_displacement, float)
        assert isinstance(m.mean_displacement, float)
        assert isinstance(m.num_steps, int)


class TestLoadModel:
    def test_load_from_checkpoint(self, tmp_path: Path) -> None:
        model = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=4,
            output_dim=3,
            hidden_dim=16,
            num_layers=2,
        )
        ckpt = tmp_path / "model.pt"
        torch.save(model.state_dict(), ckpt)

        loaded = load_model(ckpt, hidden_dim=16, num_layers=2)
        assert isinstance(loaded, MeshGraphNet)

        # Verify forward pass works
        graph: dict[str, Tensor] = {
            "x": torch.randn(4, 4),
            "edge_index": torch.randint(0, 4, (2, 6)),
            "edge_attr": torch.randn(6, 4),
        }
        out = loaded(graph)
        assert out.shape == (4, 3)
