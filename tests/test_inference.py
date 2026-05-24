"""Tests for inference module — rollout and FFT analysis."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from vibration_poc.dataset.config import NormStats
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


# ── T18: rollout tests ──────────────────────────────────────────────


class TestRollout:
    def test_rollout_returns_list_of_correct_length(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        results = rollout(tiny_model, tiny_graph, num_steps=3, norm_stats=norm_stats)
        assert len(results) == 3

    def test_rollout_result_keys(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        results = rollout(tiny_model, tiny_graph, num_steps=1, norm_stats=norm_stats)
        step = results[0]
        assert "world_pos" in step
        assert "predicted_displacement" in step
        assert "mesh_pos" in step

    def test_rollout_shapes(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        N = tiny_graph["x"].shape[0]
        results = rollout(tiny_model, tiny_graph, num_steps=2, norm_stats=norm_stats)
        for step in results:
            assert step["world_pos"].shape == (N, 3)
            assert step["predicted_displacement"].shape == (N, 3)
            assert step["mesh_pos"].shape == (N, 3)

    def test_rollout_mesh_pos_unchanged(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        original_mesh_pos = tiny_graph["mesh_pos"].clone()
        results = rollout(tiny_model, tiny_graph, num_steps=3, norm_stats=norm_stats)
        for step in results:
            assert torch.allclose(step["mesh_pos"], original_mesh_pos)

    def test_rollout_world_pos_updates(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        results = rollout(tiny_model, tiny_graph, num_steps=2, norm_stats=norm_stats)
        # world_pos at step 1 should differ from step 0 (model predicts nonzero displacement)
        assert not torch.allclose(results[0]["world_pos"], results[1]["world_pos"])

    def test_rollout_device_cpu(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        from vibration_poc.inference.predict import rollout

        device = torch.device("cpu")
        results = rollout(tiny_model, tiny_graph, num_steps=1, norm_stats=norm_stats, device=device)
        assert results[0]["world_pos"].device == device

    def test_rollout_bc_enforcement(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph_with_bc: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        """Boundary nodes (type=1) should have zero displacement when bc_node_types=[1]."""
        from vibration_poc.inference.predict import rollout

        results = rollout(
            tiny_model, tiny_graph_with_bc, num_steps=3, norm_stats=norm_stats, bc_node_types=[1]
        )
        for step in results:
            bc_disp = step["predicted_displacement"][2:]
            assert torch.allclose(bc_disp, torch.zeros_like(bc_disp))

    def test_rollout_bc_none_no_enforcement(
        self,
        tiny_model: MeshGraphNet,
        tiny_graph_with_bc: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        """Without bc_node_types, no clamping happens."""
        from vibration_poc.inference.predict import rollout

        results = rollout(tiny_model, tiny_graph_with_bc, num_steps=1, norm_stats=norm_stats)
        assert len(results) == 1


# ── T19: FFT analysis tests ─────────────────────────────────────────


class TestDisplacementTimeSeries:
    def test_output_shape(self) -> None:
        from vibration_poc.inference.analyze import displacement_time_series

        # Simulate rollout_results: 5 steps, 4 nodes
        rollout_results: list[dict[str, Tensor]] = [
            {
                "world_pos": torch.randn(4, 3),
                "predicted_displacement": torch.randn(4, 3),
                "mesh_pos": torch.randn(4, 3),
            }
            for _ in range(5)
        ]
        ts = displacement_time_series(rollout_results)
        assert ts.shape == (5, 4)

    def test_nonnegative_magnitudes(self) -> None:
        from vibration_poc.inference.analyze import displacement_time_series

        rollout_results: list[dict[str, Tensor]] = [
            {
                "world_pos": torch.zeros(3, 3),
                "predicted_displacement": torch.ones(3, 3),
                "mesh_pos": torch.zeros(3, 3),
            }
        ]
        ts = displacement_time_series(rollout_results)
        assert np.all(ts >= 0)


class TestComputeFFT:
    def test_output_shapes(self) -> None:
        from vibration_poc.inference.analyze import compute_fft

        T, N = 16, 4
        time_series = np.random.randn(T, N)
        freqs, mags = compute_fft(time_series, dt=0.01)
        assert freqs.shape == (T // 2,)
        assert mags.shape == (T // 2, N)

    def test_frequencies_positive(self) -> None:
        from vibration_poc.inference.analyze import compute_fft

        time_series = np.random.randn(20, 3)
        freqs, _ = compute_fft(time_series, dt=1.0)
        assert np.all(freqs >= 0)


class TestDominantFrequencies:
    def test_returns_correct_count(self) -> None:
        from vibration_poc.inference.analyze import compute_fft, dominant_frequencies

        time_series = np.random.randn(32, 4)
        freqs, mags = compute_fft(time_series)
        result = dominant_frequencies(freqs, mags, top_k=3)
        assert len(result) == 3

    def test_returns_tuples(self) -> None:
        from vibration_poc.inference.analyze import compute_fft, dominant_frequencies

        time_series = np.random.randn(32, 4)
        freqs, mags = compute_fft(time_series)
        result = dominant_frequencies(freqs, mags, top_k=2)
        for freq, mag in result:
            assert isinstance(freq, float)
            assert isinstance(mag, float)

    def test_sorted_by_magnitude_descending(self) -> None:
        from vibration_poc.inference.analyze import compute_fft, dominant_frequencies

        time_series = np.random.randn(64, 8)
        freqs, mags = compute_fft(time_series)
        result = dominant_frequencies(freqs, mags, top_k=5)
        magnitudes = [mag for _, mag in result]
        assert magnitudes == sorted(magnitudes, reverse=True)
