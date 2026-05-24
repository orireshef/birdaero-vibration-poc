"""Smoke test: exercises the full notebook pipeline on CPU with synthetic data.

Covers: model creation → training (1 epoch) → rollout → FFT → visualization.
Runtime: ~5 seconds on CPU, no dataset download required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from vibration_poc.dataset.config import NormStats
from vibration_poc.inference.analyze import (
    compute_fft,
    displacement_time_series,
    dominant_frequencies,
)
from vibration_poc.inference.predict import rollout
from vibration_poc.model.meshgraphnet import MeshGraphNet

NUM_NODES = 20
NUM_EDGES = 80
HIDDEN_DIM = 16
NUM_LAYERS = 2
ROLLOUT_STEPS = 20


@pytest.fixture
def synthetic_graph() -> dict[str, Tensor]:
    """Graph large enough for meaningful FFT but small enough for fast CPU test."""
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    return {
        "x": torch.randn(NUM_NODES, 4),
        "edge_index": edge_index,
        "edge_attr": torch.randn(NUM_EDGES, 4),
        "y": torch.randn(NUM_NODES, 3),
        "target_stress": torch.randn(NUM_NODES, 1),
        "mesh_pos": torch.randn(NUM_NODES, 3),
    }


@pytest.fixture
def model() -> MeshGraphNet:
    return MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=4,
        output_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def norm_stats() -> NormStats:
    return NormStats(
        node_mean=[0.0, 0.0, 0.0, 0.0],
        node_std=[1.0, 1.0, 1.0, 1.0],
        edge_mean=[0.0, 0.0, 0.0, 0.0],
        edge_std=[1.0, 1.0, 1.0, 1.0],
    )


class TestNotebookPipeline:
    """End-to-end test mirroring the demo notebook flow."""

    def test_model_creation(self, model: MeshGraphNet) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert all(p.requires_grad for p in model.parameters())

    def test_single_forward_pass(
        self, model: MeshGraphNet, synthetic_graph: dict[str, Tensor]
    ) -> None:
        model.eval()
        with torch.no_grad():
            pred = model(synthetic_graph)
        assert pred.shape == (NUM_NODES, 3)

    def test_training_step(self, model: MeshGraphNet, synthetic_graph: dict[str, Tensor]) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        pred = model(synthetic_graph)
        loss = F.mse_loss(pred, synthetic_graph["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_rollout(
        self,
        model: MeshGraphNet,
        synthetic_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        results = rollout(model, synthetic_graph, num_steps=ROLLOUT_STEPS, norm_stats=norm_stats)
        assert len(results) == ROLLOUT_STEPS
        assert results[0]["world_pos"].shape == (NUM_NODES, 3)
        assert results[0]["predicted_displacement"].shape == (NUM_NODES, 3)
        assert results[0]["mesh_pos"].shape == (NUM_NODES, 3)

    def test_fft_analysis(
        self,
        model: MeshGraphNet,
        synthetic_graph: dict[str, Tensor],
        norm_stats: NormStats,
    ) -> None:
        results = rollout(model, synthetic_graph, num_steps=ROLLOUT_STEPS, norm_stats=norm_stats)
        series = displacement_time_series(results)
        assert series.shape == (ROLLOUT_STEPS, NUM_NODES)

        freqs, magnitudes = compute_fft(series, dt=1.0)
        assert len(freqs) > 0
        assert magnitudes.shape[1] == NUM_NODES

        top = dominant_frequencies(freqs, magnitudes, top_k=3)
        assert len(top) == 3
        for freq, mag in top:
            assert freq >= 0
            assert mag >= 0

    def test_visualization_no_crash(
        self,
        model: MeshGraphNet,
        synthetic_graph: dict[str, Tensor],
        norm_stats: NormStats,
        tmp_path: Path,
    ) -> None:
        """Verify visualization functions run without error (non-interactive)."""
        import matplotlib

        matplotlib.use("Agg")

        from vibration_poc.visualization.error_maps import plot_error_map
        from vibration_poc.visualization.frequency_analysis import (
            plot_frequency_spectrum,
            plot_mode_shapes,
        )

        results = rollout(model, synthetic_graph, num_steps=ROLLOUT_STEPS, norm_stats=norm_stats)
        mesh_np = results[0]["mesh_pos"].cpu().numpy()
        pred_np = results[0]["predicted_displacement"].cpu().numpy()
        gt_np = synthetic_graph["y"].numpy()

        # Error map
        error_path = tmp_path / "error_map.png"
        plot_error_map(mesh_np, gt_np, pred_np, save_path=error_path)
        assert error_path.exists()

        # Frequency spectrum
        series = displacement_time_series(results)
        freqs, magnitudes = compute_fft(series, dt=1.0)
        spectrum_path = tmp_path / "spectrum.png"
        plot_frequency_spectrum(freqs, magnitudes, top_k=3, save_path=spectrum_path)
        assert spectrum_path.exists()

        # Mode shapes
        modes_path = tmp_path / "modes.png"
        plot_mode_shapes(mesh_np, magnitudes, freqs, top_k=3, save_path=modes_path)
        assert modes_path.exists()

    def test_full_pipeline_checkpoint(
        self,
        synthetic_graph: dict[str, Tensor],
        norm_stats: NormStats,
        tmp_path: Path,
    ) -> None:
        """Train → save → load → rollout → FFT — full notebook flow."""
        model = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=4,
            output_dim=3,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )

        # Train 1 step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        pred = model(synthetic_graph)
        loss = F.mse_loss(pred, synthetic_graph["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        ckpt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh model
        loaded = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=4,
            output_dim=3,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        )
        loaded.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Rollout + FFT
        results = rollout(loaded, synthetic_graph, num_steps=ROLLOUT_STEPS, norm_stats=norm_stats)
        series = displacement_time_series(results)
        freqs, magnitudes = compute_fft(series, dt=1.0)
        top = dominant_frequencies(freqs, magnitudes, top_k=3)

        assert len(results) == ROLLOUT_STEPS
        assert len(top) == 3
