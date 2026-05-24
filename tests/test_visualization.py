"""Tests for visualization modules — ensure plots don't crash under Agg backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import Tensor

matplotlib.use("Agg")


# ── T20: deformation visualization ──────────────────────────────────


class TestDeformationSnapshot:
    def test_runs_without_error(self) -> None:
        from vibration_poc.visualization.deformation import plot_deformation_snapshot

        mesh_pos = np.random.randn(10, 3)
        displacement = np.random.randn(10, 3)
        plot_deformation_snapshot(mesh_pos, displacement, title="test")

    def test_saves_to_file(self, tmp_path: Path) -> None:
        from vibration_poc.visualization.deformation import plot_deformation_snapshot

        save_path = tmp_path / "snapshot.png"
        mesh_pos = np.random.randn(10, 3)
        displacement = np.random.randn(10, 3)
        plot_deformation_snapshot(mesh_pos, displacement, save_path=save_path)
        assert save_path.exists()


class TestDeformationGif:
    def test_creates_gif(self, tmp_path: Path) -> None:
        from vibration_poc.visualization.deformation import create_deformation_gif

        rollout_results: list[dict[str, Tensor]] = [
            {
                "world_pos": torch.randn(6, 3),
                "predicted_displacement": torch.randn(6, 3),
                "mesh_pos": torch.randn(6, 3),
            }
            for _ in range(5)
        ]
        save_path = tmp_path / "deformation.gif"
        create_deformation_gif(rollout_results, save_path, fps=5)
        assert save_path.exists()


# ── T21: error maps ─────────────────────────────────────────────────


class TestErrorMap:
    def test_runs_without_error(self) -> None:
        from vibration_poc.visualization.error_maps import plot_error_map

        mesh_pos = np.random.randn(10, 3)
        gt = np.random.randn(10, 3)
        pred = np.random.randn(10, 3)
        plot_error_map(mesh_pos, gt, pred)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        from vibration_poc.visualization.error_maps import plot_error_map

        save_path = tmp_path / "error.png"
        mesh_pos = np.random.randn(10, 3)
        gt = np.random.randn(10, 3)
        pred = np.random.randn(10, 3)
        plot_error_map(mesh_pos, gt, pred, save_path=save_path)
        assert save_path.exists()


# ── T22: frequency analysis visualization ───────────────────────────


class TestFrequencySpectrum:
    def test_runs_without_error(self) -> None:
        from vibration_poc.visualization.frequency_analysis import plot_frequency_spectrum

        freqs = np.linspace(0, 10, 50)
        mags = np.random.rand(50, 4)
        plot_frequency_spectrum(freqs, mags, top_k=3)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        from vibration_poc.visualization.frequency_analysis import plot_frequency_spectrum

        save_path = tmp_path / "spectrum.png"
        freqs = np.linspace(0, 10, 50)
        mags = np.random.rand(50, 4)
        plot_frequency_spectrum(freqs, mags, top_k=3, save_path=save_path)
        assert save_path.exists()


class TestModeShapes:
    def test_runs_without_error(self) -> None:
        from vibration_poc.visualization.frequency_analysis import plot_mode_shapes

        mesh_pos = np.random.randn(10, 3)
        freqs = np.linspace(0, 10, 50)
        mags = np.random.rand(50, 10)
        plot_mode_shapes(mesh_pos, mags, freqs, top_k=2)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        from vibration_poc.visualization.frequency_analysis import plot_mode_shapes

        save_path = tmp_path / "modes.png"
        mesh_pos = np.random.randn(10, 3)
        freqs = np.linspace(0, 10, 50)
        mags = np.random.rand(50, 10)
        plot_mode_shapes(mesh_pos, mags, freqs, top_k=2, save_path=save_path)
        assert save_path.exists()
