"""Smoke tests for Plotly 3D visualization helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor

from vibration_poc.visualization.plotly_viz import (
    error_comparison_figure,
    frequency_spectrum_figure,
    mesh_scatter_3d,
    rollout_animation_figure,
)


def _fake_rollout(num_steps: int = 5, num_nodes: int = 10) -> list[dict[str, Tensor]]:
    return [
        {
            "world_pos": torch.randn(num_nodes, 3),
            "predicted_displacement": torch.randn(num_nodes, 3),
            "mesh_pos": torch.randn(num_nodes, 3),
        }
        for _ in range(num_steps)
    ]


class TestMeshScatter3D:
    def test_returns_figure(self) -> None:
        pos = np.random.randn(20, 3)
        vals = np.random.randn(20)
        fig = mesh_scatter_3d(pos, vals)
        assert isinstance(fig, go.Figure)

    def test_has_data(self) -> None:
        pos = np.random.randn(20, 3)
        vals = np.random.randn(20)
        fig = mesh_scatter_3d(pos, vals)
        assert len(fig.data) > 0


class TestRolloutAnimationFigure:
    def test_returns_figure(self) -> None:
        rollout = _fake_rollout(num_steps=3, num_nodes=8)
        fig = rollout_animation_figure(rollout)
        assert isinstance(fig, go.Figure)

    def test_has_frames(self) -> None:
        num_steps = 5
        rollout = _fake_rollout(num_steps=num_steps, num_nodes=8)
        fig = rollout_animation_figure(rollout)
        assert len(fig.frames) == num_steps


class TestErrorComparisonFigure:
    def test_returns_figure(self) -> None:
        pos = np.random.randn(15, 3)
        gt = np.random.randn(15, 3)
        pred = np.random.randn(15, 3)
        fig = error_comparison_figure(pos, gt, pred)
        assert isinstance(fig, go.Figure)

    def test_has_three_traces(self) -> None:
        pos = np.random.randn(15, 3)
        gt = np.random.randn(15, 3)
        pred = np.random.randn(15, 3)
        fig = error_comparison_figure(pos, gt, pred)
        assert len(fig.data) == 3


class TestFrequencySpectrumFigure:
    def test_returns_figure(self) -> None:
        freqs = np.linspace(0, 100, 50)
        mags = np.random.randn(50)
        fig = frequency_spectrum_figure(freqs, mags)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self) -> None:
        freqs = np.linspace(0, 100, 50)
        mags = np.random.randn(50)
        fig = frequency_spectrum_figure(freqs, mags)
        assert len(fig.data) == 2
