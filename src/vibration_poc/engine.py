"""Evaluation engine — agent-callable API for design analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from vibration_poc.dataset.config import NormStats
from vibration_poc.inference.analyze import (
    compute_fft,
    displacement_time_series,
    dominant_frequencies,
)
from vibration_poc.inference.predict import rollout
from vibration_poc.model.meshgraphnet import MeshGraphNet


class DesignMetrics(BaseModel):
    """Structured evaluation results for a design."""

    dominant_frequencies: list[tuple[float, float]]
    max_displacement: float
    mean_displacement: float
    num_steps: int

    model_config = {"arbitrary_types_allowed": True}


def load_model(
    checkpoint: Path,
    hidden_dim: int = 64,
    num_layers: int = 8,
    predict_stress: bool = False,
    device: str = "cpu",
) -> MeshGraphNet:
    """Load MeshGraphNet from checkpoint file."""
    model = MeshGraphNet(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        predict_stress=predict_stress,
    )
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model


def evaluate_design(
    graph: dict[str, Tensor],
    model: MeshGraphNet,
    norm_stats: NormStats,
    num_steps: int = 50,
    bc_node_types: list[int] | None = None,
    dt: float = 1.0,
) -> tuple[DesignMetrics, list[dict[str, Tensor]]]:
    """Run rollout + FFT analysis. Returns (metrics, raw_rollout_results)."""
    results = rollout(
        model, graph, num_steps=num_steps, norm_stats=norm_stats, bc_node_types=bc_node_types
    )

    ts = displacement_time_series(results)
    freqs, mags = compute_fft(ts, dt=dt)
    top_freqs = dominant_frequencies(freqs, mags, top_k=5)

    all_disps: np.ndarray = np.concatenate(
        [
            np.linalg.norm(step["predicted_displacement"].detach().cpu().numpy(), axis=1)
            for step in results
        ]
    )

    metrics = DesignMetrics(
        dominant_frequencies=top_freqs,
        max_displacement=float(all_disps.max()),
        mean_displacement=float(all_disps.mean()),
        num_steps=num_steps,
    )
    return metrics, results
