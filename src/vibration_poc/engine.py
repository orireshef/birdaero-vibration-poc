"""Evaluation engine — agent-callable API for design analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor

from vibration_poc.dataset.config import GridConfig, GridNormStats, NormStats
from vibration_poc.inference.analyze import (
    compute_fft,
    displacement_time_series,
    dominant_frequencies,
)
from vibration_poc.inference.predict import rollout
from vibration_poc.inference.predict_fno import rollout_fno
from vibration_poc.model.fno import FNO3d
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


def load_fno_model(
    checkpoint: Path,
    hidden_dim: int = 64,
    num_layers: int = 4,
    modes: tuple[int, int, int] = (12, 12, 8),
    device: str = "cpu",
) -> FNO3d:
    """Load FNO3d from checkpoint file."""
    model = FNO3d(hidden_dim=hidden_dim, num_layers=num_layers, modes=modes)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model


def evaluate_design(
    graph: dict[str, Tensor],
    model: MeshGraphNet | FNO3d,
    norm_stats: NormStats | GridNormStats,
    num_steps: int = 50,
    bc_node_types: list[int] | None = None,
    dt: float = 1.0,
    grid_config: GridConfig | None = None,
) -> tuple[DesignMetrics, list[dict[str, Tensor]]]:
    """Run rollout + FFT analysis. Returns (metrics, raw_rollout_results)."""
    if isinstance(model, FNO3d):
        if not isinstance(norm_stats, GridNormStats):
            msg = "FNO3d requires GridNormStats"
            raise TypeError(msg)
        if grid_config is None:
            grid_config = GridConfig()
        results = rollout_fno(
            model,
            graph,
            num_steps=num_steps,
            grid_norm_stats=norm_stats,
            grid_config=grid_config,
            bc_node_types=bc_node_types,
        )
    else:
        if not isinstance(norm_stats, NormStats):
            msg = "MeshGraphNet requires NormStats"
            raise TypeError(msg)
        results = rollout(
            model,
            graph,
            num_steps=num_steps,
            norm_stats=norm_stats,
            bc_node_types=bc_node_types,
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
