"""FFT analysis utilities for vibration rollout results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft as scipy_fft

if TYPE_CHECKING:
    from torch import Tensor


def displacement_time_series(rollout_results: list[dict[str, Tensor]]) -> np.ndarray:
    """Extract displacement magnitude time series from rollout results.

    Returns [T, N] array of L2 displacement magnitudes per node per step.
    """
    series: list[np.ndarray] = []
    for step in rollout_results:
        disp = step["predicted_displacement"].detach().cpu().numpy()  # [N, 3]
        mag: np.ndarray = np.linalg.norm(disp, axis=1)  # [N]
        series.append(mag)
    return np.stack(series, axis=0)  # [T, N]


def compute_fft(time_series: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT of displacement time series.

    Args:
        time_series: [T, N] array
        dt: time step

    Returns:
        (frequencies [T//2], magnitudes [T//2, N])
    """
    t_len = time_series.shape[0]
    half = t_len // 2

    spectrum = scipy_fft.rfft(time_series, axis=0)  # [T//2+1, N]
    magnitudes: np.ndarray = np.abs(spectrum)[:half]  # [T//2, N]

    freqs: np.ndarray = scipy_fft.rfftfreq(t_len, d=dt)[:half]  # [T//2]

    return freqs, magnitudes


def dominant_frequencies(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    top_k: int = 5,
) -> list[tuple[float, float]]:
    """Find top-k dominant frequencies by mean magnitude across nodes.

    Returns list of (frequency, mean_magnitude) sorted by magnitude descending.
    """
    mean_mag: np.ndarray = magnitudes.mean(axis=1)  # [T//2]
    indices = np.argsort(mean_mag)[::-1][:top_k]

    result: list[tuple[float, float]] = [
        (float(frequencies[i]), float(mean_mag[i])) for i in indices
    ]
    return result
