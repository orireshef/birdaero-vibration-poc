"""Frequency spectrum and mode shape visualization."""
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def plot_frequency_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    top_k: int = 5,
    save_path: Path | None = None,
) -> None:
    """Mean magnitude vs frequency with top-k peaks marked."""
    mean_mag: np.ndarray = magnitudes.mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(frequencies, mean_mag, label="Mean magnitude")

    # Find top-k peaks
    peak_indices: np.ndarray = np.argsort(mean_mag)[-top_k:]
    for idx in peak_indices:
        ax.axvline(frequencies[idx], color="red", linestyle="--", alpha=0.6)
        ax.plot(frequencies[idx], mean_mag[idx], "ro", markersize=8)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean Magnitude")
    ax.set_title("Frequency Spectrum")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_mode_shapes(
    mesh_pos: np.ndarray,
    fft_magnitudes: np.ndarray,
    frequencies: np.ndarray,
    top_k: int = 3,
    save_path: Path | None = None,
) -> None:
    """3D scatter per top-k frequency, colored by per-node FFT magnitude."""
    mean_mag: np.ndarray = fft_magnitudes.mean(axis=1)
    peak_indices: np.ndarray = np.argsort(mean_mag)[-top_k:][::-1]

    fig = plt.figure(figsize=(6 * top_k, 5))
    for i, idx in enumerate(peak_indices):
        ax = fig.add_subplot(1, top_k, i + 1, projection="3d")
        node_mag: np.ndarray = fft_magnitudes[idx, :]
        sc = ax.scatter(
            mesh_pos[:, 0],
            mesh_pos[:, 1],
            mesh_pos[:, 2],
            c=node_mag,
            cmap="viridis",
        )
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(f"f={frequencies[idx]:.2f}")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)
