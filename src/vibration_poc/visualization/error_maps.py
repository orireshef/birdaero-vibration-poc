"""Ground truth vs predicted error map visualization."""
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def plot_error_map(
    mesh_pos: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """3-panel scatter: GT magnitude | Predicted magnitude | Absolute error."""
    gt_mag: np.ndarray = np.linalg.norm(ground_truth, axis=1)
    pred_mag: np.ndarray = np.linalg.norm(predicted, axis=1)
    error_mag: np.ndarray = np.linalg.norm(ground_truth - predicted, axis=1)

    # Shared color limits for GT and predicted
    vmin = float(min(gt_mag.min(), pred_mag.min()))
    vmax = float(max(gt_mag.max(), pred_mag.max()))

    fig = plt.figure(figsize=(18, 5))
    panels: list[tuple[str, np.ndarray, float | None, float | None]] = [
        ("Ground Truth", gt_mag, vmin, vmax),
        ("Predicted", pred_mag, vmin, vmax),
        ("Absolute Error", error_mag, None, None),
    ]

    for i, (label, vals, lo, hi) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        sc = ax.scatter(
            mesh_pos[:, 0],
            mesh_pos[:, 1],
            mesh_pos[:, 2],
            c=vals,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(label)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)
