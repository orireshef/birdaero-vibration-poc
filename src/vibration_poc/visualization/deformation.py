"""Deformation snapshot and animated GIF visualization."""
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


def plot_deformation_snapshot(
    mesh_pos: np.ndarray,
    displacement: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """3D scatter plot colored by displacement magnitude."""
    mag: np.ndarray = np.linalg.norm(displacement, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        mesh_pos[:, 0],
        mesh_pos[:, 1],
        mesh_pos[:, 2],
        c=mag,
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax, label="Displacement magnitude")
    if title:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def create_deformation_gif(
    rollout_results: list[dict[str, Tensor]],
    save_path: Path,
    fps: int = 10,
) -> None:
    """Create animated GIF from rollout results."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Pre-compute numpy arrays
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for step in rollout_results:
        pos: np.ndarray = step["mesh_pos"].detach().cpu().numpy()
        disp: np.ndarray = step["predicted_displacement"].detach().cpu().numpy()
        frames.append((pos, disp))

    # Compute global color limits
    all_mags = [np.linalg.norm(d, axis=1) for _, d in frames]
    vmin = float(min(m.min() for m in all_mags))
    vmax = float(max(m.max() for m in all_mags))

    def update(frame_idx: int) -> list[Any]:
        ax.clear()
        pos, disp = frames[frame_idx]
        mag: np.ndarray = np.linalg.norm(disp, axis=1)
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=mag,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Step {frame_idx}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(str(save_path), writer=writer)
    plt.close(fig)
