"""Plotly 3D visualization helpers for Streamlit demo."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from torch import Tensor


def mesh_scatter_3d(
    positions: np.ndarray,
    values: np.ndarray,
    title: str = "",
    colorscale: str = "Viridis",
    colorbar_title: str = "Value",
) -> go.Figure:
    """Interactive 3D scatter plot colored by values.

    positions: [N, 3] xyz coordinates
    values: [N] scalar values for coloring
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "color": values,
                    "colorscale": colorscale,
                    "colorbar": {"title": colorbar_title},
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def rollout_animation_figure(
    rollout_results: list[dict[str, Tensor]],
) -> go.Figure:
    """Plotly figure with slider for stepping through rollout frames.

    Each frame shows 3D scatter colored by displacement magnitude at that step.
    """
    frames: list[go.Frame] = []

    # Build frames
    for i, step in enumerate(rollout_results):
        pos: np.ndarray = step["world_pos"].detach().cpu().numpy()
        disp: np.ndarray = step["predicted_displacement"].detach().cpu().numpy()
        mag: np.ndarray = np.linalg.norm(disp, axis=1)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        z=pos[:, 2],
                        mode="markers",
                        marker={
                            "size": 3,
                            "color": mag,
                            "colorscale": "Turbo",
                            "colorbar": {"title": "Displacement"},
                        },
                    )
                ],
                name=str(i),
            )
        )

    # Initial frame data
    first_pos: np.ndarray = rollout_results[0]["world_pos"].detach().cpu().numpy()
    first_disp: np.ndarray = rollout_results[0]["predicted_displacement"].detach().cpu().numpy()
    first_mag: np.ndarray = np.linalg.norm(first_disp, axis=1)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=first_pos[:, 0],
                y=first_pos[:, 1],
                z=first_pos[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "color": first_mag,
                    "colorscale": "Turbo",
                    "colorbar": {"title": "Displacement"},
                },
            )
        ],
        frames=frames,
    )

    # Slider
    sliders = [
        {
            "active": 0,
            "steps": [
                {
                    "method": "animate",
                    "args": [
                        [str(i)],
                        {"mode": "immediate", "frame": {"duration": 100}},
                    ],
                    "label": str(i),
                }
                for i in range(len(rollout_results))
            ],
            "currentvalue": {"prefix": "Step: "},
        }
    ]

    fig.update_layout(
        title="Rollout Animation",
        scene={"aspectmode": "data"},
        sliders=sliders,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def error_comparison_figure(
    mesh_pos: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> go.Figure:
    """3-subplot figure: GT displacement, predicted displacement, error magnitude.

    mesh_pos: [N, 3]
    ground_truth: [N, 3] displacement
    predicted: [N, 3] displacement
    """
    from plotly.subplots import make_subplots

    gt_mag: np.ndarray = np.linalg.norm(ground_truth, axis=1)
    pred_mag: np.ndarray = np.linalg.norm(predicted, axis=1)
    error_mag: np.ndarray = np.linalg.norm(ground_truth - predicted, axis=1)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[
            [
                {"type": "scatter3d"},
                {"type": "scatter3d"},
                {"type": "scatter3d"},
            ]
        ],
        subplot_titles=["Ground Truth", "Predicted", "Error"],
    )

    for col, (values, cscale) in enumerate(
        [(gt_mag, "Viridis"), (pred_mag, "Viridis"), (error_mag, "Reds")], 1
    ):
        fig.add_trace(
            go.Scatter3d(
                x=mesh_pos[:, 0],
                y=mesh_pos[:, 1],
                z=mesh_pos[:, 2],
                mode="markers",
                marker={"size": 3, "color": values, "colorscale": cscale},
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Error Comparison",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def frequency_spectrum_figure(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    top_k: int = 5,
) -> go.Figure:
    """Interactive frequency spectrum with peak markers.

    frequencies: [F] Hz
    magnitudes: [F, N] per-node magnitudes — we plot mean across nodes
    """
    mean_mag: np.ndarray = magnitudes.mean(axis=1) if magnitudes.ndim > 1 else magnitudes

    # Find top-k peaks
    top_indices: np.ndarray = np.argsort(mean_mag)[::-1][:top_k]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=mean_mag,
            mode="lines",
            name="Mean Magnitude",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies[top_indices],
            y=mean_mag[top_indices],
            mode="markers+text",
            marker={"size": 10, "color": "red"},
            text=[f"{frequencies[i]:.1f} Hz" for i in top_indices],
            textposition="top center",
            name="Peaks",
        )
    )

    fig.update_layout(
        title="Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
