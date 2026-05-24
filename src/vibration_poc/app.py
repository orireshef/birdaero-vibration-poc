"""Bird Aero Vibration Analysis Platform — Streamlit demo."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from torch import Tensor

from vibration_poc.dataset.config import NormStats
from vibration_poc.dataset.mesh_loader import load_stl, mesh_to_graph
from vibration_poc.engine import DesignMetrics, evaluate_design, load_model
from vibration_poc.inference.analyze import compute_fft, displacement_time_series
from vibration_poc.visualization.plotly_viz import (
    error_comparison_figure,
    frequency_spectrum_figure,
    mesh_scatter_3d,
)


def _parse_uploaded_mesh(mesh_file: Any) -> dict[str, Tensor] | None:
    """Parse uploaded .pt or .stl file into a graph dict."""
    name: str = mesh_file.name
    if name.endswith(".pt"):
        graph: dict[str, Tensor] = torch.load(mesh_file, map_location="cpu", weights_only=False)
        return graph
    if name.endswith(".stl"):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp.write(mesh_file.read())
            tmp_path = Path(tmp.name)
        vertices, faces = load_stl(tmp_path)
        return mesh_to_graph(vertices, faces)
    return None


def _has_ground_truth(graph: dict[str, Tensor]) -> bool:
    return "y" in graph


def _render_sidebar() -> tuple[bool, str, int, int, str, int, str, bool, str]:
    """Render sidebar controls. Returns config tuple."""
    with st.sidebar:
        st.header("Configuration")

        # File upload
        st.subheader("Mesh Input")
        mesh_file = st.file_uploader("Upload mesh (.pt or .stl)", type=["pt", "stl"])

        if mesh_file is not None:
            graph = _parse_uploaded_mesh(mesh_file)
            if graph is not None:
                st.session_state["graph"] = graph

        # Model checkpoint
        st.subheader("Model")
        checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/best.pt")
        hidden_dim = st.number_input("Hidden dim", value=64, min_value=8, step=8)
        num_layers = st.number_input("Num layers", value=8, min_value=1, step=1)

        # Norm stats
        norm_stats_path = st.text_input(
            "Norm stats path",
            value="data/processed/deforming_plate/norm_stats.json",
        )

        # Simulation config
        st.subheader("Simulation")
        num_steps = st.slider("Rollout steps", min_value=1, max_value=200, value=50)
        device = st.selectbox("Device", ["cpu", "cuda"])
        assert isinstance(device, str)

        # Physics
        st.subheader("Physics")
        enforce_bc = st.checkbox("Enforce boundary conditions", value=False)
        bc_types_str = st.text_input("BC node types (comma-separated)", value="1,2,3")

        # Run button
        run_clicked = st.button("Run Inference", type="primary", use_container_width=True)

        # Results summary
        if "metrics" in st.session_state:
            st.subheader("Results")
            m: DesignMetrics = st.session_state["metrics"]
            st.metric("Max Displacement", f"{m.max_displacement:.4f}")
            st.metric("Mean Displacement", f"{m.mean_displacement:.4f}")
            if "run_time" in st.session_state:
                run_time: float = st.session_state["run_time"]
                st.metric("Run Time", f"{run_time:.1f}s")

    return (
        run_clicked,
        str(checkpoint_path),
        int(hidden_dim),
        int(num_layers),
        str(norm_stats_path),
        int(num_steps),
        device,
        bool(enforce_bc),
        str(bc_types_str),
    )


def _run_inference(
    graph: dict[str, Tensor],
    checkpoint_path: str,
    hidden_dim: int,
    num_layers: int,
    norm_stats_path: str,
    num_steps: int,
    device: str,
    enforce_bc: bool,
    bc_types_str: str,
) -> None:
    """Load model, run rollout, cache results in session_state."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        return

    ns_path = Path(norm_stats_path)
    if not ns_path.exists():
        st.error(f"Norm stats not found: {ns_path}")
        return

    with open(ns_path) as f:
        ns_data: dict[str, list[float]] = json.load(f)
    norm_stats = NormStats(**ns_data)

    with st.spinner("Loading model..."):
        model = load_model(
            ckpt_path,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
        )

    bc_node_types: list[int] | None = None
    if enforce_bc:
        bc_node_types = [int(x.strip()) for x in bc_types_str.split(",") if x.strip()]

    with st.spinner(f"Running rollout ({num_steps} steps)..."):
        start_time = time.time()
        metrics, results = evaluate_design(
            graph,
            model,
            norm_stats,
            num_steps=num_steps,
            bc_node_types=bc_node_types,
        )
        elapsed = time.time() - start_time

    st.session_state["metrics"] = metrics
    st.session_state["results"] = results
    st.session_state["run_time"] = elapsed

    # Compute FFT
    ts = displacement_time_series(results)
    freqs, mags = compute_fft(ts)
    st.session_state["fft_freqs"] = freqs
    st.session_state["fft_mags"] = mags

    st.success(f"Done in {elapsed:.1f}s")
    st.rerun()


def _tab_mesh_viewer() -> None:
    """Render mesh viewer tab."""
    graph: dict[str, Tensor] | None = st.session_state.get("graph")
    if graph is not None:
        positions: np.ndarray = graph["x"][:, :3].detach().cpu().numpy()
        node_types: np.ndarray = graph["x"][:, 3].detach().cpu().numpy()
        fig = mesh_scatter_3d(
            positions,
            node_types,
            title="Mesh — Node Types",
            colorbar_title="Node Type",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Nodes: {positions.shape[0]} | Edges: {graph['edge_index'].shape[1]}")
    else:
        st.info("Upload a mesh file to visualize.")


def _tab_rollout() -> None:
    """Render rollout tab."""
    if "results" not in st.session_state:
        st.info("Run inference to see rollout results.")
        return

    results: list[dict[str, Tensor]] = st.session_state["results"]
    step_idx = st.slider("Step", 0, len(results) - 1, 0, key="step_slider")
    step = results[step_idx]

    pos: np.ndarray = step["world_pos"].detach().cpu().numpy()
    disp: np.ndarray = step["predicted_displacement"].detach().cpu().numpy()
    mag: np.ndarray = np.linalg.norm(disp, axis=1)

    fig = mesh_scatter_3d(
        pos,
        mag,
        title=f"Step {step_idx} — Displacement Magnitude",
        colorscale="Turbo",
        colorbar_title="Displacement",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Displacement over time chart
    ts = displacement_time_series(results)
    mean_disp: np.ndarray = ts.mean(axis=1)
    st.line_chart(mean_disp, x_label="Step", y_label="Mean Displacement")


def _tab_frequency() -> None:
    """Render frequency analysis tab."""
    if "fft_freqs" not in st.session_state:
        st.info("Run inference to see frequency analysis.")
        return

    freqs: np.ndarray = st.session_state["fft_freqs"]
    mags: np.ndarray = st.session_state["fft_mags"]
    fig = frequency_spectrum_figure(freqs, mags)
    st.plotly_chart(fig, use_container_width=True)

    if "metrics" in st.session_state:
        st.subheader("Dominant Frequencies")
        m: DesignMetrics = st.session_state["metrics"]
        for freq, mag_val in m.dominant_frequencies:
            st.write(f"**{freq:.2f} Hz** — magnitude {mag_val:.4f}")


def _tab_error() -> None:
    """Render error analysis tab."""
    if "results" not in st.session_state:
        st.info("Run inference with a .pt file containing ground truth to see error analysis.")
        return

    graph: dict[str, Tensor] | None = st.session_state.get("graph")
    if graph is None or not _has_ground_truth(graph):
        st.info("Ground truth not available in uploaded file.")
        return

    results: list[dict[str, Tensor]] = st.session_state["results"]
    mesh_pos: np.ndarray = graph["mesh_pos"].detach().cpu().numpy()
    gt: np.ndarray = graph["y"].detach().cpu().numpy()
    pred: np.ndarray = results[0]["predicted_displacement"].detach().cpu().numpy()
    fig = error_comparison_figure(mesh_pos, gt, pred)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Entry point for the Streamlit demo app."""
    st.set_page_config(page_title="Bird Aero Vibration Analysis", layout="wide")
    st.title("Bird Aero Vibration Analysis Platform")

    (
        run_clicked,
        checkpoint_path,
        hidden_dim,
        num_layers,
        norm_stats_path,
        num_steps,
        device,
        enforce_bc,
        bc_types_str,
    ) = _render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Mesh Viewer", "Rollout", "Frequency Analysis", "Error Analysis"]
    )

    with tab1:
        _tab_mesh_viewer()

    # Run inference if requested
    graph: dict[str, Tensor] | None = st.session_state.get("graph")
    if run_clicked and graph is not None:
        _run_inference(
            graph,
            checkpoint_path,
            hidden_dim,
            num_layers,
            norm_stats_path,
            num_steps,
            device,
            enforce_bc,
            bc_types_str,
        )

    with tab2:
        _tab_rollout()

    with tab3:
        _tab_frequency()

    with tab4:
        _tab_error()


if __name__ == "__main__":
    main()
