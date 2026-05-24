"""TFRecord reader, graph builder, normalization, and full preprocess pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from vibration_poc.dataset.config import DatasetConfig, NormStats

# ---------------------------------------------------------------------------
# T4: TFRecord reader
# ---------------------------------------------------------------------------


def load_meta(meta_path: Path) -> dict[str, dict[str, str | list[int]]]:
    """Load meta.json schema — returns the features dict."""
    with open(meta_path) as f:
        raw: dict[str, object] = json.load(f)
    if "features" in raw:
        return raw["features"]  # type: ignore[return-value]
    return raw  # type: ignore[return-value]


def parse_tfrecord(
    path: Path,
    meta: dict[str, dict[str, str | list[int]]],
) -> Iterator[dict[str, np.ndarray]]:
    """Parse TFRecord into trajectory dicts using meta schema."""
    import tfrecord.reader

    loader = tfrecord.reader.tfrecord_loader(str(path), None, None)
    for record in loader:
        decoded: dict[str, np.ndarray] = {}
        for key, spec in meta.items():
            if key not in record:
                continue
            raw: bytes = record[key]
            dtype = str(spec["dtype"])
            raw_shape = spec["shape"]
            if isinstance(raw_shape, list):
                shape: tuple[int, ...] = tuple(int(s) for s in raw_shape)
            else:
                shape = (int(raw_shape),)
            decoded[key] = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
        yield decoded


# ---------------------------------------------------------------------------
# T5: Graph builder
# ---------------------------------------------------------------------------

_TET_PAIRS: list[tuple[int, int]] = list(combinations(range(4), 2))


def cells_to_edges(cells: np.ndarray) -> Tensor:
    """Convert tetrahedral cells [C,4] to undirected edge_index [2, E].

    Each tet (4 nodes) -> 6 edges; make bidirectional, deduplicate.
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    for tet in cells:
        for i, j in _TET_PAIRS:
            src_list.append(int(tet[i]))
            dst_list.append(int(tet[j]))
            src_list.append(int(tet[j]))
            dst_list.append(int(tet[i]))

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    edges = torch.stack([src, dst], dim=0)  # [2, E_raw]

    # Deduplicate
    edge_set: Tensor = edges.T.unique(dim=0)  # type: ignore[no-untyped-call]  # [E, 2]
    return edge_set.T.contiguous()  # [2, E]


def build_graph(
    mesh_pos: np.ndarray,
    node_type: np.ndarray,
    cells: np.ndarray,
    world_pos: np.ndarray,
    target_world_pos: np.ndarray,
    target_stress: np.ndarray,
) -> dict[str, Tensor]:
    """Build graph dict.

    Returns dict with keys:
    - x: [N,4] — world_pos(3) + node_type(1)
    - edge_index: [2,E]
    - edge_attr: [E,4] — relative mesh_pos(3) + L2 norm(1)
    - y: [N,3] — target_world_pos - world_pos
    - target_stress: [N,1]
    - mesh_pos: [N,3]
    """
    t_mesh_pos = torch.from_numpy(mesh_pos.astype(np.float32))  # [N,3]
    t_node_type = torch.from_numpy(node_type.astype(np.float32))  # [N,1]
    t_world_pos = torch.from_numpy(world_pos.astype(np.float32))  # [N,3]
    t_target_wp = torch.from_numpy(target_world_pos.astype(np.float32))  # [N,3]
    t_target_stress = torch.from_numpy(target_stress.astype(np.float32))  # [N,1]

    x = torch.cat([t_world_pos, t_node_type], dim=1)  # [N,4]
    y = t_target_wp - t_world_pos  # [N,3]

    edge_index = cells_to_edges(cells.astype(np.int64))  # [2,E]
    src, dst = edge_index[0], edge_index[1]
    rel_pos = t_mesh_pos[dst] - t_mesh_pos[src]  # [E,3]
    norm = rel_pos.norm(dim=1, keepdim=True)  # [E,1]
    edge_attr = torch.cat([rel_pos, norm], dim=1)  # [E,4]

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
        "target_stress": t_target_stress,
        "mesh_pos": t_mesh_pos,
    }


# ---------------------------------------------------------------------------
# T6: Normalization
# ---------------------------------------------------------------------------


def compute_norm_stats(graphs: list[dict[str, Tensor]]) -> NormStats:
    """Compute per-feature mean/std over list of graph dicts."""
    all_x = torch.cat([g["x"] for g in graphs], dim=0)  # [total_N, 4]
    all_e = torch.cat([g["edge_attr"] for g in graphs], dim=0)  # [total_E, 4]

    node_mean: list[float] = all_x.mean(dim=0).tolist()
    node_std_raw = all_x.std(dim=0)
    node_std: list[float] = torch.clamp(node_std_raw, min=1e-8).tolist()

    edge_mean: list[float] = all_e.mean(dim=0).tolist()
    edge_std_raw = all_e.std(dim=0)
    edge_std: list[float] = torch.clamp(edge_std_raw, min=1e-8).tolist()

    return NormStats(
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
    )


def normalize_graph(graph: dict[str, Tensor], stats: NormStats) -> dict[str, Tensor]:
    """Normalize x and edge_attr. Returns new dict (no mutation)."""
    node_mean = torch.tensor(stats.node_mean, dtype=torch.float32)
    node_std = torch.tensor(stats.node_std, dtype=torch.float32)
    edge_mean = torch.tensor(stats.edge_mean, dtype=torch.float32)
    edge_std = torch.tensor(stats.edge_std, dtype=torch.float32)

    result = dict(graph)
    result["x"] = (graph["x"] - node_mean) / node_std
    result["edge_attr"] = (graph["edge_attr"] - edge_mean) / edge_std
    return result


# ---------------------------------------------------------------------------
# T7: Full preprocess pipeline
# ---------------------------------------------------------------------------


def _squeeze_static(arr: np.ndarray) -> np.ndarray:
    """Remove leading dim=1 from static fields (e.g. [1,N,D] -> [N,D])."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        result: np.ndarray = arr[0]
        return result
    return arr


def _iter_graphs(
    tfrecord_path: Path,
    meta: dict[str, dict[str, str | list[int]]],
) -> Iterator[dict[str, Tensor]]:
    """Yield graph dicts from a TFRecord file, one per frame pair."""
    for traj in parse_tfrecord(tfrecord_path, meta):
        world_pos: np.ndarray = traj["world_pos"]
        stress: np.ndarray = traj["stress"]
        mesh_pos = _squeeze_static(traj["mesh_pos"])
        node_type = _squeeze_static(traj["node_type"])
        cells = _squeeze_static(traj["cells"])

        t_len = world_pos.shape[0]
        for t in range(t_len - 1):
            yield build_graph(
                mesh_pos=mesh_pos,
                node_type=node_type,
                cells=cells,
                world_pos=world_pos[t],
                target_world_pos=world_pos[t + 1],
                target_stress=stress[t + 1],
            )


def _compute_streaming_stats(graph_dir: Path) -> NormStats:
    """Compute norm stats from saved .pt files using streaming sums."""
    files = sorted(graph_dir.glob("*.pt"))
    x_sum = torch.zeros(4, dtype=torch.float64)
    x_sq_sum = torch.zeros(4, dtype=torch.float64)
    e_sum = torch.zeros(4, dtype=torch.float64)
    e_sq_sum = torch.zeros(4, dtype=torch.float64)
    n_nodes = 0
    n_edges = 0
    for f in files:
        g: dict[str, Tensor] = torch.load(f, weights_only=True)
        x = g["x"].double()
        e = g["edge_attr"].double()
        x_sum += x.sum(dim=0)
        x_sq_sum += (x**2).sum(dim=0)
        e_sum += e.sum(dim=0)
        e_sq_sum += (e**2).sum(dim=0)
        n_nodes += x.shape[0]
        n_edges += e.shape[0]
    x_mean = x_sum / n_nodes
    x_std = torch.clamp(torch.sqrt(x_sq_sum / n_nodes - x_mean**2), min=1e-8)
    e_mean = e_sum / n_edges
    e_std = torch.clamp(torch.sqrt(e_sq_sum / n_edges - e_mean**2), min=1e-8)
    return NormStats(
        node_mean=x_mean.float().tolist(),
        node_std=x_std.float().tolist(),
        edge_mean=e_mean.float().tolist(),
        edge_std=e_std.float().tolist(),
    )


def preprocess_split(
    split: str,
    config: DatasetConfig,
    stats: NormStats | None = None,
) -> tuple[list[dict[str, Tensor]], NormStats | None]:
    """Read TFRecord for split, build graphs for consecutive frame pairs.

    Streams graphs to disk one at a time to avoid OOM on large datasets.
    For train split without stats: saves raw graphs, computes stats from disk,
    then normalizes in a second pass.
    """
    out_dir = config.processed_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = config.raw_dir / "meta.json"
    tfrecord_path = config.raw_dir / f"{split}.tfrecord"

    meta = load_meta(meta_path) if meta_path.exists() else {}

    need_stats = split == "train" and stats is None

    # Pass 1: save graphs to disk (normalized if stats known, raw if not)
    idx = 0
    for graph in _iter_graphs(tfrecord_path, meta):
        saved = normalize_graph(graph, stats) if stats is not None else graph
        torch.save(saved, out_dir / f"{idx:06d}.pt")
        idx += 1

    # Compute stats from saved raw graphs, then normalize them in place
    computed_stats: NormStats | None = None
    if need_stats and idx > 0:
        computed_stats = _compute_streaming_stats(out_dir)
        for f in sorted(out_dir.glob("*.pt")):
            g = torch.load(f, weights_only=True)
            g = normalize_graph(g, computed_stats)
            torch.save(g, f)

    # Return empty list to avoid loading all graphs into memory
    return [], computed_stats


def preprocess_dataset(config: DatasetConfig) -> None:
    """Process all splits. Compute stats on train, apply to all. Save norm_stats.json."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    # Train first to get stats
    _, stats = preprocess_split("train", config, stats=None)

    # Remaining splits
    for split in config.splits:
        if split == "train":
            continue
        preprocess_split(split, config, stats=stats)

    # Save norm_stats
    if stats is not None:
        (config.processed_dir / "norm_stats.json").write_text(stats.model_dump_json())
