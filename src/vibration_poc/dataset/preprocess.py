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


def preprocess_split(
    split: str,
    config: DatasetConfig,
    stats: NormStats | None = None,
) -> tuple[list[dict[str, Tensor]], NormStats | None]:
    """Read TFRecord for split, build graphs for consecutive frame pairs.

    For each trajectory: iterate frames 0..T-2, build graph from frame t -> t+1.
    If stats provided, normalize. If split=="train" and stats is None, compute stats.
    Cache graphs to config.processed_dir/{split}/ as .pt files.
    """
    out_dir = config.processed_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = config.raw_dir / "meta.json"
    tfrecord_path = config.raw_dir / f"{split}.tfrecord"

    meta = load_meta(meta_path) if meta_path.exists() else {}

    graphs: list[dict[str, Tensor]] = []
    idx = 0
    for traj in parse_tfrecord(tfrecord_path, meta):
        world_pos: np.ndarray = traj["world_pos"]  # [T, N, 3]
        stress: np.ndarray = traj["stress"]  # [T, N, 1]
        mesh_pos_raw: np.ndarray = traj["mesh_pos"]
        node_type_raw: np.ndarray = traj["node_type"]
        cells_raw: np.ndarray = traj["cells"]
        mesh_pos = (
            mesh_pos_raw[0]
            if mesh_pos_raw.ndim == 3 and mesh_pos_raw.shape[0] == 1
            else mesh_pos_raw
        )
        node_type = (
            node_type_raw[0]
            if node_type_raw.ndim == 3 and node_type_raw.shape[0] == 1
            else node_type_raw
        )
        cells = cells_raw[0] if cells_raw.ndim == 3 and cells_raw.shape[0] == 1 else cells_raw

        T = world_pos.shape[0]
        for t in range(T - 1):
            g = build_graph(
                mesh_pos=mesh_pos,
                node_type=node_type,
                cells=cells,
                world_pos=world_pos[t],
                target_world_pos=world_pos[t + 1],
                target_stress=stress[t + 1],
            )
            graphs.append(g)
            idx += 1

    # Compute stats if training and none provided
    computed_stats: NormStats | None = None
    if split == "train" and stats is None and graphs:
        computed_stats = compute_norm_stats(graphs)
        stats = computed_stats

    # Normalize if stats available
    if stats is not None:
        graphs = [normalize_graph(g, stats) for g in graphs]

    # Cache to disk
    for i, g in enumerate(graphs):
        torch.save(g, out_dir / f"{i:06d}.pt")

    return graphs, computed_stats


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
