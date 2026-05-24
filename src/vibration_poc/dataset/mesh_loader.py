"""STL mesh loader — parse STL files into graph dicts for inference."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def load_stl(
    path: Path,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    np.ndarray[tuple[int, int], np.dtype[np.intp]],
]:
    """Parse binary STL -> (vertices [N,3], faces [F,3]).

    Deduplicates vertices. Returns unique vertex positions and face indices.
    """
    with open(path, "rb") as f:
        f.read(80)  # header
        num_triangles = struct.unpack("<I", f.read(4))[0]

        all_verts = np.zeros((num_triangles * 3, 3), dtype=np.float32)
        for i in range(num_triangles):
            f.read(12)  # normal vector (skip)
            for j in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                all_verts[i * 3 + j] = [x, y, z]
            f.read(2)  # attribute byte count

    # Deduplicate vertices
    unique_verts, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    return unique_verts, faces


def faces_to_edges(faces: np.ndarray[tuple[int, int], np.dtype[np.intp]]) -> Tensor:
    """Convert triangle faces [F,3] to undirected edge_index [2,E].

    Each triangle edge appears in both directions.
    Duplicate edges from shared faces are removed.
    """
    edges_set: set[tuple[int, int]] = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges_set.add((a, b))
            edges_set.add((b, a))

    edge_list = sorted(edges_set)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def mesh_to_graph(
    vertices: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    faces: np.ndarray[tuple[int, int], np.dtype[np.intp]],
    node_type_default: float = 0.0,
) -> dict[str, Tensor]:
    """Build graph dict from mesh vertices/faces.

    Returns dict with x, edge_index, edge_attr, mesh_pos —
    same format rollout() expects.
    """
    n = vertices.shape[0]
    world_pos = torch.tensor(vertices, dtype=torch.float32)  # [N, 3]
    node_type = torch.full((n, 1), node_type_default, dtype=torch.float32)
    x = torch.cat([world_pos, node_type], dim=1)  # [N, 4]

    edge_index = faces_to_edges(faces)  # [2, E]

    # Edge attributes: relative position vectors + distance
    src, dst = edge_index[0], edge_index[1]
    rel_pos = world_pos[dst] - world_pos[src]  # [E, 3]
    dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [E, 1]
    edge_attr = torch.cat([rel_pos, dist], dim=1)  # [E, 4]

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "mesh_pos": world_pos,
    }
