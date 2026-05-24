"""Tests for STL mesh loader."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from vibration_poc.dataset.mesh_loader import faces_to_edges, load_stl, mesh_to_graph


def _write_binary_stl(
    path: Path,
    triangles: list[tuple[list[float], list[float], list[float]]],
) -> None:
    """Write minimal binary STL file."""
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)  # header
        f.write(struct.pack("<I", len(triangles)))
        for v0, v1, v2 in triangles:
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))  # normal
            for v in [v0, v1, v2]:
                f.write(struct.pack("<fff", *v))
            f.write(struct.pack("<H", 0))  # attr byte count


# Two triangles sharing an edge: a square in XY plane
SQUARE_TRIS: list[tuple[list[float], list[float], list[float]]] = [
    ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
]


@pytest.fixture
def stl_path(tmp_path: Path) -> Path:
    """Create a temp STL with 2 triangles (square)."""
    p = tmp_path / "test.stl"
    _write_binary_stl(p, SQUARE_TRIS)
    return p


class TestLoadStl:
    """Tests for load_stl."""

    def test_shapes(self, stl_path: Path) -> None:
        verts, faces = load_stl(stl_path)
        # 2 triangles, 6 raw verts, but 4 unique corners
        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)

    def test_vertex_dedup(self, stl_path: Path) -> None:
        verts, _faces = load_stl(stl_path)
        # Should have exactly 4 unique vertices for a square
        assert len(verts) == 4

    def test_vertex_dtype(self, stl_path: Path) -> None:
        verts, _faces = load_stl(stl_path)
        assert verts.dtype == np.float32

    def test_single_triangle(self, tmp_path: Path) -> None:
        p = tmp_path / "tri.stl"
        _write_binary_stl(p, [([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])])
        verts, faces = load_stl(p)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)


class TestFacesToEdges:
    """Tests for faces_to_edges."""

    def test_undirected(self) -> None:
        faces = np.array([[0, 1, 2]])
        edge_index = faces_to_edges(faces)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
        # Each edge should appear in both directions
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            assert (a, b) in edges
            assert (b, a) in edges

    def test_single_triangle_count(self) -> None:
        faces = np.array([[0, 1, 2]])
        edge_index = faces_to_edges(faces)
        # 1 triangle = 3 undirected edges = 6 directed
        assert edge_index.shape == (2, 6)

    def test_shared_edge_dedup(self) -> None:
        # Two triangles sharing edge (0,1): should not double-count shared edges
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        edge_index = faces_to_edges(faces)
        # 5 undirected edges = 10 directed
        # edges: 0-1 (shared), 1-2, 0-2, 1-3, 0-3
        assert edge_index.shape == (2, 10)

    def test_output_contiguous(self) -> None:
        faces = np.array([[0, 1, 2]])
        edge_index = faces_to_edges(faces)
        assert edge_index.is_contiguous()
        assert edge_index.dtype == torch.long


class TestMeshToGraph:
    """Tests for mesh_to_graph."""

    def test_keys(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        graph = mesh_to_graph(verts, faces)
        assert set(graph.keys()) == {"x", "edge_index", "edge_attr", "mesh_pos"}

    def test_shapes(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        graph = mesh_to_graph(verts, faces)
        n, e = 3, 6  # 3 nodes, 6 directed edges
        assert graph["x"].shape == (n, 4)  # 3 pos + 1 node_type
        assert graph["edge_index"].shape == (2, e)
        assert graph["edge_attr"].shape == (e, 4)  # 3 rel_pos + 1 dist
        assert graph["mesh_pos"].shape == (n, 3)

    def test_edge_attr_dim(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        graph = mesh_to_graph(verts, faces)
        assert graph["edge_attr"].shape[1] == 4

    def test_node_type_default(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        graph = mesh_to_graph(verts, faces, node_type_default=5.0)
        # Last column of x should be node_type
        assert torch.all(graph["x"][:, 3] == 5.0)

    def test_mesh_pos_matches_vertices(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        graph = mesh_to_graph(verts, faces)
        np.testing.assert_array_equal(graph["mesh_pos"].numpy(), verts)
