"""Tests for preprocess.py — TDD RED phase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# T4: load_meta
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_returns_dict(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import load_meta

        meta = {"mesh_pos": {"shape": [10, 3], "dtype": "float32", "type": "static"}}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        result = load_meta(tmp_path / "meta.json")
        assert isinstance(result, dict)

    def test_fields_preserved(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import load_meta

        meta = {
            "mesh_pos": {"shape": [10, 3], "dtype": "float32", "type": "static"},
            "world_pos": {"shape": [400, 10, 3], "dtype": "float32", "type": "dynamic"},
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        result = load_meta(tmp_path / "meta.json")
        assert "mesh_pos" in result
        assert result["mesh_pos"]["dtype"] == "float32"
        assert result["world_pos"]["type"] == "dynamic"

    def test_shape_is_list(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import load_meta

        meta = {"cells": {"shape": [5, 4], "dtype": "int32", "type": "static"}}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        result = load_meta(tmp_path / "meta.json")
        assert result["cells"]["shape"] == [5, 4]


# ---------------------------------------------------------------------------
# T4: parse_tfrecord
# ---------------------------------------------------------------------------


def _make_meta() -> dict[str, dict[str, Any]]:
    return {
        "mesh_pos": {"shape": [3, 3], "dtype": "float32", "type": "static"},
        "node_type": {"shape": [3, 1], "dtype": "int32", "type": "static"},
        "cells": {"shape": [1, 4], "dtype": "int32", "type": "static"},
        "world_pos": {"shape": [2, 3, 3], "dtype": "float32", "type": "dynamic"},
        "stress": {"shape": [2, 3, 1], "dtype": "float32", "type": "dynamic"},
    }


def _make_fake_record(meta: dict[str, dict[str, Any]]) -> dict[str, bytes]:
    record: dict[str, bytes] = {}
    for key, spec in meta.items():
        arr = np.zeros(spec["shape"], dtype=spec["dtype"])  # type: ignore[arg-type]
        record[key] = arr.tobytes()
    return record


class TestParseTfrecord:
    def test_returns_iterator(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import parse_tfrecord

        meta = _make_meta()
        fake_record = _make_fake_record(meta)

        with patch("tfrecord.reader.tfrecord_loader", return_value=iter([fake_record])):
            result = parse_tfrecord(tmp_path / "train.tfrecord", meta)
        import collections.abc

        assert isinstance(result, collections.abc.Iterator)

    def test_yields_dict_with_arrays(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import parse_tfrecord

        meta = _make_meta()
        fake_record = _make_fake_record(meta)

        with patch("tfrecord.reader.tfrecord_loader", return_value=iter([fake_record])):
            records = list(parse_tfrecord(tmp_path / "train.tfrecord", meta))

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, dict)
        for key in meta:
            assert key in rec
            assert isinstance(rec[key], np.ndarray)

    def test_shapes_match_meta(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import parse_tfrecord

        meta = _make_meta()
        fake_record = _make_fake_record(meta)

        with patch("tfrecord.reader.tfrecord_loader", return_value=iter([fake_record])):
            records = list(parse_tfrecord(tmp_path / "train.tfrecord", meta))

        rec = records[0]
        for key, spec in meta.items():
            assert list(rec[key].shape) == spec["shape"], f"{key} shape mismatch"

    def test_dtypes_match_meta(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import parse_tfrecord

        meta = _make_meta()
        fake_record = _make_fake_record(meta)

        with patch("tfrecord.reader.tfrecord_loader", return_value=iter([fake_record])):
            records = list(parse_tfrecord(tmp_path / "train.tfrecord", meta))

        rec = records[0]
        for key, spec in meta.items():
            assert str(rec[key].dtype) == spec["dtype"], f"{key} dtype mismatch"

    def test_multiple_records(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.preprocess import parse_tfrecord

        meta = _make_meta()
        fake_records = [_make_fake_record(meta), _make_fake_record(meta)]

        with patch("tfrecord.reader.tfrecord_loader", return_value=iter(fake_records)):
            records = list(parse_tfrecord(tmp_path / "train.tfrecord", meta))

        assert len(records) == 2


# ---------------------------------------------------------------------------
# T5: cells_to_edges
# ---------------------------------------------------------------------------


class TestCellsToEdges:
    def test_one_tet_edge_count(self) -> None:
        from vibration_poc.dataset.preprocess import cells_to_edges

        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
        edge_index = cells_to_edges(cells)
        # 6 undirected * 2 = 12
        assert edge_index.shape == (2, 12)

    def test_bidirectional(self) -> None:
        from vibration_poc.dataset.preprocess import cells_to_edges

        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
        edge_index = cells_to_edges(cells)
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst, strict=True):
            assert d in src, f"reverse of ({s},{d}) missing from src"
            assert s in dst, f"reverse of ({s},{d}) missing from dst"

    def test_no_self_loops(self) -> None:
        from vibration_poc.dataset.preprocess import cells_to_edges

        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
        edge_index = cells_to_edges(cells)
        assert not (edge_index[0] == edge_index[1]).any()

    def test_two_tets_deduplication(self) -> None:
        from vibration_poc.dataset.preprocess import cells_to_edges

        # Two tets sharing face 0-1-2 → fewer total edges than 2*12
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
        edge_index = cells_to_edges(cells)
        n_edges = edge_index.shape[1]
        # 7 unique undirected edges * 2 = 14  (6+6 - 3 shared face edges = 9, < 24)
        assert n_edges < 24, "should deduplicate"

    def test_returns_tensor(self) -> None:
        from vibration_poc.dataset.preprocess import cells_to_edges

        cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
        edge_index = cells_to_edges(cells)
        assert isinstance(edge_index, torch.Tensor)


# ---------------------------------------------------------------------------
# T5: build_graph
# ---------------------------------------------------------------------------


def _one_tet_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    N = 4
    mesh_pos = np.random.randn(N, 3).astype(np.float32)
    node_type = np.zeros((N, 1), dtype=np.float32)
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    world_pos = np.random.randn(N, 3).astype(np.float32)
    target_world_pos = np.random.randn(N, 3).astype(np.float32)
    target_stress = np.random.randn(N, 1).astype(np.float32)
    return mesh_pos, node_type, cells, world_pos, target_world_pos, target_stress


class TestBuildGraph:
    def test_x_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["x"].shape == (4, 4)

    def test_edge_index_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["edge_index"].shape == (2, 12)

    def test_edge_attr_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["edge_attr"].shape == (12, 4)

    def test_y_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["y"].shape == (4, 3)

    def test_y_is_velocity(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, target_world_pos, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, target_world_pos, ts)
        expected = torch.from_numpy(target_world_pos - world_pos)
        assert torch.allclose(g["y"], expected)

    def test_target_stress_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["target_stress"].shape == (4, 1)

    def test_mesh_pos_shape(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert g["mesh_pos"].shape == (4, 3)

    def test_edge_attr_last_col_positive(self) -> None:
        """Last col of edge_attr should be L2 norm (non-negative)."""
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        assert (g["edge_attr"][:, 3] >= 0).all()

    def test_all_values_are_tensors(self) -> None:
        from vibration_poc.dataset.preprocess import build_graph

        mesh_pos, node_type, cells, world_pos, twp, ts = _one_tet_inputs()
        g = build_graph(mesh_pos, node_type, cells, world_pos, twp, ts)
        for k, v in g.items():
            assert isinstance(v, torch.Tensor), f"{k} not a Tensor"


# ---------------------------------------------------------------------------
# T6: compute_norm_stats / normalize_graph
# ---------------------------------------------------------------------------


def _make_test_graphs() -> list[dict[str, torch.Tensor]]:
    """3 tiny graphs with known values."""
    graphs = []
    for i in range(3):
        x = torch.full((4, 4), float(i + 1))
        edge_attr = torch.full((12, 4), float(i + 1) * 2)
        graphs.append(
            {
                "x": x,
                "edge_attr": edge_attr,
                "edge_index": torch.zeros(2, 12, dtype=torch.long),
                "y": torch.zeros(4, 3),
                "target_stress": torch.zeros(4, 1),
                "mesh_pos": torch.zeros(4, 3),
            }
        )
    return graphs


class TestComputeNormStats:
    def test_returns_norm_stats(self) -> None:
        from vibration_poc.dataset.config import NormStats
        from vibration_poc.dataset.preprocess import compute_norm_stats

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        assert isinstance(stats, NormStats)

    def test_node_mean_length(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        assert len(stats.node_mean) == 4  # 4 node features

    def test_edge_mean_length(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        assert len(stats.edge_mean) == 4  # 4 edge features

    def test_node_mean_value(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        # x values are 1,2,3 → mean=2.0
        assert pytest.approx(stats.node_mean[0], abs=1e-4) == 2.0

    def test_std_min_floor(self) -> None:
        """Std should be at least 1e-8."""
        from vibration_poc.dataset.preprocess import compute_norm_stats

        # All same value → computed std=0, should be floored to 1e-8
        graphs = [
            {
                "x": torch.ones(4, 4),
                "edge_attr": torch.ones(12, 4),
                "edge_index": torch.zeros(2, 12, dtype=torch.long),
                "y": torch.zeros(4, 3),
                "target_stress": torch.zeros(4, 1),
                "mesh_pos": torch.zeros(4, 3),
            }
        ]
        stats = compute_norm_stats(graphs)
        for v in stats.node_std:
            assert v >= 1e-9  # floored; float32 precision may give ~9.999e-9
        for v in stats.edge_std:
            assert v >= 1e-9


class TestNormalizeGraph:
    def test_returns_new_dict(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats, normalize_graph

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        g = graphs[0]
        result = normalize_graph(g, stats)
        assert result is not g

    def test_x_normalized(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats, normalize_graph

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        g = graphs[1]  # x=2, mean=2 → normalized x should be ~0
        result = normalize_graph(g, stats)
        assert torch.allclose(result["x"], torch.zeros_like(result["x"]), atol=1e-4)

    def test_original_not_mutated(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats, normalize_graph

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        g = graphs[0]
        original_x = g["x"].clone()
        normalize_graph(g, stats)
        assert torch.equal(g["x"], original_x)

    def test_non_feature_keys_preserved(self) -> None:
        from vibration_poc.dataset.preprocess import compute_norm_stats, normalize_graph

        graphs = _make_test_graphs()
        stats = compute_norm_stats(graphs)
        g = graphs[0]
        result = normalize_graph(g, stats)
        assert "edge_index" in result
        assert "y" in result
        assert "mesh_pos" in result


# ---------------------------------------------------------------------------
# T7: preprocess_split / preprocess_dataset
# ---------------------------------------------------------------------------


def _minimal_trajectory() -> dict[str, np.ndarray]:
    """2-timestep trajectory with 4 nodes, 1 tet."""
    return {
        "mesh_pos": np.random.randn(4, 3).astype(np.float32),
        "node_type": np.zeros((4, 1), dtype=np.int32),
        "cells": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "world_pos": np.random.randn(2, 4, 3).astype(np.float32),
        "stress": np.random.randn(2, 4, 1).astype(np.float32),
    }


class TestPreprocessSplit:
    def test_creates_pt_files(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.preprocess import preprocess_split

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["train"],
        )
        cfg.processed_dir.mkdir(parents=True)
        (cfg.raw_dir).mkdir(parents=True)

        traj = _minimal_trajectory()
        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                return_value=iter([traj]),
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            graphs, _ = preprocess_split("train", cfg, stats=None)

        pt_dir = cfg.processed_dir / "train"
        assert pt_dir.exists()
        pt_files = list(pt_dir.glob("*.pt"))
        # 2 timesteps → 1 frame pair (t=0→1)
        assert len(pt_files) == len(graphs)

    def test_returns_graphs_list(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.preprocess import preprocess_split

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["train"],
        )
        cfg.processed_dir.mkdir(parents=True)
        (cfg.raw_dir).mkdir(parents=True)

        traj = _minimal_trajectory()
        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                return_value=iter([traj]),
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            graphs, _stats = preprocess_split("train", cfg, stats=None)

        assert isinstance(graphs, list)
        assert len(graphs) == 1  # 2 frames → 1 pair

    def test_computes_stats_for_train(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig, NormStats
        from vibration_poc.dataset.preprocess import preprocess_split

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["train"],
        )
        cfg.processed_dir.mkdir(parents=True)
        (cfg.raw_dir).mkdir(parents=True)

        traj = _minimal_trajectory()
        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                return_value=iter([traj]),
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            _, stats = preprocess_split("train", cfg, stats=None)

        assert isinstance(stats, NormStats)

    def test_applies_provided_stats(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig, NormStats
        from vibration_poc.dataset.preprocess import preprocess_split

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["valid"],
        )
        cfg.processed_dir.mkdir(parents=True)
        (cfg.raw_dir).mkdir(parents=True)

        stats = NormStats(
            node_mean=[0.0] * 4,
            node_std=[1.0] * 4,
            edge_mean=[0.0] * 4,
            edge_std=[1.0] * 4,
        )
        traj = _minimal_trajectory()
        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                return_value=iter([traj]),
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            _, returned_stats = preprocess_split("valid", cfg, stats=stats)

        assert returned_stats is None  # non-train with provided stats returns None


class TestPreprocessDataset:
    def test_saves_norm_stats_json(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.preprocess import preprocess_dataset

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["train"],
        )
        (cfg.raw_dir).mkdir(parents=True)

        traj = _minimal_trajectory()
        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                return_value=iter([traj]),
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            preprocess_dataset(cfg)

        assert (cfg.processed_dir / "norm_stats.json").exists()

    def test_processes_all_splits(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.preprocess import preprocess_dataset

        cfg = DatasetConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            splits=["train", "valid"],
        )
        (cfg.raw_dir).mkdir(parents=True)

        traj = _minimal_trajectory()

        def _fake_parse(*_args: object, **_kwargs: object) -> list[dict[str, np.ndarray]]:
            return [traj]

        with (
            patch(
                "vibration_poc.dataset.preprocess.parse_tfrecord",
                side_effect=_fake_parse,
            ),
            patch(
                "vibration_poc.dataset.preprocess.load_meta",
                return_value={},
            ),
        ):
            preprocess_dataset(cfg)

        for split in ["train", "valid"]:
            assert (cfg.processed_dir / split).exists()
