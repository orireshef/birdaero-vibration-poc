"""Tests for dataset config and download — written BEFORE implementation (TDD RED)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Task 1: DatasetConfig & NormStats
# ---------------------------------------------------------------------------


class TestDatasetConfigDefaults:
    def test_default_raw_dir(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.raw_dir == Path("data/raw/deforming_plate")

    def test_default_processed_dir(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.processed_dir == Path("data/processed/deforming_plate")

    def test_default_base_url(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.base_url == "https://storage.googleapis.com/dm-meshgraphnets/deforming_plate"

    def test_default_splits(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.splits == ["train", "valid", "test"]

    def test_default_num_train_samples(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.num_train_samples == 1000

    def test_default_num_steps(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.num_steps == 400


class TestDatasetConfigCustomValues:
    def test_custom_raw_dir(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig(raw_dir=Path("/tmp/raw"))
        assert cfg.raw_dir == Path("/tmp/raw")

    def test_custom_splits(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig(splits=["train"])
        assert cfg.splits == ["train"]

    def test_custom_num_train_samples(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig(num_train_samples=500)
        assert cfg.num_train_samples == 500

    def test_num_train_samples_must_be_positive(self) -> None:
        from pydantic import ValidationError

        from vibration_poc.dataset.config import DatasetConfig

        with pytest.raises(ValidationError):
            DatasetConfig(num_train_samples=0)

    def test_num_steps_must_be_positive(self) -> None:
        from pydantic import ValidationError

        from vibration_poc.dataset.config import DatasetConfig

        with pytest.raises(ValidationError):
            DatasetConfig(num_steps=-1)

    def test_string_path_coerced(self) -> None:
        from vibration_poc.dataset.config import DatasetConfig

        cfg = DatasetConfig(raw_dir="some/path")  # type: ignore[arg-type]
        assert cfg.raw_dir == Path("some/path")


class TestNormStatsRoundtrip:
    def test_json_roundtrip(self) -> None:
        from vibration_poc.dataset.config import NormStats

        stats = NormStats(
            node_mean=[0.1, 0.2],
            node_std=[1.0, 1.1],
            edge_mean=[0.3],
            edge_std=[0.9],
        )
        data = json.loads(stats.model_dump_json())
        restored = NormStats(**data)
        assert restored == stats

    def test_fields_preserved(self) -> None:
        from vibration_poc.dataset.config import NormStats

        stats = NormStats(node_mean=[1.0], node_std=[2.0], edge_mean=[3.0], edge_std=[4.0])
        assert stats.node_mean == [1.0]
        assert stats.node_std == [2.0]
        assert stats.edge_mean == [3.0]
        assert stats.edge_std == [4.0]


# ---------------------------------------------------------------------------
# Task 2: download_dataset
# ---------------------------------------------------------------------------


class TestDownloadDataset:
    def test_calls_urlretrieve_for_meta_json(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        cfg = DatasetConfig(raw_dir=tmp_path, splits=[])
        with patch("urllib.request.urlretrieve") as mock_dl:
            download_dataset(cfg)
        expected_url = f"{cfg.base_url}/meta.json"
        urls_called = [c.args[0] for c in mock_dl.call_args_list]
        assert expected_url in urls_called

    def test_calls_urlretrieve_for_each_split(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        cfg = DatasetConfig(raw_dir=tmp_path, splits=["train", "valid"])
        with patch("urllib.request.urlretrieve") as mock_dl:
            download_dataset(cfg)
        urls_called = [c.args[0] for c in mock_dl.call_args_list]
        assert f"{cfg.base_url}/train.tfrecord" in urls_called
        assert f"{cfg.base_url}/valid.tfrecord" in urls_called

    def test_creates_raw_dir(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        target = tmp_path / "deep" / "nested"
        cfg = DatasetConfig(raw_dir=target, splits=[])
        with patch("urllib.request.urlretrieve"):
            download_dataset(cfg)
        assert target.exists()

    def test_skip_existing_nonempty_file(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        cfg = DatasetConfig(raw_dir=tmp_path, splits=[])
        # Pre-create meta.json with content
        meta = tmp_path / "meta.json"
        meta.write_text('{"exists": true}')
        with patch("urllib.request.urlretrieve") as mock_dl:
            download_dataset(cfg)
        urls_called = [c.args[0] for c in mock_dl.call_args_list]
        assert f"{cfg.base_url}/meta.json" not in urls_called

    def test_redownload_empty_file(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        cfg = DatasetConfig(raw_dir=tmp_path, splits=[])
        meta = tmp_path / "meta.json"
        meta.write_text("")  # empty → should re-download
        with patch("urllib.request.urlretrieve") as mock_dl:
            download_dataset(cfg)
        urls_called = [c.args[0] for c in mock_dl.call_args_list]
        assert f"{cfg.base_url}/meta.json" in urls_called

    def test_correct_dest_path(self, tmp_path: Path) -> None:
        from vibration_poc.dataset.config import DatasetConfig
        from vibration_poc.dataset.download import download_dataset

        cfg = DatasetConfig(raw_dir=tmp_path, splits=["test"])
        with patch("urllib.request.urlretrieve") as mock_dl:
            download_dataset(cfg)
        dest_paths = [str(c.args[1]) for c in mock_dl.call_args_list]
        assert str(tmp_path / "test.tfrecord") in dest_paths
