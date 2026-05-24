"""Dataset download utilities for the deforming-plate dataset."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from vibration_poc.dataset.config import DatasetConfig


def _should_download(dest: Path) -> bool:
    """Return True if dest is missing or empty (size == 0)."""
    return not dest.exists() or dest.stat().st_size == 0


def download_dataset(config: DatasetConfig) -> None:
    """Download meta.json and split tfrecords from GCS.

    Idempotent: skips files that already exist and have size > 0.
    Creates ``config.raw_dir`` if it does not exist.
    """
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, Path]] = [
        (f"{config.base_url}/meta.json", config.raw_dir / "meta.json"),
    ]
    for split in config.splits:
        filename = f"{split}.tfrecord"
        files.append((f"{config.base_url}/{filename}", config.raw_dir / filename))

    for url, dest in files:
        if _should_download(dest):
            urllib.request.urlretrieve(url, dest)
