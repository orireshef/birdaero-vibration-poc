"""Pydantic configuration models for the deforming-plate dataset."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    raw_dir: Path = Path("data/raw/deforming_plate")
    processed_dir: Path = Path("data/processed/deforming_plate")
    base_url: str = "https://storage.googleapis.com/dm-meshgraphnets/deforming_plate"
    splits: list[str] = ["train", "valid", "test"]
    num_train_samples: int = Field(default=1000, gt=0)
    num_steps: int = Field(default=400, gt=0)
    max_trajectories: int | None = None


class NormStats(BaseModel):
    node_mean: list[float]
    node_std: list[float]
    edge_mean: list[float]
    edge_std: list[float]


class GridConfig(BaseModel):
    resolution: tuple[int, int, int] = (32, 64, 16)
    padding_ratio: float = Field(default=0.05, ge=0)
    modes: tuple[int, int, int] = (12, 12, 8)


class GridNormStats(BaseModel):
    channel_mean: list[float]
    channel_std: list[float]
    target_mean: list[float]
    target_std: list[float]
