"""Training entry point."""

from __future__ import annotations

from pathlib import Path

import yaml

from vibration_poc.dataset.config import DatasetConfig
from vibration_poc.training.trainer import TrainingConfig, train


def main() -> None:
    config_path = Path("configs/poc.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    training_dict = raw.get("training", {})
    if "physics" in raw:
        training_dict["physics"] = raw["physics"]
    training_config = TrainingConfig(**training_dict)
    dataset_config = DatasetConfig(**raw.get("dataset", {}))
    best = train(training_config, dataset_config)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
