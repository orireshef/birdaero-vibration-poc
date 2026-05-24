"""CLI rollout script for MeshGraphNet inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vibration_poc.dataset.config import NormStats
from vibration_poc.dataset.dataloader import GraphDataset
from vibration_poc.inference.analyze import displacement_time_series
from vibration_poc.inference.predict import rollout
from vibration_poc.model.meshgraphnet import MeshGraphNet


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="MeshGraphNet rollout prediction")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint .pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/deforming_plate"),
        help="Processed data dir",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--num-steps", type=int, default=50, help="Rollout steps")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rollout"),
        help="Output directory",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dim")
    parser.add_argument("--num-layers", type=int, default=8, help="MP layers")
    parser.add_argument(
        "--bc-node-types",
        type=int,
        nargs="+",
        default=None,
        help="Boundary node types to zero during rollout",
    )
    return parser.parse_args()


def main() -> None:
    """Run rollout prediction and save results."""
    args = parse_args()

    device = torch.device(args.device)

    # Load model
    model = MeshGraphNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load first graph from dataset
    dataset = GraphDataset(args.data_dir, args.split)
    if len(dataset) == 0:
        print(f"No data found in {args.data_dir / args.split}")
        return
    initial_graph = dataset[0]
    print(f"Loaded initial graph from {args.split} split ({len(dataset)} samples)")

    # Load norm stats
    norm_stats_path = args.data_dir / "norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = NormStats(**json.load(f))
    print(f"Loaded norm stats from {norm_stats_path}")

    # Run rollout
    bc_types: list[int] | None = args.bc_node_types
    results = rollout(
        model, initial_graph, args.num_steps, norm_stats, device, bc_node_types=bc_types
    )
    print(f"Rollout complete: {len(results)} steps")

    # Save results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, step_result in enumerate(results):
        torch.save(step_result, output_dir / f"step_{i:04d}.pt")
    print(f"Saved {len(results)} step files to {output_dir}")

    # Summary
    series = displacement_time_series(results)
    mean_disp = float(np.mean(series))
    max_disp = float(np.max(series))
    print("\nSummary:")
    print(f"  Steps:              {len(results)}")
    print(f"  Mean displacement:  {mean_disp:.6f}")
    print(f"  Max displacement:   {max_disp:.6f}")


if __name__ == "__main__":
    main()
