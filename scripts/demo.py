"""End-to-end demo pipeline: download, preprocess, train, predict, visualize."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from vibration_poc.dataset.config import DatasetConfig, NormStats
from vibration_poc.dataset.dataloader import GraphDataset
from vibration_poc.dataset.download import download_dataset
from vibration_poc.dataset.preprocess import preprocess_dataset
from vibration_poc.inference.analyze import compute_fft, displacement_time_series
from vibration_poc.inference.predict import rollout
from vibration_poc.model.meshgraphnet import MeshGraphNet
from vibration_poc.physics import PhysicsConfig
from vibration_poc.training.trainer import TrainingConfig, train
from vibration_poc.visualization.deformation import create_deformation_gif
from vibration_poc.visualization.error_maps import plot_error_map
from vibration_poc.visualization.frequency_analysis import (
    plot_frequency_spectrum,
    plot_mode_shapes,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="End-to-end vibration POC demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--num-steps", type=int, default=20, help="Rollout steps")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def _generate_visualizations(
    results: list[dict[str, Tensor]],
    initial_graph: dict[str, Tensor],
    output_dir: Path,
) -> np.ndarray:
    """Generate all visualization outputs. Returns displacement series."""
    print("\n=== Step 6: Generate visualizations ===")

    # Deformation GIF
    gif_path = output_dir / "deformation.gif"
    create_deformation_gif(results, gif_path)
    print(f"  Deformation GIF: {gif_path}")

    # Error map (first step: GT vs predicted)
    error_path = output_dir / "error_map.png"
    first_step = results[0]
    mesh_pos_np: np.ndarray = first_step["mesh_pos"].detach().cpu().numpy()
    predicted_np: np.ndarray = first_step["predicted_displacement"].detach().cpu().numpy()
    gt_np: np.ndarray = initial_graph["y"].detach().cpu().numpy()
    plot_error_map(mesh_pos_np, gt_np, predicted_np, save_path=error_path)
    print(f"  Error map: {error_path}")

    # FFT spectrum
    spectrum_path = output_dir / "frequency_spectrum.png"
    series = displacement_time_series(results)
    freqs, magnitudes = compute_fft(series)
    plot_frequency_spectrum(freqs, magnitudes, save_path=spectrum_path)
    print(f"  Frequency spectrum: {spectrum_path}")

    # Mode shapes
    modes_path = output_dir / "mode_shapes.png"
    plot_mode_shapes(mesh_pos_np, magnitudes, freqs, save_path=modes_path)
    print(f"  Mode shapes: {modes_path}")

    return series


def main() -> None:
    """Run full demo pipeline."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = DatasetConfig()
    device = torch.device(args.device)

    # 1. Download dataset
    print("=== Step 1: Download dataset ===")
    download_dataset(dataset_config)
    print(f"Dataset downloaded to {dataset_config.raw_dir}")

    # 2. Preprocess dataset
    print("\n=== Step 2: Preprocess dataset ===")
    preprocess_dataset(dataset_config)
    print(f"Preprocessed data saved to {dataset_config.processed_dir}")

    # 3. Train model
    print("\n=== Step 3: Train model ===")
    physics_config = PhysicsConfig(
        bc_loss_weight=1.0,
        stress_loss_weight=0.1,
        smoothness_weight=0.01,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        device=args.device,
        log_interval=1,
        physics=physics_config,
    )
    best_checkpoint = train(training_config, dataset_config)
    print(f"Best checkpoint: {best_checkpoint}")

    # 4. Load best checkpoint
    print("\n=== Step 4: Load model ===")
    model = MeshGraphNet(
        hidden_dim=training_config.hidden_dim,
        num_layers=training_config.num_layers,
    )
    state_dict = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    print("Model loaded")

    # 5. Get first test graph and norm stats
    test_dataset = GraphDataset(dataset_config.processed_dir, "test")
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return
    initial_graph = test_dataset[0]
    print(f"Test dataset: {len(test_dataset)} samples")

    norm_stats_path = dataset_config.processed_dir / "norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = NormStats(**json.load(f))

    # 6. Run rollout
    print(f"\n=== Step 5: Rollout ({args.num_steps} steps) ===")
    results = rollout(
        model,
        initial_graph,
        args.num_steps,
        norm_stats,
        device,
        bc_node_types=physics_config.bc_node_types,
    )
    print(f"Rollout complete: {len(results)} steps")

    # 7. Generate visualizations
    series = _generate_visualizations(results, initial_graph, output_dir)

    # 8. Summary
    mean_disp = float(np.mean(series))
    max_disp = float(np.max(series))
    print("\n=== Summary ===")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Rollout steps:      {len(results)}")
    print(f"  Mean displacement:  {mean_disp:.6f}")
    print(f"  Max displacement:   {max_disp:.6f}")
    print(f"  Outputs saved to:   {output_dir}")


if __name__ == "__main__":
    main()
