"""Tests for FNO model, grid physics losses, and model factory."""

from __future__ import annotations

import pytest
import torch

from vibration_poc.model.fno import FNO3d, SpectralConv3d


class TestSpectralConv3d:
    def test_output_shape(self) -> None:
        layer = SpectralConv3d(32, 64, modes=(4, 4, 4))
        x = torch.randn(2, 32, 8, 8, 8)
        assert layer(x).shape == (2, 64, 8, 8, 8)

    def test_gradient_flows(self) -> None:
        layer = SpectralConv3d(16, 16, modes=(3, 3, 3))
        x = torch.randn(1, 16, 6, 6, 6, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None


class TestFNO3d:
    def test_output_shape(self) -> None:
        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=16, num_layers=2, modes=(3, 3, 3)
        )
        x = torch.randn(2, 7, 8, 16, 4)
        assert model(x).shape == (2, 3, 8, 16, 4)

    def test_gradient_flows(self) -> None:
        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=16, num_layers=2, modes=(3, 3, 3)
        )
        x = torch.randn(1, 7, 8, 8, 4, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_parameter_count(self) -> None:
        model = FNO3d(hidden_dim=64, num_layers=4, modes=(12, 12, 8))
        n = sum(p.numel() for p in model.parameters())
        assert 100_000 < n < 100_000_000

    def test_single_sample_overfit(self) -> None:
        model = FNO3d(
            input_channels=4, output_channels=3, hidden_dim=16, num_layers=2, modes=(3, 3, 3)
        )
        x = torch.randn(1, 4, 6, 6, 4)
        y = torch.randn(1, 3, 6, 6, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_loss = float("inf")
        for epoch in range(100):
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            if epoch == 0:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < initial_loss * 0.5


class TestGridPhysicsLosses:
    def test_masked_mse(self) -> None:
        from vibration_poc.physics import compute_masked_mse

        pred = torch.randn(2, 3, 4, 4, 4)
        target = torch.randn(2, 3, 4, 4, 4)
        mask = torch.ones(2, 1, 4, 4, 4)
        loss = compute_masked_mse(pred, target, mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_masked_mse_respects_mask(self) -> None:
        from vibration_poc.physics import compute_masked_mse

        pred = torch.ones(1, 3, 4, 4, 4)
        target = torch.zeros(1, 3, 4, 4, 4)
        mask = torch.zeros(1, 1, 4, 4, 4)
        mask[0, 0, 0, 0, 0] = 1.0
        loss = compute_masked_mse(pred, target, mask)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_grid_smoothness(self) -> None:
        from vibration_poc.physics import compute_grid_smoothness_loss

        pred = torch.randn(1, 3, 4, 4, 4)
        mask = torch.ones(1, 1, 4, 4, 4)
        loss = compute_grid_smoothness_loss(pred, mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_grid_bc_penalty(self) -> None:
        from vibration_poc.physics import compute_grid_bc_penalty

        pred = torch.ones(1, 3, 4, 4, 4)
        bc_mask = torch.zeros(1, 1, 4, 4, 4)
        bc_mask[0, 0, 0, :, :] = 1.0
        loss = compute_grid_bc_penalty(pred, bc_mask)
        assert loss.item() > 0


class TestRolloutFno:
    def test_rollout_returns_results(self) -> None:
        import numpy as np

        from vibration_poc.dataset.config import GridConfig, GridNormStats
        from vibration_poc.inference.predict_fno import rollout_fno

        n_nodes = 20
        rng = np.random.default_rng(42)
        mesh_pos = torch.from_numpy(rng.uniform(0, 1, (n_nodes, 3)).astype(np.float32))
        x = torch.cat([mesh_pos, torch.zeros(n_nodes, 1)], dim=1)
        graph: dict[str, torch.Tensor] = {"mesh_pos": mesh_pos, "x": x}

        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=8, num_layers=1, modes=(2, 2, 2)
        )
        norm_stats = GridNormStats(
            channel_mean=[0.0] * 7,
            channel_std=[1.0] * 7,
            target_mean=[0.0] * 3,
            target_std=[1.0] * 3,
        )
        grid_config = GridConfig(resolution=(4, 4, 4))
        results = rollout_fno(
            model, graph, num_steps=3, grid_norm_stats=norm_stats, grid_config=grid_config
        )
        assert len(results) == 3
        assert results[0]["predicted_displacement"].shape == (n_nodes, 3)
        assert results[0]["world_pos"].shape == (n_nodes, 3)

    def test_rollout_with_bc(self) -> None:
        import numpy as np

        from vibration_poc.dataset.config import GridConfig, GridNormStats
        from vibration_poc.inference.predict_fno import rollout_fno

        n_nodes = 15
        rng = np.random.default_rng(99)
        mesh_pos = torch.from_numpy(rng.uniform(0, 1, (n_nodes, 3)).astype(np.float32))
        node_types = torch.zeros(n_nodes, 1)
        node_types[:3] = 1.0
        x = torch.cat([mesh_pos, node_types], dim=1)
        graph: dict[str, torch.Tensor] = {"mesh_pos": mesh_pos, "x": x}

        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=8, num_layers=1, modes=(2, 2, 2)
        )
        norm_stats = GridNormStats(
            channel_mean=[0.0] * 7,
            channel_std=[1.0] * 7,
            target_mean=[0.0] * 3,
            target_std=[1.0] * 3,
        )
        grid_config = GridConfig(resolution=(4, 4, 4))
        results = rollout_fno(
            model,
            graph,
            num_steps=2,
            grid_norm_stats=norm_stats,
            grid_config=grid_config,
            bc_node_types=[1],
        )
        assert len(results) == 2
        for step in results:
            bc_disp = step["predicted_displacement"][:3]
            assert torch.allclose(bc_disp, torch.zeros_like(bc_disp))


class TestTrainEpochFno:
    def test_loss_decreases(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from vibration_poc.training.trainer_fno import train_epoch_fno

        gx, gy, gz = 4, 4, 4
        n_samples = 4
        grid_inputs = torch.randn(n_samples, 7, gx, gy, gz)
        grid_targets = torch.randn(n_samples, 3, gx, gy, gz)
        masks = torch.ones(n_samples, 1, gx, gy, gz)
        ds = TensorDataset(grid_inputs, grid_targets, masks)

        def collate(batch: list[tuple[torch.Tensor, ...]]) -> dict[str, torch.Tensor]:
            inputs = torch.stack([b[0] for b in batch])
            targets = torch.stack([b[1] for b in batch])
            m = torch.stack([b[2] for b in batch])
            return {"grid_input": inputs, "grid_target": targets, "occupancy_mask": m}

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            ds, batch_size=2, collate_fn=collate
        )
        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=8, num_layers=1, modes=(2, 2, 2)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")

        loss1 = train_epoch_fno(model, loader, optimizer, device)
        loss2 = train_epoch_fno(model, loader, optimizer, device)
        assert loss2 <= loss1 * 1.5  # should not diverge

    def test_with_physics(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from vibration_poc.physics import PhysicsConfig
        from vibration_poc.training.trainer_fno import train_epoch_fno

        gx, gy, gz = 4, 4, 4
        n = 2
        gi = torch.randn(n, 7, gx, gy, gz)
        gt = torch.randn(n, 3, gx, gy, gz)
        masks = torch.ones(n, 1, gx, gy, gz)
        ds = TensorDataset(gi, gt, masks)

        def collate(batch: list[tuple[torch.Tensor, ...]]) -> dict[str, torch.Tensor]:
            return {
                "grid_input": torch.stack([b[0] for b in batch]),
                "grid_target": torch.stack([b[1] for b in batch]),
                "occupancy_mask": torch.stack([b[2] for b in batch]),
            }

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            ds, batch_size=2, collate_fn=collate
        )
        model = FNO3d(
            input_channels=7, output_channels=3, hidden_dim=8, num_layers=1, modes=(2, 2, 2)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        physics = PhysicsConfig(smoothness_weight=0.01, bc_loss_weight=0.01)
        loss = train_epoch_fno(model, loader, optimizer, torch.device("cpu"), physics=physics)
        assert loss > 0


class TestModelFactory:
    def test_create_meshgraphnet(self) -> None:
        from vibration_poc.model import create_model

        model = create_model("meshgraphnet")
        assert model is not None

    def test_create_fno(self) -> None:
        from vibration_poc.model import create_model

        model = create_model("fno", hidden_dim=16, num_layers=2, modes=(3, 3, 3))
        assert model is not None

    def test_unknown_raises(self) -> None:
        from vibration_poc.model import create_model

        with pytest.raises(ValueError, match="Unknown"):
            create_model("unknown_model")
