"""Physics constraint utilities for MeshGraphNet training and inference."""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor


class PhysicsConfig(BaseModel):
    bc_node_types: list[int] = Field(default=[1, 2, 3])
    bc_loss_weight: float = Field(default=0.0, ge=0)
    stress_loss_weight: float = Field(default=0.0, ge=0)
    smoothness_weight: float = Field(default=0.0, ge=0)


def get_boundary_mask(node_type: Tensor, bc_types: list[int]) -> Tensor:
    """Return bool mask [N] — True for boundary/fixed nodes.

    node_type: [N] or [N,1] tensor of node type values (unnormalized integer-like).
    bc_types: list of node_type values considered boundary (e.g. [1, 2, 3]).
    """
    if node_type.dim() > 1:
        node_type = node_type.squeeze(-1)
    mask = torch.zeros(node_type.shape[0], dtype=torch.bool, device=node_type.device)
    for bc_type in bc_types:
        mask = mask | (node_type == bc_type)
    return mask


def compute_bc_penalty(pred: Tensor, mask: Tensor) -> Tensor:
    """MSE of boundary node predictions vs zero.

    pred: [N, D] predicted displacement
    mask: [N] bool mask — True for boundary nodes
    Returns scalar loss.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return (pred[mask] ** 2).mean()


def compute_smoothness_loss(pred: Tensor, edge_index: Tensor) -> Tensor:
    """L2 penalty on displacement differences across edges.

    pred: [N, D] predicted displacement
    edge_index: [2, E] source-dest pairs
    Returns scalar loss.
    """
    src, dst = edge_index[0], edge_index[1]
    diff = pred[src] - pred[dst]
    return (diff**2).mean()


def compute_masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Occupancy-masked MSE. pred/target: [B,C,Gx,Gy,Gz], mask: [B,1,Gx,Gy,Gz]."""
    diff = (pred - target) ** 2 * mask
    return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)


def compute_grid_smoothness_loss(pred: Tensor, mask: Tensor) -> Tensor:
    """Finite-difference gradient penalty on grid. pred: [B,C,Gx,Gy,Gz]."""
    dx = (pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]) * mask[:, :, 1:, :, :]
    dy = (pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]) * mask[:, :, :, 1:, :]
    dz = (pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) * mask[:, :, :, :, 1:]
    return (dx**2).mean() + (dy**2).mean() + (dz**2).mean()


def compute_grid_bc_penalty(pred: Tensor, bc_mask: Tensor) -> Tensor:
    """Penalize non-zero displacement at boundary grid points."""
    masked_pred = pred * bc_mask
    return (masked_pred**2).sum() / (bc_mask.sum() * pred.shape[1] + 1e-8)
