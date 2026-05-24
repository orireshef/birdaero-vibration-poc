"""Model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def create_model(model_type: str, **kwargs: object) -> nn.Module:
    """Create a model by type name."""
    if model_type == "meshgraphnet":
        from vibration_poc.model.meshgraphnet import MeshGraphNet

        return MeshGraphNet(**kwargs)  # type: ignore[arg-type]
    if model_type == "fno":
        from vibration_poc.model.fno import FNO3d

        return FNO3d(**kwargs)  # type: ignore[arg-type]
    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)
