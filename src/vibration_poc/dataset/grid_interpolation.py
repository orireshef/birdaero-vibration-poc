"""Mesh-to-grid and grid-to-mesh interpolation for FNO."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


class MeshToGridInterpolator:
    """Interpolate irregular mesh node values onto a regular 3D grid."""

    def __init__(
        self,
        mesh_pos: npt.NDArray[np.float32],
        resolution: tuple[int, int, int],
        padding_ratio: float = 0.05,
    ) -> None:
        mins = mesh_pos.min(axis=0)
        maxs = mesh_pos.max(axis=0)
        extent = maxs - mins
        pad = extent * padding_ratio
        self._bounds: npt.NDArray[np.float64] = np.array(
            [
                mins[0] - pad[0],
                maxs[0] + pad[0],
                mins[1] - pad[1],
                maxs[1] + pad[1],
                mins[2] - pad[2],
                maxs[2] + pad[2],
            ]
        )
        self._resolution = resolution
        self._gx = np.linspace(self._bounds[0], self._bounds[1], resolution[0])
        self._gy = np.linspace(self._bounds[2], self._bounds[3], resolution[1])
        self._gz = np.linspace(self._bounds[4], self._bounds[5], resolution[2])
        xx, yy, zz = np.meshgrid(self._gx, self._gy, self._gz, indexing="ij")
        self._grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        self._mesh_pos = mesh_pos

        ones_interp = LinearNDInterpolator(mesh_pos, np.ones(len(mesh_pos)))
        occ = ones_interp(self._grid_points)
        occ = np.nan_to_num(occ, nan=0.0)
        self._occupancy: npt.NDArray[np.float32] = (
            (occ > 0.5).astype(np.float32).reshape(1, *resolution)
        )

    def interpolate(self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Map [N, C] mesh values to [C, Gx, Gy, Gz] grid."""
        n_channels = values.shape[1] if values.ndim > 1 else 1
        if values.ndim == 1:
            values = values[:, None]
        result: npt.NDArray[np.float32] = np.zeros(
            (n_channels, *self._resolution), dtype=np.float32
        )
        for c in range(n_channels):
            interp = LinearNDInterpolator(self._mesh_pos, values[:, c])
            grid_vals = interp(self._grid_points)
            grid_vals = np.nan_to_num(grid_vals, nan=0.0).astype(np.float32)
            result[c] = grid_vals.reshape(self._resolution)
        return result

    def occupancy_mask(self) -> npt.NDArray[np.float32]:
        """Return [1, Gx, Gy, Gz] binary occupancy mask."""
        return self._occupancy

    @property
    def grid_bounds(self) -> npt.NDArray[np.float64]:
        """Return [6] array: xmin, xmax, ymin, ymax, zmin, zmax."""
        return self._bounds


class GridToMeshInterpolator:
    """Interpolate regular grid values back to irregular mesh positions."""

    def __init__(
        self,
        grid_bounds: npt.NDArray[np.float64],
        resolution: tuple[int, int, int],
    ) -> None:
        self._gx = np.linspace(grid_bounds[0], grid_bounds[1], resolution[0])
        self._gy = np.linspace(grid_bounds[2], grid_bounds[3], resolution[1])
        self._gz = np.linspace(grid_bounds[4], grid_bounds[5], resolution[2])

    def interpolate(
        self,
        grid_values: npt.NDArray[np.float32],
        query_points: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Map [C, Gx, Gy, Gz] grid to [N, C] at query_points [N, 3]."""
        n_channels = grid_values.shape[0]
        result: npt.NDArray[np.float32] = np.zeros(
            (len(query_points), n_channels), dtype=np.float32
        )
        for c in range(n_channels):
            interp = RegularGridInterpolator(
                (self._gx, self._gy, self._gz),
                grid_values[c],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            result[:, c] = interp(query_points).astype(np.float32)
        return result
