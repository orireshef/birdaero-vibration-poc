"""Fourier Neural Operator for 3D structural vibration prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralConv3d(nn.Module):
    """3D spectral convolution via truncated FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes
        scale = 1.0 / (in_channels * out_channels)
        self.w1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat
            )
        )
        self.w2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat
            )
        )
        self.w3 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat
            )
        )
        self.w4 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat
            )
        )

    @staticmethod
    def _compl_mul3d(a: Tensor, b: Tensor) -> Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        gx, gy, gz = x.size(2), x.size(3), x.size(4)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            gx,
            gy,
            gz // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.w1)
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.w2)
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.w3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.w4)

        result: Tensor = torch.fft.irfftn(out_ft, s=(gx, gy, gz), dim=(-3, -2, -1))
        return result


class FNO3d(nn.Module):
    """3D Fourier Neural Operator."""

    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        modes: tuple[int, int, int] = (12, 12, 8),
    ) -> None:
        super().__init__()
        self.lifting = nn.Linear(input_channels, hidden_dim)
        self.spectral_convs = nn.ModuleList(
            [SpectralConv3d(hidden_dim, hidden_dim, modes) for _ in range(num_layers)]
        )
        self.local_convs = nn.ModuleList(
            [nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.InstanceNorm3d(hidden_dim, affine=True) for _ in range(num_layers)]
        )
        self.proj1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj2 = nn.Linear(hidden_dim * 2, output_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.lifting(x)
        x = x.permute(0, 4, 1, 2, 3)

        layers = zip(self.spectral_convs, self.local_convs, self.norms, strict=False)
        for spectral, local, norm in layers:
            x = F.gelu(norm(spectral(x) + local(x)))

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        result: Tensor = x.permute(0, 4, 1, 2, 3)
        return result
