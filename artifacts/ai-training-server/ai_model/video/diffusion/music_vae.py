"""
Music-Domain VAE for MaxCore Diffusion.

Compresses 256×256×3 RGB frames into 32×32×4 latent representations.
Trained on music video aesthetics from the pdim dataset — narrow domain
means a much smaller model is needed compared to general-purpose VAEs.

Architecture:
  Encoder:  3 × stride-2 conv blocks (256→128→64→32) + project to 2*latent_C
  Decoder:  Upsample + conv blocks (32→64→128→256) + tanh output
  Latent:   32×32×4 (8× spatial compression, 4 channels)

Uses standard PyTorch conv throughout (HyperGPU is reserved for the DiT
where temporal attention dominates compute).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


LATENT_C: int = 4
LATENT_H: int = 32
LATENT_W: int = 32


def _conv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.SiLU(),
    )


def _up_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.SiLU(),
    )


class MusicVAEEncoder(nn.Module):
    def __init__(self, latent_channels: int = LATENT_C) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.to_moments = nn.Conv2d(128, latent_channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        moments = self.to_moments(h)
        mean, log_var = moments.chunk(2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        return mean, log_var


class MusicVAEDecoder(nn.Module):
    def __init__(self, latent_channels: int = LATENT_C) -> None:
        super().__init__()
        self.from_latent = nn.Conv2d(latent_channels, 128, kernel_size=1)
        self.net = nn.Sequential(
            _up_block(128, 64),
            _up_block(64, 32),
            _up_block(32, 16),
        )
        self.out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.net(h)
        return torch.tanh(self.out(h))


class MusicVAE(nn.Module):
    def __init__(self, latent_channels: int = LATENT_C) -> None:
        super().__init__()
        self.encoder = MusicVAEEncoder(latent_channels)
        self.decoder = MusicVAEDecoder(latent_channels)
        self.latent_channels = latent_channels
        self._scale: float = 0.18215

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode x → (z, mean, log_var).  z is reparameterised sample."""
        mean, log_var = self.encoder(x)
        z = self.reparameterise(mean, log_var)
        return z * self._scale, mean, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z → [-1, 1] RGB image."""
        return self.decoder(z / self._scale)

    def reparameterise(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, log_var = self.encode(x)
        recon = self.decode(z)
        return recon, mean, log_var

    def frame_to_latent(self, frame_np: np.ndarray) -> np.ndarray:
        """Convenience: numpy HWC uint8 → latent numpy CHW float32."""
        t = torch.from_numpy(frame_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            z, _, _ = self.encode(t)
        return z.squeeze(0).numpy()

    def latent_to_frame(self, z_np: np.ndarray) -> np.ndarray:
        """Convenience: latent numpy CHW → numpy HWC uint8."""
        t = torch.from_numpy(z_np).unsqueeze(0)
        with torch.no_grad():
            out = self.decode(t)
        img = (out.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5
        return np.clip(img, 0, 255).astype(np.uint8)
