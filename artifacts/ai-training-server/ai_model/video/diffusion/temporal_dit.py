"""
Temporal Diffusion Transformer (DiT) for MaxCore Diffusion.

Processes video as a sequence of latent frames:
  Input:  [B, T, C, H, W] = [B, T, 4, 32, 32]
  Patches: 4×4 pixels → [B, T, 64, d_model]  (64 patches per frame)

Each DiTBlock contains:
  1. AdaLayerNorm    — conditioned on Visual DNA (energy/darkness/warmth/saturation)
  2. Spatial Attn   — HyperFlashAttention over 64 spatial patches (within each frame)
  3. AdaLayerNorm
  4. Temporal Attn  — HyperFlashAttention over T frames (per spatial position)
  5. AdaLayerNorm
  6. Awareness Xattn — nn.MultiheadAttention with awareness conditioning tokens as K/V
  7. FFN             — HyperGPULinear 4× expansion with GELU

HyperGPU backend handles spatial and temporal self-attention — these are the
compute-intensive paths.  Awareness cross-attention uses standard PyTorch since
the K/V sequence (N_TOKENS=8) is short enough that FlashAttention overhead isn't
worth it.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from ...gpu.hyper_backend import HyperGPUBackend, HyperFlashAttention
from ...gpu.hyper_core import PrecisionMode
from .awareness_conditioner import N_TOKENS, D_MODEL as COND_DIM

PATCH_SIZE: int = 4
LATENT_C: int = 4
LATENT_H: int = 32
LATENT_W: int = 32
N_PATCHES: int = (LATENT_H // PATCH_SIZE) * (LATENT_W // PATCH_SIZE)
PATCH_DIM: int = PATCH_SIZE * PATCH_SIZE * LATENT_C
D_MODEL: int = 256
N_HEADS: int = 4
N_LAYERS: int = 4
FFN_MULT: int = 4


def _build_backend() -> HyperGPUBackend:
    return HyperGPUBackend(
        lanes=256,
        tensor_cores=4,
        precision=PrecisionMode.MIXED,
        vram_capacity=0,
        training_mode=False,
    )


_BACKEND: Optional[HyperGPUBackend] = None


def _get_backend() -> HyperGPUBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _build_backend()
    return _BACKEND


class AdaLayerNorm(nn.Module):
    """LayerNorm with scale + shift conditioned on Visual DNA parameters."""

    def __init__(self, d_model: int = D_MODEL, n_dna: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(n_dna, 2 * d_model)

    def forward(self, x: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        params = self.proj(dna)
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-2)
        shift = shift.unsqueeze(-2)
        return (1.0 + scale) * self.norm(x) + shift


class DiTBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS) -> None:
        super().__init__()
        backend = _get_backend()

        self.ada1 = AdaLayerNorm(d_model)
        self.spatial_attn: HyperFlashAttention = backend.flash_attention(d_model, n_heads)

        self.ada2 = AdaLayerNorm(d_model)
        self.temporal_attn: HyperFlashAttention = backend.flash_attention(d_model, n_heads)

        self.ada3 = AdaLayerNorm(d_model)
        self.awareness_q = nn.Linear(d_model, d_model)
        self.awareness_kv = nn.Linear(COND_DIM, 2 * d_model)
        self.awareness_out = nn.Linear(d_model, d_model)

        self.ada4 = AdaLayerNorm(d_model)
        self.ffn = nn.Sequential(
            backend.linear(d_model, d_model * FFN_MULT),
            nn.GELU(),
            backend.linear(d_model * FFN_MULT, d_model),
        )

    def _awareness_xattn(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        BT, N, D = x.shape
        q = self.awareness_q(x)
        kv = self.awareness_kv(cond)
        k, v = kv.chunk(2, dim=-1)
        scale = math.sqrt(D // N_HEADS)
        attn = torch.softmax(
            torch.bmm(q, k.transpose(-2, -1)) / scale, dim=-1
        )
        return self.awareness_out(torch.bmm(attn, v))

    def forward(
        self,
        x: torch.Tensor,
        dna: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, D = x.shape

        # ── Spatial self-attention ─────────────────────────────────────────────
        xs = x.reshape(B * T, N, D)
        xs_n = self.ada1(xs, dna.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1))
        xs = xs + self.spatial_attn(xs_n, causal=False)

        # ── Temporal self-attention ────────────────────────────────────────────
        xt = xs.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        dna_t = dna.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        xt_n = self.ada2(xt, dna_t)
        xt = xt + self.temporal_attn(xt_n, causal=False)
        xs = xt.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)

        # ── Awareness cross-attention ──────────────────────────────────────────
        cond_bt = cond.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, N_TOKENS, COND_DIM)
        xs_n = self.ada3(xs, dna.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1))
        xs = xs + self._awareness_xattn(xs_n, cond_bt)

        # ── FFN ───────────────────────────────────────────────────────────────
        xs_n = self.ada4(xs, dna.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1))
        xs = xs + self.ffn(xs_n)

        return xs.reshape(B, T, N, D)


class TemporalDiT(nn.Module):
    """Full Temporal Diffusion Transformer.

    Input:  (x, t_emb, dna, cond)
      x:    [B, T, C, H, W]  — noisy latent frames
      t_emb:[B, d_model]      — sinusoidal timestep embedding
      dna:  [B, 4]            — Visual DNA params
      cond: [B, N_TOKENS, COND_DIM]  — awareness conditioning

    Output: [B, T, C, H, W]  — predicted noise
    """

    def __init__(self, n_layers: int = N_LAYERS, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, d_model)
        self.t_proj = nn.Linear(d_model, 4)
        self.blocks = nn.ModuleList([DiTBlock(d_model) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, PATCH_DIM)
        self.d_model = d_model

    @staticmethod
    def _sinusoidal_emb(t: torch.Tensor, d: int) -> torch.Tensor:
        half = d // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        emb = t[:, None].float() * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        ph, pw = H // PATCH_SIZE, W // PATCH_SIZE
        x = x.reshape(B, T, C, ph, PATCH_SIZE, pw, PATCH_SIZE)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(B, T, ph * pw, C * PATCH_SIZE * PATCH_SIZE)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = tokens.shape
        ph = pw = int(N ** 0.5)
        tokens = tokens.reshape(B, T, ph, pw, LATENT_C, PATCH_SIZE, PATCH_SIZE)
        tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6)
        return tokens.reshape(B, T, LATENT_C, ph * PATCH_SIZE, pw * PATCH_SIZE)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dna: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        patches = self._patchify(x)
        tokens = self.patch_embed(patches)

        t_emb = self._sinusoidal_emb(t, self.d_model)
        dna_cond = dna + self.t_proj(t_emb)

        for block in self.blocks:
            tokens = block(tokens, dna_cond, cond)

        tokens = self.out_norm(tokens)
        tokens = self.out_proj(tokens)
        return self._unpatchify(tokens)

    @torch.no_grad()
    def predict_noise(
        self,
        x: np.ndarray,
        t: int,
        dna_params: List[float],
        cond: torch.Tensor,
    ) -> np.ndarray:
        """Inference helper: numpy in → numpy out."""
        xt = torch.from_numpy(x).unsqueeze(0)
        t_tensor = torch.tensor([t], dtype=torch.long)
        dna_tensor = torch.tensor([dna_params[:4]], dtype=torch.float32)
        out = self(xt, t_tensor, dna_tensor, cond)
        return out.squeeze(0).numpy()
