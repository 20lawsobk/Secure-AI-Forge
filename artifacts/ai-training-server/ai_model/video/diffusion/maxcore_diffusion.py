"""
MaxCore Diffusion Pipeline — the primary orchestrator.

Brings together:
  DDIMScheduler        — fast 20-step denoising
  MusicVAE             — 256×256×3 ↔ 32×32×4 latent compression
  AwarenessConditioner — awareness context → conditioning tokens
  TemporalDiT          — denoising network with spatial + temporal + awareness attention
  SDEditPrior          — RCGS-grounded starting point (avoids pure-noise cold start)

Public API:
  MaxCoreDiffusionPipeline.generate(...)  → List[np.ndarray]  (HWC uint8 frames)
  get_diffusion_frame(...)                → np.ndarray | None  (single frame for compositing)

The pipeline is additive — if ANY component fails the caller receives None and
the existing procedural renderer takes over.  It can never break production.

Singleton pattern: _pipeline is lazily initialised on first use and reused
across requests.  Weights are randomly initialised until the training flywheel
populates checkpoints in uploads/diffusion/.
"""
from __future__ import annotations

import os
import threading
import numpy as np
from typing import Any, Dict, List, Optional

from .noise_scheduler import DDPMScheduler, DDIMScheduler
from .music_vae import MusicVAE, LATENT_C, LATENT_H, LATENT_W
from .awareness_conditioner import AwarenessConditioner
from .temporal_dit import TemporalDiT
from .sdedit_prior import SDEditPrior

try:
    import torch as _torch  # noqa: F401  — availability probe; real import done per-method
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

RESOLUTION: int = 256
N_FRAMES: int = 8
N_DDIM_STEPS: int = 20
NOISE_FRACTION: float = 0.40
CHECKPOINT_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "uploads", "diffusion",
)

_pipeline: Optional["MaxCoreDiffusionPipeline"] = None
_pipeline_lock = threading.Lock()


class MaxCoreDiffusionPipeline:
    """Awareness-conditioned music-domain latent video diffusion pipeline."""

    def __init__(self) -> None:
        self.scheduler_ddpm = DDPMScheduler(T=1000)
        self.scheduler = DDIMScheduler(self.scheduler_ddpm, num_steps=N_DDIM_STEPS)
        self.vae = MusicVAE(LATENT_C)
        self.conditioner = AwarenessConditioner()
        self.dit = TemporalDiT()
        self.prior = SDEditPrior(self.scheduler, NOISE_FRACTION)
        self._try_load_checkpoints()

    def _try_load_checkpoints(self) -> None:
        """Non-fatal checkpoint loading — if absent, keeps random init."""
        if not _TORCH_OK:
            return
        import torch
        import torch.nn as nn
        modules: List[tuple] = [
            ("vae", self.vae), ("dit", self.dit), ("conditioner", self.conditioner)
        ]
        for name, module in modules:
            mod: nn.Module = module
            path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
            if os.path.exists(path):
                try:
                    state = torch.load(path, map_location="cpu")
                    mod.load_state_dict(state, strict=False)
                except Exception:
                    pass

    def save_checkpoints(self) -> None:
        """Persist current weights for future restarts."""
        if not _TORCH_OK:
            return
        import torch
        import torch.nn as nn
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        modules: List[tuple] = [
            ("vae", self.vae), ("dit", self.dit), ("conditioner", self.conditioner)
        ]
        for name, module in modules:
            mod: nn.Module = module
            path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
            try:
                torch.save(mod.state_dict(), path)
            except Exception:
                pass

    def _dna_from_context(self, context: Dict[str, Any]) -> List[float]:
        """Extract Visual DNA floats from the scene context dict."""
        dna = context.get("dna", {})
        return [
            float(dna.get("energy", 0.5)),
            float(dna.get("darkness", 0.5)),
            float(dna.get("warmth", 0.5)),
            float(dna.get("saturation", 0.5)),
        ]

    def generate(
        self,
        idea: str,
        platform: str,
        tone: str,
        awareness: str,
        context: Optional[Dict[str, Any]] = None,
        n_frames: int = N_FRAMES,
        resolution: int = RESOLUTION,
    ) -> List[np.ndarray]:
        """
        Generate n_frames of awareness-conditioned video frames.

        Returns List[np.ndarray] — each frame is HWC uint8 RGB.
        Falls back to empty list on any error.
        """
        if not _TORCH_OK:
            return []
        try:
            return self._generate_inner(idea, platform, tone, awareness, context or {}, n_frames, resolution)
        except Exception:
            return []

    def _generate_inner(
        self,
        idea: str,
        platform: str,
        tone: str,
        awareness: str,
        context: Dict[str, Any],
        n_frames: int,
        resolution: int,
    ) -> List[np.ndarray]:
        dna_params = self._dna_from_context(context)
        brand = context.get("brand", "maxbooster")

        cond = self.conditioner.encode(awareness, dna_params, batch_size=1)

        x_t, t_start = self.prior.get_prior_frames(
            n_frames=n_frames,
            width=resolution,
            height=resolution,
            latent_h=LATENT_H,
            latent_w=LATENT_W,
            latent_c=LATENT_C,
            brand=brand,
            context=context,
            vae=self.vae,
        )

        t_indices = [
            i for i, ts in enumerate(self.scheduler.timesteps)
            if ts <= t_start
        ]
        if not t_indices:
            t_indices = list(range(len(self.scheduler.timesteps)))

        for idx in t_indices:
            t_val = self.scheduler.timesteps[idx]
            t_prev = self.scheduler.timesteps[idx + 1] if idx + 1 < len(self.scheduler.timesteps) else -1

            eps_pred = self.dit.predict_noise(x_t, t_val, dna_params, cond)
            x_t_new = np.zeros_like(x_t)
            for f in range(n_frames):
                x_t_new[f] = self.scheduler.ddim_step(x_t[f], t_val, t_prev, eps_pred[f])
            x_t = x_t_new

        frames = []
        for f in range(n_frames):
            try:
                frame = self.vae.latent_to_frame(x_t[f])
            except Exception:
                frame = np.full((resolution, resolution, 3), 64, dtype=np.uint8)
            frames.append(frame)
        return frames


def _get_pipeline() -> Optional[MaxCoreDiffusionPipeline]:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            try:
                _pipeline = MaxCoreDiffusionPipeline()
            except Exception:
                return None
    return _pipeline


def get_diffusion_frame(
    idea: str,
    platform: str,
    tone: str,
    awareness: str,
    width: int = RESOLUTION,
    height: int = RESOLUTION,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """
    Generate a single compositing-ready frame for the cinematic engine.

    Returns None on any failure so callers fall back to procedural backgrounds.
    This function is the public integration point used by scenes.py.
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        return None
    frames = pipeline.generate(
        idea=idea,
        platform=platform,
        tone=tone,
        awareness=awareness,
        context=context or {},
        n_frames=1,
        resolution=max(width, height),
    )
    if not frames:
        return None
    frame = frames[0]
    try:
        from PIL import Image
        img = Image.fromarray(frame).resize((width, height), Image.BILINEAR)
        return np.array(img)
    except Exception:
        return frame
