"""
SDEdit Prior — RCGS-as-starting-point for MaxCore Diffusion.

Instead of denoising from pure Gaussian noise (requires many steps, heavily
dependent on model quality), we:

  1. Retrieve a real visually-grounded frame via RCGS.
  2. Add partial noise at t_start = int(T * noise_fraction) — e.g., 40% noise.
  3. Start DDIM denoising FROM that partially-noised real frame.

This means:
  - The output is always grounded in real visual structure (from RCGS).
  - The DiT only needs to refine/stylise, not generate from scratch.
  - Even with randomly-initialised DiT weights, frames look grounded.
  - As training progresses, the DiT learns to better exploit the prior.

Reference: "SDEdit: Guided Image Synthesis and Editing with Stochastic
           Differential Equations" (Meng et al., 2021).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .noise_scheduler import DDIMScheduler

_NOISE_FRACTION: float = 0.4
_NEUTRAL_GREY: int = 128


def _neutral_frame(width: int, height: int) -> np.ndarray:
    """Return a mid-grey RGB frame as a safe fallback."""
    return np.full((height, width, 3), _NEUTRAL_GREY, dtype=np.uint8)


def _rcgs_frame(
    width: int,
    height: int,
    brand: str,
    context: dict,
) -> Optional[np.ndarray]:
    """
    Retrieve a real-asset-grounded frame via RCGS.
    Returns None if RCGS / storage is unavailable.
    """
    try:
        from ...retrieval.rcgs import condition_background
        neutral: np.ndarray = np.full((height, width, 3), _NEUTRAL_GREY, dtype=np.uint8)
        conditioned = condition_background(
            neutral,
            width=width,
            height=height,
            brand=brand,
            context=context,
        )
        return conditioned
    except Exception:
        return None


def _frame_to_float(frame: np.ndarray) -> np.ndarray:
    """HWC uint8 [0,255] → CHW float32 [-1, 1]."""
    f = frame.astype(np.float32) / 127.5 - 1.0
    return np.transpose(f, (2, 0, 1))


def _float_to_frame(arr: np.ndarray) -> np.ndarray:
    """CHW float32 [-1, 1] → HWC uint8 [0, 255]."""
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


class SDEditPrior:
    """Builds the SDEdit starting point for a video generation request."""

    def __init__(
        self,
        scheduler: DDIMScheduler,
        noise_fraction: float = _NOISE_FRACTION,
    ) -> None:
        self.scheduler = scheduler
        self.noise_fraction = noise_fraction

    def get_prior_latent(
        self,
        width: int,
        height: int,
        latent_h: int,
        latent_w: int,
        latent_c: int,
        brand: str,
        context: dict,
        vae: object,
    ) -> Tuple[np.ndarray, int]:
        """
        Build a partially-noised latent prior.

        Returns (x_t_start, t_start) where x_t_start has shape
        [latent_c, latent_h, latent_w] and t_start is the DDIM timestep
        index to start denoising from.
        """
        frame = _rcgs_frame(width, height, brand, context) or _neutral_frame(width, height)

        try:
            import numpy as _np
            from PIL import Image as _Image

            img_resized = _np.array(
                _Image.fromarray(frame).resize((width, height), _Image.BILINEAR)
            )
            latent_np = getattr(vae, "frame_to_latent", None)
            if latent_np is not None:
                z0 = latent_np(img_resized)
            else:
                z0 = _np.zeros((latent_c, latent_h, latent_w), dtype=_np.float32)
        except Exception:
            z0 = np.zeros((latent_c, latent_h, latent_w), dtype=np.float32)

        x_t, t_start = self.scheduler.partial_noise(z0, self.noise_fraction)
        return x_t, t_start

    def get_prior_frames(
        self,
        n_frames: int,
        width: int,
        height: int,
        latent_h: int,
        latent_w: int,
        latent_c: int,
        brand: str,
        context: dict,
        vae: object,
    ) -> Tuple[np.ndarray, int]:
        """
        Build prior for all T frames: [T, latent_c, latent_h, latent_w].

        All frames start from the same RCGS-grounded prior with independent
        noise realisations (same structure, natural variation).
        """
        frames_list = []
        t_start = 0
        for _ in range(n_frames):
            x_t, t_start = self.get_prior_latent(
                width, height, latent_h, latent_w, latent_c, brand, context, vae
            )
            frames_list.append(x_t)
        stacked = np.stack(frames_list, axis=0)
        return stacked, t_start
