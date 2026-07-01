"""
DDPM / DDIM noise scheduler for MaxCore Diffusion.

Pure NumPy implementation — no PyTorch dependency so it can run on any thread
without holding the GIL for tensor ops.

Schedule:  linear beta from beta_start → beta_end over T timesteps.
Forward:   q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
Reverse:   DDPM one-step update or DDIM deterministic jump.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DDPMScheduler:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    betas: np.ndarray = field(init=False)
    alphas: np.ndarray = field(init=False)
    alpha_bar: np.ndarray = field(init=False)
    sqrt_alpha_bar: np.ndarray = field(init=False)
    sqrt_one_minus_alpha_bar: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.betas = np.linspace(self.beta_start, self.beta_end, self.T, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)

    def forward_process(
        self,
        x0: np.ndarray,
        t: int,
        noise: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample q(x_t | x_0) at timestep t.

        Returns (x_t, eps) where eps is the noise that was added.
        """
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)
        a = self.sqrt_alpha_bar[t]
        b = self.sqrt_one_minus_alpha_bar[t]
        x_t = a * x0 + b * noise
        return x_t.astype(np.float32), noise.astype(np.float32)

    def ddpm_step(
        self,
        x_t: np.ndarray,
        t: int,
        eps_pred: np.ndarray,
    ) -> np.ndarray:
        """Single DDPM reverse step: p(x_{t-1} | x_t)."""
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else np.float32(1.0)
        beta_t = self.betas[t]
        coef = beta_t / self.sqrt_one_minus_alpha_bar[t]
        x0_pred = (x_t - coef * eps_pred) / math.sqrt(alpha_t)
        x0_pred = np.clip(x0_pred, -1.0, 1.0)
        mean = (
            math.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * x0_pred
            + math.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * x_t
        )
        if t > 0:
            sigma = math.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            mean = mean + sigma * np.random.randn(*x_t.shape).astype(np.float32)
        return mean.astype(np.float32)


class DDIMScheduler:
    """Deterministic DDIM sampler — allows ~20 steps instead of 1000.

    Reference: Song et al., "Denoising Diffusion Implicit Models" (2020).
    """

    def __init__(self, ddpm: DDPMScheduler, num_steps: int = 20) -> None:
        self.ddpm = ddpm
        self.num_steps = num_steps
        step_size = ddpm.T // num_steps
        self.timesteps: List[int] = list(
            range(ddpm.T - 1, -1, -step_size)
        )[:num_steps]

    def ddim_step(
        self,
        x_t: np.ndarray,
        t: int,
        t_prev: int,
        eps_pred: np.ndarray,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Single deterministic DDIM step (eta=0 → fully deterministic)."""
        ab_t = self.ddpm.alpha_bar[t]
        ab_prev = self.ddpm.alpha_bar[t_prev] if t_prev >= 0 else np.float32(1.0)
        x0_pred = (x_t - math.sqrt(1.0 - ab_t) * eps_pred) / math.sqrt(ab_t)
        x0_pred = np.clip(x0_pred, -1.0, 1.0)
        dir_xt = math.sqrt(1.0 - ab_prev - eta ** 2 * (1.0 - ab_t) / (1.0 - ab_prev + 1e-8)) * eps_pred
        x_prev = math.sqrt(ab_prev) * x0_pred + dir_xt
        if eta > 0 and t_prev >= 0:
            sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev))
            x_prev = x_prev + sigma * np.random.randn(*x_t.shape).astype(np.float32)
        return x_prev.astype(np.float32)

    def partial_noise(self, x0: np.ndarray, noise_fraction: float) -> Tuple[np.ndarray, int]:
        """Add noise at t = int(T * noise_fraction) — used by SDEdit prior."""
        t_start = int(self.ddpm.T * noise_fraction)
        t_start = max(0, min(t_start, self.ddpm.T - 1))
        x_t, _ = self.ddpm.forward_process(x0, t_start)
        return x_t, t_start
