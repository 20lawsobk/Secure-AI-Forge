"""VRC colour engine — a real node-based grade (DaVinci-Resolve behaviour).

Replaces the ad-hoc per-preset ffmpeg filter strings with an actual colour
pipeline operating on the frame pixels:

    exposure -> lift/gamma/gain (per-channel) -> colour matrix -> filmic tonemap

The colour-matrix stage (a 3x3 creative transform, e.g. teal/orange cross-talk)
is applied as a GEMM on the self-contained Digital GPU: pixels ``[N,3] @ M[3,3]``.
Deterministic; operates on and returns ``HxWx3`` uint8 frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..fabric.compute import RTACompute


@dataclass
class Grade:
    exposure: float = 0.0                       # stops
    lift: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # shadows (add)
    gamma: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # midtones (power)
    gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # highlights (mul)
    saturation: float = 1.0
    matrix: Optional[Tuple[Tuple[float, float, float], ...]] = None  # 3x3 creative
    contrast: float = 1.0


# Creative grades keyed by the presets the video engine already uses.
GRADE_PRESETS: Dict[str, Grade] = {
    "cinematic": Grade(exposure=-0.1, lift=(0.02, 0.0, 0.04), gamma=(0.95, 1.0, 1.05),
                       gain=(1.05, 1.0, 0.95), saturation=1.08, contrast=1.12,
                       matrix=((1.05, -0.02, -0.03), (-0.02, 1.0, 0.0), (0.02, 0.03, 1.06))),
    "warm": Grade(exposure=0.05, lift=(0.03, 0.01, -0.01), gamma=(1.02, 1.0, 0.96),
                  gain=(1.08, 1.02, 0.9), saturation=1.05, contrast=1.05),
    "cool": Grade(exposure=-0.05, lift=(-0.01, 0.0, 0.03), gamma=(0.97, 1.0, 1.04),
                  gain=(0.92, 1.0, 1.1), saturation=1.02, contrast=1.06),
    "neon": Grade(exposure=0.1, lift=(0.0, 0.0, 0.05), gamma=(0.9, 0.95, 1.08),
                  gain=(1.1, 1.0, 1.15), saturation=1.35, contrast=1.2,
                  matrix=((1.1, -0.05, 0.05), (-0.05, 1.02, 0.05), (0.05, -0.02, 1.15))),
    "vintage": Grade(exposure=-0.05, lift=(0.05, 0.03, 0.0), gamma=(1.05, 1.0, 0.95),
                     gain=(1.0, 0.98, 0.85), saturation=0.82, contrast=0.95),
}


def resolve_grade(name: Optional[str]) -> Grade:
    return GRADE_PRESETS.get((name or "").lower(), GRADE_PRESETS["cinematic"])


class ColorEngine:
    def __init__(self, compute: Optional[RTACompute] = None):
        self.compute = compute or RTACompute()

    def grade_frame(self, frame: np.ndarray, grade: Grade) -> np.ndarray:
        """Apply the grade to an ``HxWx3`` uint8/float frame; return uint8."""
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"grade_frame expects HxWx3, got {arr.shape}")
        rgb = arr[:, :, :3].astype(np.float64) / 255.0
        h, w, _ = rgb.shape

        # exposure (stops)
        rgb *= (2.0 ** grade.exposure)

        # lift / gamma / gain — the classic three-way grade
        lift = np.array(grade.lift, dtype=np.float64)
        gain = np.array(grade.gain, dtype=np.float64)
        gamma = np.array(grade.gamma, dtype=np.float64)
        rgb = rgb * gain + lift * (1.0 - rgb)
        rgb = np.clip(rgb, 0.0, None)
        rgb = np.power(rgb, 1.0 / np.maximum(gamma, 1e-3))

        # contrast around 0.5 pivot
        rgb = (rgb - 0.5) * grade.contrast + 0.5

        # saturation
        if abs(grade.saturation - 1.0) > 1e-3:
            luma = (rgb * np.array([0.2126, 0.7152, 0.0722])).sum(axis=2, keepdims=True)
            rgb = luma + (rgb - luma) * grade.saturation

        # creative colour matrix — routed through the Digital GPU as a GEMM
        if grade.matrix is not None:
            mat = np.array(grade.matrix, dtype=np.float64)          # [3,3]
            flat = np.clip(rgb, 0.0, 4.0).reshape(-1, 3)            # [N,3]
            graded = self.compute.gemm(flat, mat.T)                 # [N,3] on Digital GPU
            rgb = graded.reshape(h, w, 3).astype(np.float64)

        # filmic tonemap + clip
        x = np.maximum(rgb, 0.0)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        rgb = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

        out = (rgb * 255.0 + 0.5).astype(np.uint8)
        if arr.shape[2] == 4:
            out = np.dstack([out, arr[:, :, 3]])
        return out
