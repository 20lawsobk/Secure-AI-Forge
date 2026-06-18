"""
MaxCore Retrieval — in-house deterministic image embedding.

Turns a real image (file path, PIL image, or HxWx3 uint8 array) into a fixed,
L2-normalized feature vector using ONLY local CPU libraries (PIL + numpy). No
RNG, no external models, no network — the same pixels always yield the same
vector, which is what makes anchor loading idempotent and retrieval deterministic.

Vector layout (EMBED_DIM = 128), with stable named slices so callers (e.g. gap
decoding) can inspect the palette region directly:

    [ 0 : 24]  HUE_BINS  — hue histogram        ┐
    [24 : 36]  SAT_BINS  — saturation histogram │ PALETTE_SLICE = [0:48]
    [36 : 48]  VAL_BINS  — value histogram      ┘
    [48 : 96]  4x4 grid mean RGB (16 cells x 3)
    [96 :112]  4x4 grid mean luminance (16 cells)
    [112:128]  16 global stats (luminance/contrast/edge/colorfulness/aspect…)

The function is TOTAL: it never raises. On unloadable/empty input it returns
None and lets the caller skip that asset.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from PIL import Image
    _PIL_OK = True
except ImportError:  # pragma: no cover - PIL is a hard dep in this project
    _PIL_OK = False

HUE_BINS = 24
SAT_BINS = 12
VAL_BINS = 12
GRID = 4                       # 4x4 spatial grid
WORK = 96                      # working resolution (divisible by GRID)
CELL = WORK // GRID

_PALETTE_LEN = HUE_BINS + SAT_BINS + VAL_BINS          # 48
_GRID_RGB_LEN = GRID * GRID * 3                        # 48
_GRID_LUM_LEN = GRID * GRID                            # 16
_STATS_LEN = 16
EMBED_DIM = _PALETTE_LEN + _GRID_RGB_LEN + _GRID_LUM_LEN + _STATS_LEN   # 128

PALETTE_SLICE = slice(0, _PALETTE_LEN)                 # [0:48] HSV histograms


def _to_rgb(src: Any) -> Optional["Image.Image"]:
    if not _PIL_OK:
        return None
    try:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, np.ndarray):
            arr = np.asarray(src)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr.astype(np.uint8)[:, :, :3], "RGB")
        return Image.open(str(src)).convert("RGB")
    except Exception:
        return None


def image_to_vector(src: Any) -> Optional[np.ndarray]:
    """Deterministic EMBED_DIM feature vector for a real image. Never raises."""
    img = _to_rgb(src)
    if img is None:
        return None
    try:
        orig_w, orig_h = img.size
        if orig_w <= 0 or orig_h <= 0:
            return None

        small = img.resize((WORK, WORK))
        rgb = np.asarray(small, dtype=np.float32)            # (WORK, WORK, 3)
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # ── HSV histograms (palette region) ──────────────────────────────
        hue_hist, _ = np.histogram(h, bins=HUE_BINS, range=(0.0, 256.0))
        sat_hist, _ = np.histogram(s, bins=SAT_BINS, range=(0.0, 256.0))
        val_hist, _ = np.histogram(v, bins=VAL_BINS, range=(0.0, 256.0))
        n_px = float(WORK * WORK)
        hue_hist = hue_hist / n_px
        sat_hist = sat_hist / n_px
        val_hist = val_hist / n_px

        # ── 4x4 spatial grid mean RGB + luminance ────────────────────────
        cells = rgb.reshape(GRID, CELL, GRID, CELL, 3)
        grid_rgb = cells.mean(axis=(1, 3)).reshape(-1) / 255.0          # 48
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
               + 0.114 * rgb[:, :, 2])                                  # (WORK,WORK)
        grid_lum = lum.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)).reshape(-1) / 255.0  # 16

        # ── Global statistics block (16) ─────────────────────────────────
        p05, p50, p95 = np.percentile(lum, [5, 50, 95])
        edge_h = float(np.abs(np.diff(lum, axis=1)).mean()) / 255.0
        edge_v = float(np.abs(np.diff(lum, axis=0)).mean()) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = (float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
                        + 0.3 * float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))) / 255.0
        aspect = 0.5 * (np.tanh(np.log((orig_w + 1.0) / (orig_h + 1.0))) + 1.0)
        warm_ratio = float((r > b).mean())
        dark_ratio = float((lum < 64.0).mean())

        stats = np.array([
            float(lum.mean()) / 255.0,
            float(lum.std()) / 128.0,
            float(p05) / 255.0,
            float(p50) / 255.0,
            float(p95) / 255.0,
            float(p95 - p05) / 255.0,
            edge_h,
            edge_v,
            min(1.0, colorfulness),
            float(s.mean()) / 255.0,
            float(s.std()) / 128.0,
            float(v.mean()) / 255.0,
            float(v.std()) / 128.0,
            float(aspect),
            warm_ratio,
            dark_ratio,
        ], dtype=np.float32)

        vec = np.concatenate([
            hue_hist.astype(np.float32),
            sat_hist.astype(np.float32),
            val_hist.astype(np.float32),
            grid_rgb.astype(np.float32),
            grid_lum.astype(np.float32),
            stats,
        ])
        if vec.shape[0] != EMBED_DIM:
            return None
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None
        return (vec / norm).astype(np.float32)
    except Exception:
        return None
