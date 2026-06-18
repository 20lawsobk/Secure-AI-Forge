"""
MaxCore RCGS — Retrieval-Conditioned Generation for the video compositor.

This is the live query path that closes the retrieval loop. When a scene
background is generated, RCGS:

  1. embeds the procedural background into the same feature space as the index,
  2. records that query as a probe (so the CoverageWatchdog measures real usage
     and ingests gaps), and
  3. retrieves the closest REAL stored asset and uses it to *condition* the
     background — grounding the synthetic gradient in real domain structure.

Conditioning is deliberately conservative — "real grounding without harming the
scene". The retrieved asset is reduced to a desaturated, heavily blurred
*luminance structure* and applied as a gentle MULTIPLICATIVE modulation. That
keeps the scene's palette authoritative (hue ratios are preserved) and cannot
ghost text or decorative elements from library PNGs into the frame, while still
letting real assets shape the light/structure of the scene. Text is drawn on top
by ffmpeg afterwards, so legibility is untouched.

Everything here is TOTAL: any failure returns the original background unchanged,
so RCGS can only ever improve a frame, never break one (the "no broken fallback"
invariant). Path resolution is sandboxed to the approved uploads directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ai_model.image.image_engine import _UPLOADS_DIR
from ai_model.retrieval.asset_pipeline import get_asset_index
from ai_model.retrieval.image_features import image_to_vector
from ai_model.retrieval.probes import record_probe

# Strength of the multiplicative luminance modulation. With a centered structure
# map in roughly [-0.5, 0.5], a strength of 0.30 yields a per-pixel brightness
# factor within ~[0.85, 1.15] — visible grounding, palette preserved.
RCGS_ALPHA = 0.30
_MAX_BLUR_RADIUS = 24.0

# How strongly retrieval is pulled toward a brand's centroid (its learned visual
# identity) before the nearest-asset lookup. Small by design: it biases which real
# asset grounds the frame without overriding the scene's own structure.
RCGS_BRAND_WEIGHT = 0.15


def _uploads_base() -> Optional[Path]:
    try:
        return Path(_UPLOADS_DIR).resolve()
    except Exception:
        return None


def _resolve_local_path(meta: Dict[str, Any]) -> Optional[Path]:
    """
    Resolve a retrieved asset's metadata to a real file STRICTLY inside the
    approved uploads directory. URLs are ignored; only the basename of any
    ``filename``/``path`` hint is honored, so ``..`` traversal is impossible.
    """
    base = _uploads_base()
    if base is None:
        return None
    name = ""
    for key in ("filename", "path"):
        val = meta.get(key)
        if val:
            name = Path(str(val)).name  # basename only — strips any traversal
            if name:
                break
    if not name:
        return None
    try:
        cand = (base / name).resolve()
    except Exception:
        return None
    if cand != base and base not in cand.parents:
        return None
    if not cand.is_file():
        return None
    return cand


def _layout_for(width: int, height: int) -> str:
    """Map raw dimensions to a renderable layout token (for ingestion context)."""
    if height > width:
        return "vertical_9_16"
    return "square_1_1"


def _brand_biased_vector(
    index: Any, vec: np.ndarray, brand: Optional[str],
    weight: float = RCGS_BRAND_WEIGHT,
) -> np.ndarray:
    """
    Nudge the query toward the brand's centroid so retrieval favors the brand's
    established visual identity. Total: returns the original ``vec`` unchanged on
    any failure or when the brand has no centroid yet.
    """
    if not brand or weight <= 0.0:
        return vec
    try:
        centroid = index.brand_centroid(str(brand))
        if centroid is None:
            return vec
        blended = ((1.0 - weight) * np.asarray(vec, dtype=np.float32)
                   + weight * np.asarray(centroid, dtype=np.float32))
        norm = float(np.linalg.norm(blended))
        if norm <= 1e-12:
            return vec
        return (blended / norm).astype(np.float32)
    except Exception:
        return vec


def _structure_map(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    """
    Load a retrieved asset as a desaturated, heavily blurred, mean-centered
    luminance structure map in [-0.5, 0.5], cover-cropped to (height, width).
    Returns None on any failure.
    """
    try:
        from PIL import Image, ImageFilter
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    try:
        img = Image.open(str(path)).convert("L")  # desaturate
    except Exception:
        return None
    try:
        sw, sh = img.size
        if sw <= 0 or sh <= 0:
            return None
        # Cover-resize: scale to fill, then center-crop to exactly (width, height).
        scale = max(width / float(sw), height / float(sh))
        rw, rh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        img = img.resize((rw, rh), Image.BILINEAR)
        left = max(0, (rw - width) // 2)
        top = max(0, (rh - height) // 2)
        img = img.crop((left, top, left + width, top + height))
        # Heavy blur destroys any text/decoration legibility, leaving tonal
        # structure only. Radius scales with frame size, capped for speed.
        radius = min(_MAX_BLUR_RADIUS, max(4.0, max(width, height) / 64.0))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        g = np.asarray(img, dtype=np.float32)
        if g.shape[:2] != (height, width):
            return None
        mod = (g / 255.0) - float(g.mean()) / 255.0  # mean-centered → ~[-0.5, 0.5]
        return mod.astype(np.float32)
    except Exception:
        return None


def _apply_modulation(bg_arr: np.ndarray, mod: np.ndarray, alpha: float) -> np.ndarray:
    """Gentle multiplicative luminance modulation; preserves hue, clips to uint8."""
    try:
        factor = 1.0 + (alpha * 2.0) * mod        # mod ~[-0.5,0.5] → factor ~[1-α,1+α]
        factor = np.clip(factor, 0.0, 2.0)[:, :, None]
        out = bg_arr.astype(np.float32) * factor
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return bg_arr


def condition_background(
    bg_arr: np.ndarray,
    width: int,
    height: int,
    *,
    brand: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    alpha: float = RCGS_ALPHA,
) -> np.ndarray:
    """
    Condition a procedural background on the closest real retrieved asset and
    record the query as a coverage probe.

    Total: on ANY failure (no index, no usable asset, unreadable file, bad shape)
    the original ``bg_arr`` is returned unchanged — pure procedural rendering, the
    exact pre-RCGS behavior.
    """
    try:
        if not isinstance(bg_arr, np.ndarray) or bg_arr.ndim != 3:
            return bg_arr

        vec = image_to_vector(bg_arr)
        if vec is None:
            return bg_arr

        ctx: Dict[str, Any] = {"layout": _layout_for(width, height)}
        if brand:
            ctx["brand"] = str(brand)
        if isinstance(context, dict):
            for k in ("color_scheme", "platform", "prompt"):
                v = context.get(k)
                if v:
                    ctx[k] = str(v)

        # Record the live query (the TRUE demand, unbiased) so self-healing
        # measures real coverage of the query space.
        record_probe(vec, ctx)

        # Bias retrieval toward the brand's learned identity before the lookup.
        index = get_asset_index()
        q = _brand_biased_vector(index, vec, brand)
        asset = index.query(q, brand=(brand or None))
        if asset is None:
            return bg_arr

        path = _resolve_local_path(asset.metadata or {})
        if path is None:
            return bg_arr

        mod = _structure_map(path, width, height)
        if mod is None:
            return bg_arr

        return _apply_modulation(bg_arr, mod, float(alpha))
    except Exception:
        return bg_arr
