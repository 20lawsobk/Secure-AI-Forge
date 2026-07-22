"""
MaxCore Retrieval — real-asset pipeline (anchor core, seed library, ingestion).

This is the bridge between the in-house ImageEngine (real PNGs on disk) and the
AssetIndex cascade. Everything here produces REAL pixels — durable files in
uploads/images — never procedural placeholders:

  • ensure_library(index)        — renders the deterministic domain library:
        ANCHOR assets  (one per color scheme) → the cascade's always-real
                                                 fallback rung, and
        SEED   assets  (scheme x layout)      → real non-anchor coverage so the
                                                 structural gate is healthy before
                                                 the live query path exists.
    Idempotent: stable asset_ids + seeded deterministic renders mean re-runs
    replace in place (no duplicates, no litter). This is the watchdog's
    anchor_loader_fn.

  • ingest_gaps(index, gaps)     — the watchdog's ingestion_fn. Each gap is a
    feature vector retrieval could not cover. We decode the nearest renderable
    color scheme (and any provided request context), render a REAL asset to fill
    that region, and add it as a non-anchor asset. Accepts both raw-vector gaps
    ({"vector": [...]}) and richer {"vector", "context"} gaps (Phase 3+).

  • ingest_generated_asset(...)   — additive hook to fold images the system has
    already produced into the index.

All functions are total (never raise) and deterministic.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Digital GPU backend singleton ─────────────────────────────────────────────
_GPU_BACKEND = None
_GPU_BACKEND_LOCK = threading.Lock()

def _get_gpu():
    global _GPU_BACKEND
    if _GPU_BACKEND is None:
        with _GPU_BACKEND_LOCK:
            if _GPU_BACKEND is None:
                try:
                    from ai_model.gpu.torch_backend import DigitalGPUBackend
                    _GPU_BACKEND = DigitalGPUBackend()
                except Exception:
                    pass
    return _GPU_BACKEND

from ai_model.image.image_engine import (
    COLOR_SCHEMES,
    _UPLOADS_DIR,
    _gradient_array,
    ImageEngine,
    ImageRequest,
)
from ai_model.retrieval.asset_index import AssetIndex
from ai_model.retrieval.image_features import EMBED_DIM, PALETTE_SLICE, image_to_vector

# Canonical library shape.
ANCHOR_LAYOUT = "square_1_1"
SEED_LAYOUTS = ("square_1_1", "vertical_9_16")
ANCHOR_PROMPT = "MaxCore domain anchor"
SEED_PROMPT = "MaxCore promotional seed"

_engine: Optional[ImageEngine] = None
_engine_lock = threading.Lock()
_index: Optional[AssetIndex] = None
_index_lock = threading.Lock()
_sig_cache: Dict[str, np.ndarray] = {}
_sig_lock = threading.Lock()


# ------------------------------------------------------------------ #
# Singletons                                                          #
# ------------------------------------------------------------------ #

def get_asset_index() -> AssetIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = AssetIndex(dim=EMBED_DIM)
        return _index


def _get_engine() -> ImageEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ImageEngine()
        return _engine


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _seed_for(key: str) -> int:
    return int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16) % (2 ** 31)


def _vec_hash(vec: Any) -> str:
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        arr = np.round(arr, 2)
        return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=8).hexdigest()
    except Exception:
        return "0"


def _render_features(req: ImageRequest) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Render a real asset and extract its features. Returns (vec, file_meta)."""
    try:
        res = _get_engine().render(req)
    except Exception:
        return None
    if not res.success or not res.filename:
        return None
    vec = image_to_vector(Path(_UPLOADS_DIR) / res.filename)
    if vec is None:
        return None
    return vec, {"url": res.url, "filename": res.filename,
                 "width": res.width, "height": res.height}


def _scheme_palette_sig(scheme: str) -> Optional[np.ndarray]:
    """Deterministic palette signature for a color scheme (palette slice only)."""
    with _sig_lock:
        cached = _sig_cache.get(scheme)
    if cached is not None:
        return cached
    spec = COLOR_SCHEMES.get(scheme)
    if not spec:
        return None
    try:
        from PIL import Image
        top, bottom = spec[0], spec[1]
        arr = _gradient_array(64, 64, top, bottom)
        vec = image_to_vector(Image.fromarray(arr, "RGB"))
    except Exception:
        vec = None
    if vec is None:
        return None
    sig = np.asarray(vec)[PALETTE_SLICE].astype(np.float64)
    with _sig_lock:
        _sig_cache[scheme] = sig
    return sig


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    gpu = _get_gpu()
    if gpu is not None:
        engine = gpu.gpu
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b, dtype=np.float32)
        na = float(np.sqrt(abs(engine.gemm(a32.reshape(1, -1), a32.reshape(-1, 1)).ravel()[0])))
        nb = float(np.sqrt(abs(engine.gemm(b32.reshape(1, -1), b32.reshape(-1, 1)).ravel()[0])))
        if na <= 1e-12 or nb <= 1e-12:
            return -1.0
        return float(engine.gemm(a32.reshape(1, -1), b32.reshape(-1, 1)).ravel()[0]) / (na * nb)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def decode_scheme(vec: Any) -> str:
    """Pick the color scheme whose palette signature best matches `vec`."""
    try:
        pal = np.asarray(vec, dtype=np.float64).reshape(-1)[PALETTE_SLICE]
    except Exception:
        return "dark_neon"
    best, best_sim = "dark_neon", -2.0
    for scheme in COLOR_SCHEMES:
        sig = _scheme_palette_sig(scheme)
        if sig is None:
            continue
        sim = _cos(pal, sig)
        if sim > best_sim:
            best_sim, best = sim, scheme
    return best


def _parse_gap(payload: Any) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Accept raw-vector gaps and {'vector', 'context'} gaps alike."""
    if isinstance(payload, dict):
        ctx = payload.get("context") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        return payload.get("vector"), ctx
    return payload, {}


# ------------------------------------------------------------------ #
# Library construction (anchor_loader_fn)                             #
# ------------------------------------------------------------------ #

def ensure_library(index: Optional[AssetIndex] = None) -> int:
    """
    Render/refresh the deterministic domain library (anchors + seeds) and load it
    into the index. Idempotent and total. Returns the number of ANCHORS present
    (the value the coverage watchdog checks against MIN_ANCHORS).
    """
    idx = index if index is not None else get_asset_index()
    anchors = 0

    for scheme in COLOR_SCHEMES:
        aid = f"anchor::{scheme}"
        req = ImageRequest(
            prompt=ANCHOR_PROMPT, color_scheme=scheme, layout=ANCHOR_LAYOUT,
            platform="instagram", artist_name="MaxCore", intent="anchor",
            style_tags=["anchor", "cinematic"], seed=_seed_for(aid),
        )
        rf = _render_features(req)
        if rf is None:
            continue
        vec, meta = rf
        meta.update({"scheme": scheme, "layout": ANCHOR_LAYOUT, "kind": "anchor"})
        if idx.add(aid, vec, meta, is_anchor=True):
            anchors += 1

    for scheme in COLOR_SCHEMES:
        for layout in SEED_LAYOUTS:
            sid = f"seed::{scheme}::{layout}"
            req = ImageRequest(
                prompt=SEED_PROMPT, color_scheme=scheme, layout=layout,
                platform="instagram", artist_name="MaxCore", intent="promotional",
                style_tags=["seed", "cinematic"], seed=_seed_for(sid),
            )
            rf = _render_features(req)
            if rf is None:
                continue
            vec, meta = rf
            meta.update({"scheme": scheme, "layout": layout, "kind": "seed"})
            idx.add(sid, vec, meta, is_anchor=False)

    return anchors


# ------------------------------------------------------------------ #
# Ingestion (ingestion_fn)                                           #
# ------------------------------------------------------------------ #

def ingest_gaps(index: Optional[AssetIndex], gaps: List[Any]) -> int:
    """Render a real asset for each coverage gap and add it. Total; returns count."""
    idx = index if index is not None else get_asset_index()
    count = 0
    for payload in gaps or []:
        try:
            vec, ctx = _parse_gap(payload)
            if vec is None:
                continue
            scheme = str(ctx.get("color_scheme") or decode_scheme(vec))
            if scheme not in COLOR_SCHEMES:
                scheme = decode_scheme(vec)
            layout = str(ctx.get("layout") or ANCHOR_LAYOUT)
            brand = ctx.get("brand")
            brand = str(brand) if brand else None
            aid = f"ingest::{scheme}::{layout}::{_vec_hash(vec)}"
            req = ImageRequest(
                prompt=str(ctx.get("prompt") or "MaxCore coverage fill"),
                color_scheme=scheme, layout=layout,
                platform=str(ctx.get("platform") or "instagram"),
                artist_name=brand or "MaxCore", intent="coverage",
                style_tags=["ingested", "cinematic"], seed=_seed_for(aid),
            )
            rf = _render_features(req)
            if rf is None:
                continue
            fvec, meta = rf
            meta.update({"scheme": scheme, "layout": layout, "kind": "ingested"})
            if idx.add(aid, fvec, meta, is_anchor=False, brand=brand):
                count += 1
        except Exception:
            continue
    return count


def ingest_generated_asset(
    index: Optional[AssetIndex],
    image_path: Any,
    *,
    brand: Optional[str] = None,
    asset_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Fold an already-produced real image into the index. Total; never raises."""
    idx = index if index is not None else get_asset_index()
    try:
        vec = image_to_vector(image_path)
        if vec is None:
            return False
        aid = asset_id or ("gen::" + hashlib.blake2b(
            str(image_path).encode(), digest_size=10).hexdigest())
        meta: Dict[str, Any] = dict(metadata or {})
        meta.setdefault("path", str(image_path))
        meta.setdefault("kind", "generated")
        return idx.add(aid, vec, meta, is_anchor=False, brand=brand)
    except Exception:
        return False
