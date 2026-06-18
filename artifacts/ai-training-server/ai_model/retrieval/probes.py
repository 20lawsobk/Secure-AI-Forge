"""
MaxCore Retrieval — live probe ring buffer.

The compositor (and any future live query path) records every retrieval query it
issues here as a *probe*: a (vector, context) pair describing what the system
actually asked the index for. The CoverageWatchdog samples the most recent probes
each cycle to measure real-world coverage and to turn weak queries into ingestion
targets — closing the loop between what is generated and what the index holds.

Design goals (mirrors the rest of the retrieval spine):
  • Total — never raises; bad input is sanitized or dropped.
  • Deterministic order — newest-first, so a sample of the latest N is always the
    freshest N regardless of buffer churn.
  • Burst-resistant — a short-TTL signature dedupe stops one large render (many
    near-identical scenes) from flooding the buffer and biasing self-healing.
  • Bounded — a fixed-size deque caps memory; oldest probes fall off.

Probe payload shape (what :func:`recent_probes` returns):

    {"vector": [float, ...], "context": {...}, "ts": float}

``context`` is a small, JSON-safe dict carrying retrieval hints such as ``brand``,
``layout``, ``color_scheme`` and ``platform`` — exactly the keys
``asset_pipeline.ingest_gaps`` understands, so a weak probe can be ingested with
full context instead of a bare vector.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

RING_SIZE = 512            # max probes retained
DEDUPE_TTL = 2.0           # seconds; identical signatures within this window are dropped
_CONTEXT_KEYS = ("brand", "layout", "color_scheme", "platform", "prompt")

_ring: Deque[Dict[str, Any]] = deque(maxlen=RING_SIZE)
_sig_seen: Dict[str, float] = {}
_lock = threading.Lock()


def _sanitize_vector(vector: Any) -> Optional[List[float]]:
    """Coerce to a finite, non-empty 1-D float list, or None if unusable."""
    try:
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.linalg.norm(arr)) <= 1e-12:
        return None
    return arr.tolist()


def _clean_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only known, truthy, JSON-safe hint keys."""
    if not isinstance(context, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in _CONTEXT_KEYS:
        v = context.get(k)
        if v is None or v == "":
            continue
        out[k] = str(v)
    return out


def _signature(vec: List[float], ctx: Dict[str, Any]) -> str:
    """Stable signature over a rounded vector + context (for dedupe)."""
    try:
        arr = np.round(np.asarray(vec, dtype=np.float64), 2)
        h = hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=8)
        h.update(repr(sorted(ctx.items())).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return "0"


def record_probe(vector: Any, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Record one retrieval probe. Returns False (never raises) when the vector is
    unusable or when the probe is a short-TTL duplicate of a recent one.
    """
    vec = _sanitize_vector(vector)
    if vec is None:
        return False
    ctx = _clean_context(context)
    sig = _signature(vec, ctx)
    now = time.time()
    with _lock:
        last = _sig_seen.get(sig)
        if last is not None and (now - last) < DEDUPE_TTL:
            _sig_seen[sig] = now
            return False
        if _ring and _ring[-1].get("sig") == sig:
            _sig_seen[sig] = now
            return False
        _sig_seen[sig] = now
        # Opportunistically prune the dedupe map so it cannot grow unbounded.
        if len(_sig_seen) > RING_SIZE * 4:
            cutoff = now - DEDUPE_TTL
            for k in [k for k, t in _sig_seen.items() if t < cutoff]:
                _sig_seen.pop(k, None)
        _ring.append({"vector": vec, "context": ctx, "ts": now, "sig": sig})
    return True


def recent_probes(n: int = 32) -> List[Dict[str, Any]]:
    """
    Return up to ``n`` most-recent probes, newest first, as
    ``{"vector", "context"}`` payloads. Total; safe to call concurrently.
    """
    if n <= 0:
        return []
    with _lock:
        snapshot = list(_ring)
    out: List[Dict[str, Any]] = []
    for p in reversed(snapshot):
        out.append({"vector": p["vector"], "context": dict(p.get("context") or {})})
        if len(out) >= n:
            break
    return out


def probe_count() -> int:
    with _lock:
        return len(_ring)


def clear_probes() -> None:
    with _lock:
        _ring.clear()
        _sig_seen.clear()
