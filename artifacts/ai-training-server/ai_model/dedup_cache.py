"""Fleet-wide generation deduplication cache.

Identical generation requests are expensive to recompute on the Digital GPU/NumPy
inference path. This module stores successful generation results in the shared
pdim store (``storage_client``) keyed by a stable hash of the request's
semantic fields, so a repeat of the *same* request — from any node/account in
the fleet — returns the stored result instead of re-running the model.

Design guarantees (additive & non-breaking):
  * Best-effort only. Any storage error, miss, or disabled state degrades to the
    exact pre-existing behaviour (compute fresh). It never raises into callers —
    only well-formed, unexpired ``dict`` results are ever returned.
  * No request/response contract change. Callers add an optional ``cached`` flag
    and refresh ``processing_time_ms`` on a hit; every other field is the stored
    result verbatim.
  * Correctness over hit-rate. Only *transport metadata* (ids/timestamps/nonces)
    is stripped, and only from the TOP LEVEL of the request — nested values are
    semantic and can change the output, so they are never scrubbed. Two requests
    therefore share a key only when every output-affecting field matches.
  * TTL is enforced in this layer (via an expiry envelope), so results also
    expire when the pdim store is offline and ``StorageClient`` falls back to a
    non-expiring in-process dict.

When the pdim store is offline, ``StorageClient`` transparently falls back to an
in-process dict, so dedup still works within a single worker (process-local).

Tunables (env):
  * ``DEDUP_CACHE_ENABLED`` — "0"/"false" disables the cache entirely (default on).
  * ``DEDUP_CACHE_TTL``     — result time-to-live in seconds (default 3600).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

# Transport-metadata keys stripped from the TOP LEVEL of a request before
# hashing. We deliberately do NOT recurse: nested values (e.g. a text request's
# ``inputs``/``step``/``slots``) are semantic and can change the generated
# output, so stripping a nested ``id``/``time`` would let two genuinely
# different requests collide on the same cache key.
_VOLATILE_KEYS = {
    "id", "request_id", "requestid", "requestId", "trace_id", "traceId",
    "nonce", "timestamp", "ts", "time", "created_at", "createdAt",
    "processing_time_ms", "seed_nonce",
}

_ENVELOPE_VAL = "_dedup_val"
_ENVELOPE_EXP = "_dedup_exp"


def _env_enabled() -> bool:
    return os.getenv("DEDUP_CACHE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


_ENABLED = _env_enabled()
try:
    _TTL = int(os.getenv("DEDUP_CACHE_TTL", "3600"))
except ValueError:
    _TTL = 3600

_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "stores": 0, "errors": 0}


def _strip_top_level(obj: Any) -> Any:
    """Drop transport-metadata keys from the TOP LEVEL only (never recurse)."""
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k not in _VOLATILE_KEYS}
    return obj


def _normalize(req: Any) -> Any:
    """Turn a pydantic model (v1 or v2) or plain value into a JSON-safe value."""
    if hasattr(req, "model_dump"):
        try:
            return req.model_dump()
        except Exception:
            pass
    if hasattr(req, "dict"):
        try:
            return req.dict()
        except Exception:
            pass
    return req


def key_for(namespace: str, req: Any) -> Optional[str]:
    """Stable cache key for a request, or ``None`` when caching is disabled."""
    if not _ENABLED:
        return None
    try:
        payload = _strip_top_level(_normalize(req))
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return f"dedupe:{namespace}:{digest}"
    except Exception:
        with _lock:
            _stats["errors"] += 1
        return None


def get(key: Optional[str]) -> Optional[dict]:
    """Return a cached result ``dict`` for ``key`` (best-effort) or ``None``.

    Only a well-formed, unexpired envelope wrapping a ``dict`` is honoured;
    anything else (miss, corruption, wrong type, expired) is treated as a miss
    and never raised to the caller. Expired entries are best-effort deleted so
    the in-process fallback store does not grow unbounded.
    """
    if not key:
        return None
    try:
        from storage_client import get_storage
        raw = get_storage().get(key)
    except Exception:
        with _lock:
            _stats["errors"] += 1
        return None

    value: Optional[dict] = None
    if isinstance(raw, dict) and _ENVELOPE_VAL in raw and _ENVELOPE_EXP in raw:
        try:
            fresh = float(raw[_ENVELOPE_EXP]) > time.time()
        except (TypeError, ValueError):
            fresh = False
        inner = raw.get(_ENVELOPE_VAL)
        if fresh and isinstance(inner, dict):
            value = inner
        elif not fresh:
            try:
                from storage_client import get_storage
                get_storage().delete(key)
            except Exception:
                pass

    with _lock:
        if value is None:
            _stats["misses"] += 1
        else:
            _stats["hits"] += 1
    return value


def put(key: Optional[str], value: Any, ttl: Optional[int] = None) -> None:
    """Store a result ``dict`` under ``key`` with a TTL envelope (best-effort).

    Non-dict values are ignored so a hit can always be copied safely by callers.
    """
    if not key or not isinstance(value, dict):
        return
    ttl = ttl if ttl is not None else _TTL
    envelope = {_ENVELOPE_EXP: time.time() + ttl, _ENVELOPE_VAL: value}
    try:
        from storage_client import get_storage
        get_storage().set(key, envelope, ex=ttl)
        with _lock:
            _stats["stores"] += 1
    except Exception:
        with _lock:
            _stats["errors"] += 1


def stats() -> dict:
    """Observability snapshot for the dedup cache."""
    with _lock:
        hits = _stats["hits"]
        misses = _stats["misses"]
        total = hits + misses
        return {
            "enabled": _ENABLED,
            "ttl_seconds": _TTL,
            "hits": hits,
            "misses": misses,
            "stores": _stats["stores"],
            "errors": _stats["errors"],
            "hit_rate": round(hits / total, 3) if total else 0.0,
        }
