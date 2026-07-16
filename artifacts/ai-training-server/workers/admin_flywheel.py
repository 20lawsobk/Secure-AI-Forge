"""
admin_flywheel.py — Admin Content Flywheel

Automatically stores content generated via admin API keys into pdim as
training datasets. All writes are fire-and-forget (background thread pool)
so generation response latency is never affected.

Dataset namespaces in pdim (all under mb:dataset:flywheel:*):
  - mb:dataset:flywheel:scripts   — hook/body/cta text + metadata
  - mb:dataset:flywheel:social    — social variants with captions/hashtags
  - mb:dataset:flywheel:video     — scene packs with narration + visual direction
  - mb:dataset:flywheel:daw       — lyrics/hooks/beat descriptions
  - mb:dataset:flywheel:distribution — release plans + pitch copy
  - mb:dataset:flywheel:images    — image concept specs + prompts

Storage pattern (mirrors audio dataset convention):
  mb:dataset:flywheel:{type}          — list of serialised samples (lpush)
  mb:dataset:flywheel:{type}:meta     — JSON manifest: count, updated_at, description
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_fw_logger = logging.getLogger("flywheel")

# Artist identity — embedded in every flywheel sample so the model learns
# a B-Lawz-specific voice as training data accumulates.
# Override via env vars if the platform expands to multiple artists.
_ARTIST_ID    = os.environ.get("FLYWHEEL_ARTIST_ID",    "b-lawz")
_ARTIST_NAME  = os.environ.get("FLYWHEEL_ARTIST_NAME",  "B-Lawz")
_ARTIST_LABEL = os.environ.get("FLYWHEEL_ARTIST_LABEL", "B-Lawz Music")
_ARTIST_EMAIL = os.environ.get("FLYWHEEL_ARTIST_EMAIL", "blawzmusic@gmail.com")

_DATASET_DESCRIPTIONS: dict[str, str] = {
    "scripts":      "Admin-generated hook/body/cta scripts paired with topic, platform, tone, awareness",
    "social":       "Admin-generated social post variants (captions, hashtags) per platform and topic",
    "video":        "Admin-generated video scene packs with narration, visual direction, and awareness context",
    "daw":          "Admin-generated DAW content: lyrics, hooks, beat descriptions with genre/mood/key",
    "distribution": "Admin-generated distribution plans with pitch copy and platform strategy",
    "images":       "Admin-generated image concept specs with style tags, platform, and visual direction",
}


class FlywheelIngestor:
    """
    Singleton-safe ingestor — create once, call .ingest() freely from any thread.
    Uses a single background worker thread; storage errors are always swallowed.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="flywheel")
        self._meta_locks: dict[str, threading.Lock] = {}
        self._meta_lock_guard = threading.Lock()
        _fw_logger.info("[flywheel] ingestor initialised (max_workers=%d)", max_workers)

    def _meta_lock(self, content_type: str) -> threading.Lock:
        with self._meta_lock_guard:
            if content_type not in self._meta_locks:
                self._meta_locks[content_type] = threading.Lock()
            return self._meta_locks[content_type]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """Return True when the background executor is ready to accept work."""
        return not getattr(self._executor, "_shutdown", True)

    def restart(self) -> None:
        """Replace the executor with a fresh one so ingestion resumes.

        Never raises — called by the Watchdog when it detects a dead executor.
        The old executor is shut down in the background (non-blocking).
        """
        try:
            old = self._executor
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="flywheel"
            )
            try:
                old.shutdown(wait=False)
            except Exception:
                pass
            _fw_logger.info("[flywheel] executor restarted by watchdog")
        except Exception as exc:
            _fw_logger.error("[flywheel] restart failed: %s", exc)

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(
        self,
        content_type: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
        key_id: str = "admin",
    ) -> None:
        """
        Fire-and-forget: submit a storage job to the background executor.
        Returns immediately; never raises.
        """
        try:
            self._executor.submit(self._store, content_type, payload, meta, key_id)
        except Exception as exc:
            _fw_logger.debug("[flywheel] submit error (type=%s): %s", content_type, exc)

    def _store(
        self,
        content_type: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
        key_id: str,
    ) -> None:
        """Background worker — pushes one sample to pdim and updates the manifest."""
        try:
            from storage_client import get_storage  # noqa: PLC0415
        except Exception:
            return  # storage client not available (dev env without pdim)

        storage = get_storage()
        if storage is None:
            return

        ts = time.time()
        sample: dict[str, Any] = {
            "ts": ts,
            "key_id": key_id,
            "content_type": content_type,
            "artist": {
                "id":    _ARTIST_ID,
                "name":  _ARTIST_NAME,
                "label": _ARTIST_LABEL,
            },
            "meta": meta,
            "payload": _sanitise(payload),
        }

        list_key = f"mb:dataset:flywheel:{content_type}"
        meta_key = f"mb:dataset:flywheel:{content_type}:meta"

        try:
            storage.lpush(list_key, sample)
        except Exception as exc:
            _fw_logger.debug("[flywheel] lpush error (%s): %s", list_key, exc)
            return

        with self._meta_lock(content_type):
            self._update_manifest(storage, meta_key, content_type, ts)

        # ── Graduation: fold text into the live generation corpora ──────────
        # Raw flywheel samples are archival; generation actually reads the
        # `mb:phrases:<scene_type>` corpora (dataset_sampler tier 1).  Pushing
        # the generated text there makes admin content a REAL dataset source
        # immediately — and because quality_awareness.self_sufficiency()
        # counts mb:phrases:*, every graduated phrase also shrinks the
        # awareness buffer weight: the bridge recedes as B-Lawz's own corpus
        # grows.  Never raises.
        try:
            self._graduate_phrases(storage, content_type, sample["payload"])
        except Exception as exc:
            _fw_logger.debug("[flywheel] graduation error (%s): %s",
                             content_type, exc)

    # Which payload fields graduate into which phrase corpus, per type.
    # Scene types must match dataset_sampler's readers (hook/body/cta/verse…).
    _GRADUATION_MAP: dict[str, dict[str, str]] = {
        "scripts": {"hook": "hook", "body": "body", "cta": "cta"},
        "social":  {"caption": "hook"},
        "daw":     {"hook": "hook", "lyrics": "verse"},
    }

    def _graduate_phrases(self, storage: Any, content_type: str,
                          payload: dict[str, Any]) -> None:
        def _push(scene_type: str, text: Any) -> None:
            if not isinstance(text, str):
                return
            phrase = text.strip()
            if len(phrase) < 8 or len(phrase) > 280:
                return
            key = f"phrases:{scene_type}"
            existing = storage.lrange(key, 0, -1) or []
            if phrase not in existing:
                storage.lpush(key, phrase)
                storage.ltrim(key, 0, 255)

        if content_type == "video":
            # Scene packs: each scene's text graduates into its own corpus.
            for scene in (payload.get("scenes") or []):
                if isinstance(scene, dict):
                    _push(str(scene.get("type") or "body"),
                          scene.get("text") or scene.get("narration"))
            return

        field_map = self._GRADUATION_MAP.get(content_type)
        if not field_map:
            return  # images/distribution graduate through their own paths
        for field, scene_type in field_map.items():
            value = payload.get(field)
            if isinstance(value, list):
                for v in value[:5]:
                    _push(scene_type, v)
            else:
                _push(scene_type, value)
        # Social payloads often nest variants: [{platform, caption, ...}]
        if content_type == "social":
            for variant in (payload.get("variants") or [])[:8]:
                if isinstance(variant, dict):
                    _push("hook", variant.get("caption"))

    def _update_manifest(
        self,
        storage: Any,
        meta_key: str,
        content_type: str,
        ts: float,
    ) -> None:
        try:
            existing: Any = storage.get(meta_key)
            count: int = 0
            if isinstance(existing, dict):
                count = int(existing.get("count", 0))
            new_count = count + 1
            storage.set(meta_key, {
                "description": _DATASET_DESCRIPTIONS.get(content_type, content_type),
                "content_type": content_type,
                "count": new_count,
                "updated_at": ts,
                "namespace": f"mb:dataset:flywheel:{content_type}",
            })
        except Exception as exc:
            _fw_logger.debug("[flywheel] manifest update error (%s): %s", meta_key, exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

_STRIP_KEYS = {"processing_time_ms", "model_ready", "success"}

def _sanitise(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of the payload with noisy/irrelevant keys removed.
    Ensures the stored sample is clean training data, not a raw HTTP response.
    """
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k in _STRIP_KEYS:
            continue
        if isinstance(v, str) and len(v) > 20_000:
            v = v[:20_000]
        out[k] = v
    return out


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: FlywheelIngestor | None = None
_instance_lock = threading.Lock()


def get_flywheel() -> FlywheelIngestor | None:
    """Return (or lazily create) the module-level FlywheelIngestor singleton."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            try:
                _instance = FlywheelIngestor()
            except Exception as exc:
                _fw_logger.warning("[flywheel] failed to initialise: %s", exc)
    return _instance
