"""
admin_content_loop.py — Autonomous Admin Content Generation Loop

Runs as a daemon background thread after server startup. Periodically generates
content (scripts, social posts, DAW/beat descriptions) using the admin identity
and feeds it into the flywheel, so MaxBooster's own corpus grows toward the live
industry signal and the external awareness buffer retires naturally.

Retirement flow:
  quality_awareness.self_sufficiency()["buffer_weight"]
    1.0 → external Deezer/BPM awareness fully drives generation
    0.0 → admin corpus is self-sufficient; external seeding stops

The loop:
  1. Reads live industry targets (genre, BPM, mood) from quality_awareness
  2. Builds a full awareness string via platform_awareness_string()
  3. Generates awareness-conditioned content through the agent stack
  4. Injects every result into the FlywheelIngestor (→ pdim, → phrase graduation)
  5. Backs off when the corpus is self-sufficient (buffer_weight == 0)
  6. Never raises — all errors are caught and logged

Digital GPU note: all heavy inference in this loop flows through the existing
agent stack (ScriptAgent, DistributionAgent), which routes through the MaxCore
DigitalGPU backend. This file contains no direct numpy/torch calls.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

_log = logging.getLogger("admin_content_loop")

# ── Tunables (override via env) ──────────────────────────────────────────────
# How many seconds to sleep between generation cycles when the corpus is young.
_CYCLE_SECONDS         = int(os.environ.get("MB_LOOP_CYCLE_S",        "120"))
# Back-off multiplier when self-sufficient (corpus retired).
_BACKOFF_RETIRED_S     = int(os.environ.get("MB_LOOP_BACKOFF_RETIRED", "600"))
# Genres to cycle through per iteration (derived from live Deezer chart data).
_DEFAULT_GENRES = [
    "hip hop", "trap", "phonk", "r&b", "afrobeats", "drill",
    "pop", "electronic", "reggaeton", "latin",
]
# Platforms for social/distribution content.
_PLATFORMS = ["tiktok", "instagram", "youtube", "twitter", "spotify"]
# Minimum time (s) to wait after startup before first generation cycle.
_STARTUP_DELAY_S       = int(os.environ.get("MB_LOOP_STARTUP_DELAY",  "90"))

_started = False
_started_lock = threading.Lock()


def _build_awareness(genre: str, platform: str) -> str:
    """Build a rich awareness string from live industry signals. Never raises."""
    parts: list[str] = []
    try:
        from ai_model.quality_awareness import (
            platform_awareness_string,
            music_targets,
        )
        plat_aw = platform_awareness_string(platform)
        if plat_aw:
            parts.append(plat_aw)
        targets = music_targets(genre)
        if targets.get("bpm"):
            parts.append(
                f"[HIGH] Live chart BPM for {genre}: {targets['bpm']:.0f} "
                f"(range {targets.get('bpm_range', '')})"
            )
        if targets.get("energy"):
            parts.append(f"Energy level: {targets['energy']}")
    except Exception as exc:
        _log.debug("[loop] awareness build error: %s", exc)
    return "\n".join(parts)


def _generate_script(
    genre: str, platform: str, awareness: str,
    script_agent: Any,
) -> dict[str, Any] | None:
    """Generate a hook/body/CTA script via ScriptAgent. Never raises."""
    try:
        from ai_model.agents.script_agent import ScriptRequest
        sr = script_agent.run(ScriptRequest(
            idea=f"{genre} music release",
            platform=platform,
            goal="growth",
            tone="energetic",
            awareness=awareness,
        ))
        if sr and sr.hook:
            return {
                "hook":     sr.hook,
                "body":     sr.body,
                "cta":      sr.cta,
                "platform": platform,
                "genre":    genre,
                "source":   getattr(sr, "source", "model"),
            }
    except Exception as exc:
        _log.debug("[loop] script gen error: %s", exc)
    return None


def _generate_social(
    genre: str, platform: str, awareness: str,
    script_agent: Any,
) -> dict[str, Any] | None:
    """Generate a social caption variant. Reuses ScriptAgent. Never raises."""
    try:
        from ai_model.agents.script_agent import ScriptRequest
        sr = script_agent.run(ScriptRequest(
            idea=f"{genre} drop",
            platform=platform,
            goal="engagement",
            tone="authentic",
            awareness=awareness,
        ))
        if sr and sr.hook:
            return {
                "caption":  sr.hook,
                "platform": platform,
                "genre":    genre,
            }
    except Exception as exc:
        _log.debug("[loop] social gen error: %s", exc)
    return None


def _generate_daw(
    genre: str, platform: str, awareness: str,
    script_agent: Any,
) -> dict[str, Any] | None:
    """Generate a beat/DAW description (hook + lyric stub). Never raises."""
    try:
        from ai_model.agents.script_agent import ScriptRequest
        sr = script_agent.run(ScriptRequest(
            idea=f"{genre} beat",
            platform="general",
            goal="creative production",
            tone="raw",
            awareness=awareness,
        ))
        if sr and sr.hook:
            return {
                "hook":   sr.hook,
                "lyrics": sr.body,
                "genre":  genre,
            }
    except Exception as exc:
        _log.debug("[loop] daw gen error: %s", exc)
    return None


def _run_loop(get_script_agent_fn, get_distribution_agent_fn) -> None:
    """
    Main loop body. Runs forever in a daemon thread.
    Waits for the model to be ready, then generates content indefinitely.
    """
    _log.info("[loop] waiting %ds for model warmup...", _STARTUP_DELAY_S)
    time.sleep(_STARTUP_DELAY_S)

    # Wait until ScriptAgent is available.
    for _ in range(300):
        sa = get_script_agent_fn()
        if sa is not None:
            break
        time.sleep(2)
    else:
        _log.warning("[loop] ScriptAgent never became ready — loop exiting")
        return

    _log.info("[loop] starting autonomous admin content generation")

    # Cycle through genres and platforms in round-robin.
    genre_idx   = 0
    platform_idx = 0

    while True:
        try:
            from ai_model.quality_awareness import self_sufficiency, music_targets
            suff = self_sufficiency()
            bw   = suff.get("buffer_weight", 1.0)

            if suff.get("retired"):
                # Corpus is self-sufficient — slow down dramatically; no need
                # to drive the awareness bridge from external signals anymore.
                _log.info(
                    "[loop] corpus self-sufficient (own=%d/%d, bw=%.3f) — "
                    "backing off %ds",
                    suff.get("own_corpus", 0),
                    suff.get("retire_threshold", 500),
                    bw,
                    _BACKOFF_RETIRED_S,
                )
                time.sleep(_BACKOFF_RETIRED_S)
                continue

            # Pick the next genre from live chart targets (or the default list).
            try:
                live_targets = music_targets()
                live_genres  = live_targets.get("trending_genres") or _DEFAULT_GENRES
            except Exception:
                live_genres = _DEFAULT_GENRES

            genre    = live_genres[genre_idx % len(live_genres)]
            platform = _PLATFORMS[platform_idx % len(_PLATFORMS)]
            genre_idx    += 1
            platform_idx += 1

            awareness = _build_awareness(genre, platform)

            sa = get_script_agent_fn()
            if sa is None:
                time.sleep(30)
                continue

            from workers.admin_flywheel import get_flywheel
            fw = get_flywheel()
            if fw is None:
                time.sleep(30)
                continue

            # ── Generate and ingest each content type ───────────────────────
            admin_meta = {
                "genre":    genre,
                "platform": platform,
                "buffer_weight": bw,
                "own_corpus": suff.get("own_corpus", 0),
                "loop_cycle": "autonomous",
            }

            script = _generate_script(genre, platform, awareness, sa)
            if script:
                fw.ingest("scripts", script, admin_meta, key_id="admin")
                _log.info(
                    "[loop] ingested script: genre=%r platform=%r bw=%.3f",
                    genre, platform, bw,
                )

            social = _generate_social(genre, platform, awareness, sa)
            if social:
                fw.ingest("social", social, admin_meta, key_id="admin")

            daw = _generate_daw(genre, platform, awareness, sa)
            if daw:
                fw.ingest("daw", daw, admin_meta, key_id="admin")

        except Exception as exc:
            _log.warning("[loop] cycle error (will retry): %s", exc)

        time.sleep(_CYCLE_SECONDS)


def start(get_script_agent_fn, get_distribution_agent_fn=None) -> None:
    """
    Start the autonomous content generation loop in a daemon thread.
    Idempotent — safe to call multiple times (only the first call does anything).

    Args:
        get_script_agent_fn:       callable returning the ScriptAgent singleton (or None)
        get_distribution_agent_fn: callable returning the DistributionAgent singleton (or None)
    """
    global _started
    with _started_lock:
        if _started:
            return
        _started = True

    t = threading.Thread(
        target=_run_loop,
        args=(get_script_agent_fn, get_distribution_agent_fn or (lambda: None)),
        daemon=True,
        name="admin-content-loop",
    )
    t.start()
    _log.info("[loop] started (cycle=%ds, startup_delay=%ds)", _CYCLE_SECONDS, _STARTUP_DELAY_S)
