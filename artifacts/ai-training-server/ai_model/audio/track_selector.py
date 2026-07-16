"""Audio track selection logic for ``_render_audio_from_dataset``.

Extracted here so it can be unit-tested without importing the full server.py
FastAPI application.  The selector is a pure function of the stored index —
no I/O, no ffmpeg, no storage calls.
"""
from __future__ import annotations

from typing import Any


def _norm_key(k: Any) -> str:
    return str(k or "").strip().lower()


def _bpm_dist(entry: dict, want_bpm: float) -> float:
    b = float(entry.get("bpm") or 0.0)
    return abs(b - want_bpm) if b > 0 else 1e6


def _genre_score(entry: dict, preferred_genres: list[str]) -> float:
    """0.0 = at least one genre overlaps a preferred genre (best).
    1.0 = no genre overlap (neutral — still selected if nothing better).
    Partial substring matching handles "hip hop" ↔ "hip-hop", "r&b" ↔ "rnb",
    and live-chart shorthand vs full dataset genre names.
    """
    if not preferred_genres:
        return 0.0
    entry_genres = [
        str(g).lower().strip().replace("-", " ").replace("_", " ")
        for g in (entry.get("genres") or [])
        if g is not None
    ]
    for pg in preferred_genres:
        pg_n = pg.replace("-", " ").replace("_", " ")
        for eg in entry_genres:
            if pg_n in eg or eg in pg_n:
                return 0.0
    return 1.0


def select_audio_sample(
    index: list[dict],
    want_key: str,
    want_bpm: float,
    preferred_genres: list[str] | None = None,
) -> tuple[dict, bool]:
    """Select the best-matching audio sample from the index.

    Strategy (in priority order):
      1. Narrow to entries whose stored key exactly matches ``want_key``.
      2. If no key match exists, fall back to the full index (all tracks).
      3. Within the pool, rank by (genre_miss, bpm_distance, idx) so
         awareness-aligned tracks always rank ahead of equally-keyed but
         genre-mismatched alternatives.

    Returns:
        (best_entry, key_matched)  — ``key_matched`` is False when no exact
        key match was found and the selector had to fall back to the full
        index.  Callers should surface a ``selection_warning`` in the
        response when ``key_matched`` is False so producers know their key
        request could not be honoured by the current dataset.
    """
    if not index:
        raise ValueError("index is empty — cannot select a track")

    preferred = [g.lower().strip() for g in (preferred_genres or []) if g and str(g).strip()]
    nk = _norm_key(want_key)
    bpm = float(want_bpm or 0.0)

    key_matches = [e for e in index if _norm_key(e.get("key")) == nk]
    key_matched = bool(key_matches)
    pool = key_matches if key_matched else index

    best = min(
        pool,
        key=lambda e: (_genre_score(e, preferred), _bpm_dist(e, bpm), int(e.get("idx", 0))),
    )
    return best, key_matched
