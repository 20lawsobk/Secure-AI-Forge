"""Audio track selection logic for ``_render_audio_from_dataset``.

Extracted here so it can be unit-tested without importing the full server.py
FastAPI application.  The selector is a pure function of the stored index —
no I/O, no ffmpeg, no storage calls.
"""
from __future__ import annotations

from typing import Any

# ── FMA genre ID → human-readable name ────────────────────────────────────────
# The HuggingFace FMA-small dataset labels tracks with numeric genre IDs.  The
# awareness layer supplies genre NAMES (from live chart signals + user intent),
# so numeric IDs must be normalised to names or genre conditioning silently
# never matches.  Top-level + common sub-genre IDs from the FMA taxonomy.
FMA_GENRE_NAMES: dict[int, str] = {
    2: "international", 3: "blues", 4: "jazz", 5: "classical",
    8: "old-time historic", 9: "country", 10: "pop", 12: "rock",
    13: "easy listening", 14: "soul r&b", 15: "electronic", 17: "folk",
    20: "spoken", 21: "hip-hop", 38: "experimental", 1235: "instrumental",
    # Common FMA sub-genres seen in fma-small
    25: "punk", 26: "post-rock", 27: "lo-fi", 30: "avant-garde",
    31: "metal", 32: "psych-rock", 33: "psych-folk", 36: "indie-rock",
    37: "industrial", 41: "ambient", 42: "electroacoustic", 45: "loud-rock",
    46: "noise", 47: "garage", 53: "trip-hop", 58: "drum and bass",
    66: "chill-out", 70: "dance", 76: "experimental pop", 81: "techno",
    83: "house", 85: "glitch", 88: "minimalism", 89: "power-pop",
    90: "surf", 91: "shoegaze", 92: "downtempo", 94: "dubstep",
    98: "progressive", 100: "alternative", 101: "hip-hop beats",
    102: "ambient electronic", 103: "reggae", 107: "synth-pop",
    109: "no-wave", 111: "chiptune", 113: "breakbeat", 118: "idm",
    125: "singer-songwriter", 130: "americana", 137: "dream-pop",
    166: "electro-punk", 167: "grindcore", 169: "hardcore",
    170: "post-punk", 171: "krautrock", 181: "trance", 182: "turkish",
    184: "african", 185: "asia-far-east", 224: "freak-folk",
    236: "free-folk", 240: "new-age", 250: "improv", 267: "big band",
    286: "bluegrass", 296: "banjo", 297: "chamber music", 322: "choral",
    337: "composed music", 359: "gospel", 360: "grunge", 361: "hip-hop dj",
    362: "rap", 400: "sound art", 401: "nu-jazz", 428: "jazz vocal",
    439: "modern jazz", 440: "opera", 441: "20th century classical",
    442: "contemporary classical", 456: "shoegaze indie", 465: "minimal electronic",
    468: "wonky", 491: "ballad", 495: "rockabilly", 502: "salsa",
    504: "latin", 514: "romany gypsy", 524: "french", 538: "tango",
    539: "cumbia", 542: "latin america", 567: "flamenco", 580: "klezmer",
    602: "north african", 619: "middle east", 651: "reggae dub",
    659: "european music", 693: "indian", 695: "pacific", 741: "celtic",
    763: "holiday", 808: "sludge", 810: "black metal", 811: "death metal",
    906: "goth", 1032: "musique concrete",
}

# ── Awareness mood → genre affinity ───────────────────────────────────────────
# When the awareness brief supplies a mood, prefer dataset tracks whose genre
# family suits it.  This is a soft preference (tertiary sort key) — genre
# intent and key/BPM still dominate; mood only breaks ties in the right
# direction.  Names must be lowercase, hyphen/underscore-free (match the same
# normalisation applied to entry genres).
MOOD_GENRE_AFFINITY: dict[str, tuple[str, ...]] = {
    "energetic":  ("electronic", "rock", "hip hop", "dance", "house", "techno",
                   "drum and bass", "punk", "breakbeat", "trance"),
    "upbeat":     ("pop", "dance", "electronic", "house", "indie", "power pop",
                   "synth pop"),
    "calm":       ("ambient", "classical", "folk", "chill out", "downtempo",
                   "new age", "minimalism", "instrumental"),
    "chill":      ("chill out", "downtempo", "lo fi", "trip hop", "ambient",
                   "jazz", "idm"),
    "dark":       ("industrial", "metal", "noise", "goth", "dubstep",
                   "black metal", "sludge"),
    "melancholy": ("shoegaze", "dream pop", "post rock", "singer songwriter",
                   "folk", "ballad", "blues"),
    "cinematic":  ("classical", "post rock", "ambient", "orchestral",
                   "composed music", "contemporary classical", "instrumental"),
    "romantic":   ("soul r&b", "jazz vocal", "ballad", "latin", "tango",
                   "easy listening"),
    "aggressive": ("metal", "hardcore", "punk", "grindcore", "industrial",
                   "loud rock", "rap"),
    "groovy":     ("funk", "soul r&b", "disco", "house", "nu jazz", "reggae",
                   "salsa"),
    "playful":    ("chiptune", "pop", "surf", "ragtime", "wonky", "glitch"),
    "focused":    ("minimalism", "idm", "ambient electronic", "instrumental",
                   "minimal electronic", "classical"),
}


def normalize_genres(genres: Any) -> list[str]:
    """Normalise a dataset entry's genre list to lowercase names.

    Handles numeric FMA genre IDs (mapped via ``FMA_GENRE_NAMES``), numeric
    strings ("91"), and plain names ("Hip-Hop") uniformly.  Unknown numeric
    IDs are kept as their string form (harmless — they simply never match).
    Hyphens/underscores become spaces so "hip-hop" ↔ "hip hop" match.
    """
    if genres is None:
        genres = []
    elif isinstance(genres, (str, bytes, int, float)):
        # Malformed legacy data may store a scalar instead of a list —
        # treat it as a single-genre list rather than iterating chars.
        genres = [genres]
    out: list[str] = []
    for g in genres:
        if g is None:
            continue
        if isinstance(g, bool):        # bool is an int subclass — skip
            continue
        if isinstance(g, (int, float)):
            name = FMA_GENRE_NAMES.get(int(g), str(int(g)))
        else:
            s = str(g).strip()
            if s.isdigit():
                name = FMA_GENRE_NAMES.get(int(s), s)
            else:
                name = s
        out.append(name.lower().strip().replace("-", " ").replace("_", " "))
    return out


def _norm_key(k: Any) -> str:
    return str(k or "").strip().lower()


def _bpm_dist(entry: dict, want_bpm: float) -> float:
    b = float(entry.get("bpm") or 0.0)
    return abs(b - want_bpm) if b > 0 else 1e6


def _genre_score(entry: dict, preferred_genres: list[str]) -> float:
    """0.0 = at least one genre overlaps a preferred genre (best).
    1.0 = no genre overlap (neutral — still selected if nothing better).
    Partial substring matching handles "hip hop" ↔ "hip-hop", "r&b" ↔ "rnb",
    and live-chart shorthand vs full dataset genre names.  Entry genres are
    normalised via ``normalize_genres`` so numeric FMA IDs match by name.
    """
    if not preferred_genres:
        return 0.0
    entry_genres = normalize_genres(entry.get("genres"))
    for pg in preferred_genres:
        pg_n = pg.replace("-", " ").replace("_", " ")
        for eg in entry_genres:
            if pg_n in eg or eg in pg_n:
                return 0.0
    return 1.0


def _mood_score(entry: dict, preferred_mood: str) -> float:
    """0.0 = entry's genres suit the awareness mood, 1.0 = no affinity.

    Soft tertiary preference: only consulted after genre intent, so it breaks
    ties in the mood's direction without overriding explicit genre requests.
    Unknown moods score 0.0 for every entry (a no-op, never a penalty).
    """
    if not preferred_mood:
        return 0.0
    affinity = MOOD_GENRE_AFFINITY.get(
        preferred_mood.lower().strip().replace("-", " ").replace("_", " ")
    )
    if not affinity:
        return 0.0
    entry_genres = normalize_genres(entry.get("genres"))
    for fam in affinity:
        for eg in entry_genres:
            if fam in eg or eg in fam:
                return 0.0
    return 1.0


def select_audio_sample(
    index: list[dict],
    want_key: str,
    want_bpm: float,
    preferred_genres: list[str] | None = None,
    preferred_mood: str | None = None,
) -> tuple[dict, bool]:
    """Select the best-matching audio sample from the index.

    Strategy (in priority order):
      1. Narrow to entries whose stored key exactly matches ``want_key``.
      2. If no key match exists, fall back to the full index (all tracks).
      3. Within the pool, rank by (genre_miss, mood_miss, bpm_distance, idx)
         so awareness-aligned tracks always rank ahead of equally-keyed but
         genre-mismatched alternatives, and the awareness mood breaks
         remaining ties in the right direction.

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
    mood = (preferred_mood or "").strip()
    nk = _norm_key(want_key)
    bpm = float(want_bpm or 0.0)

    key_matches = [e for e in index if _norm_key(e.get("key")) == nk]
    key_matched = bool(key_matches)
    pool = key_matches if key_matched else index

    best = min(
        pool,
        key=lambda e: (
            _genre_score(e, preferred),
            _mood_score(e, mood),
            _bpm_dist(e, bpm),
            int(e.get("idx", 0)),
        ),
    )
    return best, key_matched
