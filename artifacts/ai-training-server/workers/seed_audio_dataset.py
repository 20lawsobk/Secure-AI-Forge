"""Seed real music audio into pdim storage for ``/api/generate/audio``.

Pulls real, Creative-Commons-licensed tracks from the Free Music Archive
(FMA-small) public dataset via the HuggingFace datasets-server, transcodes each
to a normalized mono MP3 of bounded length, estimates BPM/key in-house with
librosa, and stores them under the ``audio`` dataset namespace in pdim:

    mb:dataset:audio:meta         -> dataset manifest incl. a lightweight index
    mb:dataset:audio:chunk:{idx}  -> one sample {base64 mp3 + metadata}

This lets audio generation use REAL audio instead of procedural synthesis, with
no silent fallback: if the dataset is empty, generation raises explicitly.

When the HuggingFace datasets-server is unavailable (503 / network error), the
seeder automatically falls back to librosa's bundled CC-licensed example tracks
so that seeding always succeeds when the storage backend is live.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("seed_audio")

# ---------------------------------------------------------------------------
# Process-wide single-flight guard
# ---------------------------------------------------------------------------
# The seeder does a read-then-write of mb:dataset:audio:meta (start_idx +
# index snapshot), so concurrent runs from ANY entry point (HTTP endpoint,
# DataPuller auto-growth, CLI) would race and corrupt the manifest.
# This module-level lock is the canonical guard — all callers share it.

_SEED_LOCK = threading.Lock()


class AlreadySeedingError(RuntimeError):
    """Raised by seed() when a seeding run is already in progress process-wide."""


def is_seeding() -> bool:
    """Return True when a seed run is currently in progress (non-blocking probe)."""
    acquired = _SEED_LOCK.acquire(blocking=False)
    if acquired:
        _SEED_LOCK.release()
    return not acquired

_HF_DATASET = "benjamin-paine/free-music-archive-small"
_HF_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={_HF_DATASET}&config=default&split=train"
    "&offset={offset}&length={length}"
)
DATASET_NAME = "audio"
_SOURCE = f"huggingface:{_HF_DATASET}"
_SOURCE_LIBROSA = "librosa:bundled-examples"
_UA = {"User-Agent": "Mozilla/5.0"}

# Bounded clip + encoding keep each stored sample small enough for the storage
# value size while remaining a usable, listenable track.
_CLIP_SECONDS = 24.0
_SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Librosa bundled example tracks — all CC-licensed real music.
# Used as a reliable fallback when the HF datasets-server is unavailable.
# ---------------------------------------------------------------------------
_LIBROSA_EXAMPLES: list[dict[str, Any]] = [
    {
        "example_key": "fishin",
        "title": "Let's Go Fishin'",
        "artist": "Karissa Hobbs",
        "genres": ["Folk", "Country"],
        "license": "CC BY-NC-SA 4.0",
        "instrumental": True,
    },
    {
        "example_key": "brahms",
        "title": "Hungarian Dance No. 5 (String Orchestra)",
        "artist": "Brahms / Public Domain",
        "genres": ["Classical", "Orchestral"],
        "license": "CC BY-SA 3.0",
        "instrumental": True,
    },
    {
        "example_key": "nutcracker",
        "title": "Dance of the Sugar Plum Fairy",
        "artist": "Kevin MacLeod",
        "genres": ["Classical"],
        "license": "CC BY 3.0",
        "instrumental": True,
    },
    {
        "example_key": "pistachio",
        "title": "Happy Music – Pistachio Ice Cream (Ragtime)",
        "artist": "Lena Orsa",
        "genres": ["Ragtime", "Jazz"],
        "license": "CC BY 3.0",
        "instrumental": True,
    },
    {
        "example_key": "vibeace",
        "title": "Vibe Ace",
        "artist": "Kevin MacLeod",
        "genres": ["Electronic", "Ambient"],
        "license": "CC BY 3.0",
        "instrumental": True,
    },
    {
        "example_key": "choice",
        "title": "Choice (Drum & Bass)",
        "artist": "admiralbob77",
        "genres": ["Electronic", "Drum and Bass"],
        "license": "CC0 1.0",
        "instrumental": True,
    },
    {
        "example_key": "trumpet",
        "title": "Solo Trumpet No. 6",
        "artist": "Sorohanro",
        "genres": ["Classical", "Solo"],
        "license": "CC BY-SA 3.0",
        "instrumental": True,
    },
]


def _http_get(url: str, timeout: float = 90.0, retries: int = 3) -> bytes:
    """GET bytes with retries — the public dataset host can be slow/flaky from
    inside the busy server process, so a single attempt is not reliable."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # retry on timeout / transient network error
            last = exc
            logger.warning(
                "[seed_audio] GET attempt %d/%d failed: %s",
                attempt + 1, retries, exc,
            )
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {last}")


def _hf_rows_available() -> bool:
    """Probe the HF datasets-server with a minimal request.  Returns False on
    any non-200 response or network error so the caller can skip to fallback."""
    try:
        url = _HF_ROWS.format(offset=0, length=1)
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as exc:
        logger.info("[seed_audio] HF datasets-server probe failed (%s) — will use librosa fallback", exc)
        return False


def _fetch_hf_rows(offset: int, length: int) -> list[dict[str, Any]]:
    raw = _http_get(_HF_ROWS.format(offset=offset, length=length), timeout=60)
    data = json.loads(raw.decode("utf-8"))
    return [r.get("row", {}) for r in data.get("rows", [])]


def _audio_src(row: dict[str, Any]) -> Optional[str]:
    aud = row.get("audio")
    if isinstance(aud, list) and aud and isinstance(aud[0], dict):
        return aud[0].get("src")
    if isinstance(aud, dict):
        return aud.get("src")
    return None


def _librosa_row_bytes(example_key: str) -> bytes:
    """Return raw OGG/FLAC bytes from a librosa bundled example track."""
    import librosa  # already a hard dep of the server
    path = librosa.ex(example_key)
    return Path(path).read_bytes()


def _transcode(src_bytes: bytes) -> bytes:
    """Transcode arbitrary input audio to a normalized, bounded mono MP3."""
    from ai_model.video.ffmpeg_util import run_ffmpeg

    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.bin"
        op = Path(td) / "out.mp3"
        ip.write_bytes(src_bytes)
        res = run_ffmpeg(
            [
                "ffmpeg", "-y", "-i", str(ip),
                "-t", f"{_CLIP_SECONDS:.2f}",
                "-ac", "1", "-ar", str(_SAMPLE_RATE),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-codec:a", "libmp3lame", "-q:a", "5",
                str(op),
            ],
            timeout=90,
        )
        if res.returncode != 0 or not op.exists():
            raise RuntimeError(
                f"ffmpeg transcode failed (rc={res.returncode}): {res.stderr[-200:]}"
            )
        return op.read_bytes()


def _estimate_bpm_key(mp3_bytes: bytes) -> tuple[float, str]:
    """Estimate ``(bpm, 'Root mode')`` in-house. Best-effort; never raises."""
    try:
        import numpy as np
        import librosa
        from ai_model.audio.audio_analysis import analyze_audio

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tf:
            tf.write(mp3_bytes)
            tf.flush()
            y, sr = librosa.load(tf.name, sr=_SAMPLE_RATE, mono=True)
        tl = analyze_audio(np.asarray(y, dtype=np.float64), int(sr))
        return float(round(tl.bpm, 1)), f"{tl.key} {tl.mode}".strip()
    except Exception as exc:  # estimation is best-effort metadata only
        logger.warning("[seed_audio] bpm/key estimation failed: %s", exc)
        return 0.0, ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def seed(
    storage: Any,
    count: int = 12,
    *,
    replace: bool = False,
    force_source: Optional[str] = None,
    genre_targets: Optional[list] = None,
) -> dict[str, Any]:
    """Download ``count`` real tracks and store them under ``mb:dataset:audio``.

    Primary source: HuggingFace datasets-server (FMA-small).
    Fallback source: librosa's bundled CC-licensed example tracks (always
    available locally) — used automatically when the datasets-server is
    unreachable (503 / network error).

    ``force_source`` overrides the automatic probe:
      - ``"hf"``      — always attempt HF first (still falls back to librosa on error)
      - ``"librosa"`` — skip HF entirely, use librosa bundled examples
      - ``None``      — auto-detect (default: probe HF, fall back to librosa)

    ``genre_targets`` is an optional list of ``{genre, bpm, bpm_range, energy}``
    dicts sourced from the live awareness beacon (quality_awareness.audio_seeding_targets()).
    When provided, FMA rows and librosa examples are filtered/prioritised to
    match the trending genres and BPM ranges so the seeded dataset reflects
    what is charting right now rather than a random FMA slice. Samples are
    tagged ``seeded_by_awareness=True`` when selected via this filter.
    Pass ``None`` (the default) for the original unfiltered behaviour.

    Requires a live storage backend; raises if storage is unavailable so the
    caller fails explicitly instead of writing to the non-persistent in-process
    fallback.

    Raises ``AlreadySeedingError`` when another seed is already running so that
    DataPuller and other background callers can skip gracefully rather than wait.
    """
    # Single-flight: one seed at a time across all callers in this process.
    if not _SEED_LOCK.acquire(blocking=False):
        raise AlreadySeedingError(
            "audio dataset seeding already in progress — skipping this call"
        )
    try:
        return _seed_locked(storage, count=count, replace=replace,
                            force_source=force_source, genre_targets=genre_targets)
    finally:
        _SEED_LOCK.release()


def _genre_match_score(sample_genres: list, targets: list) -> float:
    """Return a match score [0, 1] between a sample's genres and awareness targets.

    A score of 1.0 means the sample's genre is listed in the targets.
    Partial credit for sub-string matches (e.g. "hip-hop" ∈ "hip hop").
    0.0 means no overlap.  The highest score across all target genres is returned.
    """
    if not targets or not sample_genres:
        return 0.5  # no targets = no preference, treat all as neutral
    norm = [g.lower().strip().replace("-", " ").replace("_", " ")
            for g in sample_genres]
    best = 0.0
    for t in targets:
        tg = t.get("genre", "").lower().strip().replace("-", " ").replace("_", " ")
        if not tg:
            continue
        for sg in norm:
            if tg == sg or tg in sg or sg in tg:
                best = 1.0
                break
            # word overlap
            t_words = set(tg.split())
            s_words = set(sg.split())
            if t_words & s_words:
                score = len(t_words & s_words) / max(len(t_words), len(s_words))
                best = max(best, score * 0.8)
        if best == 1.0:
            break
    return best


def _bpm_match_score(bpm: float, targets: list) -> float:
    """Return a BPM proximity score [0, 1] against any awareness target."""
    if not targets or bpm <= 0:
        return 0.5
    best = 0.0
    for t in targets:
        target_bpm = float(t.get("bpm") or 0)
        bpm_range = t.get("bpm_range") or []
        if len(bpm_range) >= 2:
            lo, hi = float(bpm_range[0]), float(bpm_range[1])
            if lo <= bpm <= hi:
                best = 1.0
                break
            margin = max(hi - lo, 10.0)
            dist = min(abs(bpm - lo), abs(bpm - hi))
            best = max(best, max(0.0, 1.0 - dist / margin))
        elif target_bpm > 0:
            dist = abs(bpm - target_bpm)
            best = max(best, max(0.0, 1.0 - dist / 40.0))
    return best


def _is_awareness_match(sample_genres: list, bpm: float,
                        targets: list, threshold: float = 0.4) -> bool:
    """True when a sample is a good enough match for the awareness targets."""
    g = _genre_match_score(sample_genres, targets)
    b = _bpm_match_score(bpm, targets)
    return (g * 0.7 + b * 0.3) >= threshold


def _seed_locked(
    storage: Any,
    count: int = 12,
    *,
    replace: bool = False,
    force_source: Optional[str] = None,
    genre_targets: Optional[list] = None,
) -> dict[str, Any]:
    """Internal implementation — must only be called while _SEED_LOCK is held."""
    pdim_up = getattr(storage, "is_available", False)
    disk_up = getattr(storage, "disk_store_available", False)
    if not pdim_up and not disk_up:
        raise RuntimeError(
            "storage backend unavailable — refusing to seed audio into the "
            "non-persistent in-memory fallback (pdim offline, disk store also unavailable)"
        )

    start_idx = 0
    index: list[dict[str, Any]] = []
    existing = storage.get(f"mb:dataset:{DATASET_NAME}:meta")
    if existing and not replace:
        start_idx = int(existing.get("num_chunks", 0))
        index = list(existing.get("index", []))

    # Decide which source to use for this seeding run.
    fs = (force_source or "").lower().strip()
    if fs == "librosa":
        use_hf = False
        logger.info("[seed_audio] force_source=librosa — skipping HF probe")
    elif fs == "hf":
        use_hf = True
        logger.info("[seed_audio] force_source=hf — attempting HuggingFace directly")
    else:
        use_hf = _hf_rows_available()

    if use_hf:
        logger.info("[seed_audio] using HuggingFace datasets-server as source")
        active_source = _SOURCE
    else:
        logger.info("[seed_audio] HF datasets-server unavailable — using librosa bundled examples")
        active_source = _SOURCE_LIBROSA

    stored = 0
    # When awareness targets are provided, increase the attempt budget so we
    # can afford to skip non-matching rows before finding genre/BPM matches.
    _aw_guided = bool(genre_targets)
    if _aw_guided:
        logger.info(
            "[seed_audio] awareness-guided seeding: %d genre targets (%s)",
            len(genre_targets),
            ", ".join(t.get("genre", "?") for t in (genre_targets or [])[:5]),
        )

    if use_hf:
        # ----------------------------------------------------------------
        # Primary path: stream rows from the HF datasets-server.
        # Awareness-guided: when genre_targets is provided, rows that don't
        # match any target genre/BPM are skipped (with a higher attempt cap).
        # ----------------------------------------------------------------
        offset = start_idx
        attempts = 0
        max_attempts = max(count * 4, 8) if not _aw_guided else max(count * 10, 20)
        while stored < count and attempts < max_attempts:
            try:
                batch = _fetch_hf_rows(offset, min(5, count - stored))
            except Exception as exc:
                logger.warning("[seed_audio] HF fetch error mid-run: %s — switching to librosa", exc)
                use_hf = False
                active_source = _SOURCE_LIBROSA
                break
            if not batch:
                break
            offset += len(batch)
            for row in batch:
                attempts += 1
                if stored >= count:
                    break
                src = _audio_src(row)
                if not src:
                    continue
                # ── Cheap genre pre-filter (before download) ─────────────
                # Normalise genre labels now so the awareness matcher can
                # compare them.  HF/FMA rows carry numeric IDs; normalisation
                # converts them to readable names the matcher understands.
                from ai_model.audio.track_selector import normalize_genres
                raw_genres = row.get("genres")
                norm_genres = normalize_genres(raw_genres if isinstance(raw_genres, list) else [])
                if _aw_guided and norm_genres:
                    if _genre_match_score(norm_genres, genre_targets) < 0.3:
                        logger.debug(
                            "[seed_audio] skip row (genre mismatch: %s)", norm_genres
                        )
                        continue
                try:
                    mp3 = _transcode(_http_get(src, timeout=60))
                except Exception as exc:
                    logger.warning("[seed_audio] skip HF row: %s", exc)
                    continue
                bpm, key = _estimate_bpm_key(mp3)
                # ── BPM post-filter (BPM is only known after transcode) ───
                if _aw_guided and bpm > 0:
                    if not _is_awareness_match(norm_genres, bpm, genre_targets):
                        logger.debug(
                            "[seed_audio] skip row (awareness mismatch: "
                            "bpm=%.1f genres=%s)", bpm, norm_genres
                        )
                        continue
                idx = start_idx + stored
                aw_extra: dict = {"seeded_by_awareness": True} if _aw_guided else {}
                sample = {
                    "idx": idx,
                    "title": str(row.get("title") or f"track_{idx}"),
                    "artist": str(row.get("artist") or ""),
                    "genres": norm_genres,
                    "license": str(row.get("license") or ""),
                    "instrumental": bool(row.get("instrumental")),
                    "source": active_source,
                    "bpm": bpm,
                    "key": key,
                    "duration_sec": _CLIP_SECONDS,
                    "sample_rate": _SAMPLE_RATE,
                    "format": "mp3",
                    "b64": base64.b64encode(mp3).decode("ascii"),
                    **aw_extra,
                }
                storage.set(f"mb:dataset:{DATASET_NAME}:chunk:{idx}", sample)
                index.append({
                    "idx": idx, "bpm": bpm, "key": key,
                    "genres": sample["genres"], **aw_extra,
                })
                stored += 1
                logger.info(
                    "[seed_audio] stored sample %d (%s, bpm=%s key=%s%s)",
                    idx, sample["title"], bpm, key,
                    " [awareness-guided]" if _aw_guided else "",
                )

    if not use_hf:
        # ----------------------------------------------------------------
        # Fallback path: librosa CC-licensed bundled examples.
        # Awareness-guided: sort examples by genre match score so the
        # best-matching tracks are seeded first.
        # ----------------------------------------------------------------
        examples = list(_LIBROSA_EXAMPLES)
        if _aw_guided and genre_targets:
            examples.sort(
                key=lambda e: _genre_match_score(
                    list(e.get("genres") or []), genre_targets
                ),
                reverse=True,
            )
        ex_offset = start_idx % len(examples)
        attempts = 0
        max_attempts = count * 2
        while stored < count and attempts < max_attempts:
            ex = examples[(ex_offset + stored + attempts) % len(examples)]
            attempts += 1
            try:
                raw = _librosa_row_bytes(ex["example_key"])
                mp3 = _transcode(raw)
            except Exception as exc:
                logger.warning("[seed_audio] skip librosa example %s: %s", ex["example_key"], exc)
                continue
            bpm, key = _estimate_bpm_key(mp3)
            idx = start_idx + stored
            aw_extra = {"seeded_by_awareness": True} if _aw_guided else {}
            sample = {
                "idx": idx,
                "title": str(ex.get("title") or f"track_{idx}"),
                "artist": str(ex.get("artist") or ""),
                "genres": list(ex.get("genres") or []),
                "license": str(ex.get("license") or "CC"),
                "instrumental": bool(ex.get("instrumental", True)),
                "source": active_source,
                "bpm": bpm,
                "key": key,
                "duration_sec": _CLIP_SECONDS,
                "sample_rate": _SAMPLE_RATE,
                "format": "mp3",
                "b64": base64.b64encode(mp3).decode("ascii"),
                **aw_extra,
            }
            storage.set(f"mb:dataset:{DATASET_NAME}:chunk:{idx}", sample)
            index.append({
                "idx": idx, "bpm": bpm, "key": key,
                "genres": sample["genres"], **aw_extra,
            })
            stored += 1
            logger.info(
                "[seed_audio] stored librosa sample %d (%s, bpm=%s key=%s%s)",
                idx, sample["title"], bpm, key,
                " [awareness-guided]" if _aw_guided else "",
            )

    num_chunks = start_idx + stored
    meta = {
        "name": DATASET_NAME,
        "seeded_at": time.time(),
        "description": (
            "Real music samples from the Free Music Archive (FMA-small)"
            + (" — awareness-guided genre/BPM filter" if _aw_guided else "")
            if active_source == _SOURCE
            else "Real CC-licensed music samples (librosa bundled examples)"
            + (" — awareness-guided genre sort" if _aw_guided else "")
        ),
        "content_type": "audio",
        "num_chunks": num_chunks,
        "source": active_source,
        "awareness_guided": _aw_guided,
        "index": index,
    }
    storage.set(f"mb:dataset:{DATASET_NAME}:meta", meta)
    summary = {
        "stored": stored, "total": num_chunks,
        "source": active_source, "awareness_guided": _aw_guided,
    }
    logger.info("[seed_audio] done: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    from storage_client import get_storage

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(json.dumps(seed(get_storage(), count=n), indent=2))
