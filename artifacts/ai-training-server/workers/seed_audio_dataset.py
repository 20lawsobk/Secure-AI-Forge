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
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("seed_audio")

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

def seed(storage: Any, count: int = 12, *, replace: bool = False) -> dict[str, Any]:
    """Download ``count`` real tracks and store them under ``mb:dataset:audio``.

    Primary source: HuggingFace datasets-server (FMA-small).
    Fallback source: librosa's bundled CC-licensed example tracks (always
    available locally) — used automatically when the datasets-server is
    unreachable (503 / network error).

    Requires a live storage backend; raises if storage is unavailable so the
    caller fails explicitly instead of writing to the non-persistent in-process
    fallback.
    """
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
    use_hf = _hf_rows_available()
    if use_hf:
        logger.info("[seed_audio] using HuggingFace datasets-server as source")
        active_source = _SOURCE
    else:
        logger.info("[seed_audio] HF datasets-server unavailable — using librosa bundled examples")
        active_source = _SOURCE_LIBROSA

    stored = 0

    if use_hf:
        # ----------------------------------------------------------------
        # Primary path: stream rows from the HF datasets-server
        # ----------------------------------------------------------------
        offset = start_idx
        attempts = 0
        max_attempts = max(count * 4, 8)
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
                try:
                    mp3 = _transcode(_http_get(src, timeout=60))
                except Exception as exc:
                    logger.warning("[seed_audio] skip HF row: %s", exc)
                    continue
                bpm, key = _estimate_bpm_key(mp3)
                idx = start_idx + stored
                genres = row.get("genres")
                sample = {
                    "idx": idx,
                    "title": str(row.get("title") or f"track_{idx}"),
                    "artist": str(row.get("artist") or ""),
                    "genres": genres if isinstance(genres, list) else [],
                    "license": str(row.get("license") or ""),
                    "instrumental": bool(row.get("instrumental")),
                    "source": active_source,
                    "bpm": bpm,
                    "key": key,
                    "duration_sec": _CLIP_SECONDS,
                    "sample_rate": _SAMPLE_RATE,
                    "format": "mp3",
                    "b64": base64.b64encode(mp3).decode("ascii"),
                }
                storage.set(f"mb:dataset:{DATASET_NAME}:chunk:{idx}", sample)
                index.append(
                    {"idx": idx, "bpm": bpm, "key": key, "genres": sample["genres"]}
                )
                stored += 1
                logger.info(
                    "[seed_audio] stored sample %d (%s, bpm=%s key=%s)",
                    idx, sample["title"], bpm, key,
                )

    if not use_hf:
        # ----------------------------------------------------------------
        # Fallback path: librosa CC-licensed bundled examples
        # ----------------------------------------------------------------
        # Cycle through examples (repeat if count > len) starting from
        # start_idx so that incremental seeding picks up distinct examples.
        examples = _LIBROSA_EXAMPLES
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
            }
            storage.set(f"mb:dataset:{DATASET_NAME}:chunk:{idx}", sample)
            index.append(
                {"idx": idx, "bpm": bpm, "key": key, "genres": sample["genres"]}
            )
            stored += 1
            logger.info(
                "[seed_audio] stored librosa sample %d (%s, bpm=%s key=%s)",
                idx, sample["title"], bpm, key,
            )

    num_chunks = start_idx + stored
    meta = {
        "name": DATASET_NAME,
        "description": (
            "Real music samples from the Free Music Archive (FMA-small)" if active_source == _SOURCE
            else "Real CC-licensed music samples (librosa bundled examples)"
        ),
        "content_type": "audio",
        "num_chunks": num_chunks,
        "source": active_source,
        "index": index,
    }
    storage.set(f"mb:dataset:{DATASET_NAME}:meta", meta)
    summary = {"stored": stored, "total": num_chunks, "source": active_source}
    logger.info("[seed_audio] done: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    from storage_client import get_storage

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(json.dumps(seed(get_storage(), count=n), indent=2))
