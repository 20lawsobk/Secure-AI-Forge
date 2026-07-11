"""Seed real music audio into pdim storage for ``/api/generate/audio``.

Pulls real, Creative-Commons-licensed tracks from the Free Music Archive
(FMA-small) public dataset via the HuggingFace datasets-server, transcodes each
to a normalized mono MP3 of bounded length, estimates BPM/key in-house with
librosa, and stores them under the ``audio`` dataset namespace in pdim:

    mb:dataset:audio:meta         -> dataset manifest incl. a lightweight index
    mb:dataset:audio:chunk:{idx}  -> one sample {base64 mp3 + metadata}

This lets audio generation use REAL audio instead of procedural synthesis, with
no silent fallback: if the dataset is empty, generation raises explicitly.
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
_UA = {"User-Agent": "Mozilla/5.0"}

# Bounded clip + encoding keep each stored sample small enough for the storage
# value size while remaining a usable, listenable track.
_CLIP_SECONDS = 24.0
_SAMPLE_RATE = 44100


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


def _fetch_rows(offset: int, length: int) -> list[dict[str, Any]]:
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


def seed(storage: Any, count: int = 12, *, replace: bool = False) -> dict[str, Any]:
    """Download ``count`` real tracks and store them under ``mb:dataset:audio``.

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

    stored = 0
    offset = start_idx
    attempts = 0
    max_attempts = max(count * 4, 8)
    while stored < count and attempts < max_attempts:
        batch = _fetch_rows(offset, min(5, count - stored))
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
                logger.warning("[seed_audio] skip row: %s", exc)
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
                "source": _SOURCE,
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

    num_chunks = start_idx + stored
    meta = {
        "name": DATASET_NAME,
        "description": "Real music samples from the Free Music Archive (FMA-small)",
        "content_type": "audio",
        "num_chunks": num_chunks,
        "source": _SOURCE,
        "index": index,
    }
    storage.set(f"mb:dataset:{DATASET_NAME}:meta", meta)
    summary = {"stored": stored, "total": num_chunks, "source": _SOURCE}
    logger.info("[seed_audio] done: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    from storage_client import get_storage

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(json.dumps(seed(get_storage(), count=n), indent=2))
