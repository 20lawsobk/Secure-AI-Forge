"""
In-house voice-over synthesis for video renders.
=================================================

Turns script text (hook / body / cta) into an intelligible spoken narration
track using the local eSpeak NG engine (no external APIs), then optionally
mixes it over a music soundtrack with the music ducked under the voice.

Why this module exists: the ``voiceover`` request flag used to be a silent
no-op — videos "with voice-over" shipped the procedural arpeggio soundtrack,
which users described as "birdcalls instead of words". This is the real
speech path.

Sample-rate discipline (the classic birdcall/chipmunk failure class):
every ffmpeg stage in this module passes an explicit ``-ar`` so the
narration can never be muxed at a different rate than it was synthesized.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ai_model.video.ffmpeg_util import run_ffmpeg

_logger = logging.getLogger("voiceover")

# Absolute path → posix_spawn-eligible (same fork-avoidance rationale as
# ffmpeg_util.run_ffmpeg: forking under the in-memory model can EIO/ENOMEM).
ESPEAK_BIN = shutil.which("espeak-ng") or ""

#: The one sample rate every voiceover artifact uses, matching the music
#: renderer's 44100 Hz so amix never resamples one input against the other.
VO_SAMPLE_RATE = 44100

_WPM_DEFAULT = 165  # natural narration pace

# ── Text cleaning ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"<[A-Za-z_|][A-Za-z0-9_|/]*>")
_HASHTAG_RE = re.compile(r"#\w+")
# Keep letters/digits/basic punctuation the TTS pronounces well; drop emoji
# and decorative symbols which espeak reads out loud ("fire emoji" etc. is
# engine-dependent; safest is removal).
_KEEP_RE = re.compile(r"[^A-Za-z0-9À-ÿ' \n.,!?;:()\-]")


def clean_vo_text(text: str) -> str:
    """Strip everything a narrator should not read aloud (never-raise)."""
    t = text or ""
    t = _TOKEN_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(" ", t)
    t = t.replace("—", ", ").replace("–", ", ")
    t = _KEEP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Synthesis ────────────────────────────────────────────────────────────────

def _run_espeak(args: list[str], timeout: float = 60.0) -> int:
    """Spawn espeak-ng with the same posix_spawn-friendly discipline as
    run_ffmpeg (absolute binary, no pipes, close_fds=False)."""
    with tempfile.TemporaryFile() as devnull_like:
        proc = subprocess.run(
            [ESPEAK_BIN] + args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=devnull_like,
            close_fds=False,
            timeout=timeout,
        )
    return proc.returncode


def synthesize_voiceover(
    text: str,
    out_path: str,
    sample_rate: int = VO_SAMPLE_RATE,
    wpm: int = _WPM_DEFAULT,
    voice: str = "en-us",
) -> bool:
    """Render ``text`` as spoken narration into ``out_path`` (mono WAV at
    ``sample_rate``). Returns True on success, False on any failure (raises
    nothing) — callers fall back to a music-only or silent track.
    """
    spoken = clean_vo_text(text)
    if not spoken or not ESPEAK_BIN:
        if not ESPEAK_BIN:
            _logger.warning("[voiceover] espeak-ng binary not found — narration skipped")
        return False

    raw_wav = None
    try:
        fd, raw_wav = tempfile.mkstemp(suffix=".vo_raw.wav")
        os.close(fd)
        rc = _run_espeak(
            # -a 175: louder narration so the voice clearly leads the mix
            ["-v", voice, "-s", str(int(wpm)), "-a", "175", "-w", raw_wav, spoken],
            timeout=120.0,
        )
        if rc != 0 or not os.path.getsize(raw_wav):
            _logger.warning("[voiceover] espeak-ng failed (rc=%s)", rc)
            return False

        # Resample to the pipeline rate EXPLICITLY (espeak outputs 22050 Hz;
        # muxing that against 44100 Hz music without -ar is exactly the
        # pitch-shift bug class this module guards against).
        result = run_ffmpeg(
            ["ffmpeg", "-y", "-i", raw_wav,
             "-ar", str(int(sample_rate)), "-ac", "1",
             "-sample_fmt", "s16", str(out_path)],
            timeout=60,
        )
        ok = result.returncode == 0 and Path(out_path).exists()
        if not ok:
            _logger.warning(
                "[voiceover] resample failed (rc=%s): %s",
                result.returncode, result.stderr[-200:],
            )
        return ok
    except Exception as exc:  # noqa: BLE001 - voiceover must never break renders
        _logger.warning("[voiceover] synthesis error: %s", exc)
        return False
    finally:
        if raw_wav:
            try:
                os.remove(raw_wav)
            except OSError:
                pass


def mix_voiceover_over_music(
    vo_path: str,
    music_path: str,
    out_path: str,
    duration_sec: float,
    music_gain: float = 0.18,
    sample_rate: int = VO_SAMPLE_RATE,
) -> bool:
    """Duck the music under the narration and write a combined track.
    Never-raise; returns False on failure (caller keeps the music track)."""
    try:
        dur = max(1.0, float(duration_sec or 0) or 1.0)
        result = run_ffmpeg(
            ["ffmpeg", "-y",
             "-i", str(vo_path), "-i", str(music_path),
             "-filter_complex",
             (f"[0:a]aresample={int(sample_rate)},apad[a0];"
              f"[1:a]aresample={int(sample_rate)},volume={music_gain}[a1];"
              f"[a0][a1]amix=inputs=2:duration=longest:normalize=0[aout]"),
             "-map", "[aout]", "-t", f"{dur:.3f}",
             "-ar", str(int(sample_rate)), "-ac", "1",
             str(out_path)],
            timeout=90,
        )
        return result.returncode == 0 and Path(out_path).exists()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[voiceover] mix error: %s", exc)
        return False


def voiceover_track(
    text: str,
    out_dir: str,
    job_id: str,
    duration_sec: float,
    music_path: Optional[str] = None,
) -> Optional[str]:
    """High-level entry: synthesize narration for ``text`` and (when a music
    track is supplied) duck the music under it.

    Returns the path of the audio file to mux into the video, or None when
    speech synthesis is unavailable/failed (caller keeps its existing audio).
    Total / never-raise.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        vo_wav = out / f"vo_{job_id[:12]}.wav"
        if not synthesize_voiceover(text, str(vo_wav)):
            return None
        if music_path and Path(music_path).exists():
            mixed = out / f"vomix_{job_id[:12]}.wav"
            if mix_voiceover_over_music(
                str(vo_wav), str(music_path), str(mixed), duration_sec
            ):
                return str(mixed)
            _logger.warning("[voiceover] mix failed — using narration only")
        return str(vo_wav)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[voiceover] track build error: %s", exc)
        return None
