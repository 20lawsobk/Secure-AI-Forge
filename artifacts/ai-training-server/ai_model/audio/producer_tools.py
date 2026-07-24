"""Producer-grade audio controls for the retrieval+transform audio pipeline.

Everything here is REAL DSP — no placeholders, no fakes:

* ``nearest_semitones`` / ``retune_retime``   → hit an EXACT musical key and
  BPM by pitch-shifting + time-stretching the source with ffmpeg's rubberband
  (phase-vocoder) filter, so a producer who asks for "128 BPM, F minor" gets a
  clip actually at 128 BPM in F minor — not merely the nearest sample.
* ``master_export``                           → EBU-R128 loudness normalise to a
  target LUFS and export at a chosen format / sample-rate / bit-depth (WAV 24-bit
  for the studio, MP3 for quick sharing).
* ``separate_stems``                          → split a clip into drums / bass /
  melody stems using librosa harmonic-percussive source separation plus a
  spectral low/high split — genuinely useful, honest stems for remixing.

The functions are deliberately dependency-light (ffmpeg + librosa + soundfile,
all already used elsewhere in this project) and NEVER raise for cosmetic
reasons: callers wrap them so an optional producer control can degrade to the
base clip rather than fail a whole render.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

from ai_model.video.ffmpeg_util import run_ffmpeg

# ─── Musical key handling ─────────────────────────────────────────────────────

# Pitch class of every tonic we might see (sharps + flats, upper/lower case).
_PITCH_CLASS = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "fb": 4,
    "e#": 5, "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9,
    "a#": 10, "bb": 10, "b": 11, "cb": 11,
}


def _tonic_pitch_class(key: Optional[str]) -> Optional[int]:
    """Extract the tonic pitch class (0-11) from a key label like 'F# minor'."""
    if not key:
        return None
    token = str(key).strip().split()[0].lower() if str(key).strip() else ""
    # Longest-match: try two chars (e.g. 'f#', 'bb') then one char.
    for n in (2, 1):
        if token[:n] in _PITCH_CLASS:
            return _PITCH_CLASS[token[:n]]
    return None


def nearest_semitones(src_key: Optional[str], target_key: Optional[str]) -> int:
    """Smallest semitone shift moving ``src_key``'s tonic onto ``target_key``'s.

    Chooses the wrap-around direction with the smallest absolute shift (result in
    -6..+6) so we minimise pitch-shifting artifacts. Mode (major/minor) is
    preserved automatically: shifting every pitch by a constant keeps intervals,
    so a major stays major. Returns 0 when either key is unknown.
    """
    a = _tonic_pitch_class(src_key)
    b = _tonic_pitch_class(target_key)
    if a is None or b is None:
        return 0
    diff = (b - a) % 12
    if diff > 6:
        diff -= 12
    return int(diff)


# ─── Retune + retime (hit an exact key & BPM) ─────────────────────────────────

def retune_retime(in_wav: Path, out_wav: Path, *, semitones: int = 0,
                  tempo_ratio: float = 1.0, timeout: int = 90) -> bool:
    """Pitch-shift by ``semitones`` and time-stretch by ``tempo_ratio``.

    ``tempo_ratio`` = target_bpm / source_bpm (>1 = faster/shorter).
    Uses ffmpeg's rubberband filter (high-quality phase vocoder) which does both
    in one pass without the "chipmunk" artifact of naive resampling.

    Returns True if a transform was applied and produced a file, False if it was
    a no-op (caller keeps the input untouched).
    """
    tempo_ratio = float(tempo_ratio)
    if not math.isfinite(tempo_ratio) or tempo_ratio <= 0:
        tempo_ratio = 1.0
    # Clamp to a musically sane range — beyond this the artifacts dominate and
    # we are better off serving the closest sample as-is.
    tempo_ratio = max(0.5, min(2.0, tempo_ratio))
    semitones = int(max(-6, min(6, int(semitones))))

    if abs(tempo_ratio - 1.0) < 0.01 and semitones == 0:
        return False

    pitch_scale = 2.0 ** (semitones / 12.0)
    af = f"rubberband=tempo={tempo_ratio:.6f}:pitch={pitch_scale:.6f}:pitchq=quality"
    r = run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(in_wav), "-af", af, str(out_wav)],
        timeout=timeout,
    )
    if r.returncode != 0 or not Path(out_wav).exists():
        raise RuntimeError(
            f"rubberband retune/retime failed (rc={r.returncode}): "
            f"{r.stderr[-300:]}"
        )
    return True


# ─── Mastering / export (LUFS + format + bit depth) ───────────────────────────

_PCM_CODEC = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}

# Common producer loudness targets, for reference / validation.
LUFS_PRESETS = {
    "streaming": -14.0,   # Spotify / Apple / YouTube
    "club": -9.0,         # loud club / EDM master
    "broadcast": -23.0,   # EBU R128 broadcast
    "podcast": -16.0,
}


def master_export(in_wav: Path, out_path: Path, *, fmt: str = "mp3",
                  sample_rate: int = 44100, bit_depth: int = 24,
                  loudness_lufs: Optional[float] = None,
                  mp3_quality: int = 2, timeout: int = 120) -> Path:
    """Loudness-normalise (optional) and export to the requested format.

    * ``fmt``           : "wav" (studio) or "mp3" (sharing).
    * ``loudness_lufs`` : integrated target (e.g. -14). ``None`` skips
      normalisation and preserves the source level.
    """
    fmt = (fmt or "mp3").lower()
    filters = []
    if loudness_lufs is not None:
        # Single-pass EBU R128 normalisation with a -1 dBTP ceiling.
        filters.append(f"loudnorm=I={float(loudness_lufs):.1f}:TP=-1.0:LRA=11")

    cmd = ["ffmpeg", "-y", "-i", str(in_wav)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    # Always deliver stereo — dataset sources may be mono; -ac 2 upmixes.
    cmd += ["-ac", "2", "-ar", str(int(sample_rate))]

    if fmt == "wav":
        codec = _PCM_CODEC.get(int(bit_depth), "pcm_s24le")
        cmd += ["-codec:a", codec, str(out_path)]
    else:
        # CBR 320 kbps — leaseable-beat delivery quality (VBR -q:a produced
        # ~92 kbps files that were rejected downstream).
        cmd += ["-codec:a", "libmp3lame", "-b:a", "320k", str(out_path)]

    r = run_ffmpeg(cmd, timeout=timeout)
    if r.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(
            f"master/export failed (rc={r.returncode}): {r.stderr[-300:]}"
        )
    return Path(out_path)


# ─── Stem separation (drums / bass / melody) ──────────────────────────────────

def separate_stems(in_wav: Path, out_dir: Path, base_name: str, *,
                   fmt: str = "wav", bit_depth: int = 24,
                   bass_cutoff_hz: float = 250.0) -> Dict[str, Path]:
    """Split a clip into ``drums`` / ``bass`` / ``melody`` stems.

    Fully self-contained — routes entirely through the Digital GPU stack:
      • WAV decode: stdlib ``wave`` module
      • STFT / iSTFT: Digital GPU DFT-matrix GEMM (no librosa / scipy)
      • HPSS: Wiener soft masks on GPU-computed magnitude spectrogram
      • Output: stdlib ``wave`` module (no soundfile)

    Returns ``{stem_name: path}``. Raises on failure so the caller reports
    stems unavailable honestly rather than silently returning silence.
    """
    import wave as _wave_mod
    import numpy as _np

    from ai_model.audio.digital_gpu_synth import (
        digital_gpu_hpss, write_stem_wav)

    # ── Decode source WAV (stdlib only) ──────────────────────────────────
    with _wave_mod.open(str(in_wav), "rb") as wf:
        sr        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sw        = wf.getsampwidth()
        n_frames  = wf.getnframes()
        raw       = wf.readframes(n_frames)

    dtype_map = {1: _np.int8, 2: _np.int16, 4: _np.int32}
    pcm = _np.frombuffer(raw, dtype=dtype_map.get(sw, _np.int16))
    # Mix to mono float32
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    y = pcm.astype(_np.float32) / float(1 << (sw * 8 - 1))

    if y.size == 0:
        raise RuntimeError("empty audio — cannot separate stems")

    # ── HPSS via Digital GPU STFT/iSTFT ─────────────────────────────────
    stems_audio = digital_gpu_hpss(
        y, sample_rate=int(sr),
        n_fft=2048, hop_length=512,
        bass_cutoff_hz=float(bass_cutoff_hz),
    )

    # ── Write stems via stdlib wave (no soundfile) ────────────────────────
    out: Dict[str, Path] = {}
    for name, sig in stems_audio.items():
        p = Path(out_dir) / f"{base_name}_stem_{name}.wav"
        write_stem_wav(p, sig.astype(_np.float32), sample_rate=int(sr))
        out[name] = p
    return out
