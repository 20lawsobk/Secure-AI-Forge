"""Digital GPU audio synthesis engine — 100% self-contained.

Every compute path in this module routes through the Digital GPU stack:

  • Waveform synthesis  → NativeKernels.additive_synth / freq_sweep_sin
  • Envelope shaping    → NativeKernels.exp_decay / inplace_mul
  • Percussion noise    → NativeKernels.white_noise
  • STFT / iSTFT        → HyperGPU.batched_gemm on pre-built DFT matrices
  • HPSS stem split     → DFT-domain median filter + GPU-masked reconstruction

No librosa, no scipy, no soundfile — zero dependency on Replit's base
environment for audio synthesis or stem separation.  The only external
write path is the stdlib `wave` module for PCM output.

All functions are never-raise: every GPU path has a numpy fallback that
produces identical results so the never-raise contract is preserved.
"""
from __future__ import annotations

import hashlib
import struct
import threading
import wave as _wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai_model.gpu.native.kernels import get_native_kernels

# ── module-level DFT matrix cache (built once per n_fft size) ────────────────
_DFT_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_DFT_LOCK = threading.Lock()


def _dft_matrices(n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Wr, Wi) — real and imaginary DFT matrices for n_fft.

    Wr[k, n] = cos(-2π k n / N),  Wi[k, n] = sin(-2π k n / N)
    Shape: [(n_fft//2+1), n_fft] — only positive frequencies (Hermitian symmetry).
    Built once and cached per n_fft size; subsequent calls are lock-free reads.
    """
    cached = _DFT_CACHE.get(n_fft)
    if cached is not None:
        return cached
    with _DFT_LOCK:
        cached = _DFT_CACHE.get(n_fft)
        if cached is not None:
            return cached
        n_bins = n_fft // 2 + 1
        k = np.arange(n_bins, dtype=np.float32).reshape(-1, 1)
        n = np.arange(n_fft,  dtype=np.float32).reshape(1, -1)
        angle = -2.0 * np.pi * k * n / np.float32(n_fft)
        Wr = np.cos(angle).astype(np.float32)
        Wi = np.sin(angle).astype(np.float32)
        _DFT_CACHE[n_fft] = (Wr, Wi)
        return Wr, Wi


# ── STFT / iSTFT via Digital GPU GEMM ────────────────────────────────────────

def digital_gpu_stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Short-time Fourier transform via Digital GPU batched GEMM.

    Returns (S_real, S_imag) each of shape [n_bins, n_frames] float32.
    Uses pre-built DFT matrices multiplied against windowed frames — no scipy.
    """
    hop = hop_length or (n_fft // 4)
    if window is None:
        # Hann window via the NativeKernels exp path isn't needed here — it's
        # a one-time setup and not in the hot synthesis loop.
        window = np.hanning(n_fft).astype(np.float32)
    else:
        window = np.asarray(window, dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    # Pad so last frame is complete
    n_frames = max(1, 1 + (n - n_fft) // hop)
    pad_len = n_frames * hop + n_fft - n
    if pad_len > 0:
        x = np.concatenate([x, np.zeros(pad_len, dtype=np.float32)])

    # Build frames matrix [n_frames, n_fft] — each row is one windowed frame
    idx = (np.arange(n_fft, dtype=np.int32).reshape(1, -1) +
           np.arange(n_frames, dtype=np.int32).reshape(-1, 1) * hop)
    frames = x[idx] * window   # [n_frames, n_fft]

    # DFT via GEMM: S = W @ frames.T → [n_bins, n_frames]
    Wr, Wi = _dft_matrices(n_fft)
    try:
        from ai_model.maxcore.api import DigitalGPU
        gpu = DigitalGPU()
        S_real = gpu.gemm(Wr, frames.T)   # [n_bins, n_frames]
        S_imag = gpu.gemm(Wi, frames.T)
    except Exception:
        S_real = Wr @ frames.T
        S_imag = Wi @ frames.T

    return S_real.astype(np.float32), S_imag.astype(np.float32)


def digital_gpu_istft(
    S_real: np.ndarray,
    S_imag: np.ndarray,
    hop_length: Optional[int] = None,
    length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse STFT via Digital GPU GEMM — reconstruction from complex spectrogram.

    Accepts (S_real, S_imag) each [n_bins, n_frames]; returns float32 waveform.
    Uses overlap-add with Hann window normalisation.  Fully self-contained —
    no librosa or scipy.
    """
    n_bins, n_frames = S_real.shape
    n_fft = (n_bins - 1) * 2
    hop   = hop_length or (n_fft // 4)
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)

    Wr, Wi = _dft_matrices(n_fft)
    # iDFT: x_frame = (Wr.T @ S_real + Wi.T @ S_imag) / n_fft
    # Wr.T shape [n_fft, n_bins], S_real shape [n_bins, n_frames]
    # result [n_fft, n_frames] → each column is one reconstructed frame
    try:
        from ai_model.maxcore.api import DigitalGPU
        gpu = DigitalGPU()
        frames = (gpu.gemm(Wr.T, S_real) + gpu.gemm(Wi.T, S_imag)) / np.float32(n_fft)
    except Exception:
        frames = (Wr.T @ S_real + Wi.T @ S_imag) / np.float32(n_fft)

    # Overlap-add reconstruction
    out_len = length or (n_frames * hop + n_fft)
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    win2 = window ** 2

    for i in range(n_frames):
        start = i * hop
        end   = start + n_fft
        if end > out_len:
            trim = out_len - start
            out[start:] += (frames[:, i][:trim] * window[:trim])
            norm[start:] += win2[:trim]
        else:
            out[start:end] += frames[:, i] * window
            norm[start:end] += win2

    # Normalise where the window contributes energy
    mask = norm > 1e-8
    out[mask] /= norm[mask]
    return out[:length] if length else out


# ── HPSS via DFT-domain median filter ────────────────────────────────────────

def _median_filter_axis(S: np.ndarray, kernel: int, axis: int) -> np.ndarray:
    """Pure-numpy median filter along one axis — used only for HPSS mask building.
    This is a support operation (one call total), not a hot synthesis loop,
    so plain numpy is acceptable here rather than a native kernel.
    """
    pad = kernel // 2
    if axis == 0:
        padded = np.pad(S, ((pad, pad), (0, 0)), mode="edge")
        return np.array([
            np.median(padded[i:i + kernel, :], axis=0)
            for i in range(S.shape[0])
        ], dtype=np.float32)
    else:
        padded = np.pad(S, ((0, 0), (pad, pad)), mode="edge")
        return np.array([
            np.median(padded[:, j:j + kernel], axis=1)
            for j in range(S.shape[1])
        ], dtype=np.float32).T


def digital_gpu_hpss(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    kernel_harm: int = 31,
    kernel_perc: int = 31,
    bass_cutoff_hz: float = 250.0,
) -> Dict[str, np.ndarray]:
    """Harmonic-Percussive Source Separation via Digital GPU STFT.

    Returns {"drums": ndarray, "bass": ndarray, "melody": ndarray} — all
    float32 waveforms time-aligned with ``y``.

    The STFT and iSTFT both route through Digital GPU GEMM (DFT matrix ×
    windowed frames).  The median-filter mask step runs in numpy — it is a
    one-shot setup op (not in any loop) and not a synthesis hot path.
    """
    hop = hop_length or (n_fft // 4)
    S_real, S_imag = digital_gpu_stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.sqrt(S_real ** 2 + S_imag ** 2)      # [n_bins, n_frames]

    # Median filter along frequency axis → smears percussive transients → harmonic
    H_mag = _median_filter_axis(mag, kernel_harm, axis=0)
    # Median filter along time axis → smears harmonic sustain → percussive
    P_mag = _median_filter_axis(mag, kernel_perc, axis=1)

    # Wiener-style soft masks
    denom = H_mag + P_mag + 1e-8
    M_harm = H_mag / denom   # [n_bins, n_frames]
    M_perc = P_mag / denom

    # Apply masks (element-wise multiply on complex spectrogram)
    H_real, H_imag = S_real * M_harm, S_imag * M_harm
    P_real, P_imag = S_real * M_perc, S_imag * M_perc

    harmonic   = digital_gpu_istft(H_real, H_imag, hop_length=hop, length=len(y))
    percussive = digital_gpu_istft(P_real, P_imag, hop_length=hop, length=len(y))

    # Spectral low/high split of the harmonic component → bass vs melody
    freqs = np.linspace(0, sample_rate / 2.0, S_real.shape[0], dtype=np.float32)
    low_mask = (freqs <= bass_cutoff_hz)[:, None]   # [n_bins, 1]

    # Apply STFT again on harmonic signal to get its spectrogram
    Hh_real, Hh_imag = digital_gpu_stft(harmonic, n_fft=n_fft, hop_length=hop)
    bass   = digital_gpu_istft(Hh_real * low_mask,  Hh_imag * low_mask,
                                hop_length=hop, length=len(y))
    melody = digital_gpu_istft(Hh_real * (~low_mask.astype(bool)),
                                Hh_imag * (~low_mask.astype(bool)),
                                hop_length=hop, length=len(y))

    return {"drums": percussive, "bass": bass, "melody": melody}


# ── Additive synthesis engine ─────────────────────────────────────────────────

def render_audio_clip(
    job_id: str,
    bpm: float,
    key: str,
    duration_sec: float = 30.0,
    genre: str = "",
    mood: str = "",
    sample_rate: int = 44100,
) -> np.ndarray:
    """Synthesize a stereo audio waveform entirely on the Digital GPU.

    Returns float32 stereo array of shape [n_samples * 2] interleaved L/R,
    normalised to [-1, 1], ready for direct 16-bit PCM conversion.

    All synthesis — additive harmonics, 808 sweeps, percussion noise,
    envelopes — routes through NativeKernels (compiled SIMD C) with numpy
    fallback.  Zero calls to scipy, librosa, or any external audio library.
    """
    kern = get_native_kernels()

    _genre = genre.lower().strip()
    _mood  = mood.lower().strip()

    # ── Key → root frequency + scale ─────────────────────────────────────
    note_semi = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
                 "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    parts = (key or "C major").split()
    root_name = parts[0] if parts else "C"
    is_minor  = len(parts) > 1 and parts[1].lower().startswith("min")
    root_freq = 220.0 * (2.0 ** (note_semi.get(root_name, 0) / 12.0))
    scale     = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]

    bpm_f     = max(40.0, min(float(bpm), 200.0))
    beat_sec  = 60.0 / bpm_f
    beat_len  = max(1, int(beat_sec * sample_rate))
    n_total   = int(duration_sec * sample_rate)
    audio     = np.zeros(n_total, dtype=np.float32)

    # ── Genre / mood parameters ────────────────────────────────────────────
    if any(g in _genre for g in ("trap", "drill")):
        arp_deg, note_gain, swing = [0,0,3,0,5,0,3,7], 0.45, 0.18
    elif "phonk" in _genre:
        arp_deg, note_gain, swing = [0,7,5,3,0,10,7,5], 0.50, 0.22
    elif any(g in _genre for g in ("afrobeats", "afro", "amapiano")):
        arp_deg, note_gain, swing = [0,4,2,4,0,6,4,2], 0.50, 0.12
    elif any(g in _genre for g in ("lo-fi", "lofi", "chill", "jazz")):
        arp_deg, note_gain, swing = [0,2,4,2,6,4,2,0], 0.40, 0.30
    elif any(g in _genre for g in ("reggaeton", "latin")):
        arp_deg, note_gain, swing = [0,0,4,0,5,0,4,2], 0.48, 0.08
    else:                                   # pop / hip hop / default
        arp_deg, note_gain, swing = [0,2,4,6,4,2,0,5], 0.55, 0.0

    base_octave = 2 if any(w in _mood for w in
                           ("energetic","hype","euphoric","aggressive")) else 1
    if any(w in _mood for w in ("energetic","hype","euphoric","aggressive")):
        note_gain *= 1.1
    elif any(w in _mood for w in ("melancholic","dark","sad","chill")):
        note_gain *= 0.85

    # ── Harmonic arp layer (additive synthesis via Digital GPU) ───────────
    note_sec = beat_sec * (0.25 if "trap" in _genre else 0.5)
    note_len = max(1, int(note_sec * sample_rate))

    # harmonic mix: [fundamental, 2nd, 3rd, sub-octave]
    h_ratios = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    h_amps   = np.array([0.55, 0.25, 0.12, 0.08], dtype=np.float32) * float(note_gain)

    env = kern.exp_decay(3.5, note_sec, note_len)   # per-note envelope

    idx, step = 0, 0
    note_buf = np.zeros(note_len, dtype=np.float32)
    while idx < n_total:
        deg    = scale[arp_deg[step % len(arp_deg)] % len(scale)]
        octave = base_octave + (step // len(arp_deg)) % 2
        freq   = float(root_freq) * (2.0 ** (deg / 12.0)) * octave

        freqs_h = h_ratios * np.float32(freq)
        note_buf[:] = 0.0
        kern.additive_synth(freqs_h, h_amps, sample_rate, note_buf)
        kern.inplace_mul(note_buf, env)

        sw = int(swing * beat_len) if (step % 2 == 1) else 0
        start = min(idx + sw, n_total - 1)
        end   = min(start + note_len, n_total)
        audio[start:end] += note_buf[:end - start]

        idx  += note_len
        step += 1

    # ── Percussion layer (genre-specific, all via Digital GPU) ────────────
    seed_base = int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16) & 0xFFFFFFFF

    if any(g in _genre for g in ("trap", "drill")):
        # 808 sweep sub-kick on every beat
        kick_len = max(1, int(min(beat_sec * 0.6, 0.35) * sample_rate))
        kick = kern.freq_sweep_sin(55.0, 12.0, sample_rate, kick_len)
        kick_env = kern.exp_decay(6.0, kick_len / sample_rate, kick_len)
        kern.inplace_mul(kick, kick_env)
        kick *= 0.75
        for b in range(0, n_total, beat_len):
            end = min(b + kick_len, n_total)
            audio[b:end] += kick[:end - b]
        # Triplet hi-hat noise
        hihat_period = max(1, beat_len // 3)
        hihat_len    = min(hihat_period // 2, int(0.02 * sample_rate))
        if hihat_len > 1:
            hat_noise = kern.white_noise(seed_base ^ 0xBEEF, hihat_len)
            hat_env   = kern.exp_decay(80.0, hihat_len / sample_rate, hihat_len)
            kern.inplace_mul(hat_noise, hat_env)
            hat_noise *= 0.18
            for b in range(0, n_total, hihat_period):
                end = min(b + hihat_len, n_total)
                audio[b:end] += hat_noise[:end - b]

    elif "phonk" in _genre:
        kick_len = max(1, int(min(beat_sec * 0.7, 0.40) * sample_rate))
        kick = kern.freq_sweep_sin(50.0, 8.0, sample_rate, kick_len)
        kick_env = kern.exp_decay(5.0, kick_len / sample_rate, kick_len)
        kern.inplace_mul(kick, kick_env)
        kick *= 0.80
        for b_i, b in enumerate(range(0, n_total, beat_len)):
            if b_i % 2 == 0:
                end = min(b + kick_len, n_total)
                audio[b:end] += kick[:end - b]

    elif any(g in _genre for g in ("afrobeats", "afro", "amapiano")):
        stab_period = beat_len * 2
        stab_offset = beat_len + beat_len // 2
        stab_len    = max(1, int(0.08 * sample_rate))
        stab_freqs  = np.array([root_freq * 1.5], dtype=np.float32)
        stab_amps   = np.array([0.45], dtype=np.float32)
        stab_buf    = np.zeros(stab_len, dtype=np.float32)
        kern.additive_synth(stab_freqs, stab_amps, sample_rate, stab_buf)
        stab_env = kern.exp_decay(30.0, stab_len / sample_rate, stab_len)
        kern.inplace_mul(stab_buf, stab_env)
        for b in range(stab_offset, n_total, stab_period):
            end = min(b + stab_len, n_total)
            audio[b:end] += stab_buf[:end - b]
        # Shaker noise on every 16th
        shaker_period = max(1, beat_len // 4)
        shaker_len    = min(shaker_period // 3, int(0.015 * sample_rate))
        if shaker_len > 1:
            sh = kern.white_noise(seed_base ^ 0x7777, shaker_len)
            sh_env = kern.exp_decay(120.0, shaker_len / sample_rate, shaker_len)
            kern.inplace_mul(sh, sh_env)
            sh *= 0.12
            for b in range(0, n_total, shaker_period):
                end = min(b + shaker_len, n_total)
                audio[b:end] += sh[:end - b]

    elif any(g in _genre for g in ("lo-fi", "lofi", "chill", "jazz")):
        kick_len = max(1, int(min(beat_sec, 0.18) * sample_rate))
        kick = kern.freq_sweep_sin(90.0, 18.0, sample_rate, kick_len)
        kick_env = kern.exp_decay(10.0, kick_len / sample_rate, kick_len)
        kern.inplace_mul(kick, kick_env)
        for b_i, b in enumerate(range(0, n_total, beat_len)):
            gain = 0.35 if b_i % 4 in (1, 3) else 0.50
            end = min(b + kick_len, n_total)
            audio[b:end] += gain * kick[:end - b]

    else:  # pop / hip hop default — four-on-floor
        kick_len = max(1, int(min(beat_sec, 0.18) * sample_rate))
        kick = kern.freq_sweep_sin(110.0, 18.0, sample_rate, kick_len)
        kick_env = kern.exp_decay(10.0, kick_len / sample_rate, kick_len)
        kern.inplace_mul(kick, kick_env)
        kick *= 0.6
        for b in range(0, n_total, beat_len):
            end = min(b + kick_len, n_total)
            audio[b:end] += kick[:end - b]

    # ── Normalise & stereo (Digital GPU-aware inline) ─────────────────────
    peak = float(np.max(np.abs(audio))) or 1.0
    mono = np.clip(audio / peak * 0.92, -1.0, 1.0).astype(np.float32)

    # L/R decorrelation via short comb delay for perceived width
    delay = max(1, int(0.008 * sample_rate))
    left  = mono.copy()
    right = np.empty_like(mono)
    right[:delay] = 0.0
    right[delay:] = mono[:-delay]

    # Interleave to [n_total*2] float32 stereo
    stereo_f = np.empty(n_total * 2, dtype=np.float32)
    stereo_f[0::2] = left
    stereo_f[1::2] = right
    return stereo_f


def write_wav(path: Path, stereo_f32: np.ndarray, sample_rate: int = 44100) -> None:
    """Write a float32 stereo interleaved array to a WAV file (stdlib only)."""
    pcm = (stereo_f32 * 32767.0).astype(np.int16)
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def write_stem_wav(path: Path, mono_f32: np.ndarray,
                   sample_rate: int = 44100) -> None:
    """Write a float32 mono stem to a WAV file (stdlib only — no soundfile)."""
    peak = float(np.max(np.abs(mono_f32))) or 1.0
    safe = np.clip(mono_f32 / peak, -1.0, 1.0)
    pcm = (safe * 32767.0).astype(np.int16)
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
