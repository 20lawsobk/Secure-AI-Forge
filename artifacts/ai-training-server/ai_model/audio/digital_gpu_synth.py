"""Digital GPU professional audio synthesis engine.

Every compute path routes through the Digital GPU stack:
  • Waveform generation → NativeKernels.saw_wave (polyBLEP bandlimited)
  • Filter            → NativeKernels.biquad (RBJ biquad IIR)
  • Envelopes         → NativeKernels.adsr
  • Saturation        → NativeKernels.soft_sat
  • Limiter           → NativeKernels.soft_limit
  • Compressor gain   → NativeKernels.compress_gain
  • STFT / iSTFT      → HyperGPU.gemm on DFT matrices
  • HPSS              → DFT-domain median filter

Output quality targets:
  - No aliasing: polyBLEP sawtooth oscillators
  - Real timbre: detuned unison + biquad filter sweeps
  - Arrangement: intro / verse / pre-drop / drop / breakdown / outro
  - Dynamics: per-step velocity variation, bus compression, -0.3 dBFS limit
  - Stereo: M/S widening + reverb send
"""
from __future__ import annotations

import hashlib
import struct
import threading
import wave as _wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai_model.gpu.native.kernels import NativeKernels, get_native_kernels

# ─────────────────────────────────────────────────────────────────────────────
# DFT matrix cache (for STFT / stem separation)
# ─────────────────────────────────────────────────────────────────────────────
_DFT_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_DFT_LOCK  = threading.Lock()


def _dft_matrices(n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _DFT_CACHE.get(n_fft)
    if cached:
        return cached
    with _DFT_LOCK:
        cached = _DFT_CACHE.get(n_fft)
        if cached:
            return cached
        n_bins = n_fft // 2 + 1
        k = np.arange(n_bins, dtype=np.float32).reshape(-1, 1)
        n = np.arange(n_fft,  dtype=np.float32).reshape(1, -1)
        angle = -2.0 * np.pi * k * n / np.float32(n_fft)
        Wr = np.cos(angle).astype(np.float32)
        Wi = np.sin(angle).astype(np.float32)
        _DFT_CACHE[n_fft] = (Wr, Wi)
        return Wr, Wi


# ─────────────────────────────────────────────────────────────────────────────
# STFT / iSTFT / HPSS (for stem separation — kept from v1)
# ─────────────────────────────────────────────────────────────────────────────

def digital_gpu_stft(x, n_fft=2048, hop_length=None, window=None):
    hop = hop_length or (n_fft // 4)
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    n_frames = max(1, 1 + (n - n_fft) // hop)
    pad_len  = n_frames * hop + n_fft - n
    if pad_len > 0:
        x = np.concatenate([x, np.zeros(pad_len, np.float32)])
    idx    = (np.arange(n_fft).reshape(1, -1) +
              np.arange(n_frames).reshape(-1, 1) * hop)
    frames = x[idx] * window
    Wr, Wi = _dft_matrices(n_fft)
    try:
        from ai_model.maxcore.api import DigitalGPU
        gpu    = DigitalGPU()
        S_real = gpu.gemm(Wr, frames.T)
        S_imag = gpu.gemm(Wi, frames.T)
    except Exception:
        S_real = Wr @ frames.T
        S_imag = Wi @ frames.T
    return S_real.astype(np.float32), S_imag.astype(np.float32)


def digital_gpu_istft(S_real, S_imag, hop_length=None, length=None, window=None):
    n_bins, n_frames = S_real.shape
    n_fft = (n_bins - 1) * 2
    hop   = hop_length or (n_fft // 4)
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)
    Wr, Wi = _dft_matrices(n_fft)
    try:
        from ai_model.maxcore.api import DigitalGPU
        gpu    = DigitalGPU()
        frames = (gpu.gemm(Wr.T, S_real) + gpu.gemm(Wi.T, S_imag)) / np.float32(n_fft)
    except Exception:
        frames = (Wr.T @ S_real + Wi.T @ S_imag) / np.float32(n_fft)
    out_len = length or (n_frames * hop + n_fft)
    out  = np.zeros(out_len, np.float32)
    norm = np.zeros(out_len, np.float32)
    win2 = window ** 2
    for i in range(n_frames):
        s, e = i * hop, i * hop + n_fft
        trim = min(e, out_len) - s
        out[s:s+trim]  += frames[:trim, i] * window[:trim]
        norm[s:s+trim] += win2[:trim]
    mask = norm > 1e-8
    out[mask] /= norm[mask]
    return out[:length] if length else out


def _median_filter_axis(S, kernel, axis):
    pad = kernel // 2
    if axis == 0:
        padded = np.pad(S, ((pad, pad), (0, 0)), mode="edge")
        return np.array([np.median(padded[i:i+kernel], axis=0)
                         for i in range(S.shape[0])], dtype=np.float32)
    padded = np.pad(S, ((0, 0), (pad, pad)), mode="edge")
    return np.array([np.median(padded[:, j:j+kernel], axis=1)
                     for j in range(S.shape[1])], dtype=np.float32).T


def digital_gpu_hpss(y, sample_rate, n_fft=2048, hop_length=None,
                      kernel_harm=31, kernel_perc=31, bass_cutoff_hz=250.0):
    hop    = hop_length or (n_fft // 4)
    Sr, Si = digital_gpu_stft(y, n_fft=n_fft, hop_length=hop)
    mag    = np.sqrt(Sr**2 + Si**2)
    H_mag  = _median_filter_axis(mag, kernel_harm, 0)
    P_mag  = _median_filter_axis(mag, kernel_perc, 1)
    denom  = H_mag + P_mag + 1e-8
    M_h, M_p = H_mag / denom, P_mag / denom
    harmonic   = digital_gpu_istft(Sr*M_h, Si*M_h, hop_length=hop, length=len(y))
    percussive = digital_gpu_istft(Sr*M_p, Si*M_p, hop_length=hop, length=len(y))
    freqs      = np.linspace(0, sample_rate/2, Sr.shape[0], dtype=np.float32)
    low        = (freqs <= bass_cutoff_hz)[:, None]
    Hhr, Hhi   = digital_gpu_stft(harmonic, n_fft=n_fft, hop_length=hop)
    bass       = digital_gpu_istft(Hhr*low, Hhi*low, hop_length=hop, length=len(y))
    melody     = digital_gpu_istft(Hhr*(~low.astype(bool)), Hhi*(~low.astype(bool)),
                                   hop_length=hop, length=len(y))
    return {"drums": percussive, "bass": bass, "melody": melody}


# ─────────────────────────────────────────────────────────────────────────────
# Music theory helpers
# ─────────────────────────────────────────────────────────────────────────────

_NOTE_SEMI = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
              "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# Chord voicings: (quality, [semitone intervals from root])
VOICINGS = {
    "maj":   [0, 4, 7, 12],
    "min":   [0, 3, 7, 12],
    "maj7":  [0, 4, 7, 11],
    "min7":  [0, 3, 7, 10],
    "dom7":  [0, 4, 7, 10],
    "sus2":  [0, 2, 7, 12],
    "dim":   [0, 3, 6, 12],
    "maj9":  [0, 4, 7, 11, 14],
    "min9":  [0, 3, 7, 10, 14],
}

# Genre chord progressions: list of (semitone offset, voicing)
# All offsets are relative to the track root note
CHORD_PROGS: Dict[str, List[Tuple[int, str]]] = {
    "trap":      [(0,"min7"), (-2,"maj7"), (-3,"maj"), (-2,"dom7")],
    "drill":     [(0,"min7"), (-2,"maj7"), (-3,"maj"), (-5,"min")],
    "phonk":     [(0,"min"),  (-5,"min"),  (-7,"min"),  (-3,"maj")],
    "afrobeats": [(0,"maj"),  (5,"maj"),   (7,"dom7"),  (5,"maj")],
    "amapiano":  [(0,"maj7"), (5,"maj7"),  (7,"maj"),   (2,"min7")],
    "lofi":      [(0,"maj7"), (9,"min7"),  (2,"min7"),  (7,"dom7")],
    "jazz":      [(0,"maj7"), (2,"min7"),  (7,"dom7"),  (0,"maj7")],
    "pop":       [(0,"maj"),  (7,"maj"),   (9,"min"),   (5,"maj")],
    "latin":     [(0,"min"),  (5,"min"),   (7,"maj"),   (5,"min")],
    "default":          [(0,"min7"), (-3,"maj7"), (-5,"maj"),  (-2,"dom7")],
    # F# minor cinematic: i → bVII → bIII(lift) → v  — emotionally unstoppable
    "cinematic_trap":   [(0,"min7"), (-2,"maj"),  (3,"maj7"), (-5,"min7")],
    # New genres
    "rnb":          [(2,"min9"),  (7,"maj7"),  (0,"maj7"),  (9,"min7")],
    "house":        [(9,"min"),   (5,"maj"),   (0,"maj"),   (7,"maj")],
    "reggaeton":    [(7,"min"),   (3,"maj"),   (10,"maj"),  (5,"maj")],
    "hyperpop":     [(0,"maj"),   (7,"maj"),   (9,"min"),   (5,"maj")],
    "orchestral":   [(0,"min7"),  (-4,"maj7"), (3,"maj7"),  (10,"maj7")],
}

# 16-step drum patterns (1=hit, 0=rest, per 16th note)
_D = {
    "trap": {
        "kick":  [1,0,0,0, 0,0,0,1, 0,0,1,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,0,1],
        "hat_o": [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
    },
    "drill": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,1,0],
        "hat_c": [1,1,0,1, 1,0,1,1, 0,1,1,0, 1,0,1,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0],
    },
    "phonk": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
    },
    "afrobeats": {
        "kick":  [1,0,0,0, 0,1,0,0, 1,0,0,0, 0,1,0,0],
        "snare": [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,1,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "amapiano": {
        "kick":  [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,1,0,0],
        "snare": [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
        "hat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
        "hat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "clap":  [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "lofi": {
        "kick":  [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
        "hat_c": [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "default": {
        "kick":  [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hat_o": [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    # Heavy syncopated kick, ghost-snare, dense 8th-note hats
    "cinematic_trap": {
        "kick":  [1,0,0,0, 0,0,0,1, 0,1,0,0, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,1,0, 1,0,0,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
    },
    # New genres
    "rnb": {
        "kick":  [1,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "house": {
        "kick":  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "reggaeton": {
        # Dembow: kick on 1, rim/snare on the "and" of 2 and on beat 3
        "kick":  [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "clap":  [0,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "hyperpop": {
        "kick":  [1,0,1,0, 0,0,1,0, 1,0,0,0, 0,1,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat_c": [1,1,0,1, 1,0,1,1, 1,1,0,1, 1,1,1,0],
        "hat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,1],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "orchestral": {
        # No drum kit for orchestral — all zeros
        "kick":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hat_c": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
}

# Fill patterns — used on bar 4 of every 4-bar phrase
# Drops regular hat, adds snare roll on steps 10-15, crash on next bar step 0
_FILLS: Dict[str, Dict[str, List[int]]] = {
    "trap": {
        "kick":  [1,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "hat_c": [1,0,1,0, 1,0,1,0, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # crash on next bar step 0
    },
    "drill": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "hat_c": [1,1,0,1, 1,0,1,1, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "phonk": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "hat_c": [1,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,1,0,0, 0,0,1,0, 1,0,1,0],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "cinematic_trap": {
        "kick":  [1,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "hat_c": [1,0,1,0, 1,0,1,0, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,1,1, 1,1,1,1],
        "808":   [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "afrobeats": {
        "kick":  [1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,1, 0,0,0,0, 0,0,1,1, 1,1,1,0],
        "hat_c": [1,0,1,0, 1,0,1,0, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 1,0,0,1, 0,0,1,0, 1,0,1,0],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "amapiano": {
        "kick":  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,1,0, 0,0,1,1, 1,1,1,1],
        "hat_c": [1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0],
        "hat_o": [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,1,0, 0,0,1,0, 0,0,1,1, 1,1,1,1],
        "808":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "crash": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
}

# Bass patterns: (scale_degree_index, gate_in_16ths) per 16 steps; -1 = rest
_BASS = {
    "trap":      [(0,4),(-1,0),(-1,0),(-1,0), (4,2),(-1,0),(2,2),(-1,0),
                  (0,4),(-1,0),(-1,0),(-1,0), (3,2),(-1,0),(4,2),(-1,0)],
    "drill":     [(0,4),(-1,0),(-1,0),(-1,0), (4,3),(-1,0),(-1,0),(2,1),
                  (0,6),(-1,0),(-1,0),(-1,0), (3,4),(-1,0),(-1,0),(-1,0)],
    "phonk":     [(0,2),(-1,0),(0,2),(-1,0), (-3,2),(-1,0),(-1,0),(-1,0),
                  (0,2),(-1,0),(0,2),(-1,0), (-2,4),(-1,0),(-1,0),(-1,0)],
    "afrobeats": [(0,2),(-1,0),(2,2),(-1,0), (4,2),(-1,0),(2,2),(-1,0),
                  (0,2),(-1,0),(2,2),(-1,0), (5,2),(-1,0),(4,2),(-1,0)],
    "amapiano":  [(0,4),(-1,0),(-1,0),(2,2), (4,4),(-1,0),(-1,0),(-1,0),
                  (0,4),(-1,0),(-1,0),(2,2), (3,2),(4,2),(-1,0),(-1,0)],
    "lofi":      [(0,8),(-1,0),(-1,0),(-1,0), (-1,0),(-1,0),(-1,0),(-1,0),
                  (4,6),(-1,0),(-1,0),(-1,0), (3,4),(-1,0),(-1,0),(-1,0)],
    "default":        [(0,4),(-1,0),(-1,0),(-1,0), (4,2),(-1,0),(2,2),(-1,0),
                       (0,4),(-1,0),(-1,0),(-1,0), (3,2),(-1,0),(4,2),(-1,0)],
    # Long sliding 808 notes — cinematic sub movement
    "cinematic_trap": [(0,8),(-1,0),(-1,0),(-1,0), (-1,0),(-1,0),(-3,4),(-1,0),
                       (0,10),(-1,0),(-1,0),(-1,0), (-1,0),(-1,0),(-1,0),(-1,0)],
    # New genres
    # RnB: long sustained notes, warm
    "rnb":       [(0,8),(-1,0),(-1,0),(-1,0), (-1,0),(-1,0),(-1,0),(-1,0),
                  (7,6),(-1,0),(-1,0),(-1,0), (-2,4),(-1,0),(-1,0),(-1,0)],
    # House: pump pattern
    "house":     [(0,2),(-1,0),(0,2),(-1,0), (0,2),(-1,0),(0,2),(-1,0),
                  (0,2),(-1,0),(0,2),(-1,0), (0,2),(-1,0),(-2,2),(-1,0)],
    # Reggaeton: syncopated
    "reggaeton": [(0,2),(-1,0),(-1,0),(2,1), (4,2),(-1,0),(2,2),(-1,0),
                  (0,2),(-1,0),(-1,0),(5,1), (3,2),(-1,0),(2,2),(-1,0)],
    # Hyperpop: fast driving
    "hyperpop":  [(0,1),(0,1),(2,1),(3,1), (4,2),(-1,0),(7,1),(5,1),
                  (0,1),(0,1),(2,1),(3,1), (4,2),(-1,0),(0,2),(-1,0)],
    # Orchestral: cello bass walk
    "orchestral":[(0,4),(-1,0),(-1,0),(-1,0), (2,2),(-1,0),(3,2),(-1,0),
                  (4,4),(-1,0),(-1,0),(-1,0), (3,2),(-1,0),(2,4),(-1,0)],
}

# Arrangement sections: (name, bars, elements, filter_pct, energy)
# elements: K=kick, S=snare, H=hat, 8=808, B=bass, C=chord, L=lead, R=riser
_SECTIONS = {
    "trap":      [("intro",4,"KH",0.3,0.6), ("intro2",4,"KSH8B",0.5,0.7),
                  ("verse",8,"KSH8BC",0.6,0.75), ("prechorus",4,"KSH8BCR",0.75,0.85),
                  ("chorus",8,"KSH8BCL",1.0,1.0), ("breakdown",4,"B",0.2,0.4),
                  ("verse2",8,"KSH8BC",0.6,0.75), ("prechorus2",4,"KSH8BCR",0.8,0.9),
                  ("chorus2",8,"KSH8BCL",1.0,1.0), ("outro",4,"KH",0.4,0.5)],
    "lofi":      [("intro",8,"KH",0.5,0.5), ("verse",16,"KSHBC",0.65,0.7),
                  ("chorus",16,"KSHBCL",0.85,0.85), ("verse2",8,"KSHBC",0.65,0.7),
                  ("chorus2",16,"KSHBCL",0.85,0.85), ("outro",8,"KH",0.4,0.45)],
    "afrobeats": [("intro",4,"KH",0.4,0.6), ("verse",8,"KSHBC",0.65,0.75),
                  ("prechorus",4,"KSHBCR",0.8,0.85), ("chorus",8,"KSHBCL",1.0,1.0),
                  ("breakdown",4,"BC",0.3,0.45), ("verse2",8,"KSHBC",0.65,0.75),
                  ("chorus2",8,"KSHBCL",1.0,1.0), ("outro",4,"KH",0.4,0.5)],
    "default":   [("intro",4,"KH",0.4,0.6), ("verse",8,"KSHBC",0.6,0.7),
                  ("prechorus",4,"KSHBCR",0.75,0.85), ("chorus",8,"KSHBCL",1.0,1.0),
                  ("breakdown",4,"C",0.3,0.45), ("verse2",8,"KSHBC",0.6,0.7),
                  ("prechorus2",4,"KSHBCR",0.8,0.9), ("chorus2",8,"KSHBCL",1.0,1.0),
                  ("outro",4,"KH",0.4,0.5)],
    # Suno-spec layout: intro → hook → verse → prechorus → hook2 → outro
    # T=triplet hats active in hook sections
    "cinematic_trap": [
        ("intro",     4, "KH",        0.15, 0.45),
        ("hook",      8, "KSH8TBCL",  1.00, 1.00),
        ("verse",     8, "KSH8BC",    0.65, 0.75),
        ("prechorus", 4, "KSH8TBCR",  0.82, 0.90),
        ("hook2",     8, "KSH8TBCL",  1.00, 1.00),
        ("outro",     4, "KH",        0.28, 0.38),
    ],
    # New genres
    "rnb": [
        ("intro",   4, "KH",    0.4, 0.5),
        ("verse",   8, "KSHBC", 0.6, 0.7),
        ("prechorus",4,"KSHBCR",0.75,0.85),
        ("chorus",  8, "KSHBCL",0.9, 1.0),
        ("verse2",  8, "KSHBC", 0.6, 0.7),
        ("chorus2", 8, "KSHBCL",0.9, 1.0),
        ("outro",   4, "KH",    0.4, 0.45),
    ],
    "house": [
        ("intro",    4, "KH",     0.5, 0.6),
        ("build",    4, "KSHBC",  0.7, 0.75),
        ("drop",     8, "KSHBCL", 1.0, 1.0),
        ("break",    4, "BC",     0.3, 0.5),
        ("build2",   4, "KSHBCR", 0.8, 0.85),
        ("drop2",    8, "KSHBCL", 1.0, 1.0),
        ("outro",    4, "KH",     0.5, 0.5),
    ],
    "reggaeton": [
        ("intro",    4, "KH",     0.4, 0.6),
        ("verse",    8, "KSHBC",  0.65,0.75),
        ("prechorus",4, "KSHBCR", 0.8, 0.85),
        ("chorus",   8, "KSHBCL", 1.0, 1.0),
        ("verse2",   8, "KSHBC",  0.65,0.75),
        ("chorus2",  8, "KSHBCL", 1.0, 1.0),
        ("outro",    4, "KH",     0.4, 0.5),
    ],
    "hyperpop": [
        ("intro",    2, "KH",     0.5, 0.6),
        ("verse",    4, "KSHBC",  0.7, 0.8),
        ("chorus",   8, "KSHBCL", 1.0, 1.0),
        ("breakdown",2, "BC",     0.3, 0.4),
        ("verse2",   4, "KSHBC",  0.7, 0.8),
        ("chorus2",  8, "KSHBCL", 1.0, 1.0),
        ("outro",    2, "KH",     0.5, 0.5),
    ],
    "orchestral": [
        ("intro",       8, "BC",  0.4, 0.5),
        ("theme",      16, "BCL", 0.7, 0.8),
        ("development", 8, "BC",  0.6, 0.7),
        ("theme2",     16, "BCL", 0.9, 1.0),
        ("outro",       8, "BC",  0.4, 0.45),
    ],
}

# Swing defaults per genre (0.0 = no swing)
_SWING_DEFAULTS: Dict[str, float] = {
    "lofi":       0.55,
    "jazz":       0.65,
    "afrobeats":  0.45,
    "amapiano":   0.40,
    "rnb":        0.35,
}


def _genre_key(genre: str) -> str:
    g = genre.lower().strip()
    if "cinematic" in g:          # cinematic_trap / cinematic hip-hop etc.
        return "cinematic_trap"
    if "orchestral" in g:
        return "orchestral"
    if "hyperpop" in g:
        return "hyperpop"
    if "reggaeton" in g:
        return "reggaeton"
    if "house" in g:
        return "house"
    if "rnb" in g or "r&b" in g or "r'n'b" in g:
        return "rnb"
    for k in ("trap","drill","phonk","afrobeats","amapiano","lofi","jazz","pop","latin"):
        if k in g:
            return k
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# Sound synthesis classes
# ─────────────────────────────────────────────────────────────────────────────

class SynthVoice:
    """Detuned polyBLEP saw + biquad LPF + ADSR envelope.

    ``render_note`` returns a float32 mono buffer of exactly ``n`` samples.
    All heavy lifting goes through NativeKernels; numpy used only for shaping.
    """

    def __init__(self, kern: NativeKernels, sample_rate: int = 44100):
        self.k  = kern
        self.sr = sample_rate

    def render_note(
        self,
        freq: float,
        gate_s: float,
        n: int,
        cutoff: float = 4000.0,
        resonance: float = 0.8,
        detune_cents: float = 12.0,
        n_unison: int = 5,
        attack: float = 0.008,
        decay: float = 0.12,
        sustain: float = 0.65,
        release: float = 0.25,
        drive: float = 1.6,
        amp: float = 1.0,
        glide_from_freq: Optional[float] = None,
    ) -> np.ndarray:
        if n <= 0 or freq <= 0:
            return np.zeros(max(n, 0), dtype=np.float32)

        n_osc = max(1, int(n_unison))
        if n_osc == 1:
            detune_factors = np.array([1.0], dtype=np.float32)
        else:
            cents = np.linspace(-detune_cents, detune_cents, n_osc, dtype=np.float32)
            detune_factors = (2.0 ** (cents / 1200.0)).astype(np.float32)

        freqs_arr = (np.float32(freq) * detune_factors).astype(np.float32)
        # Taper amplitude: centre osc loudest, outer oscs softer
        taper = np.hanning(n_osc + 2)[1:-1].astype(np.float32) + 0.15
        taper /= taper.sum()
        amps_arr = (taper * float(amp)).astype(np.float32)

        # ── Portamento glide (optional) ──────────────────────────────────────
        # Render in 64-sample chunks with linearly interpolating freq, threading
        # saw_wave phases for click-free audio at buffer boundaries.
        if glide_from_freq and glide_from_freq > 0 and glide_from_freq != freq:
            glide_n = min(int(0.040 * self.sr), n)   # 40ms glide window
            out = np.zeros(n, dtype=np.float32)
            phases: Optional[np.ndarray] = None
            chunk = 64
            for ci in range(0, glide_n, chunk):
                cend = min(ci + chunk, glide_n)
                sz   = cend - ci
                t    = ci / max(glide_n - 1, 1)
                f_i  = glide_from_freq + (freq - glide_from_freq) * t
                f_arr = (np.float32(f_i) * detune_factors).astype(np.float32)
                seg, phases = self.k.saw_wave(f_arr, amps_arr, float(self.sr), sz, phases)
                out[ci:cend] = seg
            # body — remaining samples at target freq with phase continuity
            if glide_n < n:
                body, _ = self.k.saw_wave(freqs_arr, amps_arr, float(self.sr),
                                          n - glide_n, phases)
                out[glide_n:] = body
        else:
            out, _ = self.k.saw_wave(freqs_arr, amps_arr, float(self.sr), n)

        # Soft saturation → LPF → ADSR
        out    = self.k.soft_sat(out, drive=float(drive))
        coeffs = self.k.lpf_coeffs(cutoff, resonance, float(self.sr))
        state  = np.zeros(2, dtype=np.float32)
        out    = self.k.biquad(coeffs, out, state)
        env    = self.k.adsr(attack, decay, sustain, release, gate_s, float(self.sr), n)
        self.k.inplace_mul(out, env)
        return out


class DrumKit:
    """Professional layered drum kit — all synthesis via Digital GPU kernels."""

    def __init__(self, kern: NativeKernels, sample_rate: int = 44100):
        self.k  = kern
        self.sr = sample_rate

    # ── Kick ─────────────────────────────────────────────────────────────────
    def kick(self, decay_s: float = 0.25, sub_freq: float = 55.0,
             punch: float = 1.0) -> np.ndarray:
        n = int(decay_s * 2.0 * self.sr)
        # Sub layer: exponential pitch sweep (80Hz → sub_freq)
        sub = self.k.freq_sweep_sin(80.0, 14.0, float(self.sr), n)
        sub_env = self.k.exp_decay(5.0 / self.sr, decay_s * 2.0, n)
        self.k.inplace_mul(sub, sub_env)

        # Body transient: mid sine burst (90Hz)
        body_len = min(n, int(0.06 * self.sr))
        body_buf = np.zeros(body_len, dtype=np.float32)
        body_freqs = np.array([90.0, 180.0], dtype=np.float32)
        body_amps  = np.array([0.6,  0.3 ], dtype=np.float32) * float(punch)
        self.k.additive_synth(body_freqs, body_amps, float(self.sr), body_buf)
        body_env = self.k.exp_decay(25.0 / self.sr, body_len / self.sr, body_len)
        self.k.inplace_mul(body_buf, body_env)

        # Click transient: 2kHz noise burst for attack definition
        click_len = min(n, int(0.008 * self.sr))
        click = self.k.white_noise(0xBABECAFE, click_len)
        click_env = self.k.exp_decay(250.0 / self.sr, click_len / self.sr, click_len)
        self.k.inplace_mul(click, click_env)
        # HP filter the click to remove muddiness
        hpf = self.k.hpf_coeffs(1000.0, 0.7, float(self.sr))
        st  = np.zeros(2, np.float32)
        click = self.k.biquad(hpf, click, st) * 0.3 * float(punch)

        out = sub.copy()
        out[:body_len] += body_buf
        out[:click_len] += click[:len(out[:click_len])]
        # Saturate for punch
        out = self.k.soft_sat(out, drive=2.2)
        out *= 0.85
        return out

    # ── 808 sub bass ─────────────────────────────────────────────────────────
    def bass_808(self, root_freq: float, gate_s: float,
                 slide_from: Optional[float] = None) -> np.ndarray:
        n = int((gate_s + 0.3) * self.sr)
        if slide_from and slide_from != root_freq:
            # Pitch glide from slide_from → root_freq over 60ms
            glide_len = min(n, int(0.06 * self.sr))
            glide = np.linspace(float(slide_from), float(root_freq),
                                glide_len, dtype=np.float32)
            normal = np.full(n - glide_len, float(root_freq), dtype=np.float32)
            freqs_t = np.concatenate([glide, normal])
        else:
            freqs_t = np.full(n, float(root_freq), dtype=np.float32)

        # Time-varying frequency using short additive segments
        out = np.zeros(n, dtype=np.float32)
        chunk = 64
        for start in range(0, n, chunk):
            end   = min(start + chunk, n)
            sz    = end - start
            f_arr = np.array([float(freqs_t[start])], dtype=np.float32)
            a_arr = np.array([1.0], dtype=np.float32)
            self.k.additive_synth(f_arr, a_arr, float(self.sr), out[start:end])

        env = self.k.adsr(0.003, 0.05, 0.85, 0.3, gate_s, float(self.sr), n)
        self.k.inplace_mul(out, env)

        # LPF at 200Hz for sub-only character
        lpf = self.k.lpf_coeffs(200.0, 0.7, float(self.sr))
        st  = np.zeros(2, np.float32)
        out = self.k.biquad(lpf, out, st)
        return out * 0.9

    # ── Snare ─────────────────────────────────────────────────────────────────
    def snare(self, vel: float = 1.0, snappy: float = 0.6) -> np.ndarray:
        n = int(0.20 * self.sr)
        # Tonal body (200Hz + 280Hz)
        tone_buf = np.zeros(n, dtype=np.float32)
        self.k.additive_synth(
            np.array([200.0, 280.0], dtype=np.float32),
            np.array([0.5, 0.3], dtype=np.float32) * float(vel),
            float(self.sr), tone_buf)
        tone_env = self.k.exp_decay(18.0 / self.sr, 0.20, n)
        self.k.inplace_mul(tone_buf, tone_env)

        # Noise layer (snappy)
        noise = self.k.white_noise(0xDEAD, n)
        noise_env = self.k.exp_decay(12.0 / self.sr, 0.20, n)
        self.k.inplace_mul(noise, noise_env)
        # Band-pass noise: HPF at 600Hz, LPF at 8kHz
        hp  = self.k.hpf_coeffs(600.0, 0.7, float(self.sr))
        st  = np.zeros(2, np.float32)
        noise = self.k.biquad(hp, noise, st)
        lp  = self.k.lpf_coeffs(8000.0, 0.5, float(self.sr))
        st2 = np.zeros(2, np.float32)
        noise = self.k.biquad(lp, noise, st2)

        out = tone_buf + noise * float(snappy) * float(vel)
        out = self.k.soft_sat(out, drive=1.8)
        return out * 0.75

    # ── Clap ─────────────────────────────────────────────────────────────────
    def clap(self, vel: float = 1.0) -> np.ndarray:
        n = int(0.12 * self.sr)
        out = np.zeros(n, dtype=np.float32)
        # Three short noise bursts 8ms apart (simulates hand position spread)
        burst_len = int(0.01 * self.sr)
        for offset in (0, int(0.008*self.sr), int(0.016*self.sr)):
            burst = self.k.white_noise(0xCAFE ^ offset, burst_len)
            burst_env = self.k.exp_decay(80.0 / self.sr, burst_len/self.sr, burst_len)
            self.k.inplace_mul(burst, burst_env)
            end = min(offset + burst_len, n)
            out[offset:end] += burst[:end-offset]
        # Band-pass 700Hz–10kHz
        hp = self.k.hpf_coeffs(700.0, 0.8, float(self.sr))
        st = np.zeros(2, np.float32)
        out = self.k.biquad(hp, out, st)
        return out * float(vel) * 0.6

    # ── Hi-hats ───────────────────────────────────────────────────────────────
    def hat_closed(self, vel: float = 1.0) -> np.ndarray:
        n = int(0.025 * self.sr)
        noise = self.k.white_noise(0xF00D, n)
        env   = self.k.exp_decay(150.0 / self.sr, 0.025, n)
        self.k.inplace_mul(noise, env)
        hp = self.k.hpf_coeffs(6000.0, 0.7, float(self.sr))
        st = np.zeros(2, np.float32)
        noise = self.k.biquad(hp, noise, st)
        return noise * float(vel) * 0.80

    def hat_open(self, vel: float = 1.0, decay_s: float = 0.18) -> np.ndarray:
        n = int(decay_s * self.sr)
        noise = self.k.white_noise(0xBEEF, n)
        env   = self.k.exp_decay(8.0 / self.sr, decay_s, n)
        self.k.inplace_mul(noise, env)
        hp = self.k.hpf_coeffs(6000.0, 0.6, float(self.sr))
        st = np.zeros(2, np.float32)
        noise = self.k.biquad(hp, noise, st)
        return noise * float(vel) * 0.85

    # ── Crash cymbal ─────────────────────────────────────────────────────────
    def crash(self, vel: float = 1.0) -> np.ndarray:
        """White noise burst (8ms) + sinusoidal body at 200Hz + exponential decay 1.5s."""
        n = int(1.5 * self.sr)
        # White noise burst (full length, decays over 1.5s)
        noise = self.k.white_noise(0xCCCC, n)
        noise_env = self.k.exp_decay(4.0 / self.sr, 1.5, n)
        self.k.inplace_mul(noise, noise_env)
        # HPF to make it shimmer
        hp = self.k.hpf_coeffs(3000.0, 0.6, float(self.sr))
        st = np.zeros(2, np.float32)
        noise = self.k.biquad(hp, noise, st)

        # Sinusoidal body at 200Hz with same decay
        body_buf = np.zeros(n, dtype=np.float32)
        self.k.additive_synth(
            np.array([200.0, 400.0, 600.0], dtype=np.float32),
            np.array([0.3, 0.15, 0.05], dtype=np.float32) * float(vel),
            float(self.sr), body_buf)
        body_env = self.k.exp_decay(3.0 / self.sr, 1.5, n)
        self.k.inplace_mul(body_buf, body_env)

        out = noise * float(vel) * 0.7 + body_buf
        return out * 0.65


class BassVoice:
    """Sub + mid-range filtered bass — follows the chord progression root."""

    def __init__(self, kern: NativeKernels, sample_rate: int = 44100):
        self.k   = kern
        self.sr  = sample_rate
        self._lpf_state = np.zeros(2, dtype=np.float32)

    def render_note(self, freq: float, gate_s: float, n: int,
                    cutoff: float = 700.0, drive: float = 2.2,
                    amp: float = 1.0) -> np.ndarray:
        if n <= 0 or freq <= 0:
            return np.zeros(max(n, 0), dtype=np.float32)

        # Sub layer: pure sine near the root
        sub_buf = np.zeros(n, dtype=np.float32)
        self.k.additive_synth(
            np.array([freq, freq*2.0], dtype=np.float32),
            np.array([0.70, 0.20], dtype=np.float32) * float(amp),
            float(self.sr), sub_buf)

        # Mid layer: polyBLEP saw, an octave up, filtered
        mid, _ = self.k.saw_wave(
            np.array([freq*2.0, freq*2.0*1.002], dtype=np.float32),
            np.array([0.3, 0.3], dtype=np.float32) * float(amp),
            float(self.sr), n)

        out = sub_buf + mid

        # Drive / saturation for mid-bass growl
        out = self.k.soft_sat(out, drive=float(drive))

        # LPF
        coeffs = self.k.lpf_coeffs(float(cutoff), 0.7, float(self.sr))
        state  = np.zeros(2, dtype=np.float32)
        out    = self.k.biquad(coeffs, out, state)

        # ADSR — punchy attack, decent sustain
        env = self.k.adsr(0.004, 0.06, 0.75, 0.20, gate_s, float(self.sr), n)
        self.k.inplace_mul(out, env)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Effects (reverb, compression, M/S widening) — all never-raise
# ─────────────────────────────────────────────────────────────────────────────

def apply_reverb(x: np.ndarray, kern: NativeKernels, sr: int,
                 room_size: float = 0.55, wet: float = 0.18,
                 seed: int = 42) -> np.ndarray:
    """FFT convolution reverb with a synthetic exponential noise IR.

    The IR is built from kern.white_noise + kern.exp_decay — no scipy.
    Convolution uses numpy FFT (zero-padded, O(N log N)).
    """
    try:
        ir_len = int(sr * room_size * 2.5)
        ir     = kern.white_noise(seed, ir_len).astype(np.float64)
        env    = kern.exp_decay(5.5 / max(room_size, 0.01) / sr,
                                room_size * 2.5, ir_len).astype(np.float64)
        ir    *= env
        # Pre-delay (8ms) and stereo diffusion
        pre   = int(0.008 * sr)
        if pre > 0:
            ir[:pre] *= 0.0
        peak = float(np.max(np.abs(ir)))
        if peak < 1e-9:
            return x
        ir /= peak

        n_fft  = 1 << int(np.ceil(np.log2(len(x) + ir_len)))
        X      = np.fft.rfft(x.astype(np.float64), n=n_fft)
        IR     = np.fft.rfft(ir, n=n_fft)
        wet_s  = np.fft.irfft(X * IR)[:len(x)].astype(np.float32)
        return kern.mix2(x, 1.0 - wet, wet_s, wet)
    except Exception:
        return x   # never-raise


def apply_glitch_transition(
    mix_L: np.ndarray, mix_R: np.ndarray,
    boundary_samples: List[int], bar_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Micro-glitch stutter at section boundaries — futuristic transition flair."""
    try:
        n       = len(mix_L)
        slice_n = max(1, bar_len // 8)
        for pos in boundary_samples:
            src_s = pos - slice_n
            if src_s < 0 or pos + slice_n * 3 >= n:
                continue
            src_L = mix_L[src_s:pos].copy()
            src_R = mix_R[src_s:pos].copy()
            for rep in range(3):
                dst_s = pos + rep * slice_n
                dst_e = dst_s + slice_n
                if dst_e > n:
                    break
                blend = float(0.55 - rep * 0.13)
                mix_L[dst_s:dst_e] = (mix_L[dst_s:dst_e] * (1.0 - blend)
                                      + src_L * blend)
                mix_R[dst_s:dst_e] = (mix_R[dst_s:dst_e] * (1.0 - blend)
                                      + src_R * blend)
    except Exception:
        pass
    return mix_L, mix_R


def apply_compressor(x: np.ndarray, kern: NativeKernels, sr: int,
                     threshold_db: float = -18.0, ratio: float = 4.0,
                     attack_ms: float = 5.0, release_ms: float = 80.0,
                     makeup_db: float = 4.0) -> np.ndarray:
    """RMS bus compressor — glues the mix, adds punch."""
    try:
        makeup = float(10 ** (makeup_db / 20.0))
        thr    = float(10 ** (threshold_db / 20.0))
        win    = max(1, int(0.008 * sr))
        rms_sq = np.convolve(x.astype(np.float64)**2,
                             np.ones(win, np.float64)/win, mode='same')
        rms    = np.sqrt(np.abs(rms_sq)).astype(np.float32)
        gain   = kern.compress_gain(rms, thr, ratio, attack_ms, release_ms, float(sr))
        return x * gain * makeup
    except Exception:
        return x   # never-raise


def apply_ms_width(L: np.ndarray, R: np.ndarray,
                   width: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    """M/S stereo widening — widens the stereo image without phase issues."""
    mid  = (L + R) * 0.5
    side = (L - R) * 0.5
    side *= width
    return (mid + side), (mid - side)


# ─────────────────────────────────────────────────────────────────────────────
# C. Sidechain compression (kick ducking bass/pads)
# ─────────────────────────────────────────────────────────────────────────────

def apply_sidechain(bus: np.ndarray, trigger: np.ndarray, sr: int,
                    attack_ms: float = 2.0, release_ms: float = 80.0,
                    ratio: float = 4.0, threshold: float = 0.3) -> np.ndarray:
    """Duck bus signal when kick trigger exceeds threshold.

    Never-raise. Uses only numpy.
    """
    try:
        bus = np.asarray(bus, dtype=np.float32)
        trigger = np.asarray(trigger, dtype=np.float32)
        n = len(bus)
        if n == 0:
            return bus

        attack_coef  = np.float32(np.exp(-1.0 / max(1, attack_ms  * sr / 1000.0)))
        release_coef = np.float32(np.exp(-1.0 / max(1, release_ms * sr / 1000.0)))
        depth = np.float32(1.0 / ratio)

        gain_env = np.ones(n, dtype=np.float32)
        g = np.float32(1.0)
        tlen = len(trigger)
        for i in range(n):
            trig_val = abs(float(trigger[i])) if i < tlen else 0.0
            if trig_val > threshold:
                target = depth
                coef   = attack_coef
            else:
                target = np.float32(1.0)
                coef   = release_coef
            g = coef * g + (np.float32(1.0) - coef) * target
            gain_env[i] = g

        return bus * gain_env
    except Exception:
        return bus


# ─────────────────────────────────────────────────────────────────────────────
# D. Delay / Echo FX
# ─────────────────────────────────────────────────────────────────────────────

def apply_delay(x: np.ndarray, sr: int, delay_ms: float,
                feedback: float = 0.35, mix: float = 0.25,
                ping_pong: bool = True) -> np.ndarray:
    """Ping-pong delay. x must be stereo-interleaved [L,R,L,R,...] or mono.

    Never-raise. Uses only numpy. Max 6 repeats.
    """
    try:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        delay_samples = max(1, int(delay_ms * sr / 1000.0))
        is_stereo = (x.ndim == 1 and x.size % 2 == 0)
        # Work on mono representation, then handle stereo ping-pong
        if x.ndim == 1 and x.size % 2 == 0:
            # Interleaved stereo
            L = x[0::2].copy()
            R = x[1::2].copy()
        else:
            L = x.copy()
            R = x.copy()

        n = len(L)
        out_L = L.copy()
        out_R = R.copy()

        fb_sig = (L + R) * 0.5  # mono sum for feedback chain
        cur_fb = fb_sig.copy()
        for rep in range(1, 7):  # max 6 repeats
            offset = delay_samples * rep
            if offset >= n:
                break
            level = (feedback ** rep) * mix
            if ping_pong:
                if rep % 2 == 1:
                    # Odd repeats go to right
                    out_R[offset:] += cur_fb[:n - offset] * level
                else:
                    # Even repeats go to left
                    out_L[offset:] += cur_fb[:n - offset] * level
            else:
                out_L[offset:] += cur_fb[:n - offset] * level
                out_R[offset:] += cur_fb[:n - offset] * level

        # Rebuild interleaved
        if x.ndim == 1 and x.size % 2 == 0:
            result = np.empty(x.size, dtype=np.float32)
            result[0::2] = out_L
            result[1::2] = out_R
            return result
        else:
            return (out_L + out_R) * 0.5
    except Exception:
        return x


# ─────────────────────────────────────────────────────────────────────────────
# E. Parametric EQ (RBJ biquad — numpy only)
# ─────────────────────────────────────────────────────────────────────────────

def _rbj_biquad_coeffs(band: dict, sr: int) -> Optional[Tuple[float,float,float,float,float]]:
    """Compute biquad coefficients from an RBJ-style band dict.

    Returns (b0, b1, b2, a1, a2) normalised by a0, or None on error.
    """
    try:
        btype  = band.get("type", "peak")
        freq   = float(band.get("freq", 1000.0))
        gain_db= float(band.get("gain_db", 0.0))
        q      = float(band.get("q", 1.0))
        q      = max(0.01, q)
        freq   = max(1.0, min(freq, sr * 0.499))

        w0     = 2.0 * np.pi * freq / sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha  = sin_w0 / (2.0 * q)
        A      = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain

        if btype == "hp":
            b0 =  (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 =  (1.0 + cos_w0) / 2.0
            a0 =   1.0 + alpha
            a1 =  -2.0 * cos_w0
            a2 =   1.0 - alpha
        elif btype == "lp":
            b0 =  (1.0 - cos_w0) / 2.0
            b1 =   1.0 - cos_w0
            b2 =  (1.0 - cos_w0) / 2.0
            a0 =   1.0 + alpha
            a1 =  -2.0 * cos_w0
            a2 =   1.0 - alpha
        elif btype == "peak":
            b0 =  1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 =  1.0 - alpha * A
            a0 =  1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 =  1.0 - alpha / A
        elif btype == "shelf_hi":
            sqA   = np.sqrt(A)
            alpha2= sin_w0 / 2.0 * np.sqrt((A + 1.0/A) * (1.0/q - 1.0) + 2.0)
            b0 =  A * ((A+1) + (A-1)*cos_w0 + 2*sqA*alpha2)
            b1 = -2*A* ((A-1) + (A+1)*cos_w0)
            b2 =  A * ((A+1) + (A-1)*cos_w0 - 2*sqA*alpha2)
            a0 =       (A+1) - (A-1)*cos_w0 + 2*sqA*alpha2
            a1 =  2  * ((A-1) - (A+1)*cos_w0)
            a2 =       (A+1) - (A-1)*cos_w0 - 2*sqA*alpha2
        elif btype == "shelf_lo":
            sqA   = np.sqrt(A)
            alpha2= sin_w0 / 2.0 * np.sqrt((A + 1.0/A) * (1.0/q - 1.0) + 2.0)
            b0 =  A * ((A+1) - (A-1)*cos_w0 + 2*sqA*alpha2)
            b1 =  2*A*((A-1) - (A+1)*cos_w0)
            b2 =  A * ((A+1) - (A-1)*cos_w0 - 2*sqA*alpha2)
            a0 =       (A+1) + (A-1)*cos_w0 + 2*sqA*alpha2
            a1 = -2  * ((A-1) + (A+1)*cos_w0)
            a2 =       (A+1) + (A-1)*cos_w0 - 2*sqA*alpha2
        else:
            return None

        inv_a0 = 1.0 / a0
        return (b0*inv_a0, b1*inv_a0, b2*inv_a0, a1*inv_a0, a2*inv_a0)
    except Exception:
        return None


def _apply_biquad_np(x: np.ndarray, b0: float, b1: float, b2: float,
                     a1: float, a2: float) -> np.ndarray:
    """Transposed-form II biquad filter, numpy fallback."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    s1, s2 = 0.0, 0.0
    for i in range(len(x)):
        v  = b0 * x[i] + s1
        s1 = b1 * x[i] - a1 * v + s2
        s2 = b2 * x[i] - a2 * v
        y[i] = v
    return y.astype(np.float32)


def apply_eq(x: np.ndarray, sr: int, bands: list) -> np.ndarray:
    """Apply a parametric EQ chain. Never-raise. numpy only.

    Each band dict: {type: hp/lp/peak/shelf_hi/shelf_lo, freq, gain_db, q}
    """
    try:
        x = np.asarray(x, dtype=np.float32)
        out = x.copy()
        for band in bands:
            coeffs = _rbj_biquad_coeffs(band, sr)
            if coeffs is None:
                continue
            b0, b1, b2, a1, a2 = coeffs
            out = _apply_biquad_np(out, b0, b1, b2, a1, a2)
        return out
    except Exception:
        return x


# ─────────────────────────────────────────────────────────────────────────────
# F. Multi-band compressor
# ─────────────────────────────────────────────────────────────────────────────

def _rms_compress_np(x: np.ndarray, sr: int, threshold: float, ratio: float,
                     attack_ms: float, release_ms: float) -> np.ndarray:
    """Simple RMS compressor, numpy only, returns gain-applied signal."""
    try:
        win = max(1, int(0.008 * sr))
        rms_sq = np.convolve(x.astype(np.float64)**2,
                             np.ones(win) / win, mode='same')
        rms = np.sqrt(np.abs(rms_sq)).astype(np.float32)
        n = len(x)
        attack_coef  = float(np.exp(-1.0 / max(1, attack_ms  * sr / 1000.0)))
        release_coef = float(np.exp(-1.0 / max(1, release_ms * sr / 1000.0)))
        gain = np.ones(n, dtype=np.float32)
        g = 1.0
        ratio_inv = 1.0 / ratio
        for i in range(n):
            r = float(rms[i])
            if r > threshold and r > 1e-10:
                # Gain reduction
                target = (threshold / r) * (1.0 - ratio_inv) + ratio_inv
            else:
                target = 1.0
            coef = attack_coef if target < g else release_coef
            g = coef * g + (1.0 - coef) * target
            gain[i] = max(0.0, min(1.0, g))
        return x * gain
    except Exception:
        return x


def apply_multiband_compressor(L: np.ndarray, R: np.ndarray,
                                sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """3-band compressor: sub (<120Hz), mid (120Hz–5kHz), hi (>5kHz).

    Uses biquad LPF/HPF crossover (numpy biquad). Never-raise.
    Returns (L_out, R_out).
    """
    try:
        def _lpf(sig, freq):
            c = _rbj_biquad_coeffs({"type": "lp", "freq": freq, "gain_db": 0, "q": 0.707}, sr)
            if c is None:
                return sig
            return _apply_biquad_np(sig, *c)

        def _hpf(sig, freq):
            c = _rbj_biquad_coeffs({"type": "hp", "freq": freq, "gain_db": 0, "q": 0.707}, sr)
            if c is None:
                return sig
            return _apply_biquad_np(sig, *c)

        results = []
        for ch in (L, R):
            ch = np.asarray(ch, dtype=np.float32)
            # Split into 3 bands
            sub = _lpf(ch, 120.0)
            hi  = _hpf(ch, 5000.0)
            mid_raw = ch - sub - hi  # mid by subtraction (complementary)

            # Compress each band independently
            sub_c = _rms_compress_np(sub,     sr, threshold=0.25, ratio=3.0,
                                     attack_ms=10.0, release_ms=200.0)
            mid_c = _rms_compress_np(mid_raw, sr, threshold=0.30, ratio=2.0,
                                     attack_ms=5.0,  release_ms=100.0)
            hi_c  = _rms_compress_np(hi,      sr, threshold=0.35, ratio=1.5,
                                     attack_ms=2.0,  release_ms=50.0)

            results.append((sub_c + mid_c + hi_c).astype(np.float32))

        return results[0], results[1]
    except Exception:
        return L, R


# ─────────────────────────────────────────────────────────────────────────────
# G. Dithering
# ─────────────────────────────────────────────────────────────────────────────

def apply_dither(x: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """TPDF dither: add triangular PDF noise at ±1 LSB before quantisation.

    Two uniform random values in ±0.5 LSB, summed → triangular distribution.
    """
    try:
        x = np.asarray(x, dtype=np.float32)
        lsb = 1.0 / (2 ** bit_depth)
        # Two independent uniform random draws in [-0.5 LSB, +0.5 LSB]
        rng = np.random.default_rng(0xD17E4)
        u1 = rng.uniform(-0.5 * lsb, 0.5 * lsb, size=x.shape).astype(np.float32)
        u2 = rng.uniform(-0.5 * lsb, 0.5 * lsb, size=x.shape).astype(np.float32)
        return x + u1 + u2
    except Exception:
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Arrangement engine
# ─────────────────────────────────────────────────────────────────────────────

def _scale_degree_freq(root_freq: float, scale: List[int],
                       degree: int, octave_offset: int = 0) -> float:
    semis = scale[degree % len(scale)] + octave_offset * 12
    return root_freq * (2.0 ** (semis / 12.0))


def _chord_freqs(root_freq: float, chord_semi: int,
                 voicing_key: str) -> List[float]:
    """Return a list of frequencies for a chord voicing."""
    chord_root = root_freq * (2.0 ** (chord_semi / 12.0))
    intervals  = VOICINGS.get(voicing_key, VOICINGS["min7"])
    return [chord_root * (2.0 ** (s / 12.0)) for s in intervals]


def _sections_for(genre_key: str, duration_sec: float,
                  bpm: float) -> List[Tuple[str, int, str, float, float]]:
    """Return sections scaled so total bars fit in duration_sec."""
    secs_per_bar = 60.0 / bpm * 4.0
    target_bars  = int(duration_sec / secs_per_bar)
    template     = _SECTIONS.get(genre_key, _SECTIONS["default"])
    total_template = sum(s[1] for s in template)
    scale_factor   = target_bars / max(total_template, 1)
    result = []
    for name, bars, elems, filt, energy in template:
        scaled = max(2, round(bars * scale_factor))
        result.append((name, scaled, elems, filt, energy))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# K. Stem renderer (refactored from render_full_track)
# ─────────────────────────────────────────────────────────────────────────────

def render_stems(
    job_id: str,
    bpm: float,
    key: str = "C minor",
    duration_sec: float = 180.0,
    genre: str = "",
    mood: str = "",
    sample_rate: int = 44100,
) -> Dict[str, np.ndarray]:
    """Render all buses separately and return them as a dict.

    Keys: "drums", "bass", "pads", "lead", "fx"
    Each value is a stereo-interleaved float32 array (not normalised yet).
    """
    kern = get_native_kernels()
    rng  = np.random.default_rng(
        int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16))

    # ── Key / scale ──────────────────────────────────────────────────────────
    parts     = (key or "C minor").split()
    root_name = parts[0] if parts else "C"
    is_minor  = len(parts) > 1 and parts[1].lower().startswith("min")
    root_freq = 220.0 * (2.0 ** (_NOTE_SEMI.get(root_name, 0) / 12.0))
    scale     = MINOR_SCALE if is_minor else MAJOR_SCALE
    _genre    = _genre_key(genre)
    _mood     = (mood or "").lower()

    bpm_f     = max(60.0, min(float(bpm), 200.0))
    step_sec  = 60.0 / bpm_f / 4.0
    step_len  = int(step_sec * sample_rate)
    bar_len   = step_len * 16
    n_total   = int(duration_sec * sample_rate)
    n_total   = max(n_total, bar_len * 8)

    # Swing: compute swing offset per step
    swing = _SWING_DEFAULTS.get(_genre, 0.0)

    def _swing_offset(step: int) -> int:
        """Return sample offset for odd steps when swing > 0."""
        if swing > 0.0 and step % 2 == 1:
            return int(swing * (step_len / 2.0))
        return 0

    # ── Chord / bass / drum patterns ─────────────────────────────────────────
    prog = CHORD_PROGS.get(_genre, CHORD_PROGS["default"])
    chord_bars = len(prog)
    dp   = _D.get(_genre, _D["default"])
    fill_dp = _FILLS.get(_genre, None)
    bp   = _BASS.get(_genre, _BASS["default"])

    # ── Arrangement sections ─────────────────────────────────────────────────
    sections = _sections_for(_genre, duration_sec, bpm_f)

    # ── Instrument constructors ──────────────────────────────────────────────
    drums  = DrumKit(kern, sample_rate)
    bass_v = BassVoice(kern, sample_rate)
    synth  = SynthVoice(kern, sample_rate)

    # ── Pre-render drum one-shots ─────────────────────────────────────────────
    if _genre == "cinematic_trap":
        kick_buf  = drums.kick(decay_s=0.35, sub_freq=40.0, punch=1.3)
        snare_buf = drums.snare(vel=1.0, snappy=0.75)
    elif _genre in ("trap", "drill"):
        kick_buf  = drums.kick(decay_s=0.30, sub_freq=45.0, punch=1.1)
        snare_buf = drums.snare(vel=1.0, snappy=0.65)
    elif _genre == "phonk":
        kick_buf  = drums.kick(decay_s=0.35, sub_freq=40.0, punch=1.2)
        snare_buf = drums.snare(vel=0.9, snappy=0.5)
    elif _genre in ("afrobeats", "amapiano"):
        kick_buf  = drums.kick(decay_s=0.20, sub_freq=55.0, punch=0.9)
        snare_buf = drums.snare(vel=0.85, snappy=0.4)
    elif _genre in ("lofi", "jazz", "rnb"):
        kick_buf  = drums.kick(decay_s=0.15, sub_freq=65.0, punch=0.7)
        snare_buf = drums.snare(vel=0.7, snappy=0.3)
    elif _genre == "house":
        kick_buf  = drums.kick(decay_s=0.22, sub_freq=50.0, punch=1.0)
        snare_buf = drums.snare(vel=0.85, snappy=0.5)
    elif _genre == "hyperpop":
        kick_buf  = drums.kick(decay_s=0.18, sub_freq=55.0, punch=1.2)
        snare_buf = drums.snare(vel=1.0, snappy=0.9)
    elif _genre == "orchestral":
        kick_buf  = np.zeros(int(0.01 * sample_rate), dtype=np.float32)
        snare_buf = np.zeros(int(0.01 * sample_rate), dtype=np.float32)
    else:
        kick_buf  = drums.kick(decay_s=0.22, sub_freq=50.0, punch=1.0)
        snare_buf = drums.snare(vel=0.9, snappy=0.5)

    clap_buf   = drums.clap(vel=1.0)
    hat_c_buf  = drums.hat_closed(vel=1.0)
    hat_o_buf  = drums.hat_open(vel=1.0)
    crash_buf  = drums.crash(vel=1.0)

    # ── Stem buses (stereo, flat) ─────────────────────────────────────────────
    drum_L  = np.zeros(n_total, dtype=np.float32)
    drum_R  = np.zeros(n_total, dtype=np.float32)
    bass_L  = np.zeros(n_total, dtype=np.float32)
    bass_R  = np.zeros(n_total, dtype=np.float32)
    pad_L   = np.zeros(n_total, dtype=np.float32)
    pad_R   = np.zeros(n_total, dtype=np.float32)
    lead_L  = np.zeros(n_total, dtype=np.float32)
    lead_R  = np.zeros(n_total, dtype=np.float32)
    fx_L    = np.zeros(n_total, dtype=np.float32)
    fx_R    = np.zeros(n_total, dtype=np.float32)

    def _stamp_bus(bus_L, bus_R, buf: np.ndarray,
                   pos: int, gain_L: float, gain_R: float) -> None:
        if pos >= n_total or buf is None or buf.size == 0:
            return
        end = min(pos + len(buf), n_total)
        sz  = end - pos
        bus_L[pos:end] += buf[:sz] * gain_L
        bus_R[pos:end] += buf[:sz] * gain_R

    # ── Build section schedule ────────────────────────────────────────────────
    schedule = []
    cur = 0
    for sec in sections:
        schedule.append((cur, cur + sec[1], sec))
        cur += sec[1]
    total_bars = cur

    boundary_samples: List[int] = [
        s * bar_len for (s, e, _sec) in schedule[1:] if s > 0
    ]

    def _section_at(bar: int):
        for (s, e, sec) in schedule:
            if s <= bar < e:
                return sec
        return sections[-1]

    prev_lead_freq: Optional[float] = None
    # Track kick mono for sidechain (needed after full render)
    kick_mono = np.zeros(n_total, dtype=np.float32)

    for bar in range(total_bars):
        bar_pos = bar * bar_len
        if bar_pos >= n_total:
            break

        sec      = _section_at(bar)
        _, _, elems, filter_pct, energy = sec
        chord_semi, chord_quality = prog[bar % chord_bars]

        has_kick        = "K" in elems
        has_snare       = "S" in elems
        has_hat         = "H" in elems
        has_808         = "8" in elems
        has_bass        = "B" in elems
        has_chord       = "C" in elems
        has_lead        = "L" in elems
        has_riser       = "R" in elems
        has_triplet_hat = "T" in elems

        cutoff_hz = 600.0 + (8000.0 - 600.0) * filter_pct

        # Determine whether this is a fill bar (bar 4 of every 4-bar phrase, 0-indexed = bar 3)
        phrase_pos = bar % 4
        use_fill   = (phrase_pos == 3) and (fill_dp is not None)
        active_dp  = fill_dp if use_fill else dp

        # ── Drum grid ─────────────────────────────────────────────────────────
        for step in range(16):
            # Swing offset for odd steps
            swing_off = _swing_offset(step)
            step_pos  = bar_pos + step * step_len + swing_off

            # Deterministic per-step velocity variation
            step_hash = int(hashlib.md5(f"{job_id}{step}".encode()).hexdigest()[:8], 16)
            vel_rng_kick  = np.random.default_rng(
                int(hashlib.md5(f"{job_id}kick{step}".encode()).hexdigest()[:8], 16) % 2**32)
            vel_rng_snare = np.random.default_rng(
                int(hashlib.md5(f"{job_id}snare{step}".encode()).hexdigest()[:8], 16) % 2**32)
            vel_rng_clap  = np.random.default_rng(
                int(hashlib.md5(f"{job_id}clap{step}".encode()).hexdigest()[:8], 16) % 2**32)
            vel_rng_hat   = np.random.default_rng(
                int(hashlib.md5(f"{job_id}hat{step}".encode()).hexdigest()[:8], 16) % 2**32)

            kick_vel  = float(vel_rng_kick.uniform(0.82, 1.0))
            snare_vel = float(vel_rng_snare.uniform(0.82, 1.0))
            clap_vel  = float(vel_rng_clap.uniform(0.82, 1.0))
            hat_vel   = float(vel_rng_hat.uniform(0.82, 1.0))

            if has_kick and active_dp["kick"][step]:
                gain = energy * 0.85 * kick_vel
                _stamp_bus(drum_L, drum_R, kick_buf, step_pos, gain, gain)
                # Record kick mono for sidechain
                if step_pos < n_total:
                    end_k = min(step_pos + len(kick_buf), n_total)
                    sz_k  = end_k - step_pos
                    kick_mono[step_pos:end_k] += kick_buf[:sz_k] * gain

            if has_snare and active_dp["snare"][step]:
                gain = energy * snare_vel * 0.75
                _stamp_bus(drum_L, drum_R, snare_buf, step_pos,
                           gain * 0.95, gain * 0.75)
                if step in (4, 12):
                    cpos = step_pos + int(0.002 * sample_rate)
                    _stamp_bus(drum_L, drum_R, clap_buf, cpos,
                               gain * 0.50 * clap_vel, gain * 0.70 * clap_vel)

            if has_hat:
                if active_dp["hat_o"][step]:
                    gain = energy * hat_vel * 0.55
                    _stamp_bus(drum_L, drum_R, hat_o_buf, step_pos, gain, gain)
                elif active_dp["hat_c"][step]:
                    gain = energy * hat_vel * 0.45
                    _stamp_bus(drum_L, drum_R, hat_c_buf, step_pos,
                               gain * 0.7, gain)

            # Crash at step 0 of fill bar (the next bar's start — place at step_pos of step 0 next bar)
            if use_fill and step == 0 and has_kick:
                next_bar_pos = (bar + 1) * bar_len
                if next_bar_pos < n_total:
                    crash_gain = energy * 0.7
                    _stamp_bus(drum_L, drum_R, crash_buf, next_bar_pos,
                               crash_gain, crash_gain * 0.9)

            # Triplet hi-hat rolls (T flag)
            if has_triplet_hat and step % 4 == 0:
                tri_step = int(step_len * 4 / 3)
                for tri in (1, 2):
                    tri_pos = step_pos + tri * tri_step
                    tri_vel = energy * (0.38 + rng.random() * 0.15)
                    pan = 0.55 + (tri % 2) * 0.35
                    _stamp_bus(drum_L, drum_R, hat_c_buf, tri_pos,
                               tri_vel * hat_vel * (1.0 - pan * 0.3),
                               tri_vel * hat_vel * pan)

        # ── 808 bass ──────────────────────────────────────────────────────────
        if has_808 and active_dp["808"][0]:
            chord_root_freq = root_freq * (2.0 ** (chord_semi / 12.0)) * 0.5
            for step in range(16):
                if active_dp["808"][step]:
                    swing_off = _swing_offset(step)
                    gate_s = step_sec * (14 if _genre == "cinematic_trap" else 6)
                    buf_808 = drums.bass_808(chord_root_freq * 0.5, gate_s)
                    step_pos = bar_pos + step * step_len + swing_off
                    gain = energy * 0.80
                    _stamp_bus(bass_L, bass_R, buf_808, step_pos, gain, gain)

        # ── Bass line ─────────────────────────────────────────────────────────
        if has_bass:
            chord_root_freq = root_freq * (2.0 ** (chord_semi / 12.0))
            step = 0
            while step < 16:
                bp_note, bp_gate = bp[step]
                if bp_note >= -90 and bp_gate > 0:
                    note_freq = chord_root_freq * (2.0 ** (bp_note / 12.0))
                    gate_s  = step_sec * bp_gate
                    note_n  = int(gate_s * sample_rate) + int(0.2 * sample_rate)
                    note_cut = max(300.0, cutoff_hz * 0.5)
                    buf_b   = bass_v.render_note(
                        note_freq * 0.5, gate_s, note_n,
                        cutoff=note_cut, drive=2.0, amp=energy * 0.8)
                    swing_off = _swing_offset(step)
                    step_pos = bar_pos + step * step_len + swing_off
                    _stamp_bus(bass_L, bass_R, buf_b, step_pos, 0.9, 0.9)
                    step += max(1, bp_gate)
                else:
                    step += 1

        # ── Chord pad ─────────────────────────────────────────────────────────
        if has_chord:
            c_freqs = _chord_freqs(root_freq, chord_semi, chord_quality)
            gate_s  = min(bar_len / sample_rate, 2.5)
            note_n  = bar_len + int(0.4 * sample_rate)
            n_uni   = 5 if energy >= 0.85 else 3
            pad_cut = cutoff_hz * 0.9

            pad_buf = np.zeros(note_n, dtype=np.float32)
            for v_idx, freq in enumerate(c_freqs):
                oct_off = 0 if v_idx < 2 else 1
                v_buf = synth.render_note(
                    freq * (2**oct_off), gate_s, note_n,
                    cutoff=pad_cut, resonance=0.65,
                    detune_cents=12.0, n_unison=n_uni,
                    attack=0.08, decay=0.3, sustain=0.6, release=0.4,
                    drive=1.3, amp=0.28)
                pad_buf += v_buf

            lp_ch = kern.lpf_coeffs(pad_cut * 0.7, 0.7, float(sample_rate))
            st_ch = np.zeros(2, np.float32)
            pad_ch_L = kern.biquad(lp_ch, pad_buf.copy(), st_ch)
            pad_ch_R = pad_buf

            gain_ch = energy * 0.50
            _stamp_bus(pad_L, pad_R, pad_ch_L, bar_pos, gain_ch, 0.0)
            _stamp_bus(pad_L, pad_R, pad_ch_R, bar_pos, 0.0, gain_ch)

        # ── Lead / arp ────────────────────────────────────────────────────────
        if has_lead:
            chord_root_freq = root_freq * (2.0 ** (chord_semi / 12.0))
            n_lead_notes = 8
            note_dur     = step_sec * 2

            arp_freqs = _chord_freqs(root_freq * 2.0, chord_semi, chord_quality)
            arp_freqs.append(chord_root_freq * 2.0 * (2 ** (10/12)))

            # Section name detection for delay-in-chorus
            sec_name = sec[0].lower() if sec and len(sec) > 0 else ""
            is_hook_section = any(x in sec_name for x in
                                  ("chorus", "hook", "drop"))

            for n_i in range(min(n_lead_notes, len(arp_freqs) * 2)):
                freq_l  = arp_freqs[n_i % len(arp_freqs)]
                gate_s  = note_dur * 0.85
                note_n  = int((note_dur + 0.12) * sample_rate)
                lead_buf = synth.render_note(
                    freq_l, gate_s, note_n,
                    cutoff=min(cutoff_hz * 1.2, 12000.0), resonance=0.9,
                    detune_cents=5.0, n_unison=3,
                    attack=0.004, decay=0.08, sustain=0.45, release=0.12,
                    drive=1.8, amp=0.4,
                    glide_from_freq=prev_lead_freq)
                prev_lead_freq = freq_l
                lpos = bar_pos + int(n_i * step_len * 2)
                pan_l = 0.9 if n_i % 2 == 0 else 0.6
                pan_r = 0.6 if n_i % 2 == 0 else 0.9
                _stamp_bus(lead_L, lead_R, lead_buf, lpos,
                           pan_l * energy, pan_r * energy)

            # Apply delay to hook/chorus sections
            if is_hook_section:
                delay_ms = 60000.0 / bpm_f * 0.75
                # Apply delay to the portion of lead we just wrote in this bar
                seg_start = bar_pos
                seg_end   = min(bar_pos + bar_len + int(0.2 * sample_rate), n_total)
                seg_L = lead_L[seg_start:seg_end].copy()
                seg_R = lead_R[seg_start:seg_end].copy()
                # Build interleaved for apply_delay
                interleaved = np.empty(len(seg_L) * 2, dtype=np.float32)
                interleaved[0::2] = seg_L
                interleaved[1::2] = seg_R
                delayed = apply_delay(interleaved, sample_rate, delay_ms,
                                      feedback=0.35, mix=0.25, ping_pong=True)
                lead_L[seg_start:seg_end] = delayed[0::2]
                lead_R[seg_start:seg_end] = delayed[1::2]

        # ── Riser ─────────────────────────────────────────────────────────────
        if has_riser:
            riser_n   = bar_len
            rise_noise = kern.white_noise(0xBEAD ^ bar, riser_n)
            rise_env   = np.linspace(0.0, 1.0, riser_n, dtype=np.float32)
            kern.inplace_mul(rise_noise, rise_env)
            n_chunks = 8
            chunk_sz = riser_n // n_chunks
            for ci in range(n_chunks):
                frac   = (ci + 1) / n_chunks
                hp_cut = 2000.0 + (10000.0 - 2000.0) * frac
                hp_c   = kern.hpf_coeffs(hp_cut, 0.6, float(sample_rate))
                st_r   = np.zeros(2, np.float32)
                s, e   = ci * chunk_sz, (ci + 1) * chunk_sz
                rise_noise[s:e] = kern.biquad(hp_c, rise_noise[s:e].copy(), st_r)
            gain_r = energy * 0.18
            _stamp_bus(fx_L, fx_R, rise_noise, bar_pos, gain_r, gain_r)

    # ── Per-bus EQ ───────────────────────────────────────────────────────────
    lead_eq_bands = [{"type": "hp", "freq": 200.0, "gain_db": 0.0, "q": 0.7}]
    pad_eq_bands  = [{"type": "hp", "freq": 150.0, "gain_db": 0.0, "q": 0.7}]
    lead_L = apply_eq(lead_L, sample_rate, lead_eq_bands)
    lead_R = apply_eq(lead_R, sample_rate, lead_eq_bands)
    pad_L  = apply_eq(pad_L,  sample_rate, pad_eq_bands)
    pad_R  = apply_eq(pad_R,  sample_rate, pad_eq_bands)

    # ── Sidechain: duck pads + lead with kick ─────────────────────────────────
    pad_L = apply_sidechain(pad_L, kick_mono, sample_rate,
                             attack_ms=2.0, release_ms=80.0,
                             ratio=4.0, threshold=0.3)
    pad_R = apply_sidechain(pad_R, kick_mono, sample_rate,
                             attack_ms=2.0, release_ms=80.0,
                             ratio=4.0, threshold=0.3)
    lead_L = apply_sidechain(lead_L, kick_mono, sample_rate,
                              attack_ms=2.0, release_ms=80.0,
                              ratio=3.0, threshold=0.3)
    lead_R = apply_sidechain(lead_R, kick_mono, sample_rate,
                              attack_ms=2.0, release_ms=80.0,
                              ratio=3.0, threshold=0.3)

    # ── Interleave each stem ──────────────────────────────────────────────────
    def _interleave(L, R):
        s = np.empty(n_total * 2, dtype=np.float32)
        s[0::2] = L
        s[1::2] = R
        return s

    # Public audio buses only — no internal scheduler keys
    return {
        "drums": _interleave(drum_L, drum_R),
        "bass":  _interleave(bass_L, bass_R),
        "pads":  _interleave(pad_L,  pad_R),
        "lead":  _interleave(lead_L, lead_R),
        "fx":    _interleave(fx_L,   fx_R),
    }


def _render_stems_internal(
    job_id: str,
    bpm: float,
    key: str = "C minor",
    duration_sec: float = 180.0,
    genre: str = "",
    mood: str = "",
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """Internal variant of render_stems that also returns scheduler metadata.

    render_full_track uses this to access boundary_samples, bar_len, etc.
    External callers should use render_stems() which returns audio buses only.
    """
    kern = get_native_kernels()
    rng  = np.random.default_rng(
        int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16))

    parts     = (key or "C minor").split()
    root_name = parts[0] if parts else "C"
    is_minor  = len(parts) > 1 and parts[1].lower().startswith("min")
    root_freq = 220.0 * (2.0 ** (_NOTE_SEMI.get(root_name, 0) / 12.0))
    scale     = MINOR_SCALE if is_minor else MAJOR_SCALE
    _genre    = _genre_key(genre)

    # Delegate to the full render_stems implementation via a shim that
    # re-exposes internal scheduler data we stripped from the public API.
    # We re-call the internal render loop directly here to avoid code dup.
    stems = render_stems(job_id, bpm, key, duration_sec, genre, mood, sample_rate)

    # Re-derive scheduler constants (fast, no audio math)
    bpm_f    = max(60.0, min(float(bpm), 200.0))
    step_sec = 60.0 / bpm_f / 4.0
    step_len = int(step_sec * sample_rate)
    bar_len  = step_len * 16
    sections = _sections_for(_genre, duration_sec, bpm_f)
    # IMPORTANT: n_total must match what render_stems actually produced.
    # render_stems uses max(int(duration_sec * sample_rate), bar_len * 8).
    # Re-deriving from sections yields a DIFFERENT value at low BPMs, causing
    # a broadcast error in render_full_track when it tries to write mix_L
    # (stems-length) into a stereo buffer sized from this mismatched n_total.
    # Use the actual rendered length from the stems to guarantee consistency.
    n_total  = len(stems["drums"]) // 2  # stems are interleaved stereo
    if n_total == 0:
        n_total = int(duration_sec * sample_rate)

    # Boundary samples: bar index where each section starts
    boundary_samples: List[int] = []
    bar = 0
    for sec in sections:
        boundary_samples.append(bar * bar_len)
        bar += int(sec[1])

    # Kick mono: extract from drums bus (L channel)
    kick_mono = stems["drums"][0::2].copy()

    return {
        **stems,
        "_boundary_samples": boundary_samples,
        "_bar_len":          bar_len,
        "_genre":            _genre,
        "_n_total":          n_total,
        "_kick_mono":        kick_mono,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_full_track(
    job_id: str,
    bpm: float,
    key: str = "C minor",
    duration_sec: float = 180.0,
    genre: str = "",
    mood: str = "",
    sample_rate: int = 44100,
) -> np.ndarray:
    """Render a full professional stereo track — every compute path on Digital GPU.

    Returns float32 stereo-interleaved array [L0,R0,L1,R1,...] normalised to
    [-1, 1], ready for direct conversion to 16-bit PCM.
    """
    kern = get_native_kernels()

    # Render individual buses via internal variant (includes scheduler metadata)
    stems = _render_stems_internal(job_id, bpm, key, duration_sec, genre, mood, sample_rate)

    boundary_samples = stems["_boundary_samples"]
    bar_len          = stems["_bar_len"]
    _genre           = stems["_genre"]
    n_total          = stems["_n_total"]

    # Sum all stems
    mix_interleaved = (
        stems["drums"] +
        stems["bass"]  +
        stems["pads"]  +
        stems["lead"]  +
        stems["fx"]
    )

    mix_L = mix_interleaved[0::2].copy()
    mix_R = mix_interleaved[1::2].copy()

    # ── Post-processing: reverb ───────────────────────────────────────────────
    mix_L = apply_reverb(mix_L, kern, sample_rate, room_size=0.45, wet=0.12,
                         seed=int(hashlib.sha256(job_id.encode()).hexdigest()[8:12], 16))
    mix_R = apply_reverb(mix_R, kern, sample_rate, room_size=0.52, wet=0.12,
                         seed=int(hashlib.sha256(job_id.encode()).hexdigest()[12:16], 16))

    # ── Multi-band compression (replaces single-band) ─────────────────────────
    mix_L, mix_R = apply_multiband_compressor(mix_L, mix_R, sample_rate)

    # ── Master EQ chain ───────────────────────────────────────────────────────
    master_eq_bands = [
        {"type": "hp",      "freq":    30.0, "gain_db":  0.0, "q": 0.7},
        {"type": "peak",    "freq":   300.0, "gain_db": -3.0, "q": 1.4},
        {"type": "shelf_hi","freq": 10000.0, "gain_db":  1.5, "q": 0.7},
    ]
    mix_L = apply_eq(mix_L, sample_rate, master_eq_bands)
    mix_R = apply_eq(mix_R, sample_rate, master_eq_bands)

    # ── M/S stereo widening ───────────────────────────────────────────────────
    ms_width = 1.6 if _genre == "cinematic_trap" else 1.3
    mix_L, mix_R = apply_ms_width(mix_L, mix_R, width=ms_width)

    # ── Micro-glitch transitions (cinematic_trap only) ────────────────────────
    if _genre == "cinematic_trap" and boundary_samples:
        mix_L, mix_R = apply_glitch_transition(mix_L, mix_R, boundary_samples, bar_len)

    # ── Fade-in / fade-out ────────────────────────────────────────────────────
    fade_in  = int(0.5 * sample_rate)
    fade_out = int(2.0 * sample_rate)
    if fade_in > 0:
        fi = np.linspace(0, 1, fade_in, dtype=np.float32)
        mix_L[:fade_in] *= fi
        mix_R[:fade_in] *= fi
    if fade_out > 0 and fade_out < n_total:
        fo = np.linspace(1, 0, fade_out, dtype=np.float32)
        mix_L[-fade_out:] *= fo
        mix_R[-fade_out:] *= fo

    # ── Soft limiter ─────────────────────────────────────────────────────────
    mix_L = kern.soft_limit(mix_L)
    mix_R = kern.soft_limit(mix_R)

    # ── Normalise ─────────────────────────────────────────────────────────────
    peak = max(float(np.max(np.abs(mix_L))), float(np.max(np.abs(mix_R))), 1e-8)
    mix_L = (mix_L / peak * 0.92).astype(np.float32)
    mix_R = (mix_R / peak * 0.92).astype(np.float32)

    # Interleave stereo
    stereo = np.empty(n_total * 2, dtype=np.float32)
    stereo[0::2] = mix_L
    stereo[1::2] = mix_R
    return stereo


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases + WAV I/O
# ─────────────────────────────────────────────────────────────────────────────

def render_audio_clip(job_id: str, bpm: float, key: str = "C minor",
                      duration_sec: float = 30.0, genre: str = "",
                      mood: str = "", sample_rate: int = 44100) -> np.ndarray:
    """Alias for render_full_track — backward compatible."""
    return render_full_track(job_id, bpm, key, duration_sec, genre, mood, sample_rate)


def write_wav(path: Path, stereo_f32: np.ndarray, sample_rate: int = 44100) -> None:
    """Write float32 stereo-interleaved array to WAV (stdlib wave — no soundfile)."""
    dithered = apply_dither(np.clip(stereo_f32, -1.0, 1.0), bit_depth=16)
    pcm = (np.clip(dithered, -1.0, 1.0) * 32767.0).astype(np.int16)
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def write_stem_wav(path: Path, mono_f32: np.ndarray, sample_rate: int = 44100) -> None:
    """Write float32 mono stem to WAV (stdlib wave — no soundfile)."""
    peak = float(np.max(np.abs(mono_f32))) or 1.0
    pcm  = (np.clip(mono_f32 / peak, -1.0, 1.0) * 32767.0).astype(np.int16)
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# J. MIDI export (raw SMF Type 0, stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def _midi_var_len(value: int) -> bytes:
    """Encode integer as MIDI variable-length quantity."""
    result = bytearray()
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


def _midi_meta_event(meta_type: int, data: bytes) -> bytes:
    """Build a MIDI meta-event: FF <type> <length_vlq> <data>."""
    return b'\xFF' + bytes([meta_type]) + _midi_var_len(len(data)) + data


def _midi_note_event(delta: int, status: int, note: int, velocity: int) -> bytes:
    """Build a MIDI note-on or note-off event with delta time."""
    return _midi_var_len(delta) + bytes([status, note & 0x7F, velocity & 0x7F])


def write_midi(
    path: str,
    job_id: str,
    bpm: float,
    key: str,
    genre: str,
    duration_sec: float,
    sample_rate: int = 44100,
) -> None:
    """Write a raw SMF Type 0 MIDI file using only stdlib (struct + bytes).

    Includes: tempo meta-event, chord progression, bass pattern, arp lead.
    Uses 480 PPQ (ticks per quarter note).
    """
    PPQ = 480
    bpm_f = max(60.0, min(float(bpm), 200.0))
    tempo_us = int(60_000_000 / bpm_f)  # microseconds per quarter note

    # Key / scale
    parts     = (key or "C minor").split()
    root_name = parts[0] if parts else "C"
    is_minor  = len(parts) > 1 and parts[1].lower().startswith("min")
    root_midi = 48 + _NOTE_SEMI.get(root_name, 0)  # C3 = 48 as base
    scale     = MINOR_SCALE if is_minor else MAJOR_SCALE
    _genre    = _genre_key(genre)

    prog     = CHORD_PROGS.get(_genre, CHORD_PROGS["default"])
    bp       = _BASS.get(_genre, _BASS["default"])
    sections = _sections_for(_genre, duration_sec, bpm_f)
    total_bars = sum(s[1] for s in sections)

    # Ticks per 16th note
    ticks_per_16th = PPQ // 4

    track_data = bytearray()

    # ── Tempo meta-event at tick 0 ────────────────────────────────────────────
    tempo_bytes = struct.pack(">I", tempo_us)[1:]  # 3 bytes big-endian
    track_data += b'\x00' + _midi_meta_event(0x51, tempo_bytes)

    chord_ch = 0    # MIDI channel 0 = chords
    bass_ch  = 1    # MIDI channel 1 = bass
    lead_ch  = 2    # MIDI channel 2 = lead/arp

    prev_tick = 0

    for bar in range(min(total_bars, int(duration_sec * bpm_f / 240.0) + 1)):
        chord_semi, chord_quality = prog[bar % len(prog)]
        bar_tick = bar * PPQ * 4  # 4 quarter notes per bar

        # ── Chord notes ───────────────────────────────────────────────────────
        intervals = VOICINGS.get(chord_quality, VOICINGS["min7"])
        chord_notes = [(root_midi + chord_semi + iv) % 128 for iv in intervals]
        chord_dur_ticks = PPQ * 4  # whole bar

        for note in chord_notes:
            delta = bar_tick - prev_tick
            track_data += _midi_note_event(delta, 0x90 | chord_ch, note, 80)
            prev_tick = bar_tick

        note_off_tick = bar_tick + chord_dur_ticks - ticks_per_16th
        for note in chord_notes:
            delta = note_off_tick - prev_tick
            track_data += _midi_note_event(max(0, delta), 0x80 | chord_ch, note, 0)
            prev_tick = note_off_tick

        # ── Bass pattern ──────────────────────────────────────────────────────
        step = 0
        while step < 16:
            bp_note, bp_gate = bp[step]
            step_tick = bar_tick + step * ticks_per_16th
            if bp_note >= -90 and bp_gate > 0:
                bass_midi = max(0, min(127, root_midi + chord_semi + bp_note - 12))
                gate_ticks = bp_gate * ticks_per_16th

                delta_on = step_tick - prev_tick
                track_data += _midi_note_event(max(0, delta_on),
                                               0x90 | bass_ch, bass_midi, 90)
                prev_tick = step_tick

                off_tick = step_tick + gate_ticks - 1
                delta_off = off_tick - prev_tick
                track_data += _midi_note_event(max(0, delta_off),
                                               0x80 | bass_ch, bass_midi, 0)
                prev_tick = off_tick
                step += max(1, bp_gate)
            else:
                step += 1

        # ── Arp lead (8th notes through chord voicing) ───────────────────────
        arp_notes = chord_notes + [min(127, chord_notes[0] + 12)]
        for n_i in range(8):
            note    = arp_notes[n_i % len(arp_notes)]
            on_tick = bar_tick + n_i * (ticks_per_16th * 2)
            gate    = ticks_per_16th * 2 - (ticks_per_16th // 2)

            delta_on = on_tick - prev_tick
            track_data += _midi_note_event(max(0, delta_on),
                                           0x90 | lead_ch, note + 12, 70)
            prev_tick = on_tick

            off_tick = on_tick + gate
            delta_off = off_tick - prev_tick
            track_data += _midi_note_event(max(0, delta_off),
                                           0x80 | lead_ch, note + 12, 0)
            prev_tick = off_tick

    # End of track meta-event
    track_data += b'\x00' + _midi_meta_event(0x2F, b'')

    # ── SMF header ────────────────────────────────────────────────────────────
    track_bytes = bytes(track_data)
    header  = b'MThd'
    header += struct.pack(">I", 6)       # header length
    header += struct.pack(">H", 0)       # format Type 0
    header += struct.pack(">H", 1)       # 1 track
    header += struct.pack(">H", PPQ)     # ticks per quarter

    track_chunk  = b'MTrk'
    track_chunk += struct.pack(">I", len(track_bytes))
    track_chunk += track_bytes

    with open(str(path), "wb") as f:
        f.write(header + track_chunk)
