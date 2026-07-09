"""ARC spectral engine — STFT noise-gate restoration (iZotope RX behaviour).

A real STFT / iSTFT built from a DFT basis so the forward and inverse
transforms are matrix multiplies routed through the self-contained Digital GPU
(``framed [T,W] @ DFT[W,F]``). On top of that:

  * a noise profile estimated from the quietest frames (percentile per bin),
  * a spectral gate that attenuates bins near the noise floor (denoise),
  * Hann windowing with 75% overlap-add reconstruction (unity gain).

Operates on mono float32 in [-1, 1]. Deterministic. Never raises on musical
content; raises only on malformed input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..fabric.compute import RTACompute


@dataclass
class SpectralConfig:
    win: int = 1024
    hop: int = 256                 # 75% overlap
    reduction_db: float = 12.0     # max attenuation applied to noise bins
    noise_percentile: float = 15.0 # frames used to estimate the noise floor
    over_subtraction: float = 1.5  # gate aggressiveness above the floor


class SpectralEngine:
    def __init__(self, compute: Optional[RTACompute] = None):
        self.compute = compute or RTACompute()

    def _dft_matrices(self, win: int):
        n = np.arange(win)
        k = np.arange(win // 2 + 1)
        ang = 2.0 * np.pi * np.outer(n, k) / win     # [W, F]
        return np.cos(ang), -np.sin(ang)             # real, imag basis

    def _frame(self, x: np.ndarray, win: int, hop: int, window: np.ndarray):
        """Frame ``x`` into overlapping windows.

        Pads so that the final (partial) segment is fully covered — otherwise the
        overlap-add reconstruction leaves the tail under-reconstructed. Returns
        the framed matrix, the frame start offsets, the padded length used for
        overlap-add, and the original length to trim back to.
        """
        orig_n = len(x)
        if orig_n < win:
            x = np.pad(x, (0, win - orig_n))
        n = len(x)
        # number of frames needed to cover every sample (ceil), always >= 1
        n_frames = 1 + max(0, int(np.ceil((n - win) / hop)))
        pad_len = (n_frames - 1) * hop + win
        if pad_len > n:
            x = np.pad(x, (0, pad_len - n))
        starts = np.arange(0, pad_len - win + 1, hop)
        frames = np.stack([x[s:s + win] for s in starts], axis=0)  # [T, W]
        return frames * window[None, :], starts, pad_len, orig_n

    def denoise(self, samples: np.ndarray, sample_rate: int,
                cfg: Optional[SpectralConfig] = None) -> np.ndarray:
        cfg = cfg or SpectralConfig()
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if x.size == 0:
            raise ValueError("SpectralEngine.denoise received empty audio")
        peak = np.max(np.abs(x)) or 1.0
        x = x / peak

        win, hop = cfg.win, cfg.hop
        window = np.hanning(win)
        frames, starts, pad_len, orig_n = self._frame(x, win, hop, window)  # [T,W]
        cos_b, sin_b = self._dft_matrices(win)                # [W,F]

        # Forward STFT as two GEMMs on the Digital GPU.
        re = self.compute.gemm(frames, cos_b).astype(np.float64)   # [T,F]
        im = self.compute.gemm(frames, sin_b).astype(np.float64)   # [T,F]
        mag = np.sqrt(re * re + im * im)
        phase = np.arctan2(im, re)

        # Noise floor: per-bin percentile over the quietest frames.
        noise = np.percentile(mag, cfg.noise_percentile, axis=0, keepdims=True)  # [1,F]
        thresh = noise * cfg.over_subtraction
        min_gain = 10.0 ** (-abs(cfg.reduction_db) / 20.0)
        # soft gate: bins below floor pulled toward min_gain, above floor kept
        ratio = mag / np.maximum(thresh, 1e-9)
        gain = np.clip(ratio, 0.0, 1.0)
        gain = min_gain + (1.0 - min_gain) * (gain ** 2)
        mag_dn = mag * gain

        re2 = mag_dn * np.cos(phase)
        im2 = mag_dn * np.sin(phase)

        # Inverse DFT as GEMMs: reconstruct real frames from (re2, im2).
        # x[n] = (1/W) * sum_k [ Re*cos + Im_full*sin ] with Hermitian symmetry.
        f = win // 2 + 1
        full_re = np.zeros((re2.shape[0], win))
        full_im = np.zeros((im2.shape[0], win))
        full_re[:, :f] = re2
        full_im[:, :f] = im2
        # mirror (bins 1..W/2-1)
        if win > 2:
            full_re[:, f:] = re2[:, 1:win - f + 1][:, ::-1]
            full_im[:, f:] = -im2[:, 1:win - f + 1][:, ::-1]
        kk = np.arange(win)
        nn = np.arange(win)
        ang = 2.0 * np.pi * np.outer(kk, nn) / win    # [W(freq), W(time)]
        icos = np.cos(ang)
        isin = np.sin(ang)
        rec = (self.compute.gemm(full_re, icos) - self.compute.gemm(full_im, isin)) / win
        rec = rec.astype(np.float64) * window[None, :]  # re-window for OLA

        # overlap-add over the padded length, then trim back to the original.
        out = np.zeros(pad_len)
        norm = np.zeros(pad_len)
        w2 = window ** 2
        for i, s in enumerate(starts):
            out[s:s + win] += rec[i]
            norm[s:s + win] += w2
        norm = np.where(norm > 1e-8, norm, 1.0)
        out = out / norm
        out = out[:orig_n] * peak
        return np.clip(out, -1.0, 1.0).astype(np.float32)
