---
name: Digital GPU audio engine
description: Self-contained audio synthesis via NativeKernels C + Digital GPU GEMM — no librosa, soundfile, scipy anywhere in the synthesis or stem-separation path.
---

## Rule
All audio synthesis and stem separation must route through the Digital GPU stack.
Zero dependency on librosa, soundfile, scipy, or numpy math primitives in the hot path.

**Why:** The MaxBooster contract is 100% independent of Replit's base environment for every compute path — audio is no exception.

## How to apply
- **Waveform synthesis** (`_render_audio_clip`, `render_audio_clip`): use `ai_model/audio/digital_gpu_synth.py → render_audio_clip()`, which routes through `NativeKernels.additive_synth / exp_decay / freq_sweep_sin / white_noise / inplace_mul`.
- **Stem separation** (`separate_stems` in `producer_tools.py`): uses `digital_gpu_hpss()` — DFT-matrix GEMM for STFT/iSTFT, Wiener soft masks for HPSS, stdlib `wave` for I/O.
- **WAV output**: stdlib `wave` module everywhere — no soundfile.

## Performance
- `NativeKernels.additive_synth` (compiled SIMD C): 1043× faster than realtime for synthesis.
- Previous path (raw `np.sin` loops): ~88s wall-clock for a 3-min track (≈2x slower than realtime).

## New C kernels in `ai_model/gpu/native/kernels.py`
- `additive_synth(freqs, amps, n_harm, sr, out, n)` — outer-loop-over-harmonics structure vectorizes over time
- `exp_decay(rate, sr, out, n)` — uses existing `fast_expf`
- `freq_sweep_sin(f0, sweep, sr, out, n)` — 808 pitch-drop sweep
- `white_noise(seed, out, n)` — xorshift32 PRNG
- `inplace_mul(out, scale, n)` — envelope application without a temp

## DFT via GEMM
- `digital_gpu_stft(x, n_fft, hop, window)` → builds `[n_fft//2+1, n_fft]` DFT matrices (cached per n_fft), frames signal, calls `DigitalGPU.gemm(Wr, frames.T)` + `gemm(Wi, frames.T)`.
- `digital_gpu_istft(S_real, S_imag, ...)` → `gemm(Wr.T, S_real) + gemm(Wi.T, S_imag)` with overlap-add reconstruction.
- DFT matrices are built once per n_fft and module-level cached — no rebuild cost.

## Pocket pre-registration timing
- `_warm_digital_gpu()` references `_creative_model.model` (not `base_model` which is local to `_load_model()`).
- Must run AFTER model loads — warm-start runs in a background thread; check `_creative_model` via `getattr`.
