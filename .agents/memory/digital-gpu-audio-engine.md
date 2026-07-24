---
name: Digital GPU audio engine
description: Self-contained audio synthesis via NativeKernels C + Digital GPU GEMM — no librosa, soundfile, scipy anywhere in the synthesis or stem-separation path.
---

## Rule
All audio synthesis and stem separation must route through the Digital GPU stack.
Zero dependency on librosa, soundfile, scipy, or numpy math primitives in the hot path.

**Why:** The MaxBooster contract is 100% independent of Replit's base environment.

## Entry point
`ai_model/audio/digital_gpu_synth.py → render_full_track()` — produces a full professional
stereo track (intro/verse/prechorus/drop/breakdown/outro) with arrangement, chord voicings,
genre-specific drum patterns, bass lines, pad chords, lead arp, reverb, compression, and M/S widening.
`render_audio_clip()` is a backward-compatible alias.

## Performance (v2 professional engine)
- 30s trap track: 835ms wall = 35,938× RT
- 60s lo-fi track: 982ms wall = 61,093× RT
- 60s phonk track: 593ms wall = 101,260× RT

## C kernels in `ai_model/gpu/native/kernels.py`
**v1 (basic):** `additive_synth`, `exp_decay`, `freq_sweep_sin`, `white_noise`, `inplace_mul`
**v2 (professional):**
- `saw_wave(freqs, amps, phases_in, phases_out, n_osc, sr, out, n)` — polyBLEP bandlimited, outer-osc loop
- `biquad_filter(b0,b1,b2,a1,a2, x,y, state, n)` — transposed-form II IIR, stateful
- `compute_lpf_coeffs` / `compute_hpf_coeffs` — RBJ cookbook, uses `fast_sinf_poly`/`fast_cosf_poly`
- `adsr_envelope(attack, decay, sustain, release, gate, sr, out, n)` — sample-accurate ADSR
- `soft_sat(drive, x, y, n)` — tanh waveshaper
- `soft_limit(x, n)` — soft-knee limiter, peak ≤ 1.0
- `compress_gain(thr, ratio_inv_m1, attack_c, release_c, rms, gain, n)` — per-sample smoothed gain curve
- `mix2(a, x, b, y, out, n)` — vectorised two-bus mix
**SIMD note:** `saw_wave` inner loop branches compile to cmov on x86 (no misprediction overhead).

## Professional track architecture
- `SynthVoice` — detuned polyBLEP saw unison (n_unison up to 7) → biquad LPF → ADSR
- `DrumKit` — layered kick (sub sweep + body + click), snare (tone+noise), clap, hat_closed/open, 808
- `BassVoice` — sub sine + detuned saw mid layer → LPF → ADSR
- `apply_reverb` — FFT convolution with synthetic exp-noise IR (kern.white_noise + kern.exp_decay + numpy FFT)
- `apply_compressor` — RMS envelope → compress_gain kernel → makeup gain
- `apply_ms_width` — M/S encoding, width=1.3 default

## Arrangement data
- `_D` — per-genre 16-step drum grids (kick/snare/hat_c/hat_o/clap/808)
- `_BASS` — per-genre 16-step bass patterns (semitone offsets + gate)
- `CHORD_PROGS` — per-genre progression [(semitone_offset, voicing_name)] × 4 chords
- `VOICINGS` — chord interval sets (maj/min/maj7/min7/dom7/sus2/dim)
- `_SECTIONS` — per-genre arrangement templates (name, bars, elements, filter_pct, energy)
- Sections scale to fill `duration_sec` automatically

## Critical bug class — exp_decay rate units
`kern.exp_decay(rate, duration, n)` computes `exp(-rate * i)` where `i` is the **sample index**,
NOT `exp(-rate * t)` where t is seconds. All call sites in DrumKit and apply_reverb must pass
`rate_per_sec / sample_rate` as the first argument, not `rate_per_sec` directly.
- Wrong: `kern.exp_decay(5.0, 0.5, n)` → collapses to zero after sample 2
- Right:  `kern.exp_decay(5.0 / sr, 0.5, n)` → natural 0.5s decay
This bug caused all drum envelopes to fire for exactly one sample (inaudible after that) and
made the reverb IR a Dirac delta (no reverb). Fixed at every call site in DrumKit + apply_reverb.

## Key decisions
- PolyBLEP saw (not sine): no aliasing, sits in mix like a real synth (Serum/Massive)
- Filter cutoff automation: 600Hz closed in intro → 8kHz open in drop — gives the sweep
- Velocity ±15% variation on all drum hits (humanization)
- Bass pattern uses -1 (ASCII minus) for rests — do NOT use Unicode minus U+2212 in tuples

## Pocket pre-registration fix
- `_warm_digital_gpu()` must reference `_creative_model.model` via `getattr`, NOT `base_model`
- `base_model` is local to `_load_model()` only; accessing it from warm-start = NameError

## DFT / STFT / HPSS (stem separation path — unchanged)
- `digital_gpu_stft` → DFT-matrix GEMM → `digital_gpu_istft` → overlap-add
- `digital_gpu_hpss` → Wiener soft masks on DFT-domain median-filtered magnitude
- Stem output via stdlib `wave` only — no soundfile
