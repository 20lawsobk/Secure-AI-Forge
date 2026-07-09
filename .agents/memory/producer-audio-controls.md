---
name: Producer audio controls
description: Producer-grade controls (exact key/BPM, LUFS master, stems, seed) layered on the retrieval+transform audio pipeline
---

# Producer audio controls

The `/api/generate/audio` pipeline is **retrieval + transform**, not a neural
composer. Producer features are implemented as real DSP transforms on the
selected dataset sample, in `ai_model/audio/producer_tools.py`, wired into the
render path.

## Durable rules
- **Every producer transform is never-raise.** Retune/retime, mastering, and
  stem separation each degrade to the base clip on failure and log an honest
  `[Producer] … skipped` line. Never let an optional control fail a whole job.
  **Why:** matches the project-wide "no broken fallback" contract; a producer
  asking for stems should still get their track if HPSS fails.
- **Exact key/BPM = pitch-shift + time-stretch (ffmpeg `rubberband`), not just
  nearest-sample selection.** tempo_ratio = target_bpm/src_bpm, clamped 0.5–2.0;
  key shift = nearest semitone between tonics, clamped ±6. When a transform is
  applied, report the level actually achieved (src×clamped), NOT the requested
  target — otherwise the response lies about what was rendered.
- **The render function returns a dict** (`url`, applied bpm/key, format,
  sample_rate, bit_depth, loudness_lufs, stems map, source_sample), not a bare
  URL string. Callers must read `render["url"]`.
- **The ARC spectral-clean file helper is format-preserving**: it re-encodes
  based on the file suffix (`.wav` → PCM, else libmp3lame). It now runs on the
  intermediate WAV mid-chain, so it must not force MP3 output.
- **Stems are always lossless WAV** (drums=percussive via HPSS, bass=harmonic
  <250 Hz, melody=harmonic ≥250 Hz), regardless of the master export format.
- **New request/response fields flow through the api-server proxy for free** —
  `proxyRequest` forwards `JSON.stringify(req.body)` and returns JSON verbatim,
  so additive audio fields need NO proxy edit.
- **Reproducibility via `seed`**: a single seed base drives both BPM and key
  RNGs (key uses seed+offset). Same seed + same request ⇒ identical bpm/key/
  sample selection.
- **Validate/clamp producer inputs at the edge** (target_bpm 40–300, sample_rate
  to a supported set, bit_depth ∈ {16,24,32}, LUFS −40..0) → bad input degrades
  to safe defaults instead of hard-failing an ffmpeg stage with no fallback.

## How to apply
When extending audio generation, add transforms as never-raise helpers in
`producer_tools.py` and thread request fields → `opts` dict → render → job
result → poll endpoint. DSP available at runtime: ffmpeg `rubberband`,
`loudnorm`, `firequalizer`, `acompressor`; librosa 0.11 (`effects.hpss`,
`time_stretch`, `pitch_shift`) + soundfile.
