---
name: Conductor (audio analysis) + Resonance design
description: In-house audio→musical-timeline analyzer, what counts as "in-house", and the agreed Resonance north-star for music-video generation
---

# "In-house" interpretation (decided with user)
The user's hard rule "no 3rd party tech, no external APIs, no integrations, no stock APIs" means **no external network services / hosted models / paid APIs** — NOT "no pip libraries". Local CPU libraries already in `pyproject.toml` are in-house and allowed: `librosa`, `scipy`, `scikit-learn`, `soundfile`, `basic-pitch`, `numpy`, `pillow`, `torch`. These were added precisely for real audio/media analysis.
**Why:** lets us use robust DSP (librosa) instead of hand-rolling FFT, while honoring the "100% owned, no outside calls" intent.

# The Conductor
`ai_model/audio/audio_analysis.py` — `analyze_audio(waveform, sample_rate) -> MusicalTimeline`.
- **Total function:** never raises on ANY input (empty/silence/NaN/inf/mono/stereo/huge-amp/bad-SR). Every feature stage has its own fallback; top-level returns a deterministic `_default_timeline` (120 BPM grid) on failure. This is how "no broken fallback" is honored at the analysis layer — there is no separate fallback path, the function is just total.
- **Deterministic** (no RNG) → memoized in a bounded in-process LRU keyed by blake2b content-hash of the waveform. `librosa.beat.beat_track` is ~8s/call, so caching is the scale lever (same track → instant).
- Outputs: bpm, beats, downbeats, onsets, energy envelope, per-band envelopes (sub/low/mid/high for stem proxies), structural sections (agglomerative on chroma+mfcc), Krumhansl key/mode.
- Tests: `ai_model/audio/test_audio_analysis.py`, run `uv run python -m ai_model.audio.test_audio_analysis` (covers BPM recovery, determinism, energy/sections, totality, encoder integration + forced-fallback schema).

# Integration (additive)
Wired into `maxbooster_veo_music/model/audio_encoder.py::AudioEncoder.encode` — replaced the fixed-quarter section stub with real sections and added keys `beat_positions`, `bpm`, `key`, `mode` (existing keys unchanged). `PlatformHeads._apply_beat_sync` already consumed `beat_positions` (was linspace stub) → now gets real beats.
**Note:** `encode` still uses unseeded `np.random` for an unrelated projection matrix (pre-existing, left as-is per additive constraint).

# Resonance — agreed north-star design (phased)
Audio-conducted music-video generation, 100% in-house, "no broken fallback", scaled by data+cache not model size:
1. **Conductor** (DONE): DSP → musical timeline.
2. **Retrieval spine + all-real cascade + ingestion + coverage gate** — per-layer nearest-neighbor over real assets; cascade rungs (exact→nearest→artist brand-prior→always-loaded domain-anchor core) are ALL real pixels so it never degrades to procedural/empty. Gaps logged as ingestion targets, not fallbacks.
3. **RCGS compositor** — per-layer real-asset retrieval + classical PIL/NumPy compositing + depth-known 2.5D parallax. Insertion point: `ai_model/video/scenes.py::_render_pil_based`. Composite once per scene; FFmpeg animates frames (scale lever).
4. **Brand vector + deterministic caching + taste/coverage flywheel.**
Key reframe: reliability=100% achievable; perfect *quality* asymptotes with base density. Domain is narrow (music/social aesthetics), so dense coverage is tractable. Audio director timeline insertion point: `ai_model/video/ai_scene_builder.py::allocate_durations`.
