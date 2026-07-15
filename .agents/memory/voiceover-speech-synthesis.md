---
name: Voice-over speech synthesis
description: How video narration is synthesized in-house and the birdcall bug class to avoid
---

The video `voiceover` flag must produce REAL speech: `ai_model/audio/voiceover.py` uses local
eSpeak NG (installed as a Nix system dep, resolved by absolute path for posix_spawn eligibility —
same fork-avoidance rationale as run_ffmpeg).

**Why:** the flag used to be a silent no-op — "voiceover" videos shipped the procedural arpeggio
synth soundtrack, which users heard as "birdcalls instead of words".

**How to apply:**
- Every ffmpeg stage touching narration passes an explicit `-ar 44100` (VO_SAMPLE_RATE). eSpeak
  outputs 22050 Hz; muxing without explicit resample is the pitch-shift/birdcall bug class.
- Narration text must be cleaned (emoji/hashtags/URLs/control tokens) before TTS or the engine
  reads them aloud.
- The whole path is never-raise: on any failure the render keeps its music/silent track.
- Music is ducked under voice via amix (music_gain ~0.18, espeak -a 175); regression tests in
  tests/test_voiceover.py assert speech-band energy dominance, skipped if espeak-ng missing.
- Job poll responses are an explicit field whitelist in server.py — new job fields must be added
  there or they never reach clients.
