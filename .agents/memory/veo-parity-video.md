---
name: Veo-parity video controls
description: How image conditioning, extension, multi-sample, and native audio are wired; traps when extending the video control surface
---

## Rules

1. **Image conditioning goes through SDEdit's prior.** `init_frame_b64` in a scene's `diffusion_meta` is decoded by the SDEdit prior and denoised FROM — that is the real mechanism for first-frame / last-frame / reference-image ("ingredients") conditioning. Scene assignment: first scene ← first frame, last scene ← last frame, middle scenes cycle ≤3 reference images.
   **Why:** SDEdit = noise + denoise from an init image, so this is genuine conditioning, not metadata decoration.

2. **Caller image payloads must be bounded BEFORE decode.** Cap encoded length (~28 MB) and check `PIL.Image.open().size` (lazy, header-only) against a pixel cap before `.convert()/.resize()`; use `b64decode(validate=True)`. Never-raise → return None.
   **Why:** unbounded b64+PIL decode of caller data is a memory-exhaustion vector even inside try/except.

3. **Camera/composition/lighting condition via the awareness text.** Fold them into a "cinematography: …" line appended to the diffusion awareness string — the conditioner encodes text, so terms left only as dict fields are inert.

4. **`sample_count` variants need DERIVED seeds.** seed=None resolves to a deterministic idea-hash seed, so N copies of the same request render byte-identical videos. Fan out with base_seed + i*step.

5. **Video extension recipe:** extract source's last frame (`ffmpeg -sseof -0.5 … -frames:v 1`), use as first-frame conditioning for a continuation rendered at the source's exact dimensions, concat video-only with normalized fps/scale/format, then mux ONE continuous soundtrack over the full duration (better than stitching two tracks). Media duration without ffprobe: parse `Duration:` from `run_ffmpeg(["ffmpeg","-i",f]).stderr` (posix_spawn-safe, no PIPE).

6. **Native audio default:** when the caller supplies no track, videos auto-mux a genre/BPM-matched soundtrack from the real-audio dataset (`generate_audio: true` default); never-raise → silent render if the dataset isn't seeded.

7. **Both public video endpoints must forward the full control surface.** `/api/generate-video` (pydantic model) AND `/api/video/generate-ai` (raw body dict) — a field added to one but not the other is a silent no-op on the second (same class of bug as maxcore-conditioning-fields).

Guard: `tests/test_veo_parity.py` (fast — no rendering).
