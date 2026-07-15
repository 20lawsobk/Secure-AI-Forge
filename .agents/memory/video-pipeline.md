---
name: Video pipeline architecture
description: Key fixes for the video generation pipeline — non-blocking jobs, ffmpeg speed, text sanitization
---

## Non-blocking video endpoints
`/api/generate-video` and `/api/video/generate-ai` MUST return `job_id` immediately. The full `plan()` (model inference) + render happens in a background `threading.Thread`. Old code awaited `plan()` in the request handler, causing 3+ min HTTP timeouts.

**Why:** `plan()` calls `model.generate()` 3-5 times (one per scene). Even with KV-cache, each call is 20s = 60-100s total planning. HTTP clients time out at 30-60s.

## ffmpeg static-image encode speed
Always pass `-framerate 24` BEFORE `-i bg.png` when using `-loop 1` in `scenes.py`. Without it, ffmpeg treats the static image at a very low default rate and encodes at ~26s per scene. With `-framerate 24`, the same clip encodes in ~1.4s.

## Text sanitization (_clean in video_agent.py)
Model output must be sanitized before use as ffmpeg drawtext text:
- Strip 25+ char non-whitespace tokens (API keys, hashes) via `re.compile(r'\S{25,}')`
- Strip bare 2+ digit numbers (model vocab IDs like "128", "49")
- Cap at 10 words (_MAX_WORDS) for social media overlays
- Strip `'`, `"`, `[`, `]`, `:`, `;`, `\`, `=` — these break ffmpeg drawtext inside `text='...'`

## ffmpeg drawtext apostrophe escaping
The `_esc()` function in `scenes.py` must NOT backslash-escape apostrophes (`'` → `\'`) because `\'` terminates the single-quoted drawtext string in ffmpeg's filter parser. Instead, apostrophes must be REMOVED (replaced with empty string).

## Render speed budget (July 2026 profiling)
Per-stage [VideoRender][Timing] prints now exist in scenes.py (bg/grade/encode), cinematic_engine.py (scenes_total/composite/workers), and server._render_only (build_scenes/soundtrack) — grep logs for "Timing" before guessing at bottlenecks.
**Why:** 88s render for a 12s video turned out to be one bug: get_diffusion_frame denoised at resolution=max(w,h)=1920 though the latent grid is fixed 256-native — pure wasted pixel work (~25s/scene, zero added detail since the result is bilinear-resized anyway). Fix = always generate at native RESOLUTION and upscale; renders dropped 88s→35s with pixel-identical class of output.
**How to apply:** never pass caller output resolution into the diffusion generate(); guard is tests/test_diffusion_frame_resolution.py (dimension contract + resolution spy + <15s speed guard). Scene rendering is serial (RENDER_GATE workers=1 under memory pressure) — that's deliberate, don't parallelize without checking Watchdog memory headroom.
