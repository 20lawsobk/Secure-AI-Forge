---
name: Video scene text generation
description: How scene text is generated for multi-scene videos, memory config for stable renders, and new video job endpoints
---

## Scene text — use dataset sampler, not model inference
Use `dataset_sampler.sample_all_scenes()` as the primary (and only) path for video scene text in `plan()`. Do NOT use `model.generate()` or `model.generate_batch()` for this.

**Why:** The transformer is undertrained — it echoes the idea snippet followed by training-vocabulary noise. Dataset sampling from `synthetic.py` phrase banks produces immediately meaningful, personalised text and matches the `source: "datasets"` contract the product expects.

**How to apply:** `plan()` in `video_agent.py` calls `_sample_all_scenes(scene_sequence, idea, genre, tone, platform, artist_name)` which returns `(Dict[int,str], "datasets")`. No inference, zero memory spike, always succeeds.

## Memory configuration — required for stable renders

Three settings must ALL be in place together or the server enters an OOM crash loop that leaves render jobs stuck in "pending" forever:

| Setting | Value | File |
|---|---|---|
| Python uvicorn workers | **1** | `server.py` `worker_count = 1` |
| Node.js cluster workers | **2** | `api-server/src/index.ts` `Math.min(cpu, 2)` |
| Parallel PIL scene renders | **2** | `cinematic_engine.py` `ThreadPoolExecutor(max_workers=min(2, len(scenes)))` |

**Why:** Model = 1.7 GB × 1 worker. Node cluster = ~150 MB × 2 workers. Render threads: each PIL scene at 1080×1920 ≈ 20 MB in RAM; capped at 2 = ~40 MB peak vs 8 = ~160 MB peak. Together this keeps idle RAM at ~95% and render RAM at ~97% — within the 8 GB host without triggering the OOM killer. If any setting reverts (especially Node workers back to 4), the server will crash-loop on the next render.

**Symptom of wrong config:** Jobs stuck permanently in `pending` with `scenes: N` populated but `filename: null` and `error: null` — the render thread was OOM-killed but the job file was never updated with the error.

**Recovery when in a crash loop:** Kill orphaned Python processes (`ps aux --sort=-%mem | grep python`) keeping only the two newest (parent + worker), wait for free RAM to rise to ~1.5 GB, then restart `Start application`.

## Render timing
- Planning (dataset sampler): ~2s
- Render (PIL frames + ffmpeg composite): ~50s for 10 scenes / ~150s for 20 scenes
- These timings are for the 1080×1920 (9:16 TikTok) resolution at `-crf 20 -preset fast`

## generate_batch() — exists but not in the hot path
`CreativeModel.generate_batch()` (micro-batched, chunk_size=4, max_new_tokens=30) works correctly and is the right inference method when the model is well-trained. Do NOT use max_new_tokens=200 for batched generation — KV-cache grows to 1.3 GB and OOM-kills the worker.

## Video job endpoints (all proxied through Node /api/*)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate-video` | Start job; accepts `scenes_override: [{index, text}]` |
| GET | `/api/video-job/:id` | Poll status |
| GET | `/api/video-jobs` | List all jobs (newest first) |
| DELETE | `/api/video-job/:id` | Cancel pending or purge done+file |
| GET | `/api/video-job/:id/preview/:sceneIdx` | JPEG frame at scene midpoint |
