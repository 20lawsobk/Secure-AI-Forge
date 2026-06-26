---
name: real-audio generation pipeline
description: How /api/generate/audio renders from a real FMA dataset stored in pdim (no procedural synth fallback), how to seed it, and the gotchas
---

# Real-audio generation (no synth fallback)

`/api/generate/audio` renders output by looping/trimming a REAL source track pulled
from a dataset seeded into pdim — there is **no procedural-synthesis fallback**. If
the dataset is empty/unavailable the endpoint raises explicitly (RuntimeError).

**Why:** the user's hard requirement was real datasets, not synth, with no silent
fallbacks. Selection picks a dataset entry by key then closest BPM to the request,
then ffmpeg `-stream_loop` to reach the requested duration.

# Seeding the dataset

Admin-only: `POST /storage/datasets/audio/seed?count=N&replace=` (header
`X-Admin-Key: $ADMIN_KEY`). Runs in a daemon thread; returns immediately
(`{"status":"seeding"}` or `{"status":"already_seeding"}` — a single-flight lock
guards re-entry). The seeder downloads CC tracks from HF
`benjamin-paine/free-music-archive-small`, ffmpeg-transcodes to bounded mono MP3,
computes bpm/key with librosa, and stores:
- `mb:dataset:audio:meta` — manifest
- `mb:dataset:audio:chunk:{idx}` — per-track record; **binary is base64 in field `b64`**

Verify: `/storage/datasets` lists `audio`; or PING the pdim instance and
`KEYS mb:dataset:audio:*`.

# Gotchas

- HF datasets-server is flaky; the seeder's `_http_get` uses retries + long timeouts.
- `server.py` has **no module-level `logger`** — use `print(...)` in new endpoints.
- The seeder module is import-cached per process; after editing
  `workers/seed_audio_dataset.py` restart the workflow(s) spawning `server.py`.
- Data lives in whichever pdim instance `STORAGE_HTTP_URL` points at; if you repoint
  instances (see pdim-storage-topology) you must RE-SEED — chunks don't migrate.
