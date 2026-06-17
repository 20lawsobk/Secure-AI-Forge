---
name: MaxBooster ↔ MaxCore engine API contract
description: How the separate MaxBooster platform calls this MaxCore engine, and the alias rules the engine must honor.
---

# MaxBooster ↔ MaxCore contract

This repo is the **MaxCore engine** (Python FastAPI on 9878 behind Node API 8080). A SEPARATE
"MaxBooster" platform consumes it. MaxBooster's client/service TS files are the source of truth
for what endpoints + shapes the engine must serve. They are NOT ported into this repo (they depend
on MaxBooster's own modules) — they only define the contract.

## Rules that are NOT obvious from this engine's code
- **MaxCoreAIClient always prepends `/api`** to any endpoint that doesn't already start with `/api/`.
  So a TS call to `/infer/viral-score`, `/generate/image`, `/video-job/<id>` hits the engine's
  `/api/...` handlers — implement contract endpoints under `/api/`, not the bare `/generate/*` set.
- **Auth:** client sends `X-Admin-Key`, `X-API-Key`, and `Authorization: Bearer` (all = same key).
  Engine's `require_scope(...)` accepts X-Api-Key/X-Admin-Key; `ADMIN_KEY`/`AI_TRAINING_KEY_PROD` bypass.
- **viral-score scale:** `/api/infer/viral-score` must return `score` (and `viral_score`) in **0–1**;
  the client does `Math.min(100, score*100)`. Returning 0–100 would break it.
- **Graceful client fallbacks:** MaxBooster degrades gracefully when an endpoint 404s or returns no
  url (e.g. falls back to local TTS for `/generate/audio`, reads `outputs[0].url` if top-level url
  missing for `/generate/image`). So prefer honest behavior over faking a url that 404s.
- **Video download:** client tries `/api/video-job/<id>/download` then `/file`, `/video` for the raw
  MP4 (validates magic bytes `ftyp` at buf[4:8]). The static `/uploads/videos/<file>` mount also works.

## How the engine satisfies it (additive only)
All gaps were closed additively/backward-compatibly: new endpoints `/api/health`,
`/api/audio/analyze` (returns sections[] w/ `type`, energy_curve[], mood[], tempo, musical_key),
`/api/infer/viral-score`, `/api/video-job/{id}/download|file|video`; plus optional alias request
fields and response aliases on `/api/generate/{content,text,image}` and `status:"processing"` on the
async video/audio jobs. **Why:** "wire in the enhancements" = make the engine fulfill the contract,
not drop the TS files in. Keep every change additive so existing dashboard callers never break.

## Operational gotcha
Both the `AI Training Server` workflow AND `Start application` can bind 9878 → orphan/stale pythons.
The `AI Training Server` workflow currently owns 9878 (its child python is the live server). After
editing server.py, restart `AI Training Server` to load new code; verify with `ps aux | rg [s]erver.py`.
