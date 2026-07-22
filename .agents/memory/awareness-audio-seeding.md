---
name: Awareness-gated audio seeding
description: Audio dataset must be seeded from live awareness signals until admin flywheel corpus is self-sufficient; describes the watchdog, filtering logic, and retirement contract.
---

# Awareness-gated audio seeding

## The rule
The audio dataset (`mb:dataset:audio`) must be populated from live awareness signals (trending genres + Deezer BPM targets) while the admin corpus (`mb:phrases:*`) is below the retirement threshold. Once the admin flywheel grows enough content (`buffer_weight → 0`), external seeding stops and the dataset grows only from admin-generated audio via `_fw_ingest_audio_render`.

**Why:** The user's explicit requirement: "the creation dataset source must remain the awareness layers until the admin account generates enough to match the signal being received."

## How to apply

### Genre/BPM filtering in the seeder
`seed_audio_dataset.seed()` accepts `genre_targets: list[dict]` — output of `quality_awareness.audio_seeding_targets()`.
- HF rows: cheap genre pre-filter before download (`_genre_match_score < 0.3` → skip), BPM post-filter after transcode (`_is_awareness_match` threshold 0.4).
- Librosa fallback: examples sorted by genre match score (best match seeded first).
- Matched samples tagged `seeded_by_awareness: True` in both chunk and index.
- Attempt cap is 10× count when filtering (vs 4× unfiltered) to afford skipping mismatches.

### Auto-growth watchdog
`quality_awareness.start_audio_seeding_watchdog()` — started at server boot alongside `start_scheduler()`.
- Interval: `MB_AUDIO_SEED_INTERVAL_MIN` env var (default 30 min).
- Target: `MB_AUDIO_TARGET_CHUNKS` env var (default 24 chunks).
- Each tick: if `buffer_weight > 0` AND `num_chunks < target` → calls `seed(store, count=need, genre_targets=audio_seeding_targets())`.
- When retired (`buffer_weight == 0`): logs once and idles — no more external seeds.

### Retirement contract
`audio_seeding_targets()` returns `[]` when `self_sufficiency()["retired"]` is True. The watchdog reads this and skips seeding. The admin flywheel arm (`_fw_ingest_audio_render` in server.py) keeps growing the dataset from real admin renders regardless.

### Dynamic awareness TTL (TS layer)
`contentAwarenessService.ts` `getOrBuild()` calls the Python `/api/awareness/quality/status` endpoint (3s timeout, 5-min cache) and scales its own cache TTL:
- `buffer_weight=1.0` → TTL = 10 min (refresh external signals aggressively)
- `buffer_weight=0.0` → TTL = 45 min (model is self-sufficient, relax refresh)

### Corpus maturity in contextString
`applyMode()` appends a `=== CORPUS MATURITY ===` block to every awareness contextString with `buffer_weight`, `own_corpus/retire_threshold`, and `external_signal_priority`. Python agents can parse this to calibrate how heavily they weight external signals.

## Key files
- `artifacts/ai-training-server/ai_model/quality_awareness.py` — `audio_seeding_targets()`, `start_audio_seeding_watchdog()`
- `artifacts/ai-training-server/workers/seed_audio_dataset.py` — `seed(..., genre_targets=)`, `_genre_match_score()`, `_bpm_match_score()`, `_is_awareness_match()`
- `artifacts/ai-training-server/server.py` — startup hook for watchdog (alongside `start_scheduler`)
- `artifacts/api-server/src/services/contentAwarenessService.ts` — dynamic TTL + corpus maturity in contextString
- `artifacts/ai-training-server/workers/admin_flywheel.py` + `server.py:_fw_ingest_audio_render` — admin audio → dataset graduation (already existed)
