---
name: Live industry beacon (awareness harvest + music targets)
description: Widened quality harvest (Deezer per-genre charts + measured preview BPM), proactive refresh scheduler, chart-BPM anchoring for beat generation.
---

The awareness buffer is the platform's bridge to the live music market. It is a *beacon until retirement* (own corpus ≥ MB_AWARENESS_RETIRE_AT), per explicit user direction — do not remove the retirement contract.

- Harvest sources: Apple RSS, Deezer global + 6 per-genre charts (public API, no key), YouTube feeds, HN. Deezer 30s previews are downloaded and analyzed with the in-house beat tracker (`_analyze_preview_bpm`) → `doc["music_features"]` per-genre {bpm_median, bpm_range, energy, duration}.
  **Why measured, not looked up:** Deezer's `bpm` field is 0 for most modern tracks — only own analysis gives real numbers.
- `run_ffmpeg` result has `.returncode`, NOT `.ok` — checking `getattr(res,"ok",False)` silently fails every decode.
- Preview study is bounded: MB_HARVEST_PREVIEWS (12) + wall-clock MB_HARVEST_PREVIEW_BUDGET_S (90s). Harvest runs only in background threads.
- `quality_awareness.music_targets(genre)` — genre→substring-match→global fallback; deliberately NOT retirement-gated (dataset beacon for beats, retires only when the beat dataset itself is replaced).
- `start_scheduler()` (called in server on_startup) refreshes every MB_AWARENESS_REFRESH_H (6h); recurring scans stop once retired, bootstrap (missing doc) still allowed.
- Social + ad quality beacon: Mastodon public trending statuses/tags (reblogs+favourites = real engagement weights) + HN vote-ranked marketing headlines → `doc["social_ad_patterns"].signal_lines`, appended to every platform's awareness inside the retirement gate.
  **Why:** the beacon covers content quality broadly (music, social, ads), not just music charts — per explicit user direction.
- Signal-line tiers must be `[HIGH]/[MEDIUM]/[LOW]` — script_agent's parser silently drops `[MED]` (regression guard: tests/test_social_ad_beacon.py, also asserts tag sanitization: `[\w-]` allowlist, 40-char cap, non-English preserved).
- Beat BPM chain: explicit caller bpm > style fingerprint > chart-anchored band (chart BPM ±6 clamped inside the intent tempo band) > 120. **How to apply:** the /api/generate/audio handler always passes a nonzero bpm downstream, so chart anchoring must happen in the HANDLER's band derivation, not only in `_render_audio_from_dataset` (whose chart block is the safety net for direct callers passing 0).
