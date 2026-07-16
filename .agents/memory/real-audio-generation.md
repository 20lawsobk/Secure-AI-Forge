---
name: real-audio generation
description: How /api/generate/audio renders real audio and how the ARC spectral-cleanup gate works end-to-end
---

- `/api/generate/audio` renders from a seeded FMA dataset in pdim (no synth fallback). Seed via admin `POST /storage/datasets/audio/seed?count=N` with `X-Admin-Key` (the admin/api key works as this header). Seeding pulls FMA-small from HuggingFace (slow/flaky ~60-90s for 6 tracks); poll `GET /storage/datasets` until an `audio` manifest with `num_chunks>0` appears. Dataset field is `b64`; seeder module is import-cached so restart to reload edits.
- Audio job status endpoint is `GET /api/audio-job/{job_id}` (NOT `/api/jobs/...`, which 404s).

## ARC spectral cleanup gate (RTA_AUDIO_SPECTRAL)
- Studio audio cleanup (RTA-1 ARC / `_arc_spectral_clean_file` → `ai_model.rta.api.spectral_clean_audio`) is **opt-in via env flag `RTA_AUDIO_SPECTRAL=1`**, read from `os.environ` at request time inside `_render_audio_from_dataset`. It is deliberately NOT default-on (denoising already-clean dataset music is a creative choice).
- ARC is gentle & deterministic on real FMA audio: it slightly lowers the noise floor and leaves the signal essentially intact (tiny mean-abs-diff). Not destructive.
- **Honest fallback**: the ARC call is wrapped in try/except; on failure it logs `[RTA] ARC spectral clean skipped (serving base clip)` and still serves the un-cleaned base clip (already on disk) — never a broken file. On success it logs `[RTA] ARC spectral clean applied to <file>`.
- **Why the flag is hard to make live**: two workflows spawn `server.py` (`Start application` AND `artifacts/api-server: API Server`), both via `python-server.ts` which passes `...process.env`. A single flock singleton means only one server.py wins port 9878. If a wrapper node process started BEFORE the shared env var was set, its child server.py won't see the flag. **How to apply:** set the var in shared env (`setEnvVars`), then restart BOTH workflows so both wrappers carry it, then kill server.py so a flag-carrying wrapper respawns it. Verify the live log shows the `ARC spectral clean applied` line for a novel job.
- To validate the exact endpoint code path deterministically (independent of wrapper env propagation): import `server` as a module (safe — `init_db`/`uvicorn`/flock are under `__main__`) with `RTA_AUDIO_SPECTRAL=1`, call `_render_audio_from_dataset(...)` directly, and monkeypatch `_arc_spectral_clean_file` to raise to prove the honest fallback.

## Awareness ↔ dataset sync
- FMA/HF-seeded chunks store NUMERIC genre IDs; awareness supplies genre NAMES. Selector must normalize via `normalize_genres()` (FMA_GENRE_NAMES map) at scoring time — this also fixes already-seeded data without reseeding. Seeder normalizes at write time too.
- Selection ranking: (genre_miss, mood_miss, bpm_dist, idx). Mood is a soft tertiary tie-breaker via MOOD_GENRE_AFFINITY; genre intent always outranks mood. Mood precedence: req.mood > brief.mood > first trending awareness mood (caller leads).
- `ApiGenerateAudioRequest` had no `mood` field → caller mood was a silent no-op (extra body fields class of bug).

## Hang/deadline lessons
- urllib3 `read=` timeout is PER-SOCKET-CHUNK, not total: a large b64 payload trickling under memory pressure never times out. Fix: absolute wall-clock deadline via `ThreadPoolExecutor.submit().result(timeout=30)` — but do NOT use a `with` block (`__exit__` = shutdown(wait=True), blocks on the hung worker). Use shutdown(wait=False).
- Job-level backstop: 120s threading.Timer marks `error`. Terminal writes need an atomic claim (lock + flag, first writer wins) or timer-error vs late-done races produce nondeterministic status.

## Admin content flywheel — audio arm
- Admin (B-Lawz) renders auto-push into mb:dataset:audio via `_fw_ingest_audio_render` (parity with _fw_ingest for text/video/image). Never-raise, admin-scope gated, sha256 content dedup, module-level index lock.
- Derivation guard: renders whose source_sample is itself a flywheel entry are NOT re-ingested (prevents copy-of-copy decay). Index entries carry `source: "flywheel"` + `content_sha`.
- Verified live: render → ingested idx N → next matching request selects idx N → no second ingest.
