---
name: Concurrency & dedup on the AI server
description: Which scaling levers already exist (don't rebuild) and how the fleet-wide dedup cache is designed.
---

# Already-built scaling levers — do NOT rebuild
- **Backpressure exists**: `ai_model/adaptive_concurrency.py` — INFERENCE_GATE / RENDER_GATE are AdaptiveGates that auto-size capacity from live CPU + cgroup memory and *queue* excess work on a Condition (raise GateBusy → HTTP 503 on timeout). Heavy handlers run via `_in_thread_gated(GATE, fn, timeout)`. This is the load ceiling; do not add another.
- **Image dedup exists**: seeded image renders skip work if the output file already exists on disk.
- **pdim store is real**: `storage_client.StorageClient` (get / set(ex=TTL) / delete / incr / keys / hget / lpush…) with a graceful in-process dict fallback; `get_storage()` is the singleton.

# Honest scale thesis (recurring user debate)
Storage scales cheaply/digitally; **compute is bounded by physical cores**. A "custom cloud" only orchestrates physical machines — it doesn't manufacture compute; a better-than-Redis store is still storage, not compute. Skipping recompute (dedup) is the only honest *software* lever that raises effective throughput.

# Fleet-wide dedup cache (`ai_model/dedup_cache.py`)
Pdim-backed result cache for the synchronous text endpoints (`/api/generate/content`, `/api/generate/text` content-mode). On hit returns the stored dict + additive `cached:true` + refreshed `processing_time_ms`. Measured 2.7s → 0.1ms on a repeat.
- **Key = sha256 of the JSON-dumped request with only TOP-LEVEL transport metadata stripped** (id/ts/nonce…). **Never recurse** — nested `inputs`/`step`/`slots` are semantic; scrubbing nested `id`/`time` would let genuinely-different requests collide (architect-caught bug). `artistProfileId` is top-level and kept, so text dedup is naturally per-artist.
- **TTL must be enforced in-layer** via an `{_dedup_exp,_dedup_val}` envelope: `StorageClient`'s in-process fallback (pdim-offline = normal dev mode) ignores `ex=`, so without the envelope a result would live for the whole process. get() also best-effort `delete()`s expired keys.
- Best-effort: only a well-formed, unexpired *dict* envelope is returned; everything else (error/miss/corruption/non-dict) = miss and never raises. Only `put` when `_model_ready` so template fallbacks aren't pinned.
- Tunables `DEDUP_CACHE_ENABLED` / `DEDUP_CACHE_TTL` (default 3600s); stats at `/api/concurrency/stats`.
- **NOT built** (honest: low/negative value on CPU/NumPy): microbatching (gates already serialize work; only adds latency) and preview-first (trades quality for compute, not a capacity win).
