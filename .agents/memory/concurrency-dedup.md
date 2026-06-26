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

## Measured ceiling (single dev VM: 4 cores, INFERENCE_GATE capacity 2)
Load-tested `/api/generate/content` (the *lightest* gen endpoint; image/audio/video are strictly heavier):
- **Unique requests (cache misses): ~0.34 req/s, FLAT across 8/32/64 concurrency.** Adding concurrency does not raise throughput — it just deepens the queue (p50 latency climbed 18s→40s→46s) until the 35s gate timeout sheds overflow as 503 (backpressure working: 0/6/12 shed at 8/32/64). This is the real compute ceiling.
- **Identical repeats (cache hits): ~182 req/s, 64ms p50, 0 errors** — the dedup win, but ONLY for repeats.
- Extrapolation: 100 VMs × 0.34 ≈ 34 req/s ≈ ~3M unique gens/day. "90M *concurrent*" ≈ 45M VMs (2 in-flight each); "90M unique/day" ≈ ~3,000 VMs. The "90M easily" claim is off by ~6 orders of magnitude for unique work; cache only closes the gap to the extent traffic actually repeats.

# Default content-gen path = dedup + single-flight (not just dedup)
`/api/generate/content` and `/api/generate/text` (content-mode) now run their heavy work through `PDIMOrchestrator.compute(req, _build, namespace="api_content"/"api_text")` **by default** (when `_model_ready`), executed on a worker thread via `await _in_thread(...)`. Previously they did dedup *caching* only — concurrent identical requests all missed and all recomputed. Now N identical in-flight requests collapse to ONE compute.
- **Why:** caching only helps *sequential* repeats; under a concurrent identical burst every request raced past the cache and paid full model inference. Single-flight is the proven 90M-scale path.
- **How to apply:** wrap the whole per-request compute in a sync `_build(_request=None)` closure; acquire `INFERENCE_GATE.slot(timeout=...)` *synchronously inside* `_build` (gate.slot is a sync CM — do NOT use the async `_in_thread_gated` here, you're already in a worker thread); `GateBusy → HTTPException(503)`. Same dedup namespaces/keys as before, so cache entries are compatible. Keep the `_model_ready==False` branch calling `_build()` directly (heuristic output stays uncached). Verify live: a concurrent identical burst yields N-1 `cached:true` + dedup `stores` +1 (seen: 12 identical → 11 cached, stores 0→1).

# Fleet-wide dedup cache (`ai_model/dedup_cache.py`)
Pdim-backed result cache for the synchronous text endpoints (`/api/generate/content`, `/api/generate/text` content-mode). On hit returns the stored dict + additive `cached:true` + refreshed `processing_time_ms`. Measured 2.7s → 0.1ms on a repeat.
- **Key = sha256 of the JSON-dumped request with only TOP-LEVEL transport metadata stripped** (id/ts/nonce…). **Never recurse** — nested `inputs`/`step`/`slots` are semantic; scrubbing nested `id`/`time` would let genuinely-different requests collide (architect-caught bug). `artistProfileId` is top-level and kept, so text dedup is naturally per-artist.
- **TTL must be enforced in-layer** via an `{_dedup_exp,_dedup_val}` envelope: `StorageClient`'s in-process fallback (pdim-offline = normal dev mode) ignores `ex=`, so without the envelope a result would live for the whole process. get() also best-effort `delete()`s expired keys.
- Best-effort: only a well-formed, unexpired *dict* envelope is returned; everything else (error/miss/corruption/non-dict) = miss and never raises. Only `put` when `_model_ready` so template fallbacks aren't pinned.
- Tunables `DEDUP_CACHE_ENABLED` / `DEDUP_CACHE_TTL` (default 3600s); stats at `/api/concurrency/stats`.
- **NOT built** (honest: low/negative value on CPU/NumPy): microbatching (gates already serialize work; only adds latency) and preview-first (trades quality for compute, not a capacity win).

# In-process single-flight correctness (MaxCore PDIM orchestrator)
A claim-then-remove single-flight (winner installs an inflight Event, computes, caches, then removes the slot + sets the event; others wait on the event) has TWO races a load test surfaces but a quick read misses:
1. **Late arrival after the winner already finished**: a caller whose first cache read missed *before* the winner's write, but who acquires leadership *after* the winner removed its slot, becomes a second leader and recomputes. Fix: on winning leadership, **double-check the cache** before paying to compute.
2. **Follower timeout / non-cacheable leader result**: if followers "fall through and compute" on wait-timeout or cache-miss-after-wake, ALL of them recompute in parallel. Fix: make `compute()` a **re-contention loop** — a follower that times out re-finds the still-installed inflight Event and waits again (a slow leader never triggers a parallel recompute), and a finished-but-non-cacheable leader yields exactly ONE new leader per loop pass (the waiting set shrinks by one each pass → terminates).
- **Why:** single-flight's entire value is "compute each unique key once under contention"; both races silently break that and only show up as >1 compute-per-key under real concurrency, never in a serial unit test.
- **How to apply:** keep the inflight slot installed for the leader's whole lifetime; release (pop slot, *then* set event) in a `finally`; never let a waiter compute while an inflight slot for its key still exists. Add counters (inflight_timeout / recontend) so re-contention is observable, and assert max-per-key-computes==1 in a concurrent load test.
