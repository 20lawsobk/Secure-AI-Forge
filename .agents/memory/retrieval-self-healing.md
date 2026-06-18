---
name: Retrieval spine + self-healing convention
description: How "no broken fallback" is made real at retrieval, and the required pattern for any custom self-healing system in this repo
---

# Self-healing systems must follow the server Watchdog pattern
The user's explicit, standing directive: **any custom self-healing system in this repo must be modeled on the existing server Watchdog** (`artifacts/ai-training-server/workers/watchdog.py`).
The pattern to copy: daemon poll loop with a top-level try/except (one failing check never kills the loop) · dependencies injected AFTER construction to avoid circular imports, every check guards `if ref is None: return` · each check is detect → alert → auto-heal with the fix recorded in `fix_applied` and guarded against double-execution · rolling `*Alert` deque + stats dict under a lock + `get_status`/`get_log`/`reset_alerts` accessors · persistence to storage KV (`STATE_KEY` etc.) · module-singleton `get_*()`.
**Why:** consistency + the user pointed to it as the proven, in-house, battle-tested template. Don't invent a parallel self-healing style.
**How to apply:** new healers (coverage, ingestion, etc.) are siblings of `CoverageWatchdog` in `ai_model/retrieval/coverage_watchdog.py`, which is itself a faithful copy of the server Watchdog.

# "No broken fallback" at the retrieval layer = the all-real cascade
Retrieval (`ai_model/retrieval/asset_index.py`) answers every query via an all-real cascade: exact → nearest(within radius) → brand_prior → **always-loaded anchor core**. Every rung is a REAL stored asset (no procedural/empty placeholder). A non-empty index NEVER returns empty; `None` is returned ONLY when the index is truly empty (no assets AND no anchors).
The anchor core is the invariant that guarantees non-empty. The coverage watchdog's #1 job is to keep that core loaded (reload via injected `anchor_loader_fn`); its #2 job is to turn weak-coverage queries into durable ingestion targets (the gap→ingestion flywheel).
**Why:** the user requires "always works at 100%, no fallback that breaks." A degraded-but-real result beats an empty/procedural one.
**Constraint:** anchors must be REAL pixels. With an empty asset base there are no real anchors yet, so the watchdog truthfully alerts `anchor_core_empty` rather than faking anchors — honoring "all-real" over a fake "it works".

# Gap-queue drain must be atomic (KV store has no atomic pop)
The storage KV client exposes lpush/lrange/llen/delete but NO atomic pop. So draining the ingestion gap queue must: claim the batch under a dedicated lock (snapshot via lrange + delete the key in the same critical section, separate from the stats lock so storage I/O doesn't block alerts/status), run ingestion on the snapshot OUTSIDE the lock, and **re-enqueue the claimed batch (lpush) if ingestion raises**.
**Why:** a naive read-then-delete loses any gap enqueued between the read and the delete, and a crashing ingestion would silently drop the batch — both violate the durability the self-healing flywheel promises.

# The flywheel closes by folding successful renders back into the index
A self-healing ingestor daemon (Watchdog-pattern sibling) watches successfully-rendered generated assets and folds them back into the live retrieval index, so each generation enriches future retrieval. Its contract, beyond the Watchdog template: **enqueue must be TOTAL** (called from inside the hot generation path — it must never raise or block the gen response, swallow everything and return a bool) · in-process queue is bounded with drop-oldest-on-overflow (log the drop) + dedupe-by-path · batch claim is atomic · failed items get bounded retry then drop+free-dedupe-key (also drop rather than overshoot the bound when full).
**Why:** the enqueue lives in the render success path, so any exception there would break user-facing generation — the #4 "never raise into the gen loop" rule. Bounded everything keeps a backlog from evicting the curated anchor core or growing without limit.

# Per-brand identity centroid is an incremental accumulator — keep it rebuild-consistent
Each brand/artist gets a centroid = unit-normalized mean of its stored embedding vectors, kept as a running sum+count updated on add/replace/remove. The invariant: the incremental accumulator must ALWAYS equal a from-scratch rebuild. So replace must subtract the OLD stored unit vector (and old brand bucket) before adding the new one; remove and load_state must rebuild the accumulators from the compacted vectors rather than patch them.
**Why:** drift between the incremental sum and the actual vectors silently corrupts brand-biased retrieval, and it is invisible until a brand query returns wrong neighbors. Test both paths against a rebuild.
