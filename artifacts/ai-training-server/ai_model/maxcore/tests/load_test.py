"""90M-scale concurrency smoke / load test for the MaxCore DigitalGPU stack.

What it proves (honestly, on a CPU-only box):

  Phase 0  Real compute     — each *unique* request runs a real DigitalGPU graph
                              (text -> embedding -> 2-layer MLP) through the engine.
  Phase 1  Concurrency      — hundreds of threads hammer identical requests; the
                              PDIM single-flight guarantee must collapse them to
                              EXACTLY ONE real compute per distinct request.
  Phase 2  Scale throughput — measures the steady-state dedup fast-path (single
                              and multi-threaded), then projects to the 90,000,000
                              request target at a realistic duplication factor.
  Phase 3  Audit            — asserts ZERO engine fallbacks (the engine handled
                              everything), ZERO errors, and correct results.

At 90M requests with a bounded unique-request working set (the real traffic
shape for content generation), the system never recomputes a duplicate, so the
90M figure is absorbed by the dedup fast path + the one-time real computes.

Env overrides:
  LOAD_TARGET   (default 90_000_000)  headline request scale to project to
  LOAD_UNIQUE   (default 2000)        distinct semantic requests in the working set
  LOAD_THREADS  (default 256)         concurrency for the contention phase
  LOAD_CONC     (default 200000)      real requests issued in the contention phase
  LOAD_SAMPLE   (default 2000000)     real fast-path ops measured for projection
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
import uuid
from collections import Counter

import numpy as np

_SERVER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from ai_model.maxcore import DigitalGPU, PDIMOrchestrator  # noqa: E402
from ai_model.maxcore.integration import build_text_mlp_graph, ref_text_mlp  # noqa: E402

TARGET = int(os.getenv("LOAD_TARGET", "90000000"))
UNIQUE = int(os.getenv("LOAD_UNIQUE", "2000"))
THREADS = int(os.getenv("LOAD_THREADS", "256"))
CONC = int(os.getenv("LOAD_CONC", "200000"))
SAMPLE = int(os.getenv("LOAD_SAMPLE", "2000000"))

VOCAB, DIM, HIDDEN, OUT, T = 64, 16, 32, 8, 6


def _fmt(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def _secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def build_world():
    dg = DigitalGPU()
    rng = np.random.default_rng(0)
    embed = rng.standard_normal((VOCAB, DIM)).astype(np.float32)
    w1 = rng.standard_normal((DIM, HIDDEN)).astype(np.float32)
    b1 = rng.standard_normal((HIDDEN,)).astype(np.float32)
    w2 = rng.standard_normal((HIDDEN, OUT)).astype(np.float32)
    b2 = rng.standard_normal((OUT,)).astype(np.float32)
    graph = dg.compile(build_text_mlp_graph(dg, embed, w1, b1, w2, b2))
    weights = (embed, w1, b1, w2, b2)

    # one-hot per unique request, deterministic from its token list
    requests = []
    for i in range(UNIQUE):
        r = random.Random(i)
        tokens = [r.randrange(VOCAB) for _ in range(T)]
        requests.append({"prompt_tokens": tokens, "rid": i})

    compute_calls = Counter()
    calls_lock = threading.Lock()

    def compute_fn(req):
        with calls_lock:
            compute_calls[tuple(req["prompt_tokens"])] += 1
        onehot = np.zeros((1, T, VOCAB), np.float32)
        for ti, tok in enumerate(req["prompt_tokens"]):
            onehot[0, ti, tok] = 1.0
        out = dg.run_graph(graph, {"onehot": onehot})
        logits = list(out.values())[0].numpy()
        return {"logits": logits.ravel().tolist()}

    def reference(req):
        onehot = np.zeros((1, T, VOCAB), np.float32)
        for ti, tok in enumerate(req["prompt_tokens"]):
            onehot[0, ti, tok] = 1.0
        return ref_text_mlp(onehot, *weights).ravel()

    return dg, compute_fn, reference, requests, compute_calls


def main() -> int:
    print("=" * 72)
    print("MaxCore DigitalGPU — 90M-scale concurrency smoke / load test")
    print(f"target={_fmt(TARGET)} requests | unique working set={UNIQUE} | "
          f"threads={THREADS} | contention={_fmt(CONC)} | sample={_fmt(SAMPLE)}")
    print("=" * 72)

    dg, compute_fn, reference, requests, compute_calls = build_world()
    orch = PDIMOrchestrator()
    run = uuid.uuid4().hex[:8]               # per-run namespaces avoid cross-run cache bleed
    ns = f"loadtest_{run}"
    errors: list[str] = []

    # ── Phase 0: real compute + correctness ───────────────────────────────────
    t0 = time.time()
    for req in requests:
        res = orch.compute(req, compute_fn, namespace=ns)
        got = np.asarray(res["result"]["logits"], dtype=np.float32)
        if not np.allclose(got, reference(req), atol=1e-2):
            errors.append(f"incorrect result for rid={req['rid']}")
    warm = time.time() - t0
    print(f"\n[Phase 0] {UNIQUE} unique real graph computes in {_secs(warm)} "
          f"({UNIQUE / warm:,.0f}/s) — correctness OK={not errors}")

    # ── Phase 1: concurrency / single-flight ──────────────────────────────────
    # Use a FRESH, un-warmed namespace so concurrent first-hits to the same
    # request genuinely race — single-flight must collapse them to one compute.
    orch2 = PDIMOrchestrator()
    ns_conc = f"loadtest_conc_{run}"
    compute_calls.clear()
    src_counts = Counter()
    src_lock = threading.Lock()
    per_thread = max(1, CONC // THREADS)
    barrier = threading.Barrier(THREADS)

    def hammer(seed):
        r = random.Random(seed)
        local = Counter()
        barrier.wait()  # release all threads simultaneously for max contention
        for _ in range(per_thread):
            req = requests[r.randrange(UNIQUE)]
            res = orch2.compute(req, compute_fn, namespace=ns_conc)
            local[res["source"]] += 1
        with src_lock:
            src_counts.update(local)

    t0 = time.time()
    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conc_dur = time.time() - t0
    issued = sum(src_counts.values())
    recomputed_twice = [k for k, v in compute_calls.items() if v > 1]
    coalesced = src_counts.get("coalesced", 0)
    computed = src_counts.get("compute", 0)
    if recomputed_twice:
        errors.append(f"single-flight violation: {len(recomputed_twice)} keys computed >1x")
    if computed == 0:
        errors.append("Phase 1 did not exercise real compute (cache was pre-warmed)")
    print(f"\n[Phase 1] {_fmt(issued)} concurrent requests across {THREADS} threads "
          f"in {_secs(conc_dur)} ({issued / conc_dur:,.0f} req/s aggregate)")
    print(f"          sources: {dict(src_counts)}")
    print(f"          distinct real computes={len(compute_calls)} "
          f"(== compute source {computed}, <= unique {UNIQUE}); "
          f"coalesced (single-flight collapse)={coalesced}")
    print(f"          single-flight guarantee — any key computed twice? "
          f"{bool(recomputed_twice)} (max per-key computes="
          f"{max(compute_calls.values()) if compute_calls else 0})")

    # ── Phase 2: steady-state dedup fast-path throughput + projection ──────────
    warm_req = requests[0]
    orch2.compute(warm_req, compute_fn, namespace=ns)  # ensure cached
    t0 = time.time()
    for _ in range(SAMPLE):
        orch2.compute(warm_req, compute_fn, namespace=ns)
    st_dur = time.time() - t0
    st_ops = SAMPLE / st_dur

    # multi-threaded aggregate fast-path — bounded budget; this phase only needs
    # to demonstrate the GIL ceiling (in-process threads do not scale this path),
    # not to move the full sample (which under contention would dominate runtime).
    mt_total = int(os.getenv("LOAD_MT", "40000"))
    mt_per = max(1, mt_total // THREADS)
    mt_barrier = threading.Barrier(THREADS)

    def fast(_):
        mt_barrier.wait()
        for _ in range(mt_per):
            orch2.compute(warm_req, compute_fn, namespace=ns)

    t0 = time.time()
    mts = [threading.Thread(target=fast, args=(i,)) for i in range(THREADS)]
    for t in mts:
        t.start()
    for t in mts:
        t.join()
    mt_dur = time.time() - t0
    mt_ops = (mt_per * THREADS) / mt_dur

    print(f"\n[Phase 2] dedup fast-path: single-thread {_fmt(st_ops)} req/s, "
          f"{THREADS}-thread aggregate {_fmt(mt_ops)} req/s")

    # projection to the 90M target at the working-set duplication factor
    dup_factor = TARGET / max(1, UNIQUE)
    unique_compute_time = UNIQUE / (UNIQUE / warm)          # == warm
    dedup_time_st = (TARGET - UNIQUE) / st_ops
    dedup_time_mt = (TARGET - UNIQUE) / mt_ops
    proj_st = unique_compute_time + dedup_time_st
    proj_mt = unique_compute_time + dedup_time_mt
    print(f"\n[Projection] dedup fast-path projection over a bounded {UNIQUE}-unique "
          f"working set — NOT an executed {_fmt(TARGET)}-concurrent run")
    print(f"             {_fmt(TARGET)} requests @ dup factor {dup_factor:,.0f}x = "
          f"{UNIQUE} cold real computes ({UNIQUE / warm:,.0f}/s) + "
          f"{_fmt(TARGET - UNIQUE)} dedup hits")
    print(f"             single-thread: {_secs(proj_st)} (={_fmt(st_ops)} req/s)")
    print(f"             note: Python's GIL caps in-process thread scaling for this "
          f"pure-Python fast path ({THREADS}-thread aggregate {_fmt(mt_ops)} req/s = "
          f"{_secs(proj_mt)}); a process-per-core deployment scales near-linearly.")

    # ── Phase 3: audit (the whole point: fallbacks must be zero) ───────────────
    counters = dg.metrics().get("counters", {})
    engine_hits = {k: v for k, v in counters.items() if k.endswith(".engine")}
    fallbacks = {k: v for k, v in counters.items() if k.endswith(".engine_fallback")}
    fb_total = sum(fallbacks.values())
    numpy_paths = {k: v for k, v in counters.items() if k.endswith(".numpy")}
    print("\n[Phase 3] engine audit")
    print(f"          engine kernels used: {engine_hits}")
    print(f"          numpy (no-engine-kernel) ops: {numpy_paths}")
    print(f"          ENGINE FALLBACKS: {fallbacks} (total={fb_total})")
    if fb_total != 0:
        errors.append(f"engine fallback fired {fb_total}x — system not fully engine-served")
    if not engine_hits or sum(engine_hits.values()) == 0:
        errors.append("engine was never used")

    print("\n" + "=" * 72)
    if errors:
        print(f"RESULT: FAIL ({len(errors)} issue(s))")
        for e in errors:
            print(f"  - {e}")
        print("=" * 72)
        return 1
    print("RESULT: PASS — single-flight collapsed all duplicates, zero engine "
          "fallbacks, all results correct.")
    print(f"        {_fmt(TARGET)} requests projected at {_secs(proj_st)} single-thread "
          f"on this CPU box via the dedup fast path (process-per-core scales further).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
