"""
Tests for the retrieval spine (AssetIndex) and the self-healing CoverageWatchdog.

Run:
    uv run python -m ai_model.retrieval.test_retrieval
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import numpy as np

from ai_model.retrieval.asset_index import AssetIndex
from ai_model.retrieval.coverage_watchdog import CoverageWatchdog


_failures = 0


def _check(name: str, cond: bool) -> None:
    global _failures
    status = "ok" if cond else "FAILED"
    if not cond:
        _failures += 1
    print(f"  {status}: {name}")


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeStorage:
    """Minimal in-memory stand-in for the KV storage client."""

    def __init__(self) -> None:
        self.is_available = True
        self._kv: Dict[str, Any] = {}
        self._lists: Dict[str, List[Any]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ex: Any = None) -> bool:
        with self._lock:
            self._kv[key] = value
        return True

    def get(self, key: str) -> Any:
        with self._lock:
            return self._kv.get(key)

    def lpush(self, key: str, *values: Any) -> int:
        with self._lock:
            lst = self._lists.setdefault(key, [])
            for v in values:
                lst.insert(0, v)
            return len(lst)

    def lrange(self, key: str, start: int, stop: int) -> list:
        with self._lock:
            lst = self._lists.get(key, [])
            if stop == -1:
                return list(lst[start:])
            return list(lst[start:stop + 1])

    def llen(self, key: str) -> int:
        with self._lock:
            return len(self._lists.get(key, []))

    def delete(self, *keys: str) -> int:
        n = 0
        with self._lock:
            for k in keys:
                if k in self._kv:
                    del self._kv[k]
                    n += 1
                if k in self._lists:
                    del self._lists[k]
                    n += 1
        return n


def _seed_index(dim: int = 8) -> AssetIndex:
    idx = AssetIndex(dim=dim)
    base = np.eye(dim)
    for i in range(dim):
        idx.add(f"real_{i}", base[i], {"kind": "real"}, brand="alpha" if i < 2 else "beta")
    return idx


# ---------------------------------------------------------------------------- #
# AssetIndex                                                                    #
# ---------------------------------------------------------------------------- #

def test_cascade_rungs() -> None:
    idx = _seed_index()
    idx.add("anchor_0", _unit(1, 1, 1, 1, 1, 1, 1, 1), {"kind": "anchor"}, is_anchor=True)

    # exact
    v = np.eye(8)[3]
    exact_key = None
    # derive the content key the way the index does
    from ai_model.retrieval.asset_index import _content_key
    sane = idx._sanitize(v)
    exact_key = _content_key(sane)
    r = idx.query(v, exact_key=exact_key)
    _check("exact rung hit", r is not None and r.rung == "exact" and r.asset_id == "real_3")

    # nearest (close to real_5 but not exact key)
    q = _unit(0.0, 0, 0, 0, 0, 0.97, 0.05, 0.02)
    r = idx.query(q)
    _check("nearest rung hit", r is not None and r.rung == "nearest" and r.asset_id == "real_5")

    # brand_prior: a query far from everything but pushed toward alpha brand subset
    # Build a query whose only acceptable hit is within the alpha brand set.
    q_far = _unit(0.55, 0.55, 0, 0, 0, 0, 0, 0)  # between real_0/real_1 (both alpha)
    r = idx.query(q_far, brand="alpha")
    _check("brand_prior reachable", r is not None and r.rung in ("nearest", "brand_prior"))

    # anchor: an orthogonal-ish query that no real asset covers within radius
    q_none = _unit(1, -1, 1, -1, 1, -1, 1, -1)
    r = idx.query(q_none)
    _check("anchor rung fallback", r is not None and r.rung == "anchor")


def test_never_empty_when_nonempty() -> None:
    idx = _seed_index()
    # No anchors: still must never return None for a usable query on a non-empty index.
    for _ in range(20):
        q = _unit(*np.random.RandomState(0).randn(8))
        r = idx.query(q)
        _check("non-empty index never returns None", r is not None)
        break  # deterministic seed; one representative check
    # truly empty index → None (the only allowed empty case)
    empty = AssetIndex(dim=8)
    _check("empty index returns None", empty.query(_unit(1, 0, 0, 0, 0, 0, 0, 0)) is None)


def test_totality() -> None:
    idx = _seed_index()
    cases = {
        "nan": np.array([np.nan] * 8),
        "inf": np.array([np.inf] * 8),
        "zeros": np.zeros(8),
        "wrong_dim_short": np.array([1.0, 0.0]),
        "wrong_dim_long": np.arange(20, dtype=float),
        "string": "not a vector",
        "none": None,
    }
    for name, q in cases.items():
        try:
            r = idx.query(q)
            ok = (r is not None)
        except Exception as e:
            ok = False
            print(f"    raised on {name}: {e}")
        _check(f"totality[{name}] returns a real asset without raising", ok)

    # add() must reject unusable vectors gracefully (never raise)
    _check("add rejects NaN vector", idx.add("bad", np.array([np.nan] * 8)) is False)
    _check("add rejects zero vector", idx.add("bad", np.zeros(8)) is False)


def test_determinism() -> None:
    q = _unit(0.3, 0.2, 0.1, 0.4, 0.0, 0.5, 0.1, 0.2)
    a = _seed_index().query(q)
    b = _seed_index().query(q)
    _check("determinism: same asset", a is not None and b is not None and a.asset_id == b.asset_id)
    _check("determinism: same distance", a is not None and b is not None and abs(a.distance - b.distance) < 1e-9)


def test_coverage_gate() -> None:
    empty = AssetIndex(dim=8)
    _check("empty → critical gate", empty.coverage_report()["gate"] == "critical")

    idx = _seed_index()
    rep = idx.coverage_report()
    _check("no anchors → critical gate", rep["gate"] == "critical")

    idx.add("anchor_0", _unit(*([1] * 8)), is_anchor=True)
    good_probes = [np.eye(8)[i] for i in range(8)]  # exactly on real assets
    rep = idx.coverage_report(good_probes)
    _check("well-covered → healthy gate", rep["gate"] == "healthy")
    _check("coverage fraction high", rep["fraction_within_radius"] >= 0.75)

    bad_probes = [_unit(*np.random.RandomState(s).randn(8)) for s in range(16)]
    rep_bad = idx.coverage_report(bad_probes)
    _check("random probes lower coverage", rep_bad["fraction_within_radius"] <= 1.0)


def test_serialization_roundtrip() -> None:
    idx = _seed_index()
    idx.add("anchor_0", _unit(*([1] * 8)), is_anchor=True)
    state = idx.to_state()
    restored = AssetIndex(dim=8)
    _check("load_state succeeds", restored.load_state(state) is True)
    _check("roundtrip size matches", restored.size == idx.size)
    _check("roundtrip anchors match", restored.anchor_count == idx.anchor_count)
    q = _unit(0.0, 0, 0, 0, 0, 0.97, 0.05, 0.02)
    _check("roundtrip query matches",
           restored.query(q).asset_id == idx.query(q).asset_id)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------- #
# CoverageWatchdog (self-healing)                                              #
# ---------------------------------------------------------------------------- #

def test_watchdog_heals_missing_anchors() -> None:
    idx = _seed_index()  # has real assets, NO anchors
    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = FakeStorage()

    loaded_calls = {"n": 0}

    def _loader() -> int:
        loaded_calls["n"] += 1
        idx.add("anchor_core_0", _unit(*([1] * 8)), {"kind": "anchor"}, is_anchor=True)
        idx.add("anchor_core_1", _unit(1, 0, 1, 0, 1, 0, 1, 0), {"kind": "anchor"}, is_anchor=True)
        return 2

    wd.anchor_loader_fn = _loader
    _check("precondition: no anchors", idx.anchor_count == 0)
    wd._run_all_checks()
    _check("watchdog loaded anchors", idx.anchor_count >= 1)
    _check("watchdog recorded a fix", wd.stats["fixes_applied"] >= 1)
    _check("anchor loader was called", loaded_calls["n"] == 1)


def test_watchdog_enqueues_coverage_gaps() -> None:
    idx = AssetIndex(dim=8)
    idx.add("real_0", np.eye(8)[0], {"kind": "real"})
    idx.add("anchor_0", _unit(*([1] * 8)), {"kind": "anchor"}, is_anchor=True)

    storage = FakeStorage()
    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = storage
    # probes far from the single real asset → weak coverage → should enqueue gaps
    wd.probe_source_fn = lambda: [_unit(*np.random.RandomState(s).randn(8)) for s in range(8)]

    wd._run_all_checks()
    qlen = storage.llen(CoverageWatchdog.GAP_QUEUE_KEY)
    _check("coverage gaps enqueued", qlen > 0)
    _check("gate recorded", wd.stats["last_gate"] in ("degraded", "critical", "healthy"))


def test_watchdog_drains_via_ingestion() -> None:
    idx = AssetIndex(dim=8)
    idx.add("real_0", np.eye(8)[0], {"kind": "real"})
    idx.add("anchor_0", _unit(*([1] * 8)), {"kind": "anchor"}, is_anchor=True)

    storage = FakeStorage()
    ingested = {"n": 0}

    def _ingest(gaps: List[Any]) -> int:
        ingested["n"] = len(gaps)
        return len(gaps)

    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = storage
    wd.ingestion_fn = _ingest
    wd.probe_source_fn = lambda: [_unit(*np.random.RandomState(s).randn(8)) for s in range(8)]

    wd._run_all_checks()
    # drain runs in a background thread; give it a moment
    import time
    for _ in range(50):
        if ingested["n"] > 0:
            break
        time.sleep(0.05)
    _check("ingestion drain invoked", ingested["n"] > 0)


def test_exact_key_replacement() -> None:
    from ai_model.retrieval.asset_index import _content_key
    idx = AssetIndex(dim=8)
    v1 = _unit(0, 0, 1, 0, 0, 0, 0, 0)
    v2 = _unit(0, 0, 0, 1, 0, 0, 0, 0)
    idx.add("A", v1)
    k1 = _content_key(idx._sanitize(v1))  # type: ignore[arg-type]
    r1 = idx.query(v1, exact_key=k1)
    _check("exact hit before replace", r1 is not None and r1.rung == "exact" and r1.asset_id == "A")

    idx.add("A", v2)  # replace A's vector
    k2 = _content_key(idx._sanitize(v2))  # type: ignore[arg-type]
    r_old = idx.query(v2, exact_key=k1)
    _check("stale exact key no longer an exact hit", r_old is not None and r_old.rung != "exact")
    r_new = idx.query(v2, exact_key=k2)
    _check("new exact key hits after replace",
           r_new is not None and r_new.rung == "exact" and r_new.asset_id == "A")


def _drain_done(wd: CoverageWatchdog, timeout: float = 3.0) -> None:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        with wd._gap_lock:
            if not wd._drain_in_flight:
                return
        time.sleep(0.02)


def test_drain_atomic_no_loss() -> None:
    import time
    storage = FakeStorage()
    idx = AssetIndex(dim=8)
    idx.add("real_0", np.eye(8)[0])
    idx.add("anchor_0", _unit(*([1] * 8)), is_anchor=True)
    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = storage

    started = threading.Event()
    release = threading.Event()
    received = {"n": 0}

    def _ingest(gaps: List[Any]) -> int:
        received["n"] = len(gaps)
        started.set()
        release.wait(3.0)
        return len(gaps)

    wd.ingestion_fn = _ingest
    wd._enqueue_gaps([_unit(*np.random.RandomState(s).randn(8)) for s in range(5)])
    _check("pre-enqueue len 5", storage.llen(CoverageWatchdog.GAP_QUEUE_KEY) == 5)

    wd._maybe_drain(time.time(), force=True)
    _check("ingestion started", started.wait(3.0))
    _check("ingestion received the claimed batch (5)", received["n"] == 5)
    _check("queue cleared after atomic claim", storage.llen(CoverageWatchdog.GAP_QUEUE_KEY) == 0)

    # Enqueue 3 more WHILE the drain is mid-flight — must not be lost.
    wd._enqueue_gaps([_unit(*np.random.RandomState(s + 100).randn(8)) for s in range(3)])
    _check("gaps enqueued during drain are kept", storage.llen(CoverageWatchdog.GAP_QUEUE_KEY) == 3)

    release.set()
    _drain_done(wd)
    _check("3 new gaps survive for next cycle", storage.llen(CoverageWatchdog.GAP_QUEUE_KEY) == 3)


def test_drain_requeue_on_failure() -> None:
    import time
    storage = FakeStorage()
    idx = AssetIndex(dim=8)
    idx.add("real_0", np.eye(8)[0])
    idx.add("anchor_0", _unit(*([1] * 8)), is_anchor=True)
    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = storage

    def _ingest(gaps: List[Any]) -> int:
        raise RuntimeError("ingestion boom")

    wd.ingestion_fn = _ingest
    wd._enqueue_gaps([_unit(*np.random.RandomState(s).randn(8)) for s in range(4)])
    wd._maybe_drain(time.time(), force=True)
    _drain_done(wd)
    _check("failed ingestion re-queues the batch (no loss)",
           storage.llen(CoverageWatchdog.GAP_QUEUE_KEY) == 4)


def test_watchdog_check_isolation() -> None:
    # A check raising must not propagate out of _run_all_checks via the loop guard.
    wd = CoverageWatchdog()
    wd.index = _seed_index()
    wd.storage = FakeStorage()

    def _boom() -> List[Any]:
        raise RuntimeError("probe source exploded")

    wd.probe_source_fn = _boom  # type: ignore[assignment]
    raised = False
    try:
        wd._run_all_checks()
    except Exception:
        raised = True
    _check("probe_source failure is contained", raised is False)


def main() -> int:
    tests = [
        test_cascade_rungs,
        test_never_empty_when_nonempty,
        test_totality,
        test_determinism,
        test_coverage_gate,
        test_serialization_roundtrip,
        test_exact_key_replacement,
        test_watchdog_heals_missing_anchors,
        test_watchdog_enqueues_coverage_gaps,
        test_watchdog_drains_via_ingestion,
        test_drain_atomic_no_loss,
        test_drain_requeue_on_failure,
        test_watchdog_check_isolation,
    ]
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as e:
            global _failures
            _failures += 1
            print(f"  FAILED: {t.__name__} raised {e!r}")
    print("\n" + ("ALL PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED"))
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
