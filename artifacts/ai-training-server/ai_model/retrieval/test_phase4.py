"""
Tests for Phase 4 — brand identity centroid, embedding cache, seeded-render
dedupe, and the generated-asset ingestor.

Run:
    cd artifacts/ai-training-server
    uv run python -u -m ai_model.retrieval.test_phase4
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ai_model.image.image_engine import _UPLOADS_DIR, ImageEngine, ImageRequest
from ai_model.retrieval.asset_index import AssetIndex
from ai_model.retrieval.generated_ingestor import GeneratedIngestor
from ai_model.retrieval.image_features import (
    EMBED_DIM,
    clear_embedding_cache,
    embedding_cache_stats,
    image_to_vector,
)

_FAILS: list = []
_CLEANUP: set = set()


def _check(name: str, cond: bool) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        _FAILS.append(name)


def _new_index() -> AssetIndex:
    return AssetIndex(dim=EMBED_DIM)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


def _rand_vec(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float64)


def _render_seeded(seed: int, prompt: str = "phase4 probe") -> Path:
    eng = ImageEngine()
    res = eng.render(ImageRequest(prompt=prompt, color_scheme="dark_neon",
                                  layout="square_1_1", seed=seed))
    p = Path(_UPLOADS_DIR) / res.filename
    _CLEANUP.add(p)
    return p


# ── Brand centroid ───────────────────────────────────────────────────────────

def test_brand_centroid_mean() -> None:
    print("[test_brand_centroid_mean]")
    idx = _new_index()
    v1, v2 = _rand_vec(1), _rand_vec(2)
    idx.add("a1", v1, brand="nova")
    idx.add("a2", v2, brand="nova")
    _check("brand_count == 2", idx.brand_count("nova") == 2)
    cen = idx.brand_centroid("nova")
    _check("centroid not None", cen is not None)
    if cen is not None:
        # Index stores unit-normalized vectors; centroid is the normalized mean.
        expect = _unit((_unit(v1) + _unit(v2)) / 2.0)
        _check("centroid == normalized mean of stored vecs",
               bool(np.allclose(cen, expect, atol=1e-5)))
        _check("centroid is unit length",
               abs(float(np.linalg.norm(cen)) - 1.0) < 1e-4)
    _check("unknown brand → None", idx.brand_centroid("ghost") is None)
    _check("brand_stats reflects count", idx.brand_stats().get("nova") == 2)


def test_brand_centroid_replace_matches_rebuild() -> None:
    print("[test_brand_centroid_replace_matches_rebuild]")
    # Incrementally maintained centroid must equal a from-scratch build.
    inc = _new_index()
    inc.add("a1", _rand_vec(1), brand="nova")
    inc.add("a2", _rand_vec(2), brand="nova")
    inc.add("a1", _rand_vec(99), brand="nova")  # replace a1 in place

    fresh = _new_index()
    fresh.add("a1", _rand_vec(99), brand="nova")
    fresh.add("a2", _rand_vec(2), brand="nova")

    ci, cf = inc.brand_centroid("nova"), fresh.brand_centroid("nova")
    _check("both centroids present", ci is not None and cf is not None)
    if ci is not None and cf is not None:
        _check("incremental replace == rebuilt centroid",
               bool(np.allclose(ci, cf, atol=1e-5)))
    _check("count still 2 after replace", inc.brand_count("nova") == 2)


def test_brand_centroid_survives_state_roundtrip() -> None:
    print("[test_brand_centroid_survives_state_roundtrip]")
    idx = _new_index()
    idx.add("a1", _rand_vec(1), brand="nova")
    idx.add("a2", _rand_vec(2), brand="nova")
    before = idx.brand_centroid("nova")

    restored = _new_index()
    ok = restored.load_state(idx.to_state())
    _check("load_state ok", bool(ok))
    after = restored.brand_centroid("nova")
    _check("centroid present after restore", after is not None)
    if before is not None and after is not None:
        _check("centroid identical across state round-trip",
               bool(np.allclose(before, after, atol=1e-5)))
    _check("brand_count restored", restored.brand_count("nova") == 2)


def test_brand_centroid_remove_recomputes() -> None:
    print("[test_brand_centroid_remove_recomputes]")
    idx = _new_index()
    idx.add("a1", _rand_vec(1), brand="nova")
    idx.add("a2", _rand_vec(2), brand="nova")
    idx.remove("a1")
    _check("count drops to 1 after remove", idx.brand_count("nova") == 1)
    cen = idx.brand_centroid("nova")
    if cen is not None:
        _check("centroid == remaining vec", bool(np.allclose(cen, _unit(_rand_vec(2)), atol=1e-5)))


# ── Embedding cache ──────────────────────────────────────────────────────────

def test_embedding_cache_hit_and_copy() -> None:
    print("[test_embedding_cache_hit_and_copy]")
    clear_embedding_cache()
    path = _render_seeded(4321)

    v1 = image_to_vector(path)
    s1 = embedding_cache_stats()
    _check("first embed is a miss", s1["misses"] == 1 and s1["hits"] == 0)
    _check("vector dim correct", v1 is not None and v1.shape[0] == EMBED_DIM)

    v2 = image_to_vector(path)
    s2 = embedding_cache_stats()
    _check("second embed is a hit", s2["hits"] == 1)
    if v1 is not None and v2 is not None:
        _check("cached vector equals computed", bool(np.allclose(v1, v2)))
        # Mutating a returned vector must not corrupt the cached copy.
        v2[0] = 999.0
        v3 = image_to_vector(path)
        _check("returned vectors are independent copies",
               v3 is not None and float(v3[0]) != 999.0)


def test_embedding_cache_bypass() -> None:
    print("[test_embedding_cache_bypass]")
    clear_embedding_cache()
    path = _render_seeded(4321)
    image_to_vector(path, use_cache=False)
    s = embedding_cache_stats()
    _check("use_cache=False bypasses cache (no hits/misses recorded)",
           s["hits"] == 0 and s["misses"] == 0 and s["size"] == 0)


# ── Seeded render dedupe ─────────────────────────────────────────────────────

def test_seeded_render_reuses_file() -> None:
    print("[test_seeded_render_reuses_file]")
    eng = ImageEngine()
    req = ImageRequest(prompt="dedupe", color_scheme="dark_neon",
                       layout="square_1_1", seed=24680)
    r1 = eng.render(req)
    p = Path(_UPLOADS_DIR) / r1.filename
    _CLEANUP.add(p)
    _check("first seeded render succeeded", r1.success and p.exists())
    mtime1 = p.stat().st_mtime_ns
    r2 = eng.render(req)
    _check("identical seed → identical filename", r1.filename == r2.filename)
    _check("file reused, not re-rendered", p.stat().st_mtime_ns == mtime1)


def test_seeded_render_concurrent_dedupe() -> None:
    print("[test_seeded_render_concurrent_dedupe]")
    eng = ImageEngine()
    req = ImageRequest(prompt="race", color_scheme="dark_neon",
                       layout="square_1_1", seed=13579)
    names: list = []
    lock = threading.Lock()

    def _go() -> None:
        res = eng.render(req)
        with lock:
            names.append(res.filename)

    threads = [threading.Thread(target=_go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _check("all concurrent renders returned the same file",
           len(set(names)) == 1 and len(names) == 8)
    if names:
        p = Path(_UPLOADS_DIR) / names[0]
        _CLEANUP.add(p)
        _check("output file exists", p.exists())


# ── Generated-asset ingestor ─────────────────────────────────────────────────

def test_ingestor_enqueue_dedupe_and_drain() -> None:
    print("[test_ingestor_enqueue_dedupe_and_drain]")
    idx = _new_index()
    ing = GeneratedIngestor()
    ing.index = idx
    path = _render_seeded(11111)

    first = ing.enqueue(str(path), brand="nova", endpoint="/api/generate/image",
                        platform="instagram")
    second = ing.enqueue(str(path), brand="nova")  # same path → deduped
    _check("first enqueue accepted", first is True)
    _check("duplicate enqueue deduped", second is False)

    drained = ing._drain_once()
    _check("drain ingested one asset", drained == 1)
    _check("index grew by the generated asset", idx.size == 1)
    _check("brand centroid now exists for nova", idx.brand_centroid("nova") is not None)
    st = ing.get_status()
    _check("ingested_total == 1", st["ingested_total"] == 1)
    _check("deduped_total == 1", st["deduped_total"] == 1)
    _check("queue drained empty", st["queue_len"] == 0)


def test_ingestor_bad_path_retries_then_drops() -> None:
    print("[test_ingestor_bad_path_retries_then_drops]")
    idx = _new_index()
    ing = GeneratedIngestor()
    ing.index = idx
    bad = str(Path(_UPLOADS_DIR) / "does_not_exist_phase4.png")

    _check("bad path enqueued", ing.enqueue(bad, brand="nova") is True)
    # MAX_RETRIES=2 → attempts 1,2 re-enqueue; attempt 3 drops.
    for _ in range(3):
        ing._drain_once()
    st = ing.get_status()
    _check("nothing ingested from bad path", st["ingested_total"] == 0)
    _check("dropped after exhausting retries", st["dropped_total"] == 1)
    _check("retries were recorded", st["retried_total"] == 2)
    _check("queue empty after drop", st["queue_len"] == 0)
    _check("index unchanged by failed ingest", idx.size == 0)


def _cleanup() -> None:
    for p in _CLEANUP:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    try:
        test_brand_centroid_mean()
        test_brand_centroid_replace_matches_rebuild()
        test_brand_centroid_survives_state_roundtrip()
        test_brand_centroid_remove_recomputes()
        test_embedding_cache_hit_and_copy()
        test_embedding_cache_bypass()
        test_seeded_render_reuses_file()
        test_seeded_render_concurrent_dedupe()
        test_ingestor_enqueue_dedupe_and_drain()
        test_ingestor_bad_path_retries_then_drops()
    finally:
        _cleanup()

    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + ", ".join(_FAILS))
        raise SystemExit(1)
    print("ALL PHASE 4 TESTS PASSED")


if __name__ == "__main__":
    main()
