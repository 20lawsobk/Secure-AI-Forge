"""
Tests for the retrieval real-asset pipeline (image features + anchor/seed library
+ gap ingestion). Renders REAL images via the in-house ImageEngine, so it is a
slower integration test than test_retrieval.py.

Run:
    cd artifacts/ai-training-server
    uv run python -u -m ai_model.retrieval.test_pipeline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_model.image.image_engine import _UPLOADS_DIR, ImageEngine, ImageRequest
from ai_model.retrieval.asset_index import MIN_ANCHORS, AssetIndex
from ai_model.retrieval.asset_pipeline import (
    decode_scheme,
    ensure_library,
    ingest_gaps,
    ingest_generated_asset,
)
from ai_model.retrieval.image_features import EMBED_DIM, image_to_vector

_FAILS: list = []


def _check(name: str, cond: bool) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        _FAILS.append(name)


def _new_index() -> AssetIndex:
    return AssetIndex(dim=EMBED_DIM)


def test_features_deterministic_and_dim() -> None:
    eng = ImageEngine()
    res = eng.render(ImageRequest(prompt="determinism probe", color_scheme="dark_neon",
                                  layout="square_1_1", seed=1234))
    _check("seeded render succeeded", res.success)
    path = Path(_UPLOADS_DIR) / res.filename
    v1 = image_to_vector(path)
    v2 = image_to_vector(path)
    _check("feature vector not None", v1 is not None)
    if v1 is not None and v2 is not None:
        _check("feature dim == EMBED_DIM", v1.shape[0] == EMBED_DIM)
        _check("feature is finite", bool(np.all(np.isfinite(v1))))
        _check("feature is unit-normalized", abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4)
        _check("feature deterministic (same file → same vector)",
               bool(np.allclose(v1, v2)))


def test_seeded_render_is_idempotent_file() -> None:
    eng = ImageEngine()
    r1 = eng.render(ImageRequest(prompt="idem", color_scheme="warm_earth",
                                 layout="square_1_1", seed=77))
    r2 = eng.render(ImageRequest(prompt="idem", color_scheme="warm_earth",
                                 layout="square_1_1", seed=77))
    _check("stable filename across identical seeded renders", r1.filename == r2.filename)
    v1 = image_to_vector(Path(_UPLOADS_DIR) / r1.filename)
    v2 = image_to_vector(Path(_UPLOADS_DIR) / r2.filename)
    _check("seeded render pixels reproducible",
           v1 is not None and v2 is not None and bool(np.allclose(v1, v2)))


def test_features_total_on_garbage() -> None:
    _check("None input → None (no raise)", image_to_vector(None) is None)
    _check("missing path → None (no raise)",
           image_to_vector("/nonexistent/xyz.png") is None)
    # Black image is degenerate but valid: should not raise.
    try:
        image_to_vector(np.zeros((16, 16, 3), dtype=np.uint8))
        ok = True
    except Exception:
        ok = False
    _check("black image does not raise", ok)


def test_ensure_library_idempotent() -> None:
    idx = _new_index()
    a1 = ensure_library(idx)
    size1, anchors1 = idx.size, idx.anchor_count
    a2 = ensure_library(idx)
    size2, anchors2 = idx.size, idx.anchor_count
    _check("anchors loaded >= MIN_ANCHORS", a1 >= MIN_ANCHORS)
    _check("anchor_count stable & nonzero", anchors1 == anchors2 and anchors1 > 0)
    _check("library is idempotent (size unchanged on re-run)", size1 == size2)
    _check("real (non-anchor) seed assets present", (size1 - anchors1) > 0)
    _check("ensure_library returns anchor count", a1 == a2 == anchors1)


def test_coverage_gate_healthy_after_library() -> None:
    idx = _new_index()
    ensure_library(idx)
    report = idx.coverage_report()
    _check("has_anchors after library", report.get("has_anchors") is True)
    _check("n_real > 0 after library", report.get("n_real", 0) > 0)
    _check("structural gate healthy (no probes needed)", report.get("gate") == "healthy")


def test_decode_scheme_valid() -> None:
    from ai_model.image.image_engine import COLOR_SCHEMES, _gradient_array
    from PIL import Image
    matched = 0
    for scheme in COLOR_SCHEMES:
        top, bottom = COLOR_SCHEMES[scheme][0], COLOR_SCHEMES[scheme][1]
        arr = _gradient_array(64, 64, top, bottom)
        vec = image_to_vector(Image.fromarray(arr, "RGB"))
        decoded = decode_scheme(vec)
        _check(f"decode_scheme({scheme}) → valid scheme", decoded in COLOR_SCHEMES)
        if decoded == scheme:
            matched += 1
    _check("decode_scheme recovers most schemes (>=5/7)", matched >= 5)


def test_ingest_gaps_adds_real_asset() -> None:
    idx = _new_index()
    ensure_library(idx)
    before = idx.size
    rng = np.random.RandomState(3)
    gap_vec = rng.randn(EMBED_DIM)
    gap_vec = (gap_vec / np.linalg.norm(gap_vec)).tolist()

    n_dict = ingest_gaps(idx, [{"ts": 0.0, "vector": gap_vec}])
    _check("ingest_gaps (dict payload) added >=1 real asset", n_dict >= 1)
    _check("index grew after dict-gap ingestion", idx.size > before)

    mid = idx.size
    n_ctx = ingest_gaps(idx, [{"vector": gap_vec,
                               "context": {"color_scheme": "warm_earth",
                                           "layout": "vertical_9_16",
                                           "brand": "artistX"}}])
    _check("ingest_gaps (context payload) added asset", n_ctx >= 1)
    _check("context ingestion registered brand", "artistX" in idx.brands())
    _check("index grew after context-gap ingestion", idx.size > mid)

    _check("ingest_gaps total on empty list", ingest_gaps(idx, []) == 0)


def test_ingest_generated_asset() -> None:
    idx = _new_index()
    ensure_library(idx)
    eng = ImageEngine()
    res = eng.render(ImageRequest(prompt="real generated promo",
                                  color_scheme="high_contrast", layout="landscape_16_9"))
    before = idx.size
    ok = ingest_generated_asset(idx, Path(_UPLOADS_DIR) / res.filename,
                                brand="artistY", asset_id="gen::test1")
    _check("ingest_generated_asset succeeded", ok is True)
    _check("generated asset added to index", idx.size == before + 1)
    _check("generated asset retrievable as exact/nearest",
           idx.query(image_to_vector(Path(_UPLOADS_DIR) / res.filename)) is not None)
    _check("ingest_generated_asset total on bad path",
           ingest_generated_asset(idx, "/nope/none.png") is False)


def main() -> int:
    tests = [
        test_features_deterministic_and_dim,
        test_seeded_render_is_idempotent_file,
        test_features_total_on_garbage,
        test_ensure_library_idempotent,
        test_coverage_gate_healthy_after_library,
        test_decode_scheme_valid,
        test_ingest_gaps_adds_real_asset,
        test_ingest_generated_asset,
    ]
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as exc:  # pragma: no cover
            import traceback
            traceback.print_exc()
            _FAILS.append(f"{t.__name__}:EXC:{exc}")
    print("\n" + ("ALL PASSED" if not _FAILS else f"FAILED: {_FAILS}"))
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
