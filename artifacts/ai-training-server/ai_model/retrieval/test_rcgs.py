"""
Tests for the live RCGS query path: the probe ring buffer, retrieval-conditioned
background compositing, and the CoverageWatchdog's context-carrying probe
handling.

Run:
    cd artifacts/ai-training-server
    uv run python -u -m ai_model.retrieval.test_rcgs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai_model.image.image_engine import _UPLOADS_DIR
from ai_model.retrieval import rcgs as rcgs_mod
from ai_model.retrieval.asset_index import AssetIndex
from ai_model.retrieval.asset_pipeline import get_asset_index
from ai_model.retrieval.coverage_watchdog import CoverageWatchdog, _probe_vector
from ai_model.retrieval.image_features import EMBED_DIM, image_to_vector
from ai_model.retrieval.probes import (
    clear_probes,
    probe_count,
    recent_probes,
    record_probe,
)

_FAILS: List[str] = []


def _check(name: str, cond: bool) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        _FAILS.append(name)


def _rand_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeStorage:
    def __init__(self) -> None:
        self.is_available = True
        self._kv: Dict[str, Any] = {}
        self._lists: Dict[str, list] = {}

    def get(self, k: str) -> Any:
        return self._kv.get(k)

    def set(self, k: str, v: Any) -> None:
        self._kv[k] = v

    def delete(self, k: str) -> None:
        self._kv.pop(k, None)
        self._lists.pop(k, None)

    def lpush(self, k: str, *vals: Any) -> None:
        self._lists.setdefault(k, [])
        for v in vals:
            self._lists[k].insert(0, v)

    def lrange(self, k: str, start: int, end: int) -> list:
        items = self._lists.get(k, [])
        if end == -1:
            return list(items[start:])
        return list(items[start:end + 1])

    def llen(self, k: str) -> int:
        return len(self._lists.get(k, []))


# ── Probe ring buffer ────────────────────────────────────────────────────────

def test_probes() -> None:
    print("test_probes")
    clear_probes()

    _check("empty buffer count is 0", probe_count() == 0)

    ok = record_probe(_rand_vec(1), {"brand": "Nova", "layout": "square_1_1"})
    _check("good probe recorded", ok and probe_count() == 1)

    # Same vector + context within TTL → deduped.
    dup = record_probe(_rand_vec(1), {"brand": "Nova", "layout": "square_1_1"})
    _check("identical probe deduped within TTL", (not dup) and probe_count() == 1)

    # Same vector, different context → not deduped.
    ok2 = record_probe(_rand_vec(1), {"brand": "Other", "layout": "square_1_1"})
    _check("different context not deduped", ok2 and probe_count() == 2)

    # Unusable vectors rejected.
    _check("zero vector rejected", not record_probe(np.zeros(EMBED_DIM)))
    _check("nan vector rejected", not record_probe(np.full(EMBED_DIM, np.nan)))
    _check("empty vector rejected", not record_probe([]))

    # recent_probes: newest-first, capped, payload shape.
    clear_probes()
    for i in range(5):
        record_probe(_rand_vec(100 + i), {"layout": "vertical_9_16"})
    out = recent_probes(3)
    _check("recent_probes caps at n", len(out) == 3)
    _check("recent_probes payload shape", all(
        set(p.keys()) == {"vector", "context"} for p in out))
    newest = recent_probes(5)
    _check("recent_probes newest-first",
           np.allclose(newest[0]["vector"], _rand_vec(104)))
    _check("recent_probes context preserved",
           newest[0]["context"].get("layout") == "vertical_9_16")
    _check("recent_probes(0) is empty", recent_probes(0) == [])

    clear_probes()


# ── RCGS path resolution (sandboxing) ────────────────────────────────────────

def _make_real_png(name: str, color: tuple) -> Path:
    from PIL import Image
    base = Path(_UPLOADS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    Image.new("RGB", (96, 96), color).save(p, format="PNG")
    return p


def _make_structured_png(name: str) -> Path:
    """A PNG with real luminance structure (so it can modulate a frame)."""
    from PIL import Image
    base = Path(_UPLOADS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    ys = np.linspace(0, 255, 96, dtype=np.float32).reshape(-1, 1)
    xs = np.linspace(0, 255, 96, dtype=np.float32).reshape(1, -1)
    lum = np.clip((ys + xs) / 2.0, 0, 255).astype(np.uint8)  # diagonal gradient
    arr = np.stack([lum, lum, lum], axis=2)
    Image.fromarray(arr, "RGB").save(p, format="PNG")
    return p


def test_path_resolution() -> None:
    print("test_path_resolution")
    png = _make_real_png("rcgs_test_asset.png", (40, 120, 200))

    r1 = rcgs_mod._resolve_local_path({"filename": "rcgs_test_asset.png"})
    _check("resolves real filename under uploads", r1 is not None and r1.is_file())

    r2 = rcgs_mod._resolve_local_path({"path": str(png)})
    _check("resolves real path under uploads", r2 is not None and r2.is_file())

    r3 = rcgs_mod._resolve_local_path({"filename": "../../../etc/passwd"})
    _check("rejects traversal (basename-only forces uploads dir)",
           r3 is None or r3.parent == Path(_UPLOADS_DIR).resolve())

    r4 = rcgs_mod._resolve_local_path({"filename": "does_not_exist_xyz.png"})
    _check("rejects nonexistent file", r4 is None)

    r5 = rcgs_mod._resolve_local_path({"url": "https://example.com/a.png"})
    _check("ignores url-only metadata", r5 is None)

    r6 = rcgs_mod._resolve_local_path({})
    _check("empty metadata → None", r6 is None)

    try:
        png.unlink()
    except OSError:
        pass


# ── RCGS background conditioning ─────────────────────────────────────────────

def test_condition_background() -> None:
    print("test_condition_background")
    clear_probes()

    # A procedural gradient background (what scenes.py produces).
    h, w = 128, 96
    base: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(20, 180, h, dtype=np.uint8).reshape(-1, 1)
    base[:, :, 2] = np.linspace(60, 220, h, dtype=np.uint8).reshape(-1, 1)

    # Empty singleton index → query returns None → background unchanged, but a
    # probe IS still recorded (coverage must see the demand).
    idx = get_asset_index()
    # Make sure the singleton is empty for this assertion.
    while idx.size:
        idx.remove(idx._ids[0])
    before = probe_count()
    out_empty = rcgs_mod.condition_background(base.copy(), w, h)
    _check("empty index → unchanged background",
           np.array_equal(out_empty, base))
    _check("probe recorded even when index empty", probe_count() == before + 1)

    # Populate the singleton with a real, STRUCTURED asset on disk (a flat
    # solid-color asset correctly adds no structure → nothing to modulate).
    png = _make_structured_png("rcgs_cond_asset.png")
    vec = image_to_vector(png)
    _check("asset embeds", vec is not None)
    idx.add("rcgs_cond_asset", vec, {"filename": "rcgs_cond_asset.png"},
            is_anchor=True)

    out = rcgs_mod.condition_background(base.copy(), w, h, brand="Nova")
    _check("conditioned output keeps shape", out.shape == base.shape)
    _check("conditioned output is uint8", out.dtype == np.uint8)
    _check("conditioning actually modulated the frame",
           not np.array_equal(out, base))
    # Hue should be roughly preserved: green channel was ~0, must stay low.
    _check("palette preserved (near-zero channel stays low)",
           int(out[:, :, 1].max()) <= 8)

    # Totality: a malformed background is returned unchanged.
    bad: np.ndarray = np.zeros((10, 10), dtype=np.uint8)  # 2-D, not HxWx3
    _check("malformed bg returned unchanged",
           np.array_equal(rcgs_mod.condition_background(bad, 10, 10), bad))

    # Structure map is mean-centered and correctly shaped.
    mod = rcgs_mod._structure_map(png, w, h)
    _check("structure map shape", mod is not None and mod.shape == (h, w))
    _check("structure map mean-centered", mod is not None and abs(float(mod.mean())) < 0.05)

    idx.remove("rcgs_cond_asset")
    try:
        png.unlink()
    except OSError:
        pass
    clear_probes()


# ── Watchdog context-carrying probe handling ─────────────────────────────────

def test_watchdog_probe_context() -> None:
    print("test_watchdog_probe_context")

    _check("_probe_vector unwraps payload",
           _probe_vector({"vector": [1.0, 2.0], "context": {}}) == [1.0, 2.0])
    _check("_probe_vector passes through raw", _probe_vector([3.0]) == [3.0])

    # Index with ONLY an anchor → any off-target probe resolves on the "anchor"
    # rung, i.e. is "weak" and must be enqueued WITH its context preserved.
    idx = AssetIndex(dim=4)
    idx.add("anchor", [1.0, 0.0, 0.0, 0.0], {"kind": "anchor"}, is_anchor=True)

    wd = CoverageWatchdog()
    wd.index = idx
    wd.storage = FakeStorage()

    payloads = [
        {"vector": [0.0, 1.0, 0.0, 0.0], "context": {"brand": "Nova", "layout": "vertical_9_16"}},
        {"vector": [0.0, 0.0, 1.0, 0.0], "context": {"color_scheme": "dark_neon"}},
    ]
    wd.probe_source_fn = lambda: list(payloads)

    wd._check_coverage(time.time())
    queued = wd.storage.lrange(wd.GAP_QUEUE_KEY, 0, -1)
    _check("weak probes enqueued", len(queued) == 2)
    _check("enqueued payloads carry context",
           all(isinstance(g.get("context"), dict) and g["context"] for g in queued))
    brands = {g["context"].get("brand") for g in queued}
    _check("brand context survives to gap queue", "Nova" in brands)


def main() -> int:
    test_probes()
    test_path_resolution()
    test_condition_background()
    test_watchdog_probe_context()
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): {_FAILS}")
        return 1
    print("ALL RCGS TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
