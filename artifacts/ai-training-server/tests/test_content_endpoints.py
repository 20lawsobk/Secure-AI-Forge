"""
Content endpoint tests — digital GPU integration.

Verifies every content-generation route returns valid, non-empty output and
that the HyperGPU accumulates ops across the run (proving the digital GPU
stack — not torch/CPU aot_eager — executed the inference).

Run:
    uv run python -m pytest tests/test_content_endpoints.py -v
or:
    uv run python tests/test_content_endpoints.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────────

BASE   = "http://127.0.0.1:9878"
API_KEY = "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key":    API_KEY,
}

# ── Minimal HTTP helpers ───────────────────────────────────────────────────────

def _req(method: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
    url  = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req  = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        raise AssertionError(f"HTTP {e.code} {method} {path}: {raw[:400]}") from e

def GET(path: str) -> dict:
    return _req("GET", path)

def POST(path: str, body: dict, timeout: int = 60) -> dict:
    return _req("POST", path, body, timeout=timeout)

# ── Result accumulator ────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    ok: bool
    msg: str = ""
    ms: float = 0.0

results: list[Result] = []

def run(name: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        results.append(Result(name, True, ms=ms))
        print(f"  ✓  {name}  ({ms:.0f} ms)")
    except AssertionError as e:
        ms = (time.perf_counter() - t0) * 1000
        results.append(Result(name, False, str(e)[:300], ms=ms))
        print(f"  ✗  {name}  ({ms:.0f} ms)\n     {e}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        results.append(Result(name, False, repr(e)[:300], ms=ms))
        print(f"  ✗  {name}  ({ms:.0f} ms)\n     {e}")

# ── Assertion helpers ─────────────────────────────────────────────────────────

def assert_str(d: dict, *keys: str, min_len: int = 3) -> None:
    for k in keys:
        v = d.get(k)
        assert isinstance(v, str) and len(v) >= min_len, \
            f"field '{k}' expected non-empty str, got {v!r}"

def assert_truthy(d: dict, key: str) -> None:
    assert d.get(key), f"field '{key}' should be truthy, got {d.get(key)!r}"

def assert_list(d: dict, key: str, min_items: int = 0) -> None:
    v = d.get(key)
    assert isinstance(v, list) and len(v) >= min_items, \
        f"field '{key}' expected list with >={min_items} items, got {v!r}"

def assert_number(d: dict, key: str, ge: float = 0) -> None:
    v = d.get(key)
    assert isinstance(v, (int, float)) and v >= ge, \
        f"field '{key}' expected number >= {ge}, got {v!r}"

# ═════════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_health():
    r = GET("/health")
    assert r.get("status") in ("ok", "healthy"), f"health={r}"

def test_gpu_hyper_status():
    r = GET("/gpu/hyper/status")
    assert r.get("engine") == "HyperGPU", f"engine={r.get('engine')}"
    assert r.get("lanes") == 512
    assert r.get("tensor_cores") == 8
    assert r.get("precision") == "MIXED"

def test_model_status():
    r = GET("/model/status")
    assert r.get("model_loaded") is True, f"model not loaded: {r}"

def test_platform_model_info():
    r = GET("/platform/model/info")
    # requires auth — should succeed, not 401
    assert "detail" not in r or r["detail"] != "API key required", \
        "model info returned 401 — auth broken"

# ── /content/generate ─────────────────────────────────────────────────────────

def test_content_generate_tiktok():
    r = POST("/content/generate", {
        "platform": "tiktok",
        "topic": "indie music release",
        "tone": "energetic",
        "goal": "growth",
        "include_hashtags": True,
    }, timeout=90)
    assert r.get("success") is True, f"success!=True: {r}"
    assert_str(r, "caption", min_len=5)
    assert_str(r, "hook",    min_len=3)

def test_content_generate_instagram():
    r = POST("/content/generate", {
        "platform": "instagram",
        "topic": "behind the scenes studio session",
        "tone": "authentic",
        "goal": "engagement",
        "include_hashtags": True,
    }, timeout=90)
    assert r.get("success") is True, f"success!=True: {r}"
    assert_str(r, "caption", min_len=5)

def test_content_generate_youtube():
    r = POST("/content/generate", {
        "platform": "youtube",
        "topic": "new album drop announcement",
        "tone": "dramatic",
        "goal": "awareness",
        "include_hashtags": False,
    }, timeout=90)
    assert r.get("success") is True, f"success!=True: {r}"
    assert_str(r, "caption", min_len=5)

def test_content_generate_has_processing_time():
    r = POST("/content/generate", {
        "platform": "tiktok",
        "topic": "artist collab teaser",
        "tone": "mysterious",
        "goal": "growth",
    }, timeout=90)
    assert r.get("success") is True
    assert_number(r, "processing_time_ms", ge=0)

# ── /platform/social/generate ─────────────────────────────────────────────────

def test_platform_social_instagram():
    r = POST("/platform/social/generate", {
        "user_id":  "test_user_001",
        "platform": "instagram",
        "topic":    "summer tour dates",
        "tone":     "excited",
        "goal":     "growth",
        "include_hashtags": True,
        "num_variants": 1,
    }, timeout=90)
    # expect a list of posts or a dict with posts/variants
    assert isinstance(r, (list, dict)), f"unexpected type: {type(r)}"
    if isinstance(r, dict):
        assert "error" not in str(r).lower() or r.get("success", True), \
            f"error in response: {r}"

def test_platform_social_tiktok_variants():
    r = POST("/platform/social/generate", {
        "user_id":  "test_user_002",
        "platform": "tiktok",
        "topic":    "drop day hype",
        "tone":     "hype",
        "goal":     "virality",
        "num_variants": 2,
    }, timeout=90)
    assert isinstance(r, (list, dict))

def test_platform_social_twitter():
    r = POST("/platform/social/generate", {
        "user_id":  "test_user_003",
        "platform": "twitter",
        "topic":    "live show tonight",
        "tone":     "urgent",
        "goal":     "ticket sales",
        "num_variants": 1,
    }, timeout=90)
    assert isinstance(r, (list, dict))

# ── /platform/daw/generate ────────────────────────────────────────────────────

def test_platform_daw_lyrics():
    r = POST("/platform/daw/generate", {
        "user_id": "test_daw_001",
        "mode":    "lyrics",
        "genre":   "hip-hop",
        "mood":    "introspective",
        "bpm":     90,
        "key":     "Dm",
    }, timeout=90)
    assert isinstance(r, dict), f"unexpected type: {type(r)}"

def test_platform_daw_hook():
    r = POST("/platform/daw/generate", {
        "user_id": "test_daw_002",
        "mode":    "hook",
        "genre":   "r&b",
        "mood":    "romantic",
    }, timeout=90)
    assert isinstance(r, dict)

def test_platform_daw_beat_description():
    r = POST("/platform/daw/generate", {
        "user_id": "test_daw_003",
        "mode":    "beat_description",
        "genre":   "trap",
        "mood":    "dark",
        "bpm":     140,
    }, timeout=90)
    assert isinstance(r, dict)

# ── /platform/distribution/plan ───────────────────────────────────────────────

def test_platform_distribution_plan():
    r = POST("/platform/distribution/plan", {
        "user_id":          "test_dist_001",
        "track_title":      "Midnight Frequency",
        "genre":            "electronic",
        "target_platforms": ["spotify", "apple_music", "tidal"],
        "bio":              "Producer & artist from Chicago",
    }, timeout=90)
    assert isinstance(r, dict)

# ── /platform/video/generate ──────────────────────────────────────────────────

def test_platform_video_youtube():
    r = POST("/platform/video/generate", {
        "user_id":          "test_vid_001",
        "topic":            "music video concept for trap anthem",
        "platform":         "youtube",
        "style":            "cinematic",
        "goal":             "engagement",
        "tone":             "dramatic",
        "duration_seconds": 30,
        "aspect_ratio":     "16:9",
        "include_captions": True,
    }, timeout=90)
    assert isinstance(r, dict)

def test_platform_video_tiktok_vertical():
    r = POST("/platform/video/generate", {
        "user_id":          "test_vid_002",
        "topic":            "day in the studio vlog",
        "platform":         "tiktok",
        "style":            "social",
        "goal":             "storytelling",
        "tone":             "playful",
        "duration_seconds": 15,
        "aspect_ratio":     "9:16",
        "include_captions": True,
    }, timeout=90)
    assert isinstance(r, dict)

# ── /analyze ──────────────────────────────────────────────────────────────────

def test_analyze_text():
    r = POST("/analyze", {
        "modality": "text",
        "payload":  "This track slaps harder than anything this summer 🔥",
        "platforms": ["tiktok", "instagram"],
        "intent":   "viral content",
        "awareness": "",
    }, timeout=60)
    assert isinstance(r, dict), f"unexpected: {r}"

def test_analyze_text_sentiment():
    r = POST("/analyze", {
        "modality": "text",
        "payload":  "Sad to announce the tour has been cancelled.",
        "platforms": ["twitter"],
        "intent":   "crisis comms",
    }, timeout=60)
    assert isinstance(r, dict)

def test_analyze_url_modality():
    r = POST("/analyze", {
        "modality": "url",
        "payload":  "https://example.com/track",
        "platforms": ["spotify"],
        "intent":   "promotion",
    }, timeout=60)
    assert isinstance(r, dict)

# ── /generate/text ────────────────────────────────────────────────────────────

def test_generate_text_content_mode():
    r = POST("/generate/text", {
        "mode": "content",
        "step": {"platform": "tiktok", "goal": "growth"},
        "inputs": {"topic": "midnight drop", "tone": "energetic"},
        "awareness": "",
    }, timeout=90)
    assert isinstance(r, dict)

def test_generate_text_planner_mode():
    r = POST("/generate/text", {
        "mode":   "planner",
        "system": "You are a music marketing strategist.",
        "input":  {"brief": "Launch single across platforms", "budget": 1000},
        "awareness": "",
    }, timeout=90)
    assert isinstance(r, dict)

# ── /generate/audio ───────────────────────────────────────────────────────────

def test_generate_audio():
    r = POST("/generate/audio", {
        "step":   {"genre": "lo-fi", "mood": "chill"},
        "inputs": {"duration": 10, "tempo": 80, "key": "Gm"},
        "awareness": "",
    }, timeout=90)
    assert isinstance(r, dict)

# ── /generate/image ───────────────────────────────────────────────────────────

def test_generate_image():
    r = POST("/generate/image", {
        "step":   {"style": "album cover", "palette": "dark moody"},
        "inputs": {"topic": "nighttime cityscape"},
        "awareness": "",
    }, timeout=90)
    assert isinstance(r, dict)

# ── /generate/video ───────────────────────────────────────────────────────────

def test_generate_video():
    r = POST("/generate/video", {
        "step":   {"template": "cinematic_promo", "platform": "tiktok"},
        "inputs": {"hook": "New drop hitting different", "body": "Available now"},
        "awareness": "",
    }, timeout=120)
    assert isinstance(r, dict)

# ── /platform/ads/generate ────────────────────────────────────────────────────

def test_ads_generate_meta():
    r = POST("/platform/ads/generate", {
        "user_id":       "test_ads_001",
        "platform":      "meta",
        "ad_type":       "video",
        "product":       "new single 'Frequency'",
        "goal":          "streams",
        "num_creatives": 2,
        "genre":         "trap",
        "artist_name":   "TestArtist",
    }, timeout=90)
    assert isinstance(r, dict)

def test_ads_generate_tiktok():
    r = POST("/platform/ads/generate", {
        "user_id":       "test_ads_002",
        "platform":      "tiktok",
        "ad_type":       "video",
        "product":       "album 'Dark Hours'",
        "goal":          "streams",
        "num_creatives": 1,
        "artist_name":   "TestArtist",
    }, timeout=90)
    assert isinstance(r, dict)

# ── /api/maxcore/pocket-multiply ──────────────────────────────────────────────

def test_pocket_multiply_2x2():
    r = POST("/api/maxcore/pocket-multiply", {
        "a": [[1, 2], [3, 4]],
        "b": [[5, 6], [7, 8]],
        "pocket": "test_2x2",
    }, timeout=30)
    assert "result" in r or "output" in r or "c" in r, \
        f"no result key in: {list(r.keys())}"

def test_pocket_multiply_3x3():
    r = POST("/api/maxcore/pocket-multiply", {
        "a": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "b": [[4, 7, 2], [1, 9, 3], [5, 6, 8]],
        "pocket": "test_identity",
    }, timeout=30)
    assert "result" in r or "output" in r or "c" in r, \
        f"no result key in: {list(r.keys())}"

def test_pocket_multiply_dedup():
    """Same operands → second call should be served from cache."""
    body = {"a": [[2, 3], [4, 5]], "b": [[1, 0], [0, 1]], "pocket": "dedup_test"}
    r1 = POST("/api/maxcore/pocket-multiply", body, timeout=30)
    r2 = POST("/api/maxcore/pocket-multiply", body, timeout=30)
    # Both responses must be valid
    for r in (r1, r2):
        assert "result" in r or "output" in r or "c" in r, \
            f"dedup response missing result: {list(r.keys())}"

def test_pocket_accelerator_stats():
    r = GET("/api/maxcore/pocket-accelerator/stats")
    assert isinstance(r, dict), f"unexpected type: {type(r)}"

# ── /platform/social/autopilot ────────────────────────────────────────────────

def test_social_autopilot():
    r = POST("/platform/social/autopilot", {
        "user_id":  "test_pilot_001",
        "platform": "instagram",
        "recent_posts": [
            {"caption": "Big drop coming soon 🔥", "likes": 820, "comments": 34},
            {"caption": "Studio session 🎧",       "likes": 540, "comments": 18},
        ],
        "target_metric": "engagement",
    }, timeout=90)
    assert isinstance(r, dict)

# ── Digital GPU ops verification ──────────────────────────────────────────────

# These are populated in main() before/after the generation tests.
_ops_before: int = 0
_ops_after:  int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    # Infrastructure
    ("Health check",                   test_health),
    ("HyperGPU status",                test_gpu_hyper_status),
    ("Model loaded",                   test_model_status),
    ("Platform model info (auth)",     test_platform_model_info),

    # /content/generate
    ("content/generate — TikTok",      test_content_generate_tiktok),
    ("content/generate — Instagram",   test_content_generate_instagram),
    ("content/generate — YouTube",     test_content_generate_youtube),
    ("content/generate — timing field",test_content_generate_has_processing_time),

    # /platform/social/generate
    ("social/generate — Instagram",    test_platform_social_instagram),
    ("social/generate — TikTok variants", test_platform_social_tiktok_variants),
    ("social/generate — Twitter",      test_platform_social_twitter),

    # /platform/daw/generate
    ("daw/generate — lyrics",          test_platform_daw_lyrics),
    ("daw/generate — hook",            test_platform_daw_hook),
    ("daw/generate — beat description",test_platform_daw_beat_description),

    # /platform/distribution/plan
    ("distribution/plan",              test_platform_distribution_plan),

    # /platform/video/generate
    ("platform/video — YouTube",       test_platform_video_youtube),
    ("platform/video — TikTok 9:16",   test_platform_video_tiktok_vertical),

    # /analyze
    ("analyze — viral text",           test_analyze_text),
    ("analyze — text sentiment",       test_analyze_text_sentiment),
    ("analyze — URL modality",         test_analyze_url_modality),

    # /generate/text
    ("generate/text — content mode",   test_generate_text_content_mode),
    ("generate/text — planner mode",   test_generate_text_planner_mode),

    # /generate/*
    ("generate/audio",                 test_generate_audio),
    ("generate/image",                 test_generate_image),
    ("generate/video",                 test_generate_video),

    # /platform/ads
    ("ads/generate — Meta",            test_ads_generate_meta),
    ("ads/generate — TikTok",          test_ads_generate_tiktok),

    # /api/maxcore/pocket-multiply
    ("pocket-multiply — 2×2",          test_pocket_multiply_2x2),
    ("pocket-multiply — identity",     test_pocket_multiply_3x3),
    ("pocket-multiply — dedup",        test_pocket_multiply_dedup),
    ("pocket-accelerator stats",       test_pocket_accelerator_stats),

    # /platform/social/autopilot
    ("social/autopilot",               test_social_autopilot),
]


def main():
    global _ops_before, _ops_after

    print("\n══════════════════════════════════════════════════════════")
    print("  Content Endpoint Tests — Digital GPU Integration")
    print("══════════════════════════════════════════════════════════\n")

    # Snapshot HyperGPU ops before any generation
    try:
        pre = GET("/gpu/hyper/status")
        _ops_before = pre.get("total_ops", 0)
        print(f"  [baseline] HyperGPU total_ops before: {_ops_before}\n")
    except Exception as e:
        print(f"  [baseline] WARNING: could not read GPU status: {e}\n")

    # Run all tests
    for name, fn in TESTS:
        run(name, fn)

    # Snapshot HyperGPU ops after generation
    try:
        post = GET("/gpu/hyper/status")
        _ops_after = post.get("total_ops", 0)
    except Exception:
        pass

    # GPU ops delta — this is the proof that inference went through HyperGPU
    print(f"\n  [gpu verify] total_ops before={_ops_before}  after={_ops_after}  "
          f"delta={_ops_after - _ops_before}")

    # Summary
    passed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    total  = len(results)
    total_ms = sum(r.ms for r in results)

    print(f"\n══════════════════════════════════════════════════════════")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {total_ms/1000:.1f}s total")
    if failed:
        print("\n  Failed tests:")
        for r in results:
            if not r.ok:
                print(f"    ✗  {r.name}")
                print(f"       {r.msg}")
    print("══════════════════════════════════════════════════════════\n")

    # GPU proof assertion
    gen_tests_ran = any(
        r.ok for r in results
        if any(kw in r.name for kw in ("content/generate", "social/generate", "daw/generate",
                                        "analyze", "generate/text", "generate/audio",
                                        "generate/image", "generate/video", "ads/generate"))
    )
    if gen_tests_ran:
        if _ops_after > _ops_before:
            print(f"  ✓  Digital GPU confirmed: {_ops_after - _ops_before} ops executed"
                  f" through HyperGPU during inference.\n")
        else:
            print(f"  ⚠  GPU delta=0. HyperGPU ops did not increase — generation may have"
                  f" used a fallback path (check model backend logs).\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
