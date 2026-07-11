"""Regression tests for the release-campaign planner (``build_campaign``).

These guard the campaign against silently degrading into near-duplicate copy: a
single release fans out into 15 posts that all share one title, so it is easy for
the templated hooks and the composed bodies to drift into repetition. They also
guard against the known failure mode where the human-readable creative *brief*
(an instruction) leaks verbatim into the posted copy or gets slugged into a
hashtag.

Everything runs OFFLINE against the deterministic composer path — no live model,
no network — so it is a fast, reliable CI check.

Run:
    cd artifacts/ai-training-server
    uv run python -u -m ai_model.generation.test_campaign
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional

from ai_model.generation.campaign import build_campaign

# Two bodies are treated as near-identical above this token-set Jaccard overlap.
# The pre-fix planner produced release-day / countdown pairs at ~0.92; healthy
# output stays well under this (observed max ~0.71 across a wide input sweep).
_MAX_BODY_SIMILARITY = 0.85

_failures = 0


def _check(label: str, ok: bool) -> None:
    global _failures
    status = "ok" if ok else "FAILED"
    if not ok:
        _failures += 1
    print(f"  [{status}] {label}")


# ── Test doubles mirroring the real server wiring (kept offline) ──────────────
_PLATFORM_NORMALIZE = {"ig": "instagram", "reels": "instagram_reels", "tt": "tiktok"}


def _normalize_platform(p: str) -> str:
    return _PLATFORM_NORMALIZE.get(p.lower(), p.lower())


def _hashtags(topic: str, genre: Optional[str], platform: str) -> List[str]:
    """Mirror of server.py::_api_hashtags — compact, single-token tags."""
    tags = [f"#{topic.replace(' ', '')}", f"#{platform}"]
    if genre:
        tags.append(f"#{genre.replace(' ', '')}")
    tags += ["#music", "#newrelease", "#artist"]
    return list(dict.fromkeys(tags))[:6]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _skeleton(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _build(**overrides) -> Dict:
    params = dict(
        artist="Nova",
        title="Midnight Drive",
        genre="pop",
        tone=None,
        brand_voice=None,
        target_audience="late-night drivers",
        platforms=["instagram", "tiktok"],
        weeks=6,
        mood="nostalgic",
        bpm=120,
        key="A minor",
        release_date="2026-09-01",
        hashtag_fn=_hashtags,
        normalize_platform_fn=_normalize_platform,
        seed=42,
    )
    params.update(overrides)
    return build_campaign(**params)


def _all_posts(plan: Dict) -> List[Dict]:
    return [post for phase in plan["phases"] for post in phase["posts"]]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fifteen_posts() -> None:
    plan = _build()
    posts = _all_posts(plan)
    _check("plan has 15 posts", len(posts) == 15)
    _check("summary total_posts == 15", plan["summary"]["total_posts"] == 15)


def test_hooks_all_distinct() -> None:
    plan = _build()
    hooks = [p["hook"] for p in _all_posts(plan)]
    dupes = sorted({h for h in hooks if hooks.count(h) > 1})
    _check(f"all 15 hooks distinct (dupes={dupes})", len(set(hooks)) == len(hooks))


def test_bodies_not_near_identical() -> None:
    """No two bodies within a phase may be near-identical (the drift this
    planner exists to prevent), and no two bodies anywhere may be exactly equal."""
    plan = _build()
    worst = 0.0
    worst_pair = ("", "")
    for phase in plan["phases"]:
        bodies = [p["body"] for p in phase["posts"]]
        for a, b in itertools.combinations(bodies, 2):
            sim = _similarity(a, b)
            if sim > worst:
                worst, worst_pair = sim, (a, b)
    _check(
        f"within-phase bodies distinct (max sim={worst:.3f} < {_MAX_BODY_SIMILARITY})",
        worst < _MAX_BODY_SIMILARITY,
    )
    all_bodies = [p["body"] for p in _all_posts(plan)]
    _check("no two bodies are byte-identical", len(set(all_bodies)) == len(all_bodies))


def test_brief_never_leaks_into_copy() -> None:
    """The human-readable creative brief is an instruction — it must never appear
    inside the post's hook, body or caption (a known raw-templating failure)."""
    plan = _build()
    leaked: List[str] = []
    for p in _all_posts(plan):
        brief = _skeleton(p["brief"])
        if not brief:
            continue
        for field in ("hook", "body", "caption"):
            if brief in _skeleton(p[field]):
                leaked.append(f"{p['content_type']}.{field}")
    _check(f"brief text never leaks into copy (leaks={leaked})", not leaked)


def test_hashtags_are_clean() -> None:
    """Hashtags must be short single-token slugs, never whole-sentence slugs of a
    brief/caption (e.g. #AnnounceTheNewSingleIsComingAndRevealTheTitle)."""
    plan = _build()
    bad: List[str] = []
    for p in _all_posts(plan):
        brief_skel = _skeleton(p["brief"])
        for tag in p["hashtags"]:
            token = tag.lstrip("#")
            # A clean tag is one '#'-prefixed token: no whitespace, not a
            # sentence-length slug, and not a slug of the brief sentence.
            if (not tag.startswith("#") or " " in tag or len(token) > 30
                    or (brief_skel and len(brief_skel) > 12 and brief_skel in _skeleton(tag))):
                bad.append(tag)
    _check(f"hashtags are clean single-token slugs (bad={bad})", not bad)


def test_deterministic() -> None:
    a = _build()
    b = _build()
    _check("build_campaign is deterministic",
           [p["caption"] for p in _all_posts(a)] == [p["caption"] for p in _all_posts(b)])


def test_distinct_across_varied_inputs() -> None:
    """The distinctness guarantees hold across a wide sweep of artist/title/genre
    combinations, not just the happy-path fixture."""
    artists = ["Nova", "", "The Midnight Collective", "Ye"]
    titles = ["Midnight Drive", "Sun", "A Very Long Winding Title", "x"]
    genres = ["pop", "edm", "indie", None, "rap"]
    hook_ok = body_ok = True
    worst = 0.0
    for i, (artist, title, genre) in enumerate(itertools.product(artists, titles, genres)):
        plan = _build(artist=artist, title=title, genre=genre, mood=None, seed=i)
        posts = _all_posts(plan)
        if len(posts) != 15:
            hook_ok = False
        hooks = [p["hook"] for p in posts]
        if len(set(hooks)) != len(hooks):
            hook_ok = False
        for phase in plan["phases"]:
            bodies = [p["body"] for p in phase["posts"]]
            for x, y in itertools.combinations(bodies, 2):
                sim = _similarity(x, y)
                worst = max(worst, sim)
                if sim >= _MAX_BODY_SIMILARITY:
                    body_ok = False
    _check("hooks stay distinct across 320 input combos", hook_ok)
    _check(f"bodies stay distinct across 320 input combos (max sim={worst:.3f})", body_ok)


def main() -> int:
    tests = [
        test_fifteen_posts,
        test_hooks_all_distinct,
        test_bodies_not_near_identical,
        test_brief_never_leaks_into_copy,
        test_hashtags_are_clean,
        test_deterministic,
        test_distinct_across_varied_inputs,
    ]
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as e:  # noqa: BLE001 - a raising test is a failure
            global _failures
            _failures += 1
            print(f"  FAILED: {t.__name__} raised {e!r}")
    print("\n" + ("ALL PASSED" if _failures == 0 else f"{_failures} CHECK(S) FAILED"))
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
