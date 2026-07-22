"""Beat-marketplace conditioning — regression guards.

Covers the MaxBooster contract additions to /api/generate/content:
goal=drive_purchase aliasing, structured beat_context → awareness signal
lines (parseable, sanitized), purchase CTA anchoring, and
platform_constraints enforcement (no_link_in_bio, professional_register).
"""

from ai_model import request_intelligence as ri
from ai_model.agents.script_agent import _parse_signals_for_platform


BC = {
    "title": "Empowering Indie Type Beat",
    "genre": "trap",
    "mood": "cinematic",
    "bpm": 140,
    "key": "C Minor",
    "production_details": "rolling hi-hats, orchestral strings, punchy 808s",
    "target_artist": "storytelling artists",
    "price_usd": 29.99,
    "license_slots_remaining": 5,
    "listen_url": "https://blawz.com/marketplace/beat/abc123",
    "audio_analysis": {
        "loudness_db": -8.2, "energy": 0.87,
        "spectral_brightness": "dark", "bass_weight": "heavy",
        "detected_instruments": ["808", "strings", "hi-hats"],
    },
}


def test_purchase_goal_aliases_to_conversion():
    for g in ("drive_purchase", "purchase", "buy"):
        intent, _ = ri.classify_intent(g, "beat release promotion")
        assert intent == "drive_conversion", g


def test_beat_context_awareness_lines_parse():
    aw = ri.beat_context_awareness(BC)
    assert "C Minor" in aw and "140" in aw and "$29.99" in aw
    assert "dark" in aw and "heavy" in aw
    parsed = _parse_signals_for_platform(aw, "instagram")
    assert len(parsed) == len(aw.splitlines()) > 0


def test_beat_context_never_raises_and_sanitizes():
    assert ri.beat_context_awareness(None) == ""
    assert ri.beat_context_awareness({"bpm": "not-a-number"}) == ""
    aw = ri.beat_context_awareness({"title": "a\x00b\nc" + "x" * 500})
    assert "\x00" not in aw and len(aw) < 400
    # javascript: / non-http URLs must be dropped
    aw2 = ri.beat_context_awareness(
        {"price_usd": 10, "listen_url": "javascript:alert(1)"})
    assert "javascript" not in aw2


def test_purchase_cta_anchors():
    cta = ri.purchase_cta(BC)
    assert "$29.99" in cta and "5 licenses left" in cta
    assert "https://blawz.com/marketplace/beat/abc123" in cta
    assert ri.purchase_cta({"title": "no anchors"}) == ""
    # professional register drops the trailing emoji
    pro = ri.purchase_cta(BC, constraints={"professional_register": True})
    assert "🛒" not in pro


def test_platform_constraints():
    t = "Stream it now — link in bio 🎧"
    out = ri.apply_platform_constraints(
        t, {"no_link_in_bio": True}, "https://x.co/b/1")
    assert "link in bio" not in out.lower() and "https://x.co/b/1" in out
    out2 = ri.apply_platform_constraints(
        "Great record. Drop a 🔥 if this hits — tell me in the comments",
        {"professional_register": True})
    assert "🔥" not in out2 and "Drop a" not in out2
    assert "Great record." in out2
    # never-raise / pass-through
    assert ri.apply_platform_constraints("x", None) == "x"
    assert ri.apply_platform_constraints("", {"no_link_in_bio": True}) == ""
