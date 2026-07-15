"""Unit tests for the URL-parser label-echo garble guard.

Verifies that:
  1. Captions containing raw awareness label lines are flagged as garbled.
  2. Captions that naturally mention the resolved artist/title values are NOT
     flagged (the whitelist correctly protects resolved content words).
  3. The tier-marker prefix pattern ([HIGH] / [MED] / [LOW]) is detected.
  4. Ordinary prose that happens to contain words like "source" or "context"
     is NOT flagged.
"""
from __future__ import annotations

import sys
import os

# Ensure the ai-training-server package root is on sys.path so the import works
# both when run via pytest from the repo root and when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER_ROOT = os.path.dirname(_HERE)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from ai_model.request_intelligence import looks_garbled  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Positive cases — label echoes MUST be detected as garbled
# ---------------------------------------------------------------------------

class TestLabelEchoDetected:
    """A response containing raw awareness label lines must score as garbled."""

    def test_artist_label_line(self):
        """Artist: <name> at line start is a metadata echo."""
        text = "Artist: Drake\nTitle: Gods Plan\nStream it now on Spotify."
        assert looks_garbled(text), "Artist:/Title: label lines should be flagged"

    def test_title_label_line(self):
        text = "Title: Blinding Lights\nListen everywhere."
        assert looks_garbled(text), "Title: label should be flagged"

    def test_genre_label_line(self):
        text = "Genre: Hip-Hop\nMood: Energetic\nNew banger out now."
        assert looks_garbled(text), "Genre: label should be flagged"

    def test_intent_label_line(self):
        text = "Intent: Drive streams\nCheck the link in bio."
        assert looks_garbled(text), "Intent: label should be flagged"

    def test_source_label_line(self):
        text = "Source: Spotify Track\nNew music available now."
        assert looks_garbled(text), "Source: label should be flagged"

    def test_album_label_line(self):
        text = "Album: Certified Lover Boy\nOut everywhere."
        assert looks_garbled(text), "Album: label should be flagged"

    def test_context_label_line(self):
        text = "Context: Official music video out now on YouTube.\nWatch now."
        assert looks_garbled(text), "Context: label should be flagged"

    def test_bpm_key_year_labels(self):
        text = "BPM: 140 | Key: F# minor\nYear: 2024\nFire track incoming."
        assert looks_garbled(text), "BPM:/Key:/Year: labels should be flagged"

    def test_tier_high_source_line(self):
        """[HIGH] Source: … is the awareness header format."""
        text = "[HIGH] Source: Spotify Track\nArtist: The Weeknd\nNew drop."
        assert looks_garbled(text), "[HIGH] Source: header should be flagged"

    def test_tier_marker_alone(self):
        text = "[HIGH] Source: TikTok video\nFollow for more."
        assert looks_garbled(text), "[HIGH] tier marker line should be flagged"

    def test_med_tier_marker(self):
        text = "[MED] Source: YouTube video\nLink in bio."
        assert looks_garbled(text), "[MED] tier marker should be flagged"

    def test_low_tier_marker(self):
        text = "[LOW] Source: Instagram post\nCheck it out."
        assert looks_garbled(text), "[LOW] tier marker should be flagged"

    def test_full_awareness_block_echo(self):
        """Reproduces the exact format from _build_awareness_text()."""
        text = (
            "[HIGH] Source: Spotify Track\n"
            "Artist: Drake\n"
            "Title: Gods Plan\n"
            "Genre: Hip-hop | Mood: dark\n"
            "Intent: Drive streams\n"
            "Context: Official single with 2B+ streams."
        )
        assert looks_garbled(text), "Full awareness block echo should be flagged"

    def test_mixed_caption_with_label(self):
        """Even one label line in an otherwise fine caption should be caught."""
        text = (
            "🔥 New music alert!\n"
            "Artist: SZA\n"
            "Stream SOS on all platforms now."
        )
        assert looks_garbled(text), "Label line embedded in caption should be flagged"

    def test_case_insensitive_label(self):
        """Labels must be detected regardless of capitalisation."""
        text = "artist: kendrick lamar\ntitle: not like us"
        assert looks_garbled(text), "Lowercase label lines should be flagged"


# ---------------------------------------------------------------------------
# 2. Negative cases — legitimate captions must NOT be flagged
# ---------------------------------------------------------------------------

class TestLegitimateCaptionNotFlagged:
    """Captions that naturally mention artist/title values must NOT be garbled."""

    def test_artist_name_in_prose(self):
        """Mentioning 'Drake' naturally is fine."""
        text = "Drake just dropped something insane. Stream it now 🔥"
        assert not looks_garbled(text), "Artist name in prose should not be flagged"

    def test_artist_name_whitelisted(self):
        """With the artist in the whitelist, name in a caption is clean."""
        text = "Gods Plan by Drake is everything right now."
        assert not looks_garbled(text, whitelist="Drake Gods Plan"), \
            "Whitelisted artist/title in prose should not be flagged"

    def test_title_in_prose(self):
        text = "Blinding Lights hits different every time. Go stream it!"
        assert not looks_garbled(text), "Song title in prose should not be flagged"

    def test_source_word_in_prose(self):
        """'source' as a common English word should not trigger the guard."""
        text = "The source of all good music is creativity and passion."
        assert not looks_garbled(text), "'source' in prose should not be flagged"

    def test_context_word_in_prose(self):
        text = "In this context, the album really shines."
        assert not looks_garbled(text), "'context' in prose should not be flagged"

    def test_key_word_in_prose(self):
        text = "The key to success is consistency. New track out Friday."
        assert not looks_garbled(text), "'key' in prose should not be flagged"

    def test_year_word_in_prose(self):
        text = "This year's biggest drop is finally here. Don't sleep."
        assert not looks_garbled(text), "'year' in prose should not be flagged"

    def test_genre_word_in_prose(self):
        text = "This genre-defining record will change everything."
        assert not looks_garbled(text), "'genre' in compound should not be flagged"

    def test_intent_word_in_prose(self):
        text = "My intent was to create something timeless for the fans."
        assert not looks_garbled(text), "'intent' in prose should not be flagged"

    def test_normal_instagram_caption(self):
        text = (
            "New music Friday 🎶\n"
            "Just dropped my latest single — link in bio.\n"
            "Let me know what you think in the comments! 🔥"
        )
        assert not looks_garbled(text), "Normal caption should not be flagged"

    def test_spotify_tiktok_mention(self):
        text = "Spotify and TikTok have been going crazy for this one. Stream now!"
        assert not looks_garbled(text), "Platform names in prose should not be flagged"

    def test_label_word_mid_sentence(self):
        """'Artist:' must only be caught at line boundaries, not mid-sentence."""
        # Simulate an unlikely mid-sentence use — should still be caught since
        # our regex uses (?m)^ and this is still at line start after the newline.
        # This test ensures a colon mid-word doesn't false-positive.
        text = "She's an artist: raw, real, and unfiltered."
        # "artist:" appears after "She's an " — NOT at a line boundary with our
        # format; however our regex uses (?m)^ so let's verify.
        # After the split this is mid-line → should NOT match.
        assert not looks_garbled(text), \
            "'artist:' mid-sentence (after other words) should not be flagged"


# ---------------------------------------------------------------------------
# 3. Cross-check — guard must cover EVERY label the URL parser actually emits
# ---------------------------------------------------------------------------

import re  # noqa: E402

from ai_model.url_parser.core import _build_awareness_text  # noqa: E402
from ai_model.url_parser.models import ParsedUrl  # noqa: E402

# Matches an awareness label prefix at the start of a line segment:
# optional [HIGH]/[MED]/[LOW] tier marker, then "Some Label:".
_LABEL_PREFIX_RE = re.compile(
    r"^(?:\[(?:HIGH|MED|LOW|MEDIUM)\]\s+)?([A-Za-z][A-Za-z /-]*?)\s*:"
)


def _emitted_label_prefixes() -> set:
    """Return every label prefix `_build_awareness_text()` currently emits.

    Builds a ParsedUrl with EVERY metadata field populated so every branch of
    the awareness builder fires, then splits pipe-joined segments and pulls
    the `Label:` prefixes. If a new field is added to the builder, its label
    shows up here automatically — no manual list to keep in sync.
    """
    parsed = ParsedUrl(
        raw_url="https://open.spotify.com/track/xyz",
        canonical_url="https://open.spotify.com/track/xyz",
        platform="spotify",
        platform_label="Spotify",
        content_type="track",
        title="Gods Plan",
        artist="Drake",
        album="Scorpion",
        label="OVO Sound",
        description="Official single with 2B+ streams.",
        body_text="extra page text",
        genre="hip-hop",
        mood="dark",
        bpm=140,
        key="F# minor",
        release_year=2024,
        intent="drive_streams",
        goal="streams",
    )
    text = _build_awareness_text(parsed)
    assert text.strip(), "fully-populated ParsedUrl must produce awareness text"

    prefixes = set()
    for line in text.splitlines():
        for segment in line.split("|"):
            m = _LABEL_PREFIX_RE.match(segment.strip())
            if m:
                prefixes.add(m.group(1).strip())
    return prefixes


class TestGuardCoversAllEmittedLabels:
    """Auto-derived cross-check: iterate the label prefixes actually emitted
    by `_build_awareness_text()` and assert every one triggers
    `looks_garbled`. This test fails automatically if a new awareness field
    (e.g. `Label:`, `Featuring:`, `Producer:`) is added to the URL parser
    without extending `_URL_LABEL_RE` in request_intelligence.py."""

    def test_awareness_builder_emits_expected_baseline(self):
        """Sanity: the builder still emits the known core labels — if this
        fails, the extraction regex or the builder changed shape."""
        prefixes = _emitted_label_prefixes()
        expected_core = {
            "Source", "Artist", "Title", "Album", "Genre", "Mood",
            "BPM", "Key", "Year", "Intent", "Context",
        }
        missing = expected_core - prefixes
        assert not missing, (
            f"awareness builder no longer emits {sorted(missing)} — "
            "update this test's ParsedUrl fixture or the extraction regex"
        )

    def test_every_emitted_label_triggers_guard(self):
        """THE cross-check: each emitted label prefix, alone at line start,
        must be flagged by looks_garbled."""
        prefixes = _emitted_label_prefixes()
        assert prefixes, "no label prefixes extracted — extraction broken"
        uncovered = []
        for prefix in sorted(prefixes):
            caption = f"{prefix}: Some Value\nStream it now on all platforms."
            if not looks_garbled(caption):
                uncovered.append(prefix)
        assert not uncovered, (
            f"_URL_LABEL_RE does not cover awareness label(s) {uncovered} — "
            "a field was added to _build_awareness_text() in "
            "ai_model/url_parser/core.py without updating _URL_LABEL_RE in "
            "ai_model/request_intelligence.py"
        )

    def test_full_emitted_block_triggers_guard(self):
        """The verbatim awareness block itself must always be flagged."""
        parsed = ParsedUrl(
            platform="spotify", platform_label="Spotify",
            content_type="track", title="Gods Plan", artist="Drake",
            genre="hip-hop", intent="drive_streams",
        )
        block = _build_awareness_text(parsed)
        assert looks_garbled(block), "verbatim awareness block must be flagged"
