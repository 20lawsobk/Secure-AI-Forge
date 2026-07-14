"""ParsedUrl — the canonical output of the Universal URL Parser."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedUrl:
    """Rich structured output for any URL passed through the Universal URL Parser.

    Fields are populated on a best-effort basis.  Every field defaults to a
    safe empty value so callers can access them without guard clauses.

    Populating order (highest priority wins):
      1. Platform-specific HTML/JSON-LD extraction
      2. URL path/query-string slug decoding
      3. Hostname-only signals
      4. HTTP og:title / <title> fetch fallback
    """

    # ── Input ────────────────────────────────────────────────────────────────
    raw_url:        str = ""      # exactly as received
    canonical_url:  str = ""      # https-normalised, www-stripped

    # ── Platform ─────────────────────────────────────────────────────────────
    platform:       str = ""      # spotify | youtube | tiktok | instagram | …
    platform_label: str = ""      # "Spotify", "YouTube Music", "TikTok", …
    content_type:   str = "page"  # track | album | playlist | video | short |
                                  # profile | post | reel | article | page

    # ── Content metadata ─────────────────────────────────────────────────────
    title:       str = ""   # song/video/post/page title
    artist:      str = ""   # artist / creator / channel name
    album:       str = ""   # album or EP name
    label:       str = ""   # record label (when detectable)
    description: str = ""   # og:description / meta:description / first ¶
    body_text:   str = ""   # additional page body text

    # ── Music-specific signals ────────────────────────────────────────────────
    genre:        str          = ""    # hip-hop | pop | r&b | afrobeats | …
    mood:         str          = ""    # energetic | chill | dark | uplifting | …
    bpm:          Optional[int]  = None  # extracted or inferred BPM
    key:          str          = ""    # C major | F# minor | Am | …
    release_year: Optional[int]  = None  # 4-digit year when visible

    # ── Derived signals ───────────────────────────────────────────────────────
    topics:         List[str] = field(default_factory=list)
    content_themes: List[str] = field(default_factory=list)
    intent:         str = ""   # drive_streams | grow_followers | build_awareness …
    goal:           str = ""   # streams | fanbase | engagement | conversion | …

    # ── Ready-to-use pipeline outputs ────────────────────────────────────────
    topic_string:   str = ""   # human-readable topic for the `topic` request field
    awareness_text: str = ""   # pre-formatted multi-line awareness block

    # ── Diagnostics ──────────────────────────────────────────────────────────
    fetch_ok: bool  = False   # True when HTTP fetch succeeded
    error:    str   = ""      # non-fatal error description (empty = clean)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True when no meaningful content was extracted."""
        return not (self.title or self.artist or self.description)

    def combined_text(self) -> str:
        """All extracted text joined for downstream NLP / awareness parsing."""
        parts = [self.title, self.artist, self.album, self.description, self.body_text]
        return " ".join(p for p in parts if p)

    def as_awareness_dict(self) -> dict:
        """Return a flat dict of non-empty signals (for debug / logging)."""
        return {k: v for k, v in {
            "platform":     self.platform_label or self.platform,
            "content_type": self.content_type,
            "title":        self.title,
            "artist":       self.artist,
            "album":        self.album,
            "genre":        self.genre,
            "mood":         self.mood,
            "bpm":          str(self.bpm) if self.bpm else "",
            "key":          self.key,
            "intent":       self.intent,
        }.items() if v}
