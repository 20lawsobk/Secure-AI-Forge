---
name: Brand voice, AI-disclosure, and metadata steering
description: How the artist Brand Voice profile, AI-disclosure, producer-metadata steering (genre/mood/bpm/key), and cross-platform variant generation are wired through text/image/video generation in artifacts/ai-training-server.
---

Research (Water & Music/Moises survey, Cyanite, DropTrack, PitchPlus) surfaced four
consistent artist/producer asks that are now built in: on-brand consistent output,
AI-disclosure control, real metadata-driven creative steering, and one-pass
cross-platform variants. Scope was deliberately kept to these four — no content
calendar/auto-posting, no profile-editing UI (proposed as a follow-up), no voice
cloning.

**Brand Voice is a fallback, not an override.** `request_intelligence.load_brand_voice()`
pulls the artist's saved profile (tone/genre/vocabulary/avoid_words/ai_disclosure) via
the pre-existing `ArtistProfileClient`, and `build_brief()` only uses profile tone/genre
when the caller didn't supply one. **Why:** a per-request `tone`/`genre` param is more
specific than a standing profile default; falling back preserves that. **How to apply:**
any new generation param that should have a brand-level default follows the same
`param or brand.get(...)` pattern, never the reverse.

**Disclosure is opt-in and must survive char-budget trimming, not violate it.**
`apply_disclosure()` appends a short label only when `brief.ai_disclosure` is true, and
callers must re-check the result against any explicit `max_chars` — if the label would
blow the budget, drop the label rather than exceeding a caller-declared hard limit.
**Why:** an architect review caught disclosure-then-trim ordering silently violating
`max_chars`. **How to apply:** apply disclosure last, then bound, then fall back to the
undecorated text if still over budget.

**Producer metadata (genre/mood/bpm) blends into one 0-1 "energy" score** via
`_producer_energy()`, which shifts `tempo`/`hook_style` for text and adds style tags for
image (`visual_style_from_brief()`) — mirroring how video's `ai_scene_builder._GENRE_DNA`
already drives energy/darkness → transitions. Kept as a lightweight parallel mapping
rather than importing video's private `_GENRE_DNA`, to avoid coupling text/image to
video internals.

**BPM must be coerced once, defensively, inside `build_brief()`** — it's reachable from
an untyped raw-JSON endpoint (`/api/video/generate-ai`) where bpm can arrive as a string
or garbage. Coerce via try/except immediately, before any `:.0f` formatting or later
`float(bpm)` calls, or a bad value raises deep inside brief construction.

**Cross-platform variants: build the brief per-platform, never once globally.**
Image generation loops slots that can each specify a different platform; the brief
(and any per-slot style-tag list) must be built/copied fresh per slot, or slot N's
derived state leaks into slot N+1 (caught by architect review: a shared `style_tags`
list mutated in a loop bled across slots). Content generation added an explicit
`platforms: List[str]` request field that loops `_build(_platform=p)` and returns
`platform_variants` keyed by platform, guarding the empty-list case before indexing.

**PDIM response caching must be bypassed when a request depends on external mutable
state.** `/api/generate/content`'s PDIM cache key is derived from the request payload
only; once `build_brief()` started reading the artist's stored profile, an unchanged
request could replay a stale cached caption/disclosure after the artist edited their
profile. Fixed by skipping the cache path entirely whenever `artistProfileId` is set
(falls back to uncached `_build()`). **How to apply:** any future field that pulls from
mutable external storage inside a cached builder needs either a bypass like this or the
storage version folded into the cache key — don't cache computed output derived from
inputs the cache key doesn't cover.
