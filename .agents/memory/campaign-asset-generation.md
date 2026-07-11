---
name: Campaign per-post asset generation
description: How opt-in image/teaser generation is wired into the release-campaign planner and why it conditions on shared art_direction
---

# Campaign per-post asset generation

`build_campaign` (ai_model/generation/campaign.py) stays engine-agnostic and
never-raise: asset generation is injected via optional `image_fn` / `teaser_fn`
callables (same pattern as `hashtag_fn` / `normalize_platform_fn`). The server
endpoint supplies closures that reuse the existing ImageEngine and the
`/api/generate-video` job launcher.

**Rule:** per-post assets MUST condition on the campaign's SHARED `art_direction`
(one palette + mood for the whole rollout), not on a per-slot re-extracted
technique.
**Why:** the whole point is "every asset looks like one release." The standalone
`/api/generate/image` path re-derives a per-slot technique seed, which would make
each post's artwork drift. The campaign forces the shared `color_scheme`/`mood`
onto every ImageRequest instead.
**How to apply:** compute `art_direction` once (before the post loop) so both the
callables and the returned plan see the same DNA. Seed image renders
deterministically from stable inputs so re-running a campaign is reproducible.

**Concurrency:** images render inline (blocking PIL), so the endpoint runs the
whole `build_campaign` off the event loop (`_in_thread`) only when assets are
requested; the text-only default stays synchronous and fast (~250ms vs ~19s for
15 images). Teasers are async video render jobs — the plan returns `job_id` +
`poll_url` (`/api/video-job/{id}`), not a finished URL.
