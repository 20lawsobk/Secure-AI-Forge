---
name: Campaign copy distinctness
description: Why a release campaign's per-post bodies/hooks collapse into near-duplicates and how the planner prevents it
---

# Campaign copy distinctness

A release campaign fans one song title into ~15 posts. Hooks are templated per
content type; bodies come from `compose_caption`. Two independent repetition traps:

- **Hooks:** when the same content type is used by >1 slot (e.g. two release-day
  `out_now` posts), the single hook template produces an *identical* hook.
  **Fix:** a `_HOOKS` value may be a list of variants; each recurrence of a type
  takes the next variant (tracked by an occurrence counter).
- **Bodies:** several posts share the title and differ only by a near-identical
  `theme` phrase (e.g. "out now" vs "out now stream now", "countdown" vs "final
  countdown"). The composer ranks the same top body template for all of them, so
  they read as near-duplicates (Jaccard ~0.92).
  **Fix:** request several ranked variants from `compose_caption` and, per post,
  pick the variant *least similar* (token-set Jaccard) to the bodies already
  placed in that phase — NOT blind position-based rotation (rotation does not
  guarantee different templates because each theme's variant ordering differs).

**Why:** blind rotation still let the countdown/out-now pairs land on the same
template. Similarity-aware selection reliably drops the worst within-phase
overlap to ~0.71 across a wide artist/title/genre sweep.

**How to apply:** the regression guard is
`ai_model/generation/test_campaign.py` (offline, deterministic). It asserts 15
posts, all-distinct hooks, within-phase body Jaccard < 0.85, no `brief`
(instruction) text leaking into hook/body/caption, and clean single-token
hashtags. If you change hook templates, the blueprint, or the composer's body
candidates, re-run it: `uv run python -u -m ai_model.generation.test_campaign`.
