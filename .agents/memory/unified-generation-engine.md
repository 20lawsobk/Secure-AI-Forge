---
name: Unified generation engine (technique + campaign)
description: How the ai_model/generation package blends real assets with genre priors and how the release-campaign planner composes existing primitives without leaking instructions.
---

# Unified conditioned generation engine

`ai_model/generation/` centralizes conditioning shared by all modalities:
`technique.py` (Visual/Sonic DNA), `orchestrator.py` (build_context), and
`campaign.py` (release rollout planner). Everything is **additive + never-raise**.

## Real-asset blend must be capped
`extract_technique` blends a genre/tone **prior** with **real-asset descriptors**
retrieved from the RCGS index, weighted `buffer_weight × rung_relevance`, hard
**capped at `_REAL_WEIGHT_CAP` (0.6)**.
**Why:** a sparse/non-diverse index returns the same generic "anchor" asset for
every query; without the cap (and with a low anchor relevance, 0.3) it flattens
every genre onto one look (EDM == acoustic). The cap guarantees the genre prior
always keeps character. `source="prior"` with `real_weight=0.0` is a **valid,
by-design state** (no real match this session), NOT a regression — confirm the
compute ran by checking energy is still genre-specific (e.g. EDM≈0.8 vs
acoustic≈0.28), not the 0.5 default.

## never-raise is whole-body, not per-line
`extract_technique` wraps its ENTIRE compute body in try/except returning a
default `TechniqueProfile()`. **How to apply:** for a function contractually
"never raises," guard the whole body + the cache write; per-expression guards
miss arithmetic/type-coercion failures the architect will flag.

## Campaign planner: keep instruction text OUT of generation inputs
`build_campaign` turns one release into a multi-week, multi-platform rollout
(announce→tease→pre-save→release→sustain) by composing existing primitives:
`request_intelligence.build_brief` + `compose_caption` for **body/CTA only**,
per-content-type **templated hooks**, and `extract_technique` for shared
`art_direction`.
**Why:** feeding the human-readable creative *brief* ("announce the single and
reveal the title", "the pre-save link is live — why they should…") as
`narrative`/`topic` leaks it VERBATIM into captions and produces monster
hashtags (whole sentence slugged into one tag). Feed only natural **theme**
noun-phrases ("pre-save now", "behind the scenes"); keep the instruction brief
as an output-only field. Same principle as idea-awareness-field-separation.
**How to apply:** hashtag_fn gets the **title only**, never title+angle.
