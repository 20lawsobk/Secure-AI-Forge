---
name: Quality awareness buffer
description: World-studied content-quality buffer in pdim that blends into generation and auto-retires as the own corpus grows
---

# Quality awareness buffer (temporary, self-retiring)

The rule: the quality buffer is a TEMPORARY dataset of world-studied patterns from live top-performing content (chart rank / view counts / vote counts as REAL quality signals — never fixed made-up weights). Influence weight decays linearly as the own pdim phrase corpus grows (`MB_AWARENESS_RETIRE_AT`, default 500); at zero the buffer retires and all blending APIs return empty.

**Why:** user's garden model — robots study the best plants in the world only until the seed vault is self-sufficient, then retire. No fake data: if every harvest source fails, harvest raises rather than storing an empty world-scan.

**How to apply:**
- Blending must be never-raise + TTL-cached — it can never break or slow generation. Retirement check first.
- GRADUATION is required in EVERY path that consumes the buffer (video sampler, text hook ranking, AND image headline ranking): a used buffer pick pushes its raw TEMPLATE (not topic-filled text) into the own corpus — otherwise retirement stalls for traffic mixes that skip that path. Architect flagged this as the key gap on first review.
- Each modality graduates into its OWN corpus key (`phrases:hook` for text, `phrases:image_headline` for image, video's per-scene-type keys) even though they draw candidates from the same borrowed `scene_phrases("hook")` bank — keeps graduation counting real usage per modality instead of double-crediting one corpus.
- Study, don't republish: generation reads only derived templates/stats, never raw exemplar text.
- Staleness: a buffer older than `MB_AWARENESS_MAX_AGE_H` (default 24h) triggers a background replace-harvest while still serving the old doc.
- Storage-key gotcha reconfirmed: storage client `keys()` pattern is NOT namespaced but `llen()/get()/set()` ARE — strip the namespace prefix from keys() results before llen.
- Reddit RSS is 403-blocked from this environment; Apple charts / YouTube RSS (has `views=` in media:statistics) / HN Algolia are reliable engagement sources.
- Two distinct awareness layers exist: the Node industry-news signal layer (ephemeral context string, only via the API server) and this pdim quality buffer (engine-side, also benefits direct engine calls).
- A second borrowed-knowledge source exists: the research-distilled content playbook (`ai_model/content_playbook.py` — curated hook archetypes / CTA bank / HVC structure_score / hashtag caps from published 2025-26 engagement studies). It is BORROWED knowledge like the harvester buffer, so EVERY injection point must gate on the same `self_sufficiency()["retired"]` contract — templates via scene_phrases blending AND brief directives. Architect flagged ungated directives as the contract violation on first review; scoring heuristics (structure_score inside score_candidate) and platform hashtag caps are evaluation criteria / platform norms, not injected content, and do NOT retire.
- Playbook templates are user-facing copy (they compete in hook ranking and can win) — they must be finished clean text with only {idea}/{artist} slots, never meta/instructional language.
- THIRD borrowed source: the hand-authored "distinctive weave" in `request_intelligence.py` (`_body_candidates` release-specific narrative/artist/track/theme bodies + `hook_variants` request-specific hooks). It is borrowed knowledge like the buffer/playbook, so it MUST gate on the same `self_sufficiency()["retired"]` contract via `_weave_active()` — otherwise a hand-authored template out-scores the model FOREVER and silently pins the system to templates, defeating retirement. Compute the gate ONCE per caption in `compose_caption` and thread `weave_active` into `best_hook`/`hook_variants`/`_body_candidates`/`deterministic_candidate` (self-defaults when omitted) so one request = one retirement decision. Keep a minimal generic fallback ungated as the non-empty safety net; the agent (model/awareness) body always competes regardless of the gate.

## Flywheel graduation closes both loops
- Raw mb:dataset:flywheel:* samples were write-only archival; generation reads mb:phrases:<scene_type> (dataset_sampler tier 1). Flywheel now graduates text (scripts hook/body/cta, social captions, daw hook/lyrics, video scene texts) into those corpora on ingest.
- Because self_sufficiency() counts mb:phrases:*, every graduated phrase also shrinks buffer_weight — admin content directly drives awareness retirement (verified live: one generation → own_corpus +3, weight 0.852→0.846).
- Gotchas when verifying: own-corpus count is TTL-cached 60s (poll after 65s); with pdim offline the server uses an IN-PROCESS store — external scripts can't see it, verify via /api/awareness/quality/status.
