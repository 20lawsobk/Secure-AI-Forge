---
name: Quality awareness buffer
description: World-studied content-quality buffer in pdim that blends into generation and auto-retires as the own corpus grows
---

# Quality awareness buffer (temporary, self-retiring)

The rule: the quality buffer is a TEMPORARY dataset of world-studied patterns from live top-performing content (chart rank / view counts / vote counts as REAL quality signals — never fixed made-up weights). Influence weight decays linearly as the own pdim phrase corpus grows (`MB_AWARENESS_RETIRE_AT`, default 500); at zero the buffer retires and all blending APIs return empty.

**Why:** user's garden model — robots study the best plants in the world only until the seed vault is self-sufficient, then retire. No fake data: if every harvest source fails, harvest raises rather than storing an empty world-scan.

**How to apply:**
- Blending must be never-raise + TTL-cached — it can never break or slow generation. Retirement check first.
- GRADUATION is required in EVERY path that consumes the buffer (video sampler AND text hook ranking): a used buffer pick pushes its raw TEMPLATE (not topic-filled text) into the own corpus — otherwise retirement stalls for traffic mixes that skip that path. Architect flagged this as the key gap on first review.
- Study, don't republish: generation reads only derived templates/stats, never raw exemplar text.
- Staleness: a buffer older than `MB_AWARENESS_MAX_AGE_H` (default 24h) triggers a background replace-harvest while still serving the old doc.
- Storage-key gotcha reconfirmed: storage client `keys()` pattern is NOT namespaced but `llen()/get()/set()` ARE — strip the namespace prefix from keys() results before llen.
- Reddit RSS is 403-blocked from this environment; Apple charts / YouTube RSS (has `views=` in media:statistics) / HN Algolia are reliable engagement sources.
- Two distinct awareness layers exist: the Node industry-news signal layer (ephemeral context string, only via the API server) and this pdim quality buffer (engine-side, also benefits direct engine calls).
