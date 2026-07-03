---
name: Quality awareness buffer
description: World-studied content-quality buffer in pdim that blends into generation and auto-retires as the own corpus grows
---

# Quality awareness buffer (temporary, self-retiring)

The rule: the quality buffer is a TEMPORARY dataset — world-studied patterns from live top-performing content (Apple Music charts, YouTube music channel RSS, HN Algolia engagement), stored in pdim (`mb:awareness:quality:doc`), blended into generation with weight `max(0, 1 - own_corpus/MB_AWARENESS_RETIRE_AT)` (default 500). Buffer picks GRADUATE into `mb:phrases:*` on use, so using the buffer grows the own corpus toward the buffer's own retirement.

**Why:** user's garden model — robots study the best plants in the world only until the seed vault is self-sufficient, then they retire. No fake data: if every harvest source fails, harvest raises (never store an empty world-scan as knowledge).

**How to apply:**
- Read side lives in `ai_model/quality_awareness.py` — everything is never-raise + TTL-cached; it must never break or slow generation. Retirement check comes first: retired ⇒ all blending APIs return empty.
- Harvest: admin `POST /api/awareness/quality/harvest` (sync, ~2s, 502 if all sources fail); status: `GET /api/awareness/quality/status`. Lazy single-flight auto-harvest fires when the buffer is missing and not retired.
- Templates are data-conditioned ({idea}/{artist} slots filled from live chart genres/artists), formatted with `.format(idea=…, artist=…)` inside try/except — a bad slot skips the template, never raises.
- There are TWO awareness layers: Node `contentAwarenessService` (industry news signals, ephemeral, injected as context string via :8080) and this pdim buffer (content quality patterns, persistent, engine-side — also benefits direct :9878 calls).
- Storage-key gotcha reconfirmed: `keys()` pattern is NOT namespaced but `llen()/get()/set()` ARE — strip `mb:` from keys() results before llen.
- Reddit RSS is 403-blocked from this environment; Apple charts / YouTube RSS / HN Algolia are reliable.
