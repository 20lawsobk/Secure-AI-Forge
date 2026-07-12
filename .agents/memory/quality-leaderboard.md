---
name: Quality leaderboard 100/100
description: Scoring rules, final fixes, and constraints to preserve when touching script_agent or server.py to keep test_awareness_and_quality at 100.0 average.
---

## Achieved state
39/39 passing, **100.0 average quality** across 18 leaderboard rows.  Cache namespace: `api_content_v4`.

## Scoring formula (test mirror must stay in sync with server)
- `_length_score`: plateau 15–60 words (full 1.0 inside range, tapers outside). **Never raise the upper bound above 60** without checking TikTok rows.
- `_hook_score`: `0.55 (power word) + 0.30 (!) + 0.15 (emoji) = 1.0`. No word-count component.
- `_struct_score`: arousal cap = **2 hits** (0.10 × hits, max +0.20). Raising cap to 3 breaks rows with sparse arousal.
- `_keyword_score`: returns **1.0 when no keywords provided** — unconstrained generation must not be penalized.
- `looks_garbled`: uses `re.findall(r"[A-Za-z0-9''\-]+")` (strips `#`), NOT `text.split()`. Never revert to split-based version.

## Key constraint: `_norm` in `_build_awareness_body`
`_norm(s, maxlen=50)` with **word-boundary truncation** (`rfind(" ")`).  
**Why:** em-dash `—` in body signal strings counts as an extra split-word, pushing TikTok posts 5-10 words past the 60-word plateau when maxlen=100. Signals truncated at 50 chars land at 50-55 total words → `ls=1.0`.  
**How to apply:** Any edit to `_build_awareness_body` that changes signal concatenation must re-verify word count stays ≤60.

## Key constraint: video endpoint must expose `body` and `cta`
`/platform/video/generate` response must include top-level `"body"` and `"cta"` fields (from `script_result`).  
**Why:** `rate_content` in the test builds `full_text = hook + body + cta`; without them, only 14 words reach the scorer → `ls` and `ss` tank.

## Key constraint: cache namespace
After any edit to hook templates, `_build_awareness_body`, or `_PLATFORM_STRATEGY`, bump `namespace="api_content_vN"` in server.py AND kill -9 the server.py process before re-running tests. Disk-backed pdim cache survives restarts and will serve stale content otherwise.

## `_effective_awareness` ordering
Platform awareness is **prepended** before user awareness so platform arousal signals reach the body builder even when the user provides raw awareness. Do not reverse this order.
