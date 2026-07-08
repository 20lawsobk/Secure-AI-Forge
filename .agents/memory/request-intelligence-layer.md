---
name: Request intelligence layer
description: Shared pre-generation analysis module wired into every generation endpoint; conventions to keep consistent.
---

# Request intelligence layer

A shared, pure-python, dependency-free pre-generation module sits in front of
every generation service endpoint (content, text, image, audio, both video
endpoints). It analyses the raw request and produces a structured "brief"
(intent, audience, keywords, per-platform strategy: aspect ratio, hook/cta
style, word-count window, hashtag count, tempo, temperature) plus an
`augmented_idea` string and human-readable directives.

## Conventions any future change MUST keep consistent
- **Additive only.** Every generation response keeps all its original fields and
  merely *adds* an `intelligence` block (the brief) and, for text-like
  endpoints, a `quality_score`. Async endpoints (audio, video) also store the
  brief in the job record and echo it in the immediate `{job_id,status}` reply.
- **Cheap before the gate.** The layer runs *before* the slow, gated transformer
  inference, so it must stay deterministic and fast (no model calls inside it).
- **Multi-candidate ranking only for text/content.** `candidate_count` is 3 for
  content/text (ranking is cheap heuristic scoring) and 1 for image/audio/video
  — those modalities are too expensive to generate N times.
  **Why:** the in-house model is undertrained and inference is slow/gated;
  doing N model generations per request would blow latency and the gate budget.
- **Never feed `augmented_idea` (or `brief.directives`) into an `idea`/`topic`
  field that gets templated raw into user-facing text.** Keep `idea` a clean
  topic string; route richer context through a dedicated `awareness` param
  instead — see `idea-awareness-field-separation.md` for the parser-specific
  caveats (some awareness parsers echo bulleted directives verbatim).
  **Why:** an earlier version of this guidance said to feed `augmented_idea`
  straight to agents; that caused the pipe-joined `"topic | tone: X | goal: Y
  | audience: Z"` string (and later, directive bullets like "Optimise for: X")
  to render literally in captions/thumbnail headlines — a shipped, silent
  quality bug (200 response, garbled/leaked text).
- **Hashtag count** is capped at `min(10, max(brief.hashtags_target, len(preferred)))`
  — keep the hard `min(10, …)` so behaviour stays compatible with the old `[:10]`.

## Caption composer (content endpoint)

`compose_caption(topic, artist, brief, genre, brand_voice, agent_hook/body/cta)`
composes the caption FROM the brief instead of templating the raw topic:
- Body candidates are built from brief keywords/audience/tone/genre; the agent
  body competes but is **rejected as an echo** when its alphanumeric skeleton
  equals the topic's (punctuation/emoji-only edits still count as echoes).
- CTA candidates = agent cta + `brief.suggested_cta` + playbook bank, where the
  playbook contribution is **gated on `quality_awareness.self_sufficiency()["retired"]`**
  — same retirement contract as directives and hooks. Any new borrowed-playbook
  injection point must carry the same gate.
- Every full hook/body/cta combination is scored with `score_candidate` (whole
  caption, structure-aware) and the best wins; deterministic, no model calls.
  **Why:** ranking parts in isolation misses structure effects (HVC bonus, length
  windows) that only exist at the caption level.

## How to apply
When adding/altering a generation endpoint, call `build_brief(modality, platform,
topic, goal, tone, genre, artist, extra)` first, feed the clean `topic`/`idea`
string (not `brief.augmented_idea`) plus `brief.tone` to the agent, rank
outputs with `rank_candidates` / `best_hook` (text/content only), and return
`brief.to_dict()` under an `intelligence` key. Keyword extraction is
Unicode-aware (`[^\W_]+`), so non-Latin topics still work.

## Garble guard (undertrained-model output)
`looks_garbled(text, whitelist)` is the deterministic gate that keeps raw
undertrained-transformer garble (glued tokens like "beingpre-save",
letter+multi-digit fusions like "frequency82") out of user-facing copy.
- **Enforced at two layers:** ScriptAgent rejects a garbled model hook+body
  (falls through to awareness composition) with whitelist = request idea +
  awareness; `score_candidate` subtracts a decisive 40-point penalty with
  whitelist = brief keywords + `augmented_idea`.
- **Why the whitelist matters:** legit alphanumeric artist/track names
  ("Frequency82") match the fusion heuristic; words from the request itself
  must always be exempt or the guard suppresses valid copy. Ranking whitelist
  must include `augmented_idea` (carries the raw topic), not keywords alone —
  keyword extraction can drop numeric-suffixed names.
- **Why _is_meaningful is not enough:** it only checks length/control-tokens/
  repetition; glued-token garble passes it. Any new consumer of raw model text
  must route through the garble guard too.
