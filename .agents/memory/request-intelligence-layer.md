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
- **Feed `augmented_idea` to the agents, but always keep a deterministic
  raw-topic candidate in the ranking pool.**
  **Why:** verbose steering text can degrade the undertrained model's output, so
  a clean raw-topic candidate must be able to win the ranking as a guardrail.
- **Hashtag count** is capped at `min(10, max(brief.hashtags_target, len(preferred)))`
  — keep the hard `min(10, …)` so behaviour stays compatible with the old `[:10]`.

## How to apply
When adding/altering a generation endpoint, call `build_brief(modality, platform,
topic, goal, tone, genre, artist, extra)` first, feed `brief.augmented_idea` /
`brief.tone` to the agent, rank outputs with `rank_candidates` / `best_hook`
(text/content only), and return `brief.to_dict()` under an `intelligence` key.
Keyword extraction is Unicode-aware (`[^\W_]+`), so non-Latin topics still work.
