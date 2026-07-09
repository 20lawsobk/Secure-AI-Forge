---
name: MaxCore generation conditioning-field contract (awareness channel)
description: How to make new context actually influence MaxCore text/audio/video generation, and the silent-drop trap on the Maxcore*Request models
---

# The only reliable conditioning channel is `awareness`
MaxCore's Python agents (`script_agent.py` ScriptRequest, `visual_spec_agent` VisualSpecRequest) condition generation on an `awareness` string — it seeds the LLM prompt AND is parsed for `#hashtags`/industry signals in the fallback path. To inject any new generation-time context (artist name, releases, audience, top hooks, trending), compose it into a labeled awareness block and send it as `awareness`.

**Never** concatenate enrichment into the raw `topic`/`idea` — that corrupts the user-facing caption (see idea-awareness-field-separation.md). Structured/extra JSON fields on the MaxCore request body are a **silent no-op**: the `Maxcore*Request` pydantic models are plain (no agent reads arbitrary extra fields), so extra keys are accepted and ignored. Don't ship them as if they do something.

# The silent-drop trap
`MaxcoreTextRequest` and `MaxcoreMediaRequest` historically did NOT declare `awareness`, so any awareness the Node side sent for text/audio/video was dropped by pydantic before it reached the handler (image used `getattr`, so it partly worked). Same trap bit `ApiGenerateContentRequest` (the `/api/generate/content` path): it extended plain `BaseModel`, so the `awareness` block Node's `enrichWithAwareness` injected was dropped (`extra="ignore"`) and `getattr(req,"awareness","")` was always `""` — the endpoint never fed the bridge, which was the true root cause of the "generic body regardless of payload" symptom. Fix = extend `_AwarenessMixin` (it also normalises Node's `{contextString}` dict form).

**awareness is a GENERATION bridge, not just steering.** It spans the whole gap to the still-training in-house model (mirrors Veo for video): it seeds the model prompt AND, on garble/failure, the awareness-composed fallback builds the actual hook/body/cta. So the user's own per-request creative direction (instruction/themes) belongs IN awareness too — serialise it (cleaned via `_narrative_clause`, colons neutralised, themes as `•` bullets NOT `#hashtags` since distribution harvests awareness hashtags into shared `dist:hashtags:*`) and merge it AHEAD of external awareness (document order = signal priority).

**Rule:** a new conditioning signal only takes effect if you BOTH (a) declare the field on the relevant `Maxcore*Request` model in server.py, AND (b) pass it into `ScriptRequest(...)`/`VisualSpecRequest(...)` inside every handler (analyze, content-text, audio, video, image). Missing either half = the signal is silently ignored, no error.

# Parity across ALL generation endpoints (not just content)
The awareness bridge must be wired IDENTICALLY on every generation endpoint, personalised only by that endpoint's own direction fields. Historically ONLY `/api/generate/content` had the full wiring; `text`/`image`/`audio`/`video` each silently dropped Node awareness (plain `BaseModel`) and/or fed `brief.directives` as awareness (the parser quotes directives verbatim — wrong). Fix pattern, three parts per endpoint:
1. Request model extends `_AwarenessMixin` (+ declare `instruction`/`extra_context`/`content_themes` direction fields) — stops the silent drop.
2. Build the merged awareness with the shared `_merged_awareness_for(req)` helper (user direction via `awareness_from_direction` FIRST, then external `req.awareness`; NO directives fallback — return "" when empty, agents handle it) and thread it into that endpoint's agent request (`ScriptRequest`/`VisualSpecRequest`/`VideoAgentRequest`).
3. Thread awareness into the actual RENDER, not just the concept text. Text/image/video get this for free (agent output → render). AUDIO is the trap: awareness only shaped a metadata "concept" while the sound came from fingerprint/RNG — wire the awareness/intent-derived `brief.tempo` into BPM band selection so the live signal reaches the dataset-sample render.

**Why:** "each endpoint has an equal awareness layer" is the design intent, but code drifted so only one route implemented it — verify with `grep "_AwarenessMixin" server.py` that every `ApiGenerate*Request` extends it, and that no gen handler still does `awareness="\n".join(... brief.directives)`. (Platform endpoints at the bottom of server.py are separate and may still use the directives fallback — out of the 5-gen-route scope.)

# How to verify it actually landed
A generation asset's `metadata.source` flips to `"awareness"` (fallback parsed your block) or `"ai_model"` (LLM consumed the prompt) once awareness is wired — previously text could not report `source:"awareness"`. Model output may still look garbled (the in-house model is undertrained); that is a model-quality issue, not a wiring failure. Wiring correctness = the signal reaches the agent, not that it's echoed verbatim.
