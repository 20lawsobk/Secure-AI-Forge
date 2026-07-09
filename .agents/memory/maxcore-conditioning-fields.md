---
name: MaxCore generation conditioning-field contract (awareness channel)
description: How to make new context actually influence MaxCore text/audio/video generation, and the silent-drop trap on the Maxcore*Request models
---

# The only reliable conditioning channel is `awareness`
MaxCore's Python agents (`script_agent.py` ScriptRequest, `visual_spec_agent` VisualSpecRequest) condition generation on an `awareness` string — it seeds the LLM prompt AND is parsed for `#hashtags`/industry signals in the fallback path. To inject any new generation-time context (artist name, releases, audience, top hooks, trending), compose it into a labeled awareness block and send it as `awareness`.

**Never** concatenate enrichment into the raw `topic`/`idea` — that corrupts the user-facing caption (see idea-awareness-field-separation.md). Structured/extra JSON fields on the MaxCore request body are a **silent no-op**: the `Maxcore*Request` pydantic models are plain (no agent reads arbitrary extra fields), so extra keys are accepted and ignored. Don't ship them as if they do something.

# The silent-drop trap
`MaxcoreTextRequest` and `MaxcoreMediaRequest` historically did NOT declare `awareness`, so any awareness the Node side sent for text/audio/video was dropped by pydantic before it reached the handler (image used `getattr`, so it partly worked). 

**Rule:** a new conditioning signal only takes effect if you BOTH (a) declare the field on the relevant `Maxcore*Request` model in server.py, AND (b) pass it into `ScriptRequest(...)`/`VisualSpecRequest(...)` inside every handler (analyze, content-text, audio, video, image). Missing either half = the signal is silently ignored, no error.

# How to verify it actually landed
A generation asset's `metadata.source` flips to `"awareness"` (fallback parsed your block) or `"ai_model"` (LLM consumed the prompt) once awareness is wired — previously text could not report `source:"awareness"`. Model output may still look garbled (the in-house model is undertrained); that is a model-quality issue, not a wiring failure. Wiring correctness = the signal reaches the agent, not that it's echoed verbatim.
