---
name: Awareness-only social/ad content
description: How template escape hatches were closed so all social/ad copy comes from live awareness signals
---

Directive: social + ad content must ALWAYS compose from live awareness signals; static template pools are dead code, kept only as truly-unreachable last resorts.

**Rules:**
- Every ScriptRequest/DistributionRequest a handler builds MUST pass `awareness=` — extra body fields or omitted awareness silently route to model/template paths. Guarantee non-emptiness with `_effective_awareness(platform, _merged_awareness_for(req))` (platform buffer appended after caller signals).
- ScriptAgent awareness composition (`_awareness_compose`) is deterministic pure-Python and needs NO model — call it even when `_model_ready` is False; never gate the awareness path on model readiness.
- When awareness-composed copy is used, it is authoritative for hook, body, AND cta (no score gate vs pool hooks). Only `peak_replicated` (account's own proven top performer) outranks it for the hook.
- On composition failure with awareness present, derive copy from the raw awareness lines — do not ship pool templates.
- `_merged_awareness_for(req)` alone can be EMPTY (e.g. AdGenerateRequest with no direction fields at cold start) — always wrap with `_effective_awareness`.

**Why:** template hooks were shipping in ad creatives (`source: "template"`) because the ads handler passed raw merged awareness (empty) and the score gate let pool hooks beat awareness copy.

**How to apply:** any new generation/scoring handler that constructs ScriptRequest/DistributionRequest must pass guaranteed-non-empty awareness and use natural-theme `idea` text (never instruction-style "video ad for X" — it leaks verbatim into output).
