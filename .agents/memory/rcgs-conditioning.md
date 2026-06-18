---
name: RCGS retrieval-conditioned backgrounds + render-time probes
description: How real assets condition procedural video backgrounds, and why probes/modulation behave as they do — non-obvious testing + design gotchas
---

# What RCGS does
At render time, each procedural video background is "conditioned" with structure derived from a real retrieved asset (desaturated, heavy-blurred, mean-centered luminance map applied multiplicatively, low alpha). Separately, every conditioning call records an embedding "probe" so the self-healing CoverageWatchdog detects gaps from REAL render demand, not synthetic queries.

# Non-obvious design rules (keep these — they are intentional, not bugs)
- **Totality is mandatory.** `condition_background()` must NEVER raise into the render loop. Any failure (bad input, empty index, unreadable asset, embed/query/structure-map error) returns the ORIGINAL background unchanged. This mirrors the "no broken fallback" constraint.
- **Probe is recorded BEFORE the retrieval fallback.** So coverage sees the demand even when no asset can condition the frame (empty index, no match). Verify this ordering survives refactors — moving the probe after the fallback silently blinds coverage to the hardest gaps.
- **Flat/solid-color assets produce ZERO modulation, and that is correct.** A mean-centered structure map of a constant-luminance image is ~all-zero → multiplicative factor ~1 everywhere → frame unchanged. This is not a bug. (It cost a test cycle: a "did it modulate?" assertion must use a STRUCTURED asset, e.g. a gradient, not a solid PNG.)

# Path safety
Assets resolve STRICTLY under the uploads dir, basename-only (`Path(meta).name`), `resolve()` + parent-under-base check + `is_file()`. URL-only metadata is ignored. Traversal/symlinks rejected.

# Repo convention worth knowing
Generated media under `artifacts/ai-training-server/uploads/` (videos `*.mp4`, images `*.png`) is TRACKED by git on purpose (100+ committed). Incidental test-render outputs you create while verifying are NOT source — delete the stray file before finishing so the change set is just source edits.

# Live verification recipe
Restart the workflow owning the live python (see lock-topology note), then: `/coverage/report` gains additive `live_probes_total`/`live_probes_sampled` keys (their presence proves the reload). Fire one real `/api/generate-video`, wait for job `status:"done"`, then `/coverage/report` shows `live_probes_total > 0` and `probes > 0`. Job status is the string `"done"`, not `"completed"`.
