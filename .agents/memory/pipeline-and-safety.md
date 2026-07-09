---
name: 10-stage pipeline conformance + Stage 8 safety
description: How the inference pipeline maps to the canonical 10 stages, and where content-safety enforcement is wired
---

# 10-stage inference pipeline + content-safety (Stage 8)

The system already implements the canonical 10-stage LLM pipeline, organized
**by domain** (model core / agents / intelligence / retrieval / render fabric /
API), not by stage-named folders. The authoritative stage→file map lives in
`artifacts/ai-training-server/docs/PIPELINE.md` — update it, don't reinvent it.

**Decision: do NOT physically restructure into 10 stage-named modules.**
**Why:** it changes no behavior while forcing a large risky move of a working
system, inviting regressions across every generation path for zero functional
gain. PIPELINE.md gives the same traceability. If reorg is ever wanted, do it
incrementally behind that map, not as a big-bang rename.

## Stage 8 = content-safety, in `ai_model/safety/content_safety.py`
Deterministic, dependency-free policy engine (`ContentSafety`, singleton via
`get_safety()`). Severity NONE/REDACT/BLOCK. SEVERE categories require
instructional intent + harmful object (regex) to avoid false-positives on
creative lyrics/marketing; MODERATE = hate slurs (redacted).

**Enforcement is wired at THREE choke points — keep them wired if you touch
generation:**
1. During decode: `CreativeModel._sample_next` AND `beam_search` mask
   `bad_token_ids()` to -inf. Both decode families must apply the mask; beam
   does not call `_sample_next`, so it needs its own mask line.
2. Post-gen ranking: `request_intelligence.rank_candidates` calls
   `enforce()` once per candidate and `penalty_of(result)` (NOT `penalty()`,
   which re-screens and inflates counters).
3. Post-gen agent output: `script_agent._clean_text` runs `enforce()`.

**Mode semantics (env `MB_SAFETY_MODE`):** `enforce` (redact/refuse + mask +
penalize), `monitor` (detect + count only — `bad_token_ids` returns empty,
`penalty_of` returns 0, text passes through), `off`. `MB_SAFETY_ENABLED=0` = off.
`screen()` is the ONLY counter-incrementing method; call it once per text.

Verify: `POST /api/safety/screen {"text": "..."}` → verdict + enforced text +
live violation counters.
