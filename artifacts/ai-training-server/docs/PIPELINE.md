# Inference Pipeline — 10-Stage Map

This system implements the full canonical LLM inference pipeline. The
capabilities are organized **by domain** (model core, agents, intelligence
layer, render fabric, API) rather than under ten stage-named folders — see
["Why not stage-named folders?"](#why-not-stage-named-folders) at the bottom.
This document is the authoritative map of *which files implement each stage*.

| # | Stage | Status | Primary implementation |
|---|-------|--------|------------------------|
| 1 | Input Encoding | ✅ | `ai_model/model/transformer.py` — token embedding (`token_emb`) + RoPE positional encoding (`precompute_rope_freqs`); modality encoders feed a shared latent (image/audio) |
| 2 | Context Assembly | ✅ | `ai_model/model/creative_model.py` — prompt encode, truncation, BOS handling; awareness/brief context assembled per request |
| 3 | Semantic Projection | ✅ | `ai_model/request_intelligence.py` — `build_brief()` → `GenerationBrief` (intent, audience, entities, task, strategy) |
| 4 | Attention Computation | ✅ | `ai_model/model/transformer.py` — `RoPESelfAttention` (multi-head Q/K/V + softmax) |
| 5 | Layerwise Reasoning | ✅ | `ai_model/model/transformer.py` — `TransformerDecoderLayer` stack (SwiGLU FFN, residual, norm) |
| 6 | Latent Intent Resolution | ✅ | `ai_model/model/creative_model.py` — decoding-strategy selection (temperature/top-p/top-k, beam, contrastive); brief-driven tone/structure |
| 7 | Token Generation | ✅ | `ai_model/model/creative_model.py` — autoregressive loop with KV-cache (`generate`, `decode_one`, `_sample_next`) |
| 8 | Constraint Enforcement | ✅ | `ai_model/safety/content_safety.py` — content-safety policy applied **during** decode (logit masking) and **after** decode (screen / redact / refuse). See below. |
| 9 | Post-Processing | ✅ | `_clean_text` (`ai_model/agents/script_agent.py`), `_clean_headline` (`ai_model/image/image_engine.py`), `_clean` (`ai_model/video/video_agent.py`) |
| 10 | Delivery | ✅ | `server.py` FastAPI endpoints returning structured Pydantic responses (e.g. `/content/generate`, `/api/generate/*`) |

## Stage 8 — Constraint Enforcement (detail)

Added as a dedicated, deterministic policy engine in `ai_model/safety/`.

**Two enforcement surfaces:**

1. **During generation (logit masking).** `ContentSafety.bad_token_ids(tokenizer)`
   returns the token ids for hard-blocked terms; `CreativeModel._sample_next`
   masks them to `-inf` so they can never be emitted. Applies to every decode
   path (single, batched, beam) because they all route through `_sample_next`.
2. **After generation (screening).** `screen` / `enforce`:
   - **BLOCK** (severe: weapons/explosives, illicit drug synthesis, self-harm
     instructions, child sexual content, incitement to mass violence) →
     the whole output is replaced with a safe refusal.
   - **REDACT** (moderate: hate slurs) → the offending span is masked (`████`),
     the rest is kept.
   Wired into `request_intelligence.rank_candidates` (unsafe candidates are
   penalised so a clean variant wins; stored text is already enforced) and into
   `script_agent._clean_text` (every agent output is scrubbed regardless of
   ranking).

**Design notes**
- *No silent fakes:* violations are reported honestly (category + severity +
  matched span) and counted (`ContentSafety.stats()`).
- *Low false-positives on a creative generator:* SEVERE rules require
  instructional/exploitative intent (e.g. "how to build a bomb"), not the mere
  mention of a word that can appear in lyrics or marketing copy.

**Configuration (env)**
- `MB_SAFETY_ENABLED` — `1` (default) / `0` to disable entirely.
- `MB_SAFETY_MODE` — `enforce` (default), `monitor` (detect + count, pass text
  through unchanged), or `off`.

**Verification**
- `POST /api/safety/screen` `{"text": "..."}` → returns the verdict, the
  enforced text, and running violation counters.

## Why not stage-named folders?

A physical restructure into ten `stage_N_*` modules was considered and
**intentionally declined**. It would change no behavior while forcing a
large, risky move of a working system away from its domain-oriented layout
(model / agents / intelligence / retrieval / render fabric / API), inviting
regressions across every generation path for zero functional gain. This map
provides the same stage-to-code traceability without the churn. If a future
reorganization is desired, do it incrementally behind this map, not as a
big-bang rename.
