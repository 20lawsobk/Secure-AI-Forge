---
name: Engine-serve all heavy kernels (no-fallback discipline)
description: How to make a 2D-only in-house compute engine genuinely serve every heavy kernel, and how to make "fallbacks never fire" an auditable invariant.
---

When the directive is "built so well that fallbacks aren't needed at all," every heavy kernel must route through the in-house engine, and any safety-net path must be *observable*, not silent.

# Folding heavy ops onto a 2D-only engine
- **Batched / N-D matmul** → fold to a stack of 2D engine GEMMs: reshape `[...,M,K]@[K,N]` to one 2D GEMM; broadcast batch dims + loop per-slice for fully-batched. No NumPy matmul on the heavy path.
- **Any-axis softmax** → move the target axis to last, fold to 2D `[-1, L]`, call the engine's last-axis softmax, then unfold + swap back. (The engine kernel was validated on 2D, so folding guarantees it serves any rank.)
- **Masked / multi-head / cross attention** → decompose into engine GEMM (QKᵀ) + engine softmax + engine GEMM (probs@V). Only the score scaling and the additive mask stay NumPy — those are *elementwise*, for which the engine has no kernel; that is the implementation, not a fallback. Keep the native fused attention kernel for the common unmasked same-shape case.

# Observability discipline (so the claim is checkable)
- Distinguish two counter suffixes: `*.numpy` = "no engine kernel exists for this op" (legit elementwise/reduce), vs `*.engine_fallback` = "engine was present but a kernel call raised."
- "Fallbacks never fire" means **`engine_fallback` stays 0**; a nonzero count is a real regression, not noise.
- Back the claim with tests that assert counter *deltas* are 0 for representative heavy shapes (batched gemm, non-last-axis softmax, masked multi-head attention), not merely correctness vs NumPy.

**Why:** an audit that lumps legit elementwise NumPy together with engine failures can't prove the engine actually served the heavy compute. Separating "no kernel exists" from "engine failed" turns "no fallback" into a single checkable invariant.

**How to apply:** any new heavy kernel added to the backend must (a) route its matmul/softmax through the engine via the folding helpers, (b) use NumPy only for elementwise/reduce, and (c) ship a counter-delta test asserting `*.numpy` and `*.engine_fallback` deltas are 0 for that op.

# Load-test honesty for an in-house dedup engine
- A "90M-scale" number from a bounded unique working set dominated by cache hits is a **dedup fast-path projection**, NOT an executed N-concurrent run. Label it exactly that, and surface the *cold real-compute rate* separately from dedup hits.
- Use **per-run (uuid) namespaces** for the dedup keys so reruns don't bleed cached results from a prior run into the next (which would silently inflate the hit rate).
