---
name: PDIM pocket multiplication
description: Pocket = one orchestrator dedup namespace; nesting is pure path namespacing; payload/contract rules
---

# Pocket-dimension multiplication design

- A "pocket" is exactly one PDIMOrchestrator dedup + single-flight namespace (`pocket:<path>`). Nesting pockets inside pockets is pure path namespacing (`root/a/b/...`, unbounded depth) — no separate storage hierarchy needed; zlib-compressing every stored payload is what makes "as many pockets inside one pocket as you want" cheap.
- **Why:** user's pdim spec says compression allows unbounded pocket nesting; orchestrator namespaces already give isolation + dedup for free.
- **How to apply:** any new "pocket" style feature should map to orchestrator namespaces, not new storage layers.

Contract rules learned:
- `PDIMOrchestrator.compute(request, fn, namespace=)` requests must be JSON-dumpable → pass sha256 content digests of ndarrays, never the arrays; compute fn must return a dict (encode ndarrays as zlib+b64 with shape/dtype/codec).
- Python server API auth uses `X-Api-Key` (or `X-Admin-Key`) headers, NOT `Authorization: Bearer` — a Bearer call gets 401 "API key required".
- Don't tighten FastAPI matrix params to `list[list[float]]` — batched A is 3-D; validate via np.asarray + explicit 400/413 instead.
