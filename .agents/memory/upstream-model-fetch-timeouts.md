---
name: upstream model fetch timeouts
description: Why api-server fan-out calls to the Python model must use a long-timeout undici Agent, not bare fetch, and how the Node proxy timeouts must be set.
---

# Upstream model-call timeouts (api-server → Python engine)

**Rule:** Any api-server route that fans out to the Python model engine for
generation (analyze + per-asset generate) must issue the upstream call through an
`undici` `Agent` with `headersTimeout`/`bodyTimeout` raised well past the slowest
expected render (e.g. 900_000ms). Do NOT use the bare global `fetch`.

**Why:** Node's global `fetch` (undici) applies a default **300s headersTimeout**.
Under concurrent load a single *honest* multimodal request legitimately spends
>300s waiting for the model, so undici aborts the valid in-flight call and the
route surfaces it as an HTTP 500. Observed concretely: multimodal HIGH(10)
concurrency → all 10 returned 500 at ~315s with bare fetch; after switching to a
long-timeout undici Agent pool, HIGH(10) passed 10/10 at ~340–357s.

**How to apply:**
- Mirror the existing pool pattern (`model-proxy.ts`, `app.ts`, `keepalive.ts`):
  `new Agent({ keepAliveTimeout, keepAliveMaxTimeout, connections, pipelining,
  headersTimeout, bodyTimeout })`, then pass `dispatcher: pool` to the call.
- On the Node HTTP server (worker `app.listen`): keep `server.timeout = 0`
  (socket inactivity) — a finite value KILLS long generations because no bytes
  flow on the client socket while the model works. Keep `requestTimeout` and
  `headersTimeout` BOUNDED (govern receiving the client *request*, so slowloris
  is still rejected).
- Capacity caveat: `connections: 32` × N Node workers caps concurrent upstream
  sockets to one Python process. Fixing the timeout exposes real CPU/RAM/ffmpeg
  pressure next; add a semaphore/backpressure layer before raising concurrency
  further rather than just lifting timeouts.
- Other routes still using bare `fetch` for upstream model calls have the same
  latent 300s-abort bug; convert them the same way if they show 500s under load.
