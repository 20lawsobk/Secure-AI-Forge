---
name: api-server proxy is an explicit allowlist
description: New Python /api routes are unreachable from the dashboard until also registered in the api-server model-proxy
---

The dashboard (ai-dashboard) never talks to the Python training server directly.
Its Vite dev server proxies `/api` → api-server (Node), and api-server's
`src/routes/model-proxy.ts` forwards to the Python server (`MODEL_API_PORT`).

**The proxy is an explicit per-route allowlist, NOT a catch-all.** Every route is
registered individually (`router.post("/generate/content", ...)`, etc.). There is
no `router.all("*")` passthrough.

**Rule:** Any new endpoint added to `artifacts/ai-training-server/server.py`
under `/api/...` must ALSO get a matching handler in `model-proxy.ts` (calling
`proxyRequest(req, res, "/api/...")`), or the dashboard gets a 404 even though
the Python endpoint works. This applies to every method — GET/POST/PATCH/DELETE
each need their own registration. Forward query strings explicitly
(`new URLSearchParams(req.query).toString()`); path params via
`encodeURIComponent(req.params.x)`.

**Why:** `/api/generate/campaign` existed on the Python server but was NOT in the
proxy allowlist, so the dashboard could not reach it. Discovered when wiring the
campaign scheduling calendar.
