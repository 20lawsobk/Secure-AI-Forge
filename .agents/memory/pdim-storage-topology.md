---
name: pdim storage topology & WRONGPASS diagnosis
description: How pdim (pocketdimensionstorage.replit.app) per-instance auth works, why STORAGE_BEARER_TOKEN shows WRONGPASS, which env vars storage_client actually reads, and how availability self-heals
---

# pdim storage: multi-instance auth

pdim (`pocketdimensionstorage.replit.app`) is a Redis-like HTTP exec API
(`POST /api/redis/instances/{id}/exec` with `{"cmd","args"}`, `Authorization: Bearer <token>`).
It hosts **multiple named instances, each with its OWN token**, and tokens are
**rotated by the user on pdim** — a token that worked yesterday can WRONGPASS today.

- `GET /api/redis/instances` is **unauthenticated** and lists every instance with
  `id`, `name`, `tokenHint`, `keyCount`.
- `WRONGPASS Invalid token for this instance` (HTTP 403) means the token does NOT
  match the instance in the URL path. It is a **token/instance mismatch, NOT a dead
  server** — the instance can be perfectly healthy.

**Diagnosis recipe (decisive):** `GET /api/redis/instances`, then PING **each**
instance id using the *current* `$STORAGE_BEARER_TOKEN`. Whichever returns `PONG`
is the instance that token belongs to. Point the app there. Do NOT keep re-testing
the same token against the same instance — scan instances instead.

# Which env vars actually drive storage

`storage_client.py` reads only: `STORAGE_HTTP_URL` (the `.../instances/{id}/exec`
URL), `STORAGE_BEARER_TOKEN` (the bearer), and `STORAGE_INSTANCE` (cosmetic name in
`/storage/status`). **`STORAGE_CONNECTION_URL` is UNUSED by code** — it was observed
mangled (`pdim://pdim://<agentTok>@.../22c8e6d2…@.../f26378c8…`) and ignoring it is
safe. So to repoint instances you only need to fix `STORAGE_HTTP_URL` (+ optionally
`STORAGE_INSTANCE`); the token stays in `STORAGE_BEARER_TOKEN`.

**Why:** the token/URL can drift to different instances independently (URL still
points at the old instance after the user rotates+regenerates a token for another
instance). `available:false` with `url_configured:true` = token doesn't match the
URL's instance.

# Availability self-heals — a startup 403 is not permanent

`is_available` caches the first `ping()`, BUT a daemon `_periodic_health_check`
thread re-pings every 30s and flips `_available` back. So once `STORAGE_HTTP_URL`
+ `STORAGE_BEARER_TOKEN` are a matching pair, availability recovers within ~30s (or
immediately on a fresh process). A transient startup ping failure does not wedge it.

# Making an env-var change take effect

After `setEnvVars`, restart the workflows so a Python with the new env wins the
`/tmp/maxcore_model_9878.lock` on :9878. Multiple workflows spawn `server.py`
(`Start application` and `artifacts/api-server: API Server`); **restart BOTH** so no
old-env standby Python can grab the lock. Verify with
`curl -H "X-Admin-Key: $ADMIN_KEY" localhost:9878/storage/status` → `available:true`
and the expected `instance` name.

# Ping pool must be separate from the main request pool

`StorageClient` uses a dedicated `_ping_pool_manager` (1 connection, `read=20s`) for
`ping()` / `_periodic_health_check`, separate from the main pool (64 connections,
`read=8s`). This is required for two reasons:

1. **Cold-wake latency**: Replit deployments take 5–15 s on first request after idle.
   An 8 s read timeout fires during that window, marking pdim "offline" for the entire
   cold-wake period. The 20 s ping pool survives it.
2. **Pool saturation**: Under load the 64-connection main pool can be fully occupied.
   A dedicated 1-connection ping pool can never be starved by request traffic.

`_init_storage()` in `server.py` retries the startup ping 4× with 5 s delays (20 s
total) before falling back to local mode — gives pdim a fair chance to cold-wake before
we declare it offline at boot.

`_periodic_health_check` sleeps 5 s (not 10 s) when offline so cold-wake detection
happens within one probe window.

# Whole-app hang ≠ token mismatch

A second, distinct failure mode (seen 2026-07-08): **every** route on
`pocketdimensionstorage.replit.app` hangs — TCP connects (Replit edge accepts)
but no HTTP response ever comes back, including the unauthenticated
`GET /api/redis/instances` and plain `GET /`. That is the pdim **deployment
itself being down/hung**, not an auth problem; no token scan will help. The
user must wake/republish their pdim app. Distinguish quickly: WRONGPASS/403 =
token mismatch (fixable here); universal read-timeout = pdim app down (not
fixable from this project).

Health-check protocol: `storage_client` POSTs `{"cmd":"PING"}` to
`STORAGE_HTTP_URL` itself (the `.../exec` URL) — GETing `/ping` or `/health`
on the host is the wrong protocol and proves nothing.

# Resolution of record

App was repointed from `max-booster-training` (id `f26378c8…`) to
**`max-booster-agent`** (id `22c8e6d237afe8ae41541f87`) because that is the instance
the user's `STORAGE_BEARER_TOKEN` authenticates. Audio dataset
(`mb:dataset:audio:meta` + `:chunk:{idx}`) is seeded into that instance.
