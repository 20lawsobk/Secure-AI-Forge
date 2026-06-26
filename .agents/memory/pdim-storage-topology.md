---
name: pdim storage topology & WRONGPASS diagnosis
description: How pdim (pocketdimensionstorage.replit.app) auth works, why STORAGE_BEARER_TOKEN can be WRONGPASS, and how to make env-var token changes actually take effect
---

# pdim storage: multi-instance auth

pdim (`pocketdimensionstorage.replit.app`) is a Redis-like HTTP exec API
(`POST /api/redis/instances/{id}/exec` with `{"cmd","args"}`, `Authorization: Bearer <token>`).
It hosts **multiple named instances, each with its OWN token**.

- `GET /api/redis/instances` is **unauthenticated** and lists every instance with
  `id`, `name`, `tokenHint` (e.g. `mbs_c6fe…3a0d`), `keyCount`, `lastUsedAt`.
- `WRONGPASS Invalid token for this instance` (HTTP 403) means the token does NOT
  match the instance in the URL path. It is a **token/instance mismatch, NOT a dead
  server** — the instance can be perfectly healthy.

**Diagnosis recipe:** `GET /api/redis/instances`, then PING each instance id
against every token you hold in env (`STORAGE_BEARER_TOKEN`, `AI_TRAINING_KEY_PROD`,
`ADMIN_KEY`). Match `tokenHint` prefix/suffix to find the right pairing.

**Why:** the configured instance's correct token may live in a *differently-named*
env var. The valid token for the configured `max-booster-training` instance was the
value in `AI_TRAINING_KEY_PROD`, while `STORAGE_BEARER_TOKEN` (and the token embedded
in `STORAGE_CONNECTION_URL`) held a stale value matching no live instance.

**How to apply:** before assuming pdim is offline, verify `available:false` against
`/api/redis/instances` + per-instance PING. If a token authenticates an instance,
fix the env var to that value (storage_client reads `STORAGE_HTTP_URL` +
`STORAGE_BEARER_TOKEN`; also fix the token embedded in `STORAGE_CONNECTION_URL`).

# Making a shared env-var token change actually take effect

Changing a shared env var is not enough — `restart_workflow` does NOT reliably
reload it because the `Start application` command backgrounds the api-server with `&`,
and that detached supervisor survives the restart carrying the OLD env. It keeps
respawning `uv run python3 server.py` children that inherit the stale token, and
whichever Python grabs the `/tmp/maxcore_model_9878.lock` on :9878 wins.

**Symptom:** `/storage/status` still `available:false` after a token fix; newly
spawned `server.py` procs still show the old token.

**Fix:** find the live procs and inspect their loaded token via
`tr '\0' '\n' < /proc/<pid>/environ | grep STORAGE_BEARER_TOKEN`. Kill the entire
OLD-token supervisor tree (the detached node api-server + its `uv`/python children),
not just the workflows. A correct-token supervisor then rebinds :9878. Verify with
`curl -H "X-Admin-Key: $ADMIN_KEY" http://localhost:9878/storage/status` → `available:true`.

# What is actually in the configured instance (max-booster-training)

Only 5 keys, NO audio corpus: `received:chunks` (zset, ~114k chunk-id refs),
`received:stream` (stream of {chunkId,offset,size=64MB,receivedAt} manifest entries —
no binaries stored as keys), `mb:watchdog:state`, `mb:coverage:state(:index)`.
The 5219-key, actively-used instance is the separate `max-booster-agent` (its token,
hint `18cf…7713`, is NOT in this repl's env). Dataset retrieval code expects
`mb:dataset:{name}:meta` + `:chunk:{idx}` (lists of TEXT records) — none exist here.
Public dataset sources (MusicBench/MusicCaps) yield TEXT captions only, not waveforms.
