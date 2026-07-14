import os from "os";
import { Router, type IRouter, type Request, type Response } from "express";
import { Agent, request as undiciRequest } from "undici";
import {
  contentAwarenessService,
  type ContentGenerationMode,
} from "../services/contentAwarenessService.js";

const router: IRouter = Router();

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

// Server-side key injected when the browser hasn't provided one.
// The Node proxy runs on localhost behind Vite's /api proxy — it is a
// trusted gateway, so injecting the env key here is safe. External
// callers never reach this server directly in dev.
const _SERVER_FALLBACK_KEY =
  process.env.AI_SERVER_KEY ||
  process.env.AI_TRAINING_KEY_PROD ||
  process.env.ADMIN_KEY ||
  "";

// ─── Keep-alive connection pool ─────────────────────────────────────────────
// Reuse TCP connections to the Python server — eliminates per-request TCP
// handshake overhead.  Pool tuned for high-concurrency AI proxy:
//   connections: cpu*8 (min 64) — enough concurrent in-flight requests for
//     all Node workers × expected generation bursts without queuing at the pool.
//   keepAliveTimeout: 300s — matches uvicorn's timeout_keep_alive so the
//     server side never closes first (avoids ECONNRESET under load).
//   headersTimeout/bodyTimeout added per-request (see undiciRequest calls) so
//     long model generations never get aborted by socket inactivity.
const _keepAlivePool = new Agent({
  keepAliveTimeout: 300_000,
  keepAliveMaxTimeout: 600_000,
  connections: Math.max(64, os.cpus().length * 8),
  pipelining: 1,
});

// ─── Circuit Breaker ────────────────────────────────────────────────────────
// After CB_FAILURE_THRESHOLD consecutive upstream failures the circuit opens
// and requests fail-fast with 503 for CB_RECOVERY_MS, then enter half-open
// (one probe allowed through). Resets fully on any successful response.

const CB_FAILURE_THRESHOLD = 5;
const CB_RECOVERY_MS = 15_000;

let _cbFailures = 0;
let _cbOpenSince: number | null = null;

function _cbIsOpen(): boolean {
  if (_cbOpenSince === null) return false;
  if (Date.now() - _cbOpenSince >= CB_RECOVERY_MS) {
    // half-open: reset so the next request probes the upstream
    _cbOpenSince = null;
    _cbFailures = 0;
    return false;
  }
  return true;
}

function _cbRecordSuccess(): void {
  _cbFailures = 0;
  _cbOpenSince = null;
}

function _cbRecordFailure(): void {
  _cbFailures++;
  if (_cbFailures >= CB_FAILURE_THRESHOLD && _cbOpenSince === null) {
    _cbOpenSince = Date.now();
    console.warn(
      `[CircuitBreaker] Opened after ${_cbFailures} consecutive failures — fast-failing for ${CB_RECOVERY_MS / 1000}s`,
    );
  }
}

// ─── TTL Cache for hot read-only endpoints ─────────────────────────────────

interface CacheEntry {
  data: unknown;
  status: number;
  expiry: number;
}

const _cache = new Map<string, CacheEntry>();

const CACHE_TTL_MS: Record<string, number> = {
  "/dashboard/stats": 5_000,
  "/health": 8_000,
  "/model/status": 8_000,
  "/gpu/status": 6_000,
  "/gpu/hyper/status": 6_000,
  "/gpu/capabilities": 15_000,
  "/storage/status": 10_000,
  "/watchdog/status": 10_000,
  "/training/continuous/status": 4_000,
  "/training/puller/status": 8_000,
  "/training/puller/sources": 30_000,
};

function getCached(path: string): CacheEntry | null {
  const entry = _cache.get(path);
  if (entry && entry.expiry > Date.now()) return entry;
  _cache.delete(path);
  return null;
}

function setCached(path: string, status: number, data: unknown): void {
  const ttl = CACHE_TTL_MS[path];
  if (ttl) _cache.set(path, { data, status, expiry: Date.now() + ttl });
}

// ─── Awareness Enrichment ───────────────────────────────────────────────────
// Fetches live industry context and merges it into req.body.awareness before
// proxying to the Python AI server. Always additive — never blocks generation.
// A 3 s race guard prevents cold-cache RSS fetches from delaying responses.

// ─── Body normalisation ─────────────────────────────────────────────────────
// Renames mis-cased or aliased fields and injects missing required defaults
// before forwarding to the Python AI server.  The Python schemas use strict
// FastAPI validation (422 on any missing required field), so we fix things
// here rather than in every caller / dashboard page.

function _resolveUserId(body: Record<string, unknown>): string {
  return (
    (body["user_id"] as string | undefined)?.trim() ||
    (body["userId"] as string | undefined)?.trim() ||
    (body["artistProfileId"] as string | undefined)?.trim() ||
    "default_user"
  );
}

function normalizeBody(
  req: Request,
  renames: Array<[from: string, to: string]>,
  defaults: Array<[key: string, value: unknown]>,
): void {
  const body = req.body as Record<string, unknown>;
  for (const [from, to] of renames) {
    if (body[from] !== undefined && (body[to] === undefined || body[to] === null || body[to] === "")) {
      body[to] = body[from];
    }
    // always remove the aliased key so Python never sees both
    if (from !== to) delete body[from];
  }
  for (const [key, value] of defaults) {
    if (body[key] === undefined || body[key] === null || body[key] === "") {
      body[key] = value;
    }
  }
}

async function enrichWithAwareness(
  req: Request,
  mode: ContentGenerationMode,
): Promise<void> {
  try {
    const ctx = await Promise.race([
      contentAwarenessService.getContextForMode(mode),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), 3_000)),
    ]);
    if (ctx && ctx.confidence > 0 && ctx.contextString) {
      req.body = {
        ...(req.body as Record<string, unknown>),
        awareness: ctx.contextString,
      };
    }
  } catch {
    // Awareness enrichment is always additive — never block generation
  }
}

// ─── Safe JSON parsing (handles non-JSON upstream error bodies) ─────────────

async function parseBodyText(body: {
  text(): Promise<string>;
}): Promise<unknown> {
  const text = await body.text();
  try {
    return JSON.parse(text);
  } catch {
    return { error: "Upstream returned non-JSON", detail: text.slice(0, 300) };
  }
}

// ─── Shared error handler for proxy network failures ─────────────────────────

function handleProxyNetworkError(
  err: unknown,
  res: Response,
  path: string,
): void {
  const elapsed = Date.now();
  console.error(`[Proxy] Network error proxying to ${path}:`, err);
  const e = err as any;
  if (e.name === "AbortError" || e.code === "ABORT_ERR") {
    _cbRecordFailure();
    res.status(504).json({
      error: "Upstream timeout",
      detail: "AI training server did not respond within 45 s.",
    });
  } else if (
    (e as NodeJS.ErrnoException).code === "ECONNREFUSED" ||
    e.cause?.code === "ECONNREFUSED"
  ) {
    _cbRecordFailure();
    res.status(503).json({
      error: "AI model server unavailable",
      detail:
        "The Python AI training server is not running or still initializing.",
    });
  } else if (
    e.cause?.code === "UND_ERR_SOCKET" ||
    e.cause?.message?.includes("other side closed") ||
    e.code === "UND_ERR_SOCKET"
  ) {
    _cbRecordFailure();
    res.status(503).json({
      error: "AI model server closed connection",
      detail:
        "The request was dropped — the AI server may be busy. Please retry.",
    });
  } else {
    res.status(500).json({ error: "Proxy error", detail: String(err) });
  }
  void elapsed;
}

// ─── Core proxy function ────────────────────────────────────────────────────

async function proxyRequest(
  req: Request,
  res: Response,
  path: string,
): Promise<void> {
  const isGet = req.method === "GET" || req.method === "HEAD";
  const startTime = Date.now();

  // Serve from cache for cacheable GETs
  if (isGet) {
    const cached = getCached(path);
    if (cached) {
      res.setHeader("X-Cache", "HIT");
      res.status(cached.status).json(cached.data);
      return;
    }
  }

  // Circuit breaker — fail fast when the upstream is known to be down
  if (_cbIsOpen()) {
    const retryIn = Math.ceil(
      (CB_RECOVERY_MS - (Date.now() - (_cbOpenSince ?? Date.now()))) / 1000,
    );
    res.status(503).json({
      error: "AI model server temporarily unavailable",
      detail: `Circuit breaker open — retry in ~${retryIn}s.`,
    });
    return;
  }

  try {
    const url = `${MODEL_API_BASE}${path}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45_000);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (req.headers["x-admin-key"]) {
      headers["X-Admin-Key"] = req.headers["x-admin-key"] as string;
    } else if (req.headers["x-api-key"]) {
      headers["X-Api-Key"] = req.headers["x-api-key"] as string;
    } else if (_SERVER_FALLBACK_KEY) {
      // Browser hasn't sent an auth header (e.g. admin key not entered yet).
      // Inject the server-side key so generate endpoints don't 401.
      headers["X-Api-Key"] = _SERVER_FALLBACK_KEY;
    }

    let upstreamRes: Awaited<ReturnType<typeof undiciRequest>>;
    try {
      upstreamRes = await undiciRequest(url, {
        method: req.method as any,
        signal: controller.signal,
        dispatcher: _keepAlivePool,
        headers,
        body: !isGet && req.body ? JSON.stringify(req.body) : undefined,
        headersTimeout: 0,
        bodyTimeout: 300_000,
      });
    } finally {
      clearTimeout(timeoutId);
    }

    const data = await parseBodyText(upstreamRes.body);

    // Treat 5xx upstream responses as failures for the circuit breaker
    if (upstreamRes.statusCode >= 500) {
      _cbRecordFailure();
    } else {
      _cbRecordSuccess();
    }

    // Populate cache for successful GET responses
    if (isGet && upstreamRes.statusCode < 300) {
      setCached(path, upstreamRes.statusCode, data);
    }

    res.setHeader("X-Cache", "MISS");
    res.status(upstreamRes.statusCode).json(data);
  } catch (err) {
    console.error(
      `[Proxy] Error proxying to ${path} (${Date.now() - startTime}ms):`,
      err,
    );
    handleProxyNetworkError(err, res, path);
  }
}

// ─── Binary proxy ─────────────────────────────────────────────────────────────
// Used for endpoints that return non-JSON (e.g. image/jpeg frame previews).
// Streams the raw upstream body through with the correct Content-Type header.
// Falls back to JSON error forwarding on non-200 responses.

async function proxyBinary(
  req: Request,
  res: Response,
  path: string,
): Promise<void> {
  if (_cbIsOpen()) {
    const retryIn = Math.ceil(
      (CB_RECOVERY_MS - (Date.now() - (_cbOpenSince ?? Date.now()))) / 1000,
    );
    res.status(503).json({
      error: "AI model server temporarily unavailable",
      detail: `Circuit breaker open — retry in ~${retryIn}s.`,
    });
    return;
  }

  try {
    const url = `${MODEL_API_BASE}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45_000);

    const headers: Record<string, string> = {};
    if (req.headers["x-admin-key"]) {
      headers["X-Admin-Key"] = req.headers["x-admin-key"] as string;
    } else if (req.headers["x-api-key"]) {
      headers["X-Api-Key"] = req.headers["x-api-key"] as string;
    } else if (_SERVER_FALLBACK_KEY) {
      headers["X-Api-Key"] = _SERVER_FALLBACK_KEY;
    }

    let upstreamRes: Awaited<ReturnType<typeof undiciRequest>>;
    try {
      upstreamRes = await undiciRequest(url, {
        method: req.method as any,
        signal: controller.signal,
        dispatcher: _keepAlivePool,
        headers,
        headersTimeout: 0,
        bodyTimeout: 300_000,
      });
    } finally {
      clearTimeout(timeoutId);
    }

    if (upstreamRes.statusCode >= 500) {
      _cbRecordFailure();
    } else {
      _cbRecordSuccess();
    }

    if (upstreamRes.statusCode !== 200) {
      // Non-200: try to forward as JSON, fall back to plain text
      const text = await upstreamRes.body.text();
      try {
        res.status(upstreamRes.statusCode).json(JSON.parse(text));
      } catch {
        res.status(upstreamRes.statusCode).send(text);
      }
      return;
    }

    const contentType =
      (upstreamRes.headers["content-type"] as string | undefined) ??
      "application/octet-stream";
    res.setHeader("Content-Type", contentType);
    res.status(200);
    const buf = Buffer.from(await upstreamRes.body.arrayBuffer());
    res.send(buf);
  } catch (err) {
    handleProxyNetworkError(err, res, path);
  }
}

// ─── Streamed binary proxy (large files) ────────────────────────────────────
// Used for rendered video downloads, which can be tens of MB and must not be
// fully buffered into memory or subjected to the short 45s abort window that
// `proxyBinary` uses for small previews. Pipes the upstream body straight
// through to the response and preserves Content-Type/Content-Length/
// Content-Disposition so download filename semantics survive the proxy hop.

async function proxyBinaryStream(
  req: Request,
  res: Response,
  path: string,
): Promise<void> {
  if (_cbIsOpen()) {
    const retryIn = Math.ceil(
      (CB_RECOVERY_MS - (Date.now() - (_cbOpenSince ?? Date.now()))) / 1000,
    );
    res.status(503).json({
      error: "AI model server temporarily unavailable",
      detail: `Circuit breaker open — retry in ~${retryIn}s.`,
    });
    return;
  }

  try {
    const url = `${MODEL_API_BASE}${path}`;
    const controller = new AbortController();
    // Large video files can take a while to transfer; only abort on true
    // inactivity (no bytes for 5 minutes), not on total transfer time.
    let idleTimer: NodeJS.Timeout = setTimeout(() => {}, 0);
    const bumpIdleTimer = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => controller.abort(), 300_000);
    };
    bumpIdleTimer();

    const headers: Record<string, string> = {};
    if (req.headers["x-admin-key"]) {
      headers["X-Admin-Key"] = req.headers["x-admin-key"] as string;
    } else if (req.headers["x-api-key"]) {
      headers["X-Api-Key"] = req.headers["x-api-key"] as string;
    } else if (_SERVER_FALLBACK_KEY) {
      headers["X-Api-Key"] = _SERVER_FALLBACK_KEY;
    }

    let upstreamRes: Awaited<ReturnType<typeof undiciRequest>>;
    try {
      upstreamRes = await undiciRequest(url, {
        method: req.method as any,
        signal: controller.signal,
        dispatcher: _keepAlivePool,
        headers,
        headersTimeout: 0,
        bodyTimeout: 0,
      });
    } catch (err) {
      clearTimeout(idleTimer);
      throw err;
    }

    if (upstreamRes.statusCode >= 500) {
      _cbRecordFailure();
    } else {
      _cbRecordSuccess();
    }

    if (upstreamRes.statusCode !== 200) {
      clearTimeout(idleTimer);
      const text = await upstreamRes.body.text();
      try {
        res.status(upstreamRes.statusCode).json(JSON.parse(text));
      } catch {
        res.status(upstreamRes.statusCode).send(text);
      }
      return;
    }

    const contentType =
      (upstreamRes.headers["content-type"] as string | undefined) ??
      "application/octet-stream";
    res.setHeader("Content-Type", contentType);
    const contentLength = upstreamRes.headers["content-length"];
    if (contentLength) res.setHeader("Content-Length", contentLength as string);
    const disposition = upstreamRes.headers["content-disposition"];
    if (disposition)
      res.setHeader("Content-Disposition", disposition as string);
    res.status(200);

    for await (const chunk of upstreamRes.body) {
      bumpIdleTimer();
      if (!res.write(chunk)) {
        await new Promise((resolve) => res.once("drain", resolve));
      }
    }
    clearTimeout(idleTimer);
    res.end();
  } catch (err) {
    handleProxyNetworkError(err, res, path);
  }
}

// ─── Routes ─────────────────────────────────────────────────────────────────

router.get("/health", async (req, res) => {
  await proxyRequest(req, res, "/health");
});

router.get("/api-keys", async (req, res) => {
  await proxyRequest(req, res, "/api-keys");
});

router.post("/api-keys", async (req, res) => {
  await proxyRequest(req, res, "/api-keys");
});

router.delete("/api-keys/:keyId", async (req, res) => {
  await proxyRequest(req, res, `/api-keys/${req.params.keyId}`);
});

router.post("/api-keys/:keyId/rotate", async (req, res) => {
  await proxyRequest(req, res, `/api-keys/${req.params.keyId}/rotate`);
});

router.get("/model/status", async (req, res) => {
  await proxyRequest(req, res, "/model/status");
});

router.get("/gpu/status", async (req, res) => {
  await proxyRequest(req, res, "/gpu/status");
});

router.get("/gpu/hyper/status", async (req, res) => {
  await proxyRequest(req, res, "/gpu/hyper/status");
});

router.get("/gpu/capabilities", async (req, res) => {
  await proxyRequest(req, res, "/gpu/capabilities");
});

router.get("/training/status", async (req, res) => {
  await proxyRequest(req, res, "/training/status");
});

router.post("/training/start", async (req, res) => {
  await proxyRequest(req, res, "/training/start");
});

router.get("/training/logs", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/training/logs${req.query.limit ? `?limit=${req.query.limit}` : ""}`,
  );
});

router.post("/training/stop", async (req, res) => {
  await proxyRequest(req, res, "/training/stop");
});

router.get("/training/datasets", async (req, res) => {
  await proxyRequest(req, res, "/training/datasets");
});

router.post("/training/schedule", async (req, res) => {
  await proxyRequest(req, res, "/training/schedule");
});

// ─── Continuous Training ───────────────────────────────────────────────────

router.get("/training/continuous/status", async (req, res) => {
  await proxyRequest(req, res, "/training/continuous/status");
});

router.post("/training/continuous/start", async (req, res) => {
  await proxyRequest(req, res, "/training/continuous/start");
});

router.post("/training/continuous/stop", async (req, res) => {
  await proxyRequest(req, res, "/training/continuous/stop");
});

router.get("/training/continuous/history", async (req, res) => {
  await proxyRequest(req, res, "/training/continuous/history");
});

// ─── Data Puller ───────────────────────────────────────────────────────────

router.get("/training/puller/status", async (req, res) => {
  await proxyRequest(req, res, "/training/puller/status");
});

router.get("/training/puller/sources", async (req, res) => {
  await proxyRequest(req, res, "/training/puller/sources");
});

router.post("/training/puller/pull", async (req, res) => {
  await proxyRequest(req, res, "/training/puller/pull");
});

router.post("/training/puller/start", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/training/puller/start${req.query.interval_minutes ? `?interval_minutes=${req.query.interval_minutes}` : ""}`,
  );
});

router.post("/training/puller/stop", async (req, res) => {
  await proxyRequest(req, res, "/training/puller/stop");
});

router.get("/platform/video/generate", async (req, res) => {
  await proxyRequest(req, res, "/platform/video/generate");
});

router.post("/platform/video/generate", async (req, res) => {
  // Python: PlatformVideoRequest { user_id, topic, ... }
  // Dashboard sends: { idea, ...} with no user_id
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["idea", "topic"]],  // rename idea → topic
    [
      ["user_id", _resolveUserId(b)],
      ["topic",   (b["topic"] ?? b["idea"] ?? "") as string],
    ],
  );
  await enrichWithAwareness(req, "video_script");
  await proxyRequest(req, res, "/platform/video/generate");
});

router.post("/content/generate", async (req, res) => {
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/content/generate");
});

router.get("/boostsheets", async (req, res) => {
  await proxyRequest(req, res, "/boostsheets");
});

router.get("/dashboard/stats", async (req, res) => {
  await proxyRequest(req, res, "/dashboard/stats");
});

router.get("/storage/status", async (req, res) => {
  await proxyRequest(req, res, "/storage/status");
});

router.post("/storage/feedback", async (req, res) => {
  await proxyRequest(req, res, "/storage/feedback");
});

router.get("/storage/curriculum/:userId", async (req, res) => {
  await proxyRequest(req, res, `/storage/curriculum/${req.params.userId}`);
});

router.get("/storage/datasets", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets");
});

router.post("/storage/datasets/register", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets/register");
});

router.get("/storage/datasets/audio/status", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets/audio/status");
});

router.post("/storage/datasets/audio/seed", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets/audio/seed");
});

router.get("/storage/checkpoints", async (req, res) => {
  await proxyRequest(req, res, "/storage/checkpoints");
});

router.post("/storage/checkpoint/save", async (req, res) => {
  await proxyRequest(req, res, "/storage/checkpoint/save");
});

router.get("/storage/checkpoint/:modelId", async (req, res) => {
  await proxyRequest(req, res, `/storage/checkpoint/${req.params.modelId}`);
});

router.get("/storage/session", async (req, res) => {
  await proxyRequest(req, res, "/storage/session");
});

router.get("/storage/pipeline/status", async (req, res) => {
  await proxyRequest(req, res, "/storage/pipeline/status");
});

router.get("/storage/artist/:profileId", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/storage/artist/${encodeURIComponent(req.params.profileId)}`,
  );
});

router.post("/storage/artist/:profileId", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/storage/artist/${encodeURIComponent(req.params.profileId)}`,
  );
});

router.post("/storage/artist/:profileId/releases", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/storage/artist/${encodeURIComponent(req.params.profileId)}/releases`,
  );
});

router.post("/training/start-from-storage", async (req, res) => {
  await proxyRequest(req, res, "/training/start-from-storage");
});

// ─── Platform API Routes — Main Music Platform Integration ───────────────────

router.post("/platform/social/generate", async (req, res) => {
  // Python: PlatformSocialRequest { user_id, topic, platform, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["userId", "user_id"]],
    [
      ["user_id", _resolveUserId(b)],
      ["topic",   (b["topic"] ?? b["idea"] ?? b["content"] ?? "") as string],
    ],
  );
  await enrichWithAwareness(req, "social");
  await proxyRequest(req, res, "/platform/social/generate");
});

router.post("/platform/social/autopilot", async (req, res) => {
  // Python: PlatformAutopilotRequest { user_id, ... }
  // Dashboard sends userId (camelCase)
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["userId", "user_id"]],
    [["user_id", _resolveUserId(b)]],
  );
  await enrichWithAwareness(req, "social");
  await proxyRequest(req, res, "/platform/social/autopilot");
});

router.post("/platform/daw/generate", async (req, res) => {
  // Python: PlatformDAWRequest { user_id, mode, topic, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["userId", "user_id"]],
    [
      ["user_id", _resolveUserId(b)],
      ["mode",    "lyrics"],
    ],
  );
  await enrichWithAwareness(req, "songwriting");
  await proxyRequest(req, res, "/platform/daw/generate");
});

router.post("/platform/distribution/plan", async (req, res) => {
  // Python: PlatformDistributionRequest { user_id, track_title, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [
      ["userId",     "user_id"],
      ["title",      "track_title"],
      ["track",      "track_title"],
      ["trackTitle", "track_title"],
      ["song",       "track_title"],
    ],
    [
      ["user_id",     _resolveUserId(b)],
      ["track_title", (b["track_title"] ?? b["trackTitle"] ?? b["title"] ?? b["track"] ?? b["song"] ?? "Untitled") as string],
    ],
  );
  await enrichWithAwareness(req, "distribution");
  await proxyRequest(req, res, "/platform/distribution/plan");
});

router.get("/platform/model/info", async (req, res) => {
  await proxyRequest(req, res, "/platform/model/info");
});

router.post("/platform/model/reload", async (req, res) => {
  await proxyRequest(req, res, "/platform/model/reload");
});

// ─── AI Ad System & Autopilot ────────────────────────────────────────────────

router.post("/platform/ads/record", async (req, res) => {
  // Python: AdRecordRequest { user_id, platform, ad_type, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["userId", "user_id"]],
    [
      ["user_id",  _resolveUserId(b)],
      ["platform", "meta"],
      ["ad_type",  "video"],
    ],
  );
  await proxyRequest(req, res, "/platform/ads/record");
});

router.post("/platform/ads/generate", async (req, res) => {
  // Python: AdGenerateRequest { user_id, product, platform, goal, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [
      ["userId",     "user_id"],
      ["name",       "product"],
      ["artistName", "product"],
      ["artist",     "product"],
    ],
    [
      ["user_id",  _resolveUserId(b)],
      ["product",  (b["product"] ?? b["name"] ?? b["artistName"] ?? b["artist"] ?? "Artist") as string],
      ["platform", "meta"],
      ["goal",     "streams"],
    ],
  );
  await enrichWithAwareness(req, "ad_copy");
  await proxyRequest(req, res, "/platform/ads/generate");
});

router.post("/platform/ads/autopilot", async (req, res) => {
  // Python: AdAutopilotRequest { user_id, ... }
  // Dashboard sends userId (camelCase)
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [["userId", "user_id"]],
    [["user_id", _resolveUserId(b)]],
  );
  await enrichWithAwareness(req, "ad_copy");
  await proxyRequest(req, res, "/platform/ads/autopilot");
});

router.post("/platform/ads/audience", async (req, res) => {
  // Python: AdAudienceRequest { user_id, product, platform, goal, ... }
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [
      ["userId",     "user_id"],
      ["name",       "product"],
      ["artistName", "product"],
      ["artist",     "product"],
    ],
    [
      ["user_id",  _resolveUserId(b)],
      ["product",  (b["product"] ?? b["name"] ?? b["artistName"] ?? b["artist"] ?? "Artist") as string],
      ["platform", "meta"],
      ["goal",     "streams"],
    ],
  );
  await enrichWithAwareness(req, "advertising");
  await proxyRequest(req, res, "/platform/ads/audience");
});

router.get("/platform/ads/performance/:userId", async (req, res) => {
  const query = req.query.platform ? `?platform=${req.query.platform}` : "";
  await proxyRequest(
    req,
    res,
    `/platform/ads/performance/${req.params.userId}${query}`,
  );
});

router.post("/platform/ads/optimize", async (req, res) => {
  await enrichWithAwareness(req, "ad_copy");
  await proxyRequest(req, res, "/platform/ads/optimize");
});

// ─── Safety, Audio Analysis & Scoring ───────────────────────────────────────

router.post("/safety/screen", async (req, res) => {
  await proxyRequest(req, res, "/api/safety/screen");
});

router.post("/infer/viral-score", async (req, res) => {
  await proxyRequest(req, res, "/api/infer/viral-score");
});

// Beat/structure analysis for beat-synced video generation — distinct from
// the general "/analyze/audio" sentiment-style endpoint above.
router.post("/audio/analyze", async (req, res) => {
  await proxyRequest(req, res, "/api/audio/analyze");
});

// ─── RTA / Concurrency / Awareness stats ────────────────────────────────────

router.get("/rta/status", async (req, res) => {
  await proxyRequest(req, res, "/api/rta/status");
});

router.get("/concurrency/stats", async (req, res) => {
  await proxyRequest(req, res, "/api/concurrency/stats");
});

router.get("/awareness/quality/status", async (req, res) => {
  await proxyRequest(req, res, "/api/awareness/quality/status");
});

// ─── Watchdog ──────────────────────────────────────────────────────────────

router.get("/watchdog/status", async (req, res) => {
  await proxyRequest(req, res, "/watchdog/status");
});

router.get("/watchdog/log", async (req, res) => {
  const limit = req.query.limit ? `?limit=${req.query.limit}` : "";
  await proxyRequest(req, res, `/watchdog/log${limit}`);
});

router.post("/watchdog/reset", async (req, res) => {
  await proxyRequest(req, res, "/watchdog/reset");
});

// ─── Content Generation ────────────────────────────────────────────────────

router.post("/generate/content", async (req, res) => {
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/generate/content");
});

router.post("/generate/text", async (req, res) => {
  // Python: ApiGenerateTextRequest { mode, ... } — mode defaults to "content"
  // but FastAPI 422s if the field is explicitly absent from callers that omit it.
  normalizeBody(req,
    [],
    [["mode", "content"]],
  );
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/generate/text");
});

router.post("/content/score", async (req, res) => {
  await proxyRequest(req, res, "/api/content/score");
});

// ─── Release Campaigns ─────────────────────────────────────────────────────
// One song → a full multi-week rollout, then persisted per-artist as an
// editable, schedulable calendar (save / list / edit posts / hand off to the
// distribution layer to queue on target dates).

router.post("/generate/campaign", async (req, res) => {
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/generate/campaign");
});

router.post("/campaigns", async (req, res) => {
  await proxyRequest(req, res, "/api/campaigns");
});

router.get("/campaigns", async (req, res) => {
  const qs = new URLSearchParams(
    req.query as Record<string, string>,
  ).toString();
  await proxyRequest(req, res, `/api/campaigns${qs ? `?${qs}` : ""}`);
});

router.get("/campaigns/:id", async (req, res) => {
  const qs = new URLSearchParams(
    req.query as Record<string, string>,
  ).toString();
  await proxyRequest(
    req,
    res,
    `/api/campaigns/${encodeURIComponent(req.params.id)}${qs ? `?${qs}` : ""}`,
  );
});

router.delete("/campaigns/:id", async (req, res) => {
  const qs = new URLSearchParams(
    req.query as Record<string, string>,
  ).toString();
  await proxyRequest(
    req,
    res,
    `/api/campaigns/${encodeURIComponent(req.params.id)}${qs ? `?${qs}` : ""}`,
  );
});

router.patch("/campaigns/:id/posts/:postId", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/api/campaigns/${encodeURIComponent(req.params.id)}/posts/${encodeURIComponent(req.params.postId)}`,
  );
});

router.post("/campaigns/:id/schedule", async (req, res) => {
  await proxyRequest(
    req,
    res,
    `/api/campaigns/${encodeURIComponent(req.params.id)}/schedule`,
  );
});

// ─── Analysis ──────────────────────────────────────────────────────────────

router.post("/analyze", async (req, res) => {
  // Python: MaxcoreAnalyzeRequest { modality, payload, ... }
  // Both fields have defaults but FastAPI still 422s if they're explicitly null.
  const b = req.body as Record<string, unknown>;
  normalizeBody(req,
    [
      ["text",    "payload"],
      ["content", "payload"],
      ["input",   "payload"],
    ],
    [
      ["modality", "text"],
      ["payload",  (b["payload"] ?? b["text"] ?? b["content"] ?? b["input"] ?? "") as string],
    ],
  );
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/analyze");
});

router.post("/analyze/sentiment", async (req, res) => {
  await proxyRequest(req, res, "/api/analyze/sentiment");
});

router.post("/analyze/audio", async (req, res) => {
  // Python: ApiAnalyzeAudioRequest { audio_url }
  // Dashboard sends { url } instead.
  normalizeBody(req,
    [["url", "audio_url"]],
    [],
  );
  await proxyRequest(req, res, "/api/analyze/audio");
});

// ─── Advertising & Engagement ──────────────────────────────────────────────

router.post("/optimize/ad", async (req, res) => {
  await enrichWithAwareness(req, "ad_copy");
  await proxyRequest(req, res, "/api/optimize/ad");
});

router.post("/predict/engagement", async (req, res) => {
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/predict/engagement");
});

// ─── Media Generation ──────────────────────────────────────────────────────

router.post("/generate/image", async (req, res) => {
  await enrichWithAwareness(req, "content");
  await proxyRequest(req, res, "/api/generate/image");
});

router.post("/generate/audio", async (req, res) => {
  await enrichWithAwareness(req, "music");
  await proxyRequest(req, res, "/api/generate/audio");
});

router.post("/generate-video", async (req, res) => {
  await enrichWithAwareness(req, "video_script");
  await proxyRequest(req, res, "/api/generate-video");
});

// Canonical /generate/video alias — maps to the same AI video endpoint
router.post("/generate/video", async (req, res) => {
  await enrichWithAwareness(req, "video_script");
  await proxyRequest(req, res, "/api/video/generate-ai");
});

router.post("/video/generate-ai", async (req, res) => {
  await enrichWithAwareness(req, "video_script");
  await proxyRequest(req, res, "/api/video/generate-ai");
});

// Veo-parity scene extension: continue a previously generated video
router.post("/video/extend", async (req, res) => {
  await proxyRequest(req, res, "/api/video/extend");
});

// ─── Job Polling & Management ──────────────────────────────────────────────

router.get("/video-jobs", async (req, res) => {
  await proxyRequest(req, res, "/api/video-jobs");
});

router.get("/video-job/:jobId", async (req, res) => {
  await proxyRequest(req, res, `/api/video-job/${req.params.jobId}`);
});

router.delete("/video-job/:jobId", async (req, res) => {
  await proxyRequest(req, res, `/api/video-job/${req.params.jobId}`);
});

router.get("/video-job/:jobId/preview/:sceneIdx", async (req, res) => {
  await proxyBinary(
    req,
    res,
    `/api/video-job/${req.params.jobId}/preview/${req.params.sceneIdx}`,
  );
});

// Rendered MP4 download — the Python server registers this same handler under
// three aliased paths (download/file/video); we mirror all three so callers
// can use whichever they already reference. Uses the streamed binary proxy
// (not proxyBinary) since these files can be tens of MB — no full buffering,
// no short abort timer.
router.get("/video-job/:jobId/download", async (req, res) => {
  await proxyBinaryStream(
    req,
    res,
    `/api/video-job/${encodeURIComponent(req.params.jobId)}/download`,
  );
});

router.get("/video-job/:jobId/file", async (req, res) => {
  await proxyBinaryStream(
    req,
    res,
    `/api/video-job/${encodeURIComponent(req.params.jobId)}/file`,
  );
});

router.get("/video-job/:jobId/video", async (req, res) => {
  await proxyBinaryStream(
    req,
    res,
    `/api/video-job/${encodeURIComponent(req.params.jobId)}/video`,
  );
});

router.get("/audio-job/:jobId", async (req, res) => {
  await proxyRequest(req, res, `/api/audio-job/${req.params.jobId}`);
});

// ─── Model Weight Sync ─────────────────────────────────────────────────────

router.get("/models/social/state", async (req, res) => {
  await proxyRequest(req, res, "/api/models/social/state");
});

router.get("/models/advertising/state", async (req, res) => {
  await proxyRequest(req, res, "/api/models/advertising/state");
});

router.get("/models/content/state", async (req, res) => {
  await proxyRequest(req, res, "/api/models/content/state");
});

router.get("/models/engagement/state", async (req, res) => {
  await proxyRequest(req, res, "/api/models/engagement/state");
});

// ─── Training Feedback ─────────────────────────────────────────────────────

router.post("/train/feedback", async (req, res) => {
  await proxyRequest(req, res, "/api/train/feedback");
});

export default router;
