import fs from "fs";
import { Agent, request as undiciRequest } from "undici";

// ─── Configuration ───────────────────────────────────────────────────────────

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

// Fast heartbeat: just /health, fires every 3 s.
// Detects failures in under one poll cycle and keeps one connection warm
// without the 26-second sequential crawl.
const HEARTBEAT_INTERVAL_MS = 3_000;

// Residency sweep: ALL endpoints fired concurrently, every 60 s.
// Keeps HyperGPU kernels, RTA fabric, pocket GEMM dedup, and KV-cache
// resident between real user requests without the noise of sequential
// staggered pings.
const RESIDENCY_INTERVAL_MS = 60_000;

// Deep-warm inference pass: POST /api/warm, every 5 minutes.
// Exercises the full Digital GPU chain (transformer → flash-attn → GEMM)
// so the first real generation request hits a hot path.
const DEEP_WARM_INTERVAL_MS = 5 * 60_000;

// Status snapshot path — written by the primary after each sweep so worker
// processes can serve it via GET /api/keepalive/status without IPC.
const STATUS_FILE = "/tmp/maxcore-keepalive.json";

// ─── Endpoints ───────────────────────────────────────────────────────────────
// All GET endpoints on the Python AI server, grouped by subsystem.
// Hitting these in the residency sweep keeps TCP connections warm,
// the Python process active, Digital GPU subsystems resident,
// and the circuit breaker counter at zero.

export const PING_PATHS: readonly string[] = [
  // ── Core health ────────────────────────────────────────────────────────────
  "/health",
  "/api/health",

  // ── Dashboard ──────────────────────────────────────────────────────────────
  "/dashboard/stats",

  // ── Model ──────────────────────────────────────────────────────────────────
  "/model/status",

  // ── Digital GPU subsystems ─────────────────────────────────────────────────
  "/gpu/status",
  "/gpu/hyper/status",
  "/gpu/capabilities",
  "/api/rta/status",
  "/api/maxcore/pocket-accelerator/stats",

  // ── Concurrency & job queues ────────────────────────────────────────────────
  "/api/concurrency/stats",
  "/api/video-jobs",

  // ── Training ───────────────────────────────────────────────────────────────
  "/training/status",
  "/training/continuous/status",
  "/training/puller/status",
  "/training/puller/sources",
  "/training/datasets",
  "/training/logs?limit=1",
  "/training/continuous/history",

  // ── Watchdog ───────────────────────────────────────────────────────────────
  "/watchdog/status",

  // ── Storage / pdim ─────────────────────────────────────────────────────────
  "/storage/status",
  "/storage/datasets",
  "/storage/datasets/audio/status",
  "/storage/checkpoints",
  "/storage/session",
  "/storage/pipeline/status",

  // ── BoostSheets ─────────────────────────────────────────────────────────────
  "/boostsheets",

  // ── Platform model info ────────────────────────────────────────────────────
  "/platform/model/info",

  // ── Awareness quality buffer ───────────────────────────────────────────────
  "/api/awareness/quality/status",

  // ── Per-domain model state ─────────────────────────────────────────────────
  "/api/models/social/state",
  "/api/models/advertising/state",
  "/api/models/content/state",
  "/api/models/engagement/state",
];

// ─── Connection pools ────────────────────────────────────────────────────────
// Separate from the proxy pool so keepalive traffic never starves real requests.

// Heartbeat pool: single persistent connection, minimal overhead.
const _heartbeatPool = new Agent({
  keepAliveTimeout: 60_000,
  keepAliveMaxTimeout: 120_000,
  connections: 1,
  pipelining: 1,
});

// Residency pool: enough connections to fire all endpoints concurrently.
const _residencyPool = new Agent({
  keepAliveTimeout: 60_000,
  keepAliveMaxTimeout: 120_000,
  connections: Math.min(PING_PATHS.length, 32),
  pipelining: 1,
});

// ─── State ───────────────────────────────────────────────────────────────────

let _running = false;
let _cycleCount = 0;          // residency sweep count
let _heartbeatCount = 0;
let _heartbeatOk = true;
let _consecutiveHbFail = 0;
let _heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
let _residencyTimer: ReturnType<typeof setTimeout> | null = null;
let _lastResidencyAt: string | null = null;
let _lastDeepWarmAt: string | null = null;
let _lastDeepWarmOk: boolean | null = null;
let _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;

// Per-endpoint health from the last residency sweep
const _endpointHealth = new Map<string, boolean>(
  PING_PATHS.map((p) => [p, true]),
);

const ADMIN_KEY = process.env.ADMIN_KEY ?? "";

function _headers(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (ADMIN_KEY) h["X-Admin-Key"] = ADMIN_KEY;
  return h;
}

// ─── Fast heartbeat ──────────────────────────────────────────────────────────
// Fires every HEARTBEAT_INTERVAL_MS (3 s).  Only hits /health — cheap and
// sufficient to keep the primary TCP connection warm and detect a crash within
// one interval rather than waiting up to 20 s.

async function runHeartbeat(): Promise<void> {
  _heartbeatCount++;
  try {
    const { statusCode, body } = await undiciRequest(
      `${MODEL_API_BASE}/health`,
      {
        method: "GET",
        dispatcher: _heartbeatPool,
        headers: _headers(),
        headersTimeout: 5_000,
        bodyTimeout: 5_000,
      },
    );
    await body.dump();
    const ok = statusCode < 500;
    if (!ok && _heartbeatOk) {
      console.warn(`[Keepalive] Heartbeat #${_heartbeatCount}: /health → ${statusCode} — Python may be struggling`);
    } else if (ok && !_heartbeatOk) {
      console.log(`[Keepalive] Heartbeat #${_heartbeatCount}: /health recovered ✓`);
    }
    _heartbeatOk = ok;
    if (ok) _consecutiveHbFail = 0;
    else _consecutiveHbFail++;
  } catch {
    _heartbeatOk = false;
    _consecutiveHbFail++;
    if (_consecutiveHbFail === 1 || _consecutiveHbFail % 10 === 0) {
      console.warn(`[Keepalive] Heartbeat #${_heartbeatCount}: /health unreachable (${_consecutiveHbFail} consecutive)`);
    }
  }
}

// ─── Residency sweep ─────────────────────────────────────────────────────────
// Fires all PING_PATHS concurrently every RESIDENCY_INTERVAL_MS (60 s).
// Concurrent (not staggered) so the sweep completes in ~2 s instead of 26 s,
// then stays silent until the next interval.

async function pingOne(
  path: string,
  pool: Agent,
): Promise<boolean> {
  try {
    const { statusCode, body } = await undiciRequest(
      `${MODEL_API_BASE}${path}`,
      {
        method: "GET",
        dispatcher: pool,
        headers: _headers(),
        headersTimeout: 10_000,
        bodyTimeout: 10_000,
      },
    );
    await body.dump();
    return statusCode < 500;
  } catch {
    return false;
  }
}

async function runResidencySweep(): Promise<void> {
  _cycleCount++;

  // Fire all endpoints concurrently — results arrive together in ~RTT time
  const results = await Promise.all(
    PING_PATHS.map(async (path) => {
      const ok = await pingOne(path, _residencyPool);
      _endpointHealth.set(path, ok);
      return ok;
    }),
  );

  const ok = results.filter(Boolean).length;
  const fail = results.length - ok;

  _lastResidencyAt = new Date().toISOString();

  if (fail === PING_PATHS.length) {
    console.warn(
      `[Keepalive] Sweep #${_cycleCount}: all ${fail} endpoints unreachable — Python may be starting`,
    );
  } else if (fail > 0) {
    const failed = PING_PATHS.filter((p) => !_endpointHealth.get(p));
    console.warn(
      `[Keepalive] Sweep #${_cycleCount}: ${ok} ok, ${fail} unreachable — ${failed.join(", ")}`,
    );
  } else {
    console.log(`[Keepalive] Sweep #${_cycleCount}: all ${ok} endpoints alive`);
  }

  // Deep-warm: only when Python is actually up and the interval has elapsed
  if (fail < PING_PATHS.length && Date.now() >= _nextDeepWarmAt) {
    _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    runDeepWarm().catch(() => {});
  }

  _flushStatus(ok, fail);
}

// ─── Deep-warm pass ──────────────────────────────────────────────────────────

const DEEP_WARM_RETRY_MS = 2 * 60_000;

async function runDeepWarm(): Promise<void> {
  try {
    const { statusCode, body } = await undiciRequest(
      `${MODEL_API_BASE}/api/warm`,
      {
        method: "POST",
        dispatcher: _residencyPool,
        headers: _headers(),
        body: "{}",
        headersTimeout: 0,
        bodyTimeout: 0,
      },
    );
    const raw = await body.text();
    _lastDeepWarmOk = statusCode < 500;
    _lastDeepWarmAt = new Date().toISOString();

    if (_lastDeepWarmOk) {
      console.log(`[Keepalive] Deep-warm POST /api/warm → ${statusCode} ✓`);
      _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    } else {
      console.warn(`[Keepalive] Deep-warm POST /api/warm → ${statusCode} (body: ${raw.slice(0, 200)})`);
      _nextDeepWarmAt = Date.now() + DEEP_WARM_RETRY_MS;
    }
  } catch (err) {
    _lastDeepWarmOk = false;
    _lastDeepWarmAt = new Date().toISOString();
    _nextDeepWarmAt = Date.now() + DEEP_WARM_RETRY_MS;
    console.warn(`[Keepalive] Deep-warm POST /api/warm failed — retrying in ${DEEP_WARM_RETRY_MS / 60000}min: ${err}`);
  }
}

// ─── Status snapshot ─────────────────────────────────────────────────────────

function _flushStatus(ok: number, fail: number): void {
  try {
    const endpoints: Record<string, boolean> = {};
    for (const [path, alive] of _endpointHealth.entries()) {
      endpoints[path] = alive;
    }
    const snapshot = {
      running: _running,
      cycleCount: _cycleCount,
      heartbeatCount: _heartbeatCount,
      heartbeatOk: _heartbeatOk,
      consecutiveHeartbeatFail: _consecutiveHbFail,
      lastResidencyAt: _lastResidencyAt,
      heartbeatIntervalMs: HEARTBEAT_INTERVAL_MS,
      residencyIntervalMs: RESIDENCY_INTERVAL_MS,
      totalEndpoints: PING_PATHS.length,
      summary: { ok, fail },
      endpoints,
      deepWarm: {
        intervalMs: DEEP_WARM_INTERVAL_MS,
        lastDeepWarmAt: _lastDeepWarmAt,
        lastDeepWarmOk: _lastDeepWarmOk,
        nextDeepWarmAt: new Date(_nextDeepWarmAt).toISOString(),
      },
    };
    fs.writeFileSync(STATUS_FILE, JSON.stringify(snapshot), "utf8");
  } catch {
    // Non-fatal — status file is best-effort
  }
}

/** Return the keepalive status snapshot. */
export function getKeepaliveStatus(): Record<string, unknown> {
  try {
    const raw = fs.readFileSync(STATUS_FILE, "utf8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {
      running: _running,
      cycleCount: _cycleCount,
      heartbeatCount: _heartbeatCount,
      heartbeatOk: _heartbeatOk,
      lastResidencyAt: _lastResidencyAt,
      heartbeatIntervalMs: HEARTBEAT_INTERVAL_MS,
      residencyIntervalMs: RESIDENCY_INTERVAL_MS,
      totalEndpoints: PING_PATHS.length,
      summary: { ok: 0, fail: 0 },
      endpoints: {},
      deepWarm: {
        intervalMs: DEEP_WARM_INTERVAL_MS,
        lastDeepWarmAt: null,
        lastDeepWarmOk: null,
        nextDeepWarmAt: new Date(_nextDeepWarmAt).toISOString(),
      },
      message: "warming up — first sweep not yet complete",
    };
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function startKeepalive(): void {
  if (_running) return;
  _running = true;

  console.log(
    `[Keepalive] Starting — heartbeat every ${HEARTBEAT_INTERVAL_MS / 1000}s, ` +
    `residency sweep every ${RESIDENCY_INTERVAL_MS / 1000}s, ` +
    `deep-warm every ${DEEP_WARM_INTERVAL_MS / 60000}min`,
  );

  // ── Heartbeat loop ───────────────────────────────────────────────────────
  const scheduleHeartbeat = () => {
    _heartbeatTimer = setTimeout(async () => {
      await runHeartbeat();
      if (_running) scheduleHeartbeat();
    }, HEARTBEAT_INTERVAL_MS);
  };

  // ── Residency sweep loop ─────────────────────────────────────────────────
  const scheduleResidency = () => {
    _residencyTimer = setTimeout(async () => {
      await runResidencySweep();
      if (_running) scheduleResidency();
    }, RESIDENCY_INTERVAL_MS);
  };

  // First heartbeat: tiny delay so the server is bound before we hit it
  setTimeout(async () => {
    await runHeartbeat();
    if (_running) scheduleHeartbeat();
  }, 500);

  // First residency sweep: run immediately so GPU subsystems warm on boot,
  // then schedule the recurring interval
  runResidencySweep().then(() => {
    _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    if (_running) scheduleResidency();
  });

  // First deep-warm: run in parallel with the first sweep (don't wait 5 min)
  runDeepWarm().catch(() => {});
}

export function stopKeepalive(): void {
  _running = false;
  if (_heartbeatTimer !== null) {
    clearTimeout(_heartbeatTimer);
    _heartbeatTimer = null;
  }
  if (_residencyTimer !== null) {
    clearTimeout(_residencyTimer);
    _residencyTimer = null;
  }
  console.log("[Keepalive] Stopped");
}
