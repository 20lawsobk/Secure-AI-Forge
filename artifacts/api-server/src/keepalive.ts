import fs from "fs";
import { Agent, request as undiciRequest } from "undici";

// ─── Configuration ───────────────────────────────────────────────────────────

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

// How often the full ping cycle repeats (ms)
const PING_INTERVAL_MS = 20_000;

// Gap between individual pings inside one cycle (ms) — spreads load evenly
const PING_STAGGER_MS = 800;

// Status snapshot path — written by the primary after each cycle so worker
// processes can serve it via GET /api/keepalive/status without IPC.
const STATUS_FILE = "/tmp/maxcore-keepalive.json";

// ─── Ping paths ──────────────────────────────────────────────────────────────
// All GET endpoints on the Python AI server, grouped by subsystem.
// Hitting these keeps TCP connections warm, the Python process active,
// Digital GPU subsystems resident, and the circuit breaker counter at zero.

export const PING_PATHS: readonly string[] = [
  // ── Core health ────────────────────────────────────────────────────────────
  "/health",
  "/api/health",           // model + Digital GPU detailed health

  // ── Dashboard ──────────────────────────────────────────────────────────────
  "/dashboard/stats",

  // ── Model ──────────────────────────────────────────────────────────────────
  "/model/status",

  // ── Digital GPU subsystems ─────────────────────────────────────────────────
  // These keep HyperGPU (HyperSIMDCore, flash-attn, conv kernels), the RTA
  // rendering fabric (path tracer / video grader / spectral), and the pocket
  // accelerator (GEMM dedup cache) resident and responsive.
  "/gpu/status",
  "/gpu/hyper/status",
  "/gpu/capabilities",
  "/api/rta/status",                          // RTA rendering fabric
  "/api/maxcore/pocket-accelerator/stats",    // pocket/GEMM dedup

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

// ─── Dedicated keep-alive pool for the pinger ────────────────────────────────
// Separate from the proxy pool so pings never starve real traffic.

const _pingPool = new Agent({
  keepAliveTimeout: 60_000,
  keepAliveMaxTimeout: 120_000,
  connections: 4,
  pipelining: 1,
});

// ─── State ───────────────────────────────────────────────────────────────────

let _running = false;
let _cycleCount = 0;
let _consecutiveAllFailed = 0;
let _timer: ReturnType<typeof setTimeout> | null = null;
let _lastCycleAt: string | null = null;

// Per-endpoint health: true = last ping succeeded, false = last ping failed
const _endpointHealth = new Map<string, boolean>(
  PING_PATHS.map((p) => [p, true]),
);

// ─── Single ping ─────────────────────────────────────────────────────────────

const ADMIN_KEY = process.env.ADMIN_KEY ?? "";

async function pingOne(path: string): Promise<boolean> {
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (ADMIN_KEY) headers["X-Admin-Key"] = ADMIN_KEY;

    const { statusCode, body } = await undiciRequest(
      `${MODEL_API_BASE}${path}`,
      {
        method: "GET",
        dispatcher: _pingPool,
        headers,
        headersTimeout: 8_000,
        bodyTimeout: 8_000,
      },
    );
    // Drain the body so the connection is returned to the pool
    await body.dump();
    return statusCode < 500;
  } catch {
    return false;
  }
}

// ─── Full ping cycle ─────────────────────────────────────────────────────────

async function runCycle(): Promise<void> {
  _cycleCount++;
  let ok = 0;
  let fail = 0;

  for (let i = 0; i < PING_PATHS.length; i++) {
    const path = PING_PATHS[i]!;
    const success = await pingOne(path);
    _endpointHealth.set(path, success);
    success ? ok++ : fail++;

    // Stagger between pings — last one needs no delay
    if (i < PING_PATHS.length - 1) {
      await new Promise((r) => setTimeout(r, PING_STAGGER_MS));
    }
  }

  _lastCycleAt = new Date().toISOString();

  if (fail === PING_PATHS.length) {
    _consecutiveAllFailed++;
    console.warn(
      `[Keepalive] Cycle #${_cycleCount}: all ${fail} pings failed — ` +
        `AI server may be starting up (${_consecutiveAllFailed} consecutive all-fail cycles)`,
    );
  } else {
    _consecutiveAllFailed = 0;
    if (fail > 0) {
      const failed = PING_PATHS.filter((p) => !_endpointHealth.get(p));
      console.warn(
        `[Keepalive] Cycle #${_cycleCount}: ${ok} ok, ${fail} failed — ` +
          failed.join(", "),
      );
    } else {
      console.log(
        `[Keepalive] Cycle #${_cycleCount}: all ${ok} endpoints alive`,
      );
    }
  }

  // Write status snapshot so worker processes can serve it without IPC
  _flushStatus(ok, fail);
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
      lastCycleAt: _lastCycleAt,
      intervalMs: PING_INTERVAL_MS,
      totalEndpoints: PING_PATHS.length,
      summary: { ok, fail },
      endpoints,
    };
    fs.writeFileSync(STATUS_FILE, JSON.stringify(snapshot), "utf8");
  } catch {
    // Non-fatal — status file is best-effort
  }
}

/** Return the keepalive status snapshot.  Workers read from the file written
 *  by the primary; the primary can use this directly. */
export function getKeepaliveStatus(): Record<string, unknown> {
  try {
    const raw = fs.readFileSync(STATUS_FILE, "utf8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {
      running: _running,
      cycleCount: _cycleCount,
      lastCycleAt: _lastCycleAt,
      intervalMs: PING_INTERVAL_MS,
      totalEndpoints: PING_PATHS.length,
      summary: { ok: 0, fail: 0 },
      endpoints: {},
      message: "warming up — first cycle not yet complete",
    };
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function startKeepalive(): void {
  _running = true;
  console.log(
    `[Keepalive] Starting — pinging ${PING_PATHS.length} endpoints every ${PING_INTERVAL_MS / 1000}s`,
  );

  const schedule = () => {
    _timer = setTimeout(async () => {
      await runCycle();
      schedule(); // re-schedule after cycle completes (not setInterval) so pings never overlap
    }, PING_INTERVAL_MS);
  };

  // Run the first cycle immediately so the server is warm on startup
  runCycle().then(schedule);
}

export function stopKeepalive(): void {
  _running = false;
  if (_timer !== null) {
    clearTimeout(_timer);
    _timer = null;
    console.log("[Keepalive] Stopped");
  }
}
