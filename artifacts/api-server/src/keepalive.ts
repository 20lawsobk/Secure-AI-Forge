import fs from "fs";
import { Agent, request as undiciRequest } from "undici";

// ─── Configuration ───────────────────────────────────────────────────────────

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

// How often the full GET ping cycle repeats (ms)
const PING_INTERVAL_MS = 20_000;

// Gap between individual pings inside one cycle (ms) — spreads load evenly
const PING_STAGGER_MS = 800;

// How often a deep-warm inference pass is fired (ms).
// Every DEEP_WARM_INTERVAL_MS the keepalive POSTs /api/warm to exercise the
// Digital GPU inference chains (transformer → flash-attn → pocket GEMM) so
// KV-cache and GEMM dedup entries stay resident between real user requests.
const DEEP_WARM_INTERVAL_MS = 5 * 60_000; // 5 minutes

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

// ─── Dedicated keep-alive pool ───────────────────────────────────────────────
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
let _lastDeepWarmAt: string | null = null;
let _lastDeepWarmOk: boolean | null = null;
let _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;

// Per-endpoint health: true = last ping succeeded, false = last ping failed
const _endpointHealth = new Map<string, boolean>(
  PING_PATHS.map((p) => [p, true]),
);

// ─── Single ping ─────────────────────────────────────────────────────────────

const ADMIN_KEY = process.env.ADMIN_KEY ?? "";

async function pingOne(path: string): Promise<boolean> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (ADMIN_KEY) headers["X-Admin-Key"] = ADMIN_KEY;

    const { statusCode, body } = await undiciRequest(`${MODEL_API_BASE}${path}`, {
      method: "GET",
      dispatcher: _pingPool,
      headers,
      headersTimeout: 0,
      bodyTimeout: 0,
    });
    await body.dump();
    return statusCode < 500;
  } catch {
    return false;
  }
}

// How quickly to retry a deep-warm that failed (ms).
// Used when the first pass fires before Python is up.  Normal steady-state
// interval is DEEP_WARM_INTERVAL_MS (5 min).
const DEEP_WARM_RETRY_MS = 2 * 60_000; // 2 minutes

// ─── Deep-warm pass ──────────────────────────────────────────────────────────
// POST /api/warm exercises the Digital GPU inference chains so KV-cache,
// pocket GEMM dedup entries, and flash-attn kernel paths stay hot between
// real user requests.  Safe to call repeatedly — Python side is idempotent
// and never-raise.

async function runDeepWarm(): Promise<void> {
  const ADMIN_KEY_HDR = process.env.ADMIN_KEY ?? "";
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (ADMIN_KEY_HDR) headers["X-Admin-Key"] = ADMIN_KEY_HDR;

    const { statusCode, body } = await undiciRequest(
      `${MODEL_API_BASE}/api/warm`,
      {
        method: "POST",
        dispatcher: _pingPool,
        headers,
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
      // Success — next deep-warm at the normal steady-state interval
      _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    } else {
      console.warn(
        `[Keepalive] Deep-warm POST /api/warm → ${statusCode} (body: ${raw.slice(0, 200)})`,
      );
      // Python responded but returned an error — retry at the short interval
      _nextDeepWarmAt = Date.now() + DEEP_WARM_RETRY_MS;
    }
  } catch (err) {
    // Connection error (Python still starting) — schedule a fast retry
    _lastDeepWarmOk = false;
    _lastDeepWarmAt = new Date().toISOString();
    _nextDeepWarmAt = Date.now() + DEEP_WARM_RETRY_MS;
    console.warn(`[Keepalive] Deep-warm POST /api/warm failed — retrying in ${DEEP_WARM_RETRY_MS / 60000}min: ${err}`);
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
        `[Keepalive] Cycle #${_cycleCount}: ${ok} ok, ${fail} failed — ${failed.join(", ")}`,
      );
    } else {
      console.log(`[Keepalive] Cycle #${_cycleCount}: all ${ok} endpoints alive`);
    }
  }

  // ── Deep-warm check ────────────────────────────────────────────────────────
  // Only fire when Python is actually up (at least one endpoint responded) and
  // the interval has elapsed.  Skip during all-fail windows to avoid piling up
  // requests against a restarting server.
  if (fail < PING_PATHS.length && Date.now() >= _nextDeepWarmAt) {
    _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    // Run in background — don't block the ping cycle scheduling
    runDeepWarm().catch(() => {});
  }

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
      deepWarm: {
        intervalMs: DEEP_WARM_INTERVAL_MS,
        lastDeepWarmAt: null,
        lastDeepWarmOk: null,
        nextDeepWarmAt: new Date(_nextDeepWarmAt).toISOString(),
      },
      message: "warming up — first cycle not yet complete",
    };
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function startKeepalive(): void {
  _running = true;
  console.log(
    `[Keepalive] Starting — pinging ${PING_PATHS.length} endpoints every ${PING_INTERVAL_MS / 1000}s, ` +
      `deep-warm every ${DEEP_WARM_INTERVAL_MS / 60000}min`,
  );

  const schedule = () => {
    _timer = setTimeout(async () => {
      await runCycle();
      schedule();
    }, PING_INTERVAL_MS);
  };

  // Run the first cycle immediately so the server is warm on startup;
  // also fire the first deep-warm right away (don't wait 5 min on boot)
  runCycle().then(() => {
    _nextDeepWarmAt = Date.now() + DEEP_WARM_INTERVAL_MS;
    schedule();
  });
  // First deep-warm runs in parallel with the first ping cycle
  runDeepWarm().catch(() => {});
}

export function stopKeepalive(): void {
  _running = false;
  if (_timer !== null) {
    clearTimeout(_timer);
    _timer = null;
    console.log("[Keepalive] Stopped");
  }
}
