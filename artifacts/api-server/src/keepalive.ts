import { Agent, request as undiciRequest } from "undici";

// ─── Configuration ───────────────────────────────────────────────────────────

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

// How often the full ping cycle repeats (ms)
const PING_INTERVAL_MS = 20_000;

// Gap between individual pings inside one cycle (ms) — spreads load evenly
const PING_STAGGER_MS = 800;

// All GET endpoints that exist on the Python AI server.
// Hitting these keeps TCP connections warm, the Python process active,
// and the circuit breaker's failure counter at zero.
const PING_PATHS: string[] = [
  "/health",
  "/dashboard/stats",
  "/model/status",
  "/gpu/status",
  "/gpu/hyper/status",
  "/gpu/capabilities",
  "/training/status",
  "/training/continuous/status",
  "/training/puller/status",
  "/training/puller/sources",
  "/training/datasets",
  "/training/logs?limit=1",
  "/training/continuous/history",
  "/watchdog/status",
  "/storage/status",
  "/storage/datasets",
  "/storage/checkpoints",
  "/storage/session",
  "/storage/pipeline/status",
  "/boostsheets",
  "/platform/model/info",
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

let _cycleCount = 0;
let _consecutiveAllFailed = 0;
let _timer: ReturnType<typeof setTimeout> | null = null;

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
    const path = PING_PATHS[i];
    const success = await pingOne(path);
    success ? ok++ : fail++;

    // Stagger between pings — last one needs no delay
    if (i < PING_PATHS.length - 1) {
      await new Promise((r) => setTimeout(r, PING_STAGGER_MS));
    }
  }

  if (fail === PING_PATHS.length) {
    _consecutiveAllFailed++;
    console.warn(
      `[Keepalive] Cycle #${_cycleCount}: all ${fail} pings failed — ` +
        `AI server may be starting up (${_consecutiveAllFailed} consecutive all-fail cycles)`,
    );
  } else {
    _consecutiveAllFailed = 0;
    if (fail > 0) {
      console.warn(
        `[Keepalive] Cycle #${_cycleCount}: ${ok} ok, ${fail} failed`,
      );
    } else {
      console.log(
        `[Keepalive] Cycle #${_cycleCount}: all ${ok} endpoints alive`,
      );
    }
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function startKeepalive(): void {
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
  if (_timer !== null) {
    clearTimeout(_timer);
    _timer = null;
    console.log("[Keepalive] Stopped");
  }
}
