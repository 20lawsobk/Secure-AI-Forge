import { spawn, ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import net from "net";
import { Agent, request as undiciRequest } from "undici";

const PYTHON_PORT = parseInt(process.env.MODEL_API_PORT || "9878", 10);

// Proxy-only mode: this instance must NEVER try to own (spawn or restart) the
// Python server.  Two conditions each independently set this flag:
//
//  1. DISABLE_PYTHON_SPAWN=1   — explicit override; can be set on any workflow.
//  2. MODEL_API_PORT not set   — only the intended owners set MODEL_API_PORT:
//     the "Start application" workflow (dev) and the production "serve"
//     script (deployment).  The artifact-managed dev workflow
//     (artifacts/api-server: API Server) runs `dev` without it, so it falls
//     into proxy-only mode automatically.  This prevents the startup race
//     where two api-server instances compete for the Python lock.  Without
//     MODEL_API_PORT in the serve script, PRODUCTION has no Python owner at
//     all and the whole API 503s — do not remove it from `serve`.
const PYTHON_SPAWN_DISABLED =
  process.env.DISABLE_PYTHON_SPAWN === "1" ||
  process.env.MODEL_API_PORT === undefined;

const PYTHON_SCRIPT = (() => {
  const metaUrl = import.meta?.url;
  if (metaUrl) {
    return path.resolve(
      path.dirname(fileURLToPath(metaUrl)),
      "../../../artifacts/ai-training-server/server.py",
    );
  }
  return path.resolve(process.cwd(), "artifacts/ai-training-server/server.py");
})();

// Backoff: starts at 2 s, doubles each crash, caps at 30 s
const INITIAL_RETRY_MS = 2_000;
const MAX_RETRY_MS = 30_000;

// If the server ran at least this long without crashing, treat the next
// crash as a fresh start (reset backoff).
const MIN_HEALTHY_RUN_MS = 60_000;

// Poll interval for the health monitor (detects silent crashes)
const HEALTH_POLL_INTERVAL_MS = 15_000;

// ─── Internal HTTP pool (warm-pass calls only) ────────────────────────────────
// Dedicated undici agent so post-startup warm-up requests never share
// the keepalive pool and can't stall real proxy traffic.
const _warmPool = new Agent({
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 60_000,
  connections: 2,
  pipelining: 1,
});

const ADMIN_KEY = process.env.ADMIN_KEY ?? "";

let pythonProcess: ChildProcess | null = null;
let consecutiveCrashes = 0;
let lastStartTime = 0;
let restartScheduled = false;
let healthTimer: ReturnType<typeof setInterval> | null = null;
let shuttingDown = false;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function backoffMs(): number {
  const delay = INITIAL_RETRY_MS * Math.pow(2, consecutiveCrashes - 1);
  return Math.min(Math.round(delay), MAX_RETRY_MS);
}

function isPortOpen(port: number, timeoutMs = 1000): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    sock.setTimeout(timeoutMs);
    sock.on("connect", () => { sock.destroy(); resolve(true); });
    sock.on("error",   () => { sock.destroy(); resolve(false); });
    sock.on("timeout", () => { sock.destroy(); resolve(false); });
    sock.connect(port, "127.0.0.1");
  });
}

function waitForPort(port: number, timeoutMs = 45_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const poll = () => {
      const sock = new net.Socket();
      sock.setTimeout(1000);
      sock.on("connect", () => { sock.destroy(); resolve(); });
      sock.on("error",   () => { sock.destroy(); Date.now() < deadline ? setTimeout(poll, 500) : reject(new Error(`Timeout waiting for port ${port}`)); });
      sock.on("timeout", () => { sock.destroy(); Date.now() < deadline ? setTimeout(poll, 500) : reject(new Error(`Timeout waiting for port ${port}`)); });
      sock.connect(port, "127.0.0.1");
    };
    poll();
  });
}

async function pollUntilOpen(port: number, maxMs = 6_000): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    if (await isPortOpen(port)) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

// ─── Post-startup warm-up pass ────────────────────────────────────────────────
// After Python confirms model_loaded=true, POST /api/warm to exercise the
// Digital GPU inference chains (transformer → flash-attn → pocket GEMM) before
// any real user traffic arrives.  On a reserved VM this runs once per deploy so
// the very first request hits a hot path, not a cold one.

async function waitForModelReady(maxMs = 180_000): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (ADMIN_KEY) headers["X-Admin-Key"] = ADMIN_KEY;

  while (Date.now() < deadline) {
    try {
      const { statusCode, body } = await undiciRequest(
        `http://localhost:${PYTHON_PORT}/health`,
        { method: "GET", dispatcher: _warmPool, headers,
          headersTimeout: 0, bodyTimeout: 0 },
      );
      const raw = await body.text();
      if (statusCode === 200) {
        const parsed = JSON.parse(raw) as { model_loaded?: boolean };
        if (parsed.model_loaded === true) return true;
      }
    } catch {
      // server still starting — keep polling
    }
    await new Promise((r) => setTimeout(r, 3_000));
  }
  return false;
}

async function fireWarmPass(): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (ADMIN_KEY) headers["X-Admin-Key"] = ADMIN_KEY;

  try {
    console.log("[Python] Firing production warm-up pass (POST /api/warm)…");
    const { statusCode, body } = await undiciRequest(
      `http://localhost:${PYTHON_PORT}/api/warm`,
      {
        method: "POST",
        dispatcher: _warmPool,
        headers,
        body: "{}",
        headersTimeout: 0,
        bodyTimeout: 0,
      },
    );
    const raw = await body.text();
    if (statusCode === 200) {
      let summary = "";
      try {
        const parsed = JSON.parse(raw) as Record<string, unknown>;
        const subs = parsed["subsystems"] as Record<string, { ok: boolean; ms: number }> | undefined;
        if (subs) {
          summary = Object.entries(subs)
            .map(([k, v]) => `${k}:${v.ok ? "✓" : "✗"}(${v.ms}ms)`)
            .join(" ");
        }
      } catch { /* ignore parse errors */ }
      console.log(`[Python] Warm-up pass complete — ${summary || "ok"}`);
    } else {
      console.warn(`[Python] Warm-up pass returned ${statusCode}: ${raw.slice(0, 300)}`);
    }
  } catch (err) {
    // Never fatal — keepalive will re-warm on next deep-warm cycle
    console.warn(`[Python] Warm-up pass failed (non-fatal): ${err}`);
  }
}

// ─── Core spawn logic ─────────────────────────────────────────────────────────

function spawnPython() {
  if (shuttingDown || pythonProcess) return;

  console.log(`[Python] Starting AI training server (port ${PYTHON_PORT})...`);
  lastStartTime = Date.now();

  pythonProcess = spawn("uv", ["run", "python3", PYTHON_SCRIPT], {
    env: {
      ...process.env,
      MODEL_API_PORT: String(PYTHON_PORT),
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout?.on("data", (d: Buffer) => process.stdout.write(`[Python] ${d}`));
  pythonProcess.stderr?.on("data", (d: Buffer) => process.stderr.write(`[Python] ${d}`));

  pythonProcess.on("exit", async (code, signal) => {
    pythonProcess = null;
    restartScheduled = false;

    if (shuttingDown || signal === "SIGTERM" || signal === "SIGINT") {
      console.log("[Python] Server shut down gracefully.");
      return;
    }

    // If another process grabbed the port (e.g., standalone workflow restarted), don't fight it
    if (await isPortOpen(PYTHON_PORT)) {
      console.log(`[Python] Port ${PYTHON_PORT} is held by another process — standing by.`);
      return;
    }

    const uptime = Date.now() - lastStartTime;
    if (uptime >= MIN_HEALTHY_RUN_MS) {
      console.log(`[Python] Server was healthy for ${Math.round(uptime / 1000)}s — resetting backoff.`);
      consecutiveCrashes = 0;
    }

    consecutiveCrashes++;
    const delay = backoffMs();
    console.log(`[Python] Server exited (code ${code ?? "?"}, signal ${signal ?? "none"}). Restart #${consecutiveCrashes} in ${delay}ms…`);

    restartScheduled = true;
    setTimeout(() => {
      spawnPython();
      // After each crash-restart, re-run the warm-up pass once Python is back
      waitForModelReady(180_000).then((ready) => {
        if (ready) {
          fireWarmPass().catch(() => {});
        } else {
          console.warn("[Python] Model did not report ready within 3 min after restart — skipping warm pass");
        }
      });
    }, delay);
  });
}

// ─── Health monitor ────────────────────────────────────────────────────────────

function startHealthMonitor() {
  if (healthTimer) return;
  healthTimer = setInterval(async () => {
    if (shuttingDown || pythonProcess || restartScheduled) return;
    const up = await isPortOpen(PYTHON_PORT);
    if (!up) {
      console.log("[Python] Health monitor: server is down — restarting...");
      spawnPython();
    }
  }, HEALTH_POLL_INTERVAL_MS);
  healthTimer.unref();
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function ensurePythonServer(): Promise<void> {
  if (PYTHON_SPAWN_DISABLED) {
    console.log(
      `[Python] DISABLE_PYTHON_SPAWN=1 — proxy-only mode, waiting for Python on port ${PYTHON_PORT}…`,
    );
    const alreadyUp = await pollUntilOpen(PYTHON_PORT, 60_000);
    if (!alreadyUp) {
      console.warn(`[Python] Proxy-only: port ${PYTHON_PORT} not yet open — requests will be retried by circuit breaker.`);
    } else {
      console.log(`[Python] AI training server ready on port ${PYTHON_PORT}`);
    }
    return;
  }

  const alreadyUp = await pollUntilOpen(PYTHON_PORT, 6_000);

  if (alreadyUp) {
    console.log(`[Python] AI training server already running on port ${PYTHON_PORT} — monitoring only.`);
  } else {
    spawnPython();
    try {
      await waitForPort(PYTHON_PORT, 45_000);
      consecutiveCrashes = 0;
      console.log(`[Python] AI training server ready on port ${PYTHON_PORT}`);
    } catch {
      console.error("[Python] Warning: server did not respond in time — requests will be retried by proxy.");
    }
  }

  startHealthMonitor();

  // ── Production warm-up pass ───────────────────────────────────────────────
  // Wait for model_loaded=true, then exercise the Digital GPU inference chains
  // so the reserved VM is fully hot before the first real user request.
  // Runs in background — never blocks startKeepalive() or worker fork.
  waitForModelReady(180_000).then((ready) => {
    if (ready) {
      return fireWarmPass();
    }
    console.warn("[Python] Model did not become ready within 3 min — warm pass skipped; keepalive will retry on next deep-warm cycle");
  }).catch(() => {});
}

export function stopPythonServer() {
  shuttingDown = true;
  if (healthTimer) {
    clearInterval(healthTimer);
    healthTimer = null;
  }
  if (pythonProcess) {
    console.log("[Python] Stopping AI training server...");
    pythonProcess.kill("SIGTERM");
    pythonProcess = null;
  }
}
