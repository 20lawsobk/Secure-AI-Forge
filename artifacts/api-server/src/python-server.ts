import { spawn, ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import net from "net";

const PYTHON_PORT = parseInt(process.env.MODEL_API_PORT || "9878", 10);

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
    sock.on("connect", () => {
      sock.destroy();
      resolve(true);
    });
    sock.on("error", () => {
      sock.destroy();
      resolve(false);
    });
    sock.on("timeout", () => {
      sock.destroy();
      resolve(false);
    });
    sock.connect(port, "127.0.0.1");
  });
}

function waitForPort(port: number, timeoutMs = 45_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const poll = () => {
      const sock = new net.Socket();
      sock.setTimeout(1000);
      sock.on("connect", () => {
        sock.destroy();
        resolve();
      });
      sock.on("error", () => {
        sock.destroy();
        Date.now() < deadline
          ? setTimeout(poll, 500)
          : reject(new Error(`Timeout waiting for port ${port}`));
      });
      sock.on("timeout", () => {
        sock.destroy();
        Date.now() < deadline
          ? setTimeout(poll, 500)
          : reject(new Error(`Timeout waiting for port ${port}`));
      });
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

// ─── Core spawn logic ─────────────────────────────────────────────────────────

function spawnPython() {
  if (shuttingDown || pythonProcess) return;

  console.log(`[Python] Starting AI training server (port ${PYTHON_PORT})...`);
  lastStartTime = Date.now();

  pythonProcess = spawn("python3", [PYTHON_SCRIPT], {
    env: {
      ...process.env,
      MODEL_API_PORT: String(PYTHON_PORT),
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout?.on("data", (d: Buffer) =>
    process.stdout.write(`[Python] ${d}`),
  );
  pythonProcess.stderr?.on("data", (d: Buffer) =>
    process.stderr.write(`[Python] ${d}`),
  );

  pythonProcess.on("exit", async (code, signal) => {
    pythonProcess = null;
    restartScheduled = false;

    if (shuttingDown || signal === "SIGTERM" || signal === "SIGINT") {
      console.log("[Python] Server shut down gracefully.");
      return;
    }

    // If another process grabbed the port (e.g., standalone workflow restarted), don't fight it
    if (await isPortOpen(PYTHON_PORT)) {
      console.log(
        `[Python] Port ${PYTHON_PORT} is held by another process — standing by.`,
      );
      return;
    }

    const uptime = Date.now() - lastStartTime;
    if (uptime >= MIN_HEALTHY_RUN_MS) {
      console.log(
        `[Python] Server was healthy for ${Math.round(uptime / 1000)}s — resetting backoff.`,
      );
      consecutiveCrashes = 0;
    }

    consecutiveCrashes++;
    const delay = backoffMs();
    console.log(
      `[Python] Server exited (code ${code ?? "?"}, signal ${signal ?? "none"}). Restart #${consecutiveCrashes} in ${delay}ms…`,
    );

    restartScheduled = true;
    setTimeout(spawnPython, delay);
  });
}

// ─── Health monitor ────────────────────────────────────────────────────────────
// Detects silent crashes (process died without triggering exit, or port went
// away while the process was externally managed).

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
  healthTimer.unref(); // don't prevent Node from exiting on its own
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function ensurePythonServer(): Promise<void> {
  const alreadyUp = await pollUntilOpen(PYTHON_PORT, 6_000);

  if (alreadyUp) {
    console.log(
      `[Python] AI training server already running on port ${PYTHON_PORT} — monitoring only.`,
    );
  } else {
    spawnPython();
    try {
      await waitForPort(PYTHON_PORT, 45_000);
      consecutiveCrashes = 0;
      console.log(`[Python] AI training server ready on port ${PYTHON_PORT}`);
    } catch {
      console.error(
        "[Python] Warning: server did not respond in time — requests will be retried by proxy.",
      );
    }
  }

  startHealthMonitor();
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
