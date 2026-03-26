import { spawn, ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import net from "net";

const PYTHON_PORT = parseInt(process.env.MODEL_API_PORT || "9878", 10);

// Resolve the script path safely in both ESM (tsx dev) and CJS (production bundle).
// In CJS bundles, import.meta.url is undefined; fall back to process.cwd() which
// is always the monorepo workspace root in both environments.
const PYTHON_SCRIPT = (() => {
  const metaUrl = import.meta?.url;
  if (metaUrl) {
    return path.resolve(path.dirname(fileURLToPath(metaUrl)), "../../../artifacts/ai-training-server/server.py");
  }
  return path.resolve(process.cwd(), "artifacts/ai-training-server/server.py");
})();
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

let pythonProcess: ChildProcess | null = null;
let retries = 0;
let lastStartTime = 0;

// If the server ran successfully for at least this long, reset the retry counter
// so transient long-running crashes don't exhaust the retry budget permanently.
const MIN_HEALTHY_RUN_MS = 60_000;

function waitForPort(port: number, timeout = 30000): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    function attempt() {
      const sock = new net.Socket();
      sock.setTimeout(1000);
      sock.on("connect", () => { sock.destroy(); resolve(); });
      sock.on("error", () => {
        sock.destroy();
        if (Date.now() - start > timeout) {
          reject(new Error(`Python server did not start within ${timeout}ms`));
        } else {
          setTimeout(attempt, 500);
        }
      });
      sock.on("timeout", () => {
        sock.destroy();
        setTimeout(attempt, 500);
      });
      sock.connect(port, "127.0.0.1");
    }
    attempt();
  });
}

function startPythonServer() {
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

  pythonProcess.stdout?.on("data", (d: Buffer) => {
    process.stdout.write(`[Python] ${d}`);
  });

  pythonProcess.stderr?.on("data", (d: Buffer) => {
    process.stderr.write(`[Python] ${d}`);
  });

  pythonProcess.on("exit", async (code, signal) => {
    pythonProcess = null;
    if (signal === "SIGTERM" || signal === "SIGINT") {
      console.log("[Python] Server shut down gracefully.");
      return;
    }
    // If port is now held by another process (e.g. standalone workflow), stop retrying.
    const portTaken = await isPortAlreadyOpen(PYTHON_PORT);
    if (portTaken) {
      console.log(`[Python] Port ${PYTHON_PORT} now held by another process — reusing it.`);
      return;
    }
    // If the server was healthy for a sustained period, treat it as a fresh start
    const uptime = Date.now() - lastStartTime;
    if (uptime >= MIN_HEALTHY_RUN_MS) {
      console.log(`[Python] Server ran for ${Math.round(uptime / 1000)}s — resetting retry counter.`);
      retries = 0;
    }
    retries++;
    if (retries <= MAX_RETRIES) {
      console.log(`[Python] Server exited (code ${code}). Restarting in ${RETRY_DELAY_MS}ms... (${retries}/${MAX_RETRIES})`);
      setTimeout(startPythonServer, RETRY_DELAY_MS);
    } else {
      console.error("[Python] Max restarts reached. Check server.py for errors.");
    }
  });
}

async function isPortAlreadyOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    sock.setTimeout(1000);
    sock.on("connect", () => { sock.destroy(); resolve(true); });
    sock.on("error", () => { sock.destroy(); resolve(false); });
    sock.on("timeout", () => { sock.destroy(); resolve(false); });
    sock.connect(port, "127.0.0.1");
  });
}

// Poll for up to `maxMs` in case the standalone workflow is still starting up.
async function waitForPortOpen(port: number, maxMs = 6000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    if (await isPortAlreadyOpen(port)) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

export async function ensurePythonServer(): Promise<void> {
  // Give any concurrently-starting standalone workflow a window to bind first.
  const alreadyRunning = await waitForPortOpen(PYTHON_PORT, 6000);
  if (alreadyRunning) {
    console.log(`[Python] AI training server already running on port ${PYTHON_PORT} — skipping spawn.`);
    return;
  }

  startPythonServer();
  try {
    await waitForPort(PYTHON_PORT, 45000);
    retries = 0;
    console.log(`[Python] AI training server ready on port ${PYTHON_PORT}`);
  } catch (err) {
    console.error("[Python] Warning: server did not respond in time — proxy will retry on requests.");
  }
}

export function stopPythonServer() {
  if (pythonProcess) {
    console.log("[Python] Stopping AI training server...");
    pythonProcess.kill("SIGTERM");
    pythonProcess = null;
  }
}
