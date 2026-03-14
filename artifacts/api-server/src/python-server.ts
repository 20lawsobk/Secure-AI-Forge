import { spawn, ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import net from "net";

const PYTHON_PORT = parseInt(process.env.MODEL_API_PORT || "9878", 10);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_SCRIPT = path.resolve(__dirname, "../../../artifacts/ai-training-server/server.py");
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

let pythonProcess: ChildProcess | null = null;
let retries = 0;

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

  pythonProcess.on("exit", (code, signal) => {
    pythonProcess = null;
    if (signal === "SIGTERM" || signal === "SIGINT") {
      console.log("[Python] Server shut down gracefully.");
      return;
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

export async function ensurePythonServer(): Promise<void> {
  const alreadyRunning = await isPortAlreadyOpen(PYTHON_PORT);
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
