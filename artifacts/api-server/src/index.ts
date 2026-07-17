import cluster from "cluster";
import os from "os";
import app from "./app";
import { ensurePythonServer, stopPythonServer } from "./python-server";
import { startKeepalive, stopKeepalive } from "./keepalive";
import { startWarmService } from "./services/keepalive.js";

// ─── Required secrets / env-var gate ─────────────────────────────────────────
// Fail fast on startup rather than serving broken requests or exposing an
// unauthenticated surface.  SESSION_SECRET must be a Replit Secret (never a
// plain env var in source control) so it is available in both development and
// the production VM without being committed to the repo.
const REQUIRED_ENV: string[] = ["SESSION_SECRET"];
const missing = REQUIRED_ENV.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error(
    `[Startup] Missing required secrets: ${missing.join(", ")}. ` +
      "Set them as Replit Secrets and restart.",
  );
  process.exit(1);
}

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// Default to min(cpus, 4) Node workers.  Each worker is ~150-200 MB;
// 4 workers + the Python model (~1.7 GB) leaves ~4 GB headroom on an 8 GB
// host for PIL render threads and ffmpeg.  Override via NODE_CLUSTER_WORKERS
// env var if you need to dial back on a memory-constrained host.
const _envWorkers = parseInt(process.env["NODE_CLUSTER_WORKERS"] ?? "", 10);
const NUM_WORKERS = Math.min(
  os.cpus().length,
  Number.isFinite(_envWorkers) && _envWorkers > 0 ? _envWorkers : 4,
);

if (cluster.isPrimary) {
  console.log(
    `[Cluster] Primary ${process.pid} — forking ${NUM_WORKERS} workers on port ${port}`,
  );

  // Only the primary process owns the Python lifecycle
  ensurePythonServer().catch((err) => {
    console.error("[Python] Failed to start AI server:", err);
  });

  // Keepalive: warm all endpoints on a 20s cycle so the AI server never idles
  startKeepalive();

  // Warm-status observer: polls /api/warm/status until the deep-warm pass
  // (fired by python-server.ts) reaches a terminal state; exposes the result
  // via getWarmStatus() / waitUntilWarm() for the system-readiness endpoint.
  startWarmService();

  for (let i = 0; i < NUM_WORKERS; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker, code, signal) => {
    if (signal === "SIGTERM" || signal === "SIGINT") return;
    console.log(
      `[Cluster] Worker ${worker.process.pid} died (code=${code ?? "?"}, signal=${signal ?? "none"}) — respawning…`,
    );
    cluster.fork();
  });

  const shutdown = () => {
    console.log("[Cluster] Primary shutting down…");
    stopKeepalive();
    stopPythonServer();
    for (const id in cluster.workers) {
      cluster.workers[id]?.kill("SIGTERM");
    }
    setTimeout(() => process.exit(0), 3_000);
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
} else {
  // Worker — just runs Express; Python lifecycle managed by primary
  const server = app.listen(port, () => {
    console.log(`[Cluster] Worker ${process.pid} listening on port ${port}`);
  });

  // Self-contained environment — no external clients, no slowloris risk.
  // All timeouts disabled so long-running model generations are never aborted.
  server.requestTimeout = 0;
  server.timeout = 0;
  server.headersTimeout = 0;
  server.keepAliveTimeout = 0;

  const shutdown = () => {
    server.close(() => process.exit(0));
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}
