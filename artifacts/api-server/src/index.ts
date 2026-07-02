import cluster from "cluster";
import os from "os";
import app from "./app";
import { ensurePythonServer, stopPythonServer } from "./python-server";
import { startKeepalive, stopKeepalive } from "./keepalive";

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

  // This server's job is proxying long-running model generations. Under
  // concurrent multimodal load a single honest request can legitimately spend
  // many minutes awaiting the upstream model, during which no bytes flow on the
  // client socket. `server.timeout` (socket inactivity) MUST stay 0 or those
  // valid in-flight generations get aborted mid-render — this is why Node
  // defaults it to 0 for long-response servers. `requestTimeout` and
  // `headersTimeout` stay bounded so a slow/stalled CLIENT request (slowloris)
  // is still rejected; they govern receiving the request, not the response.
  server.requestTimeout = 300_000;
  server.timeout = 0;
  server.headersTimeout = 65_000;
  server.keepAliveTimeout = 60_000;

  const shutdown = () => {
    server.close(() => process.exit(0));
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}
