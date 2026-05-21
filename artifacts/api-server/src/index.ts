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

// Cap at 4 workers — beyond that the Python AI server becomes the bottleneck
const NUM_WORKERS = Math.min(os.cpus().length, 4);

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

  const shutdown = () => {
    server.close(() => process.exit(0));
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}
