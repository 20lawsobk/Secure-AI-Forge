import app from "./app";
import { ensurePythonServer, stopPythonServer } from "./python-server";

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

async function main() {
  const server = app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
  });

  const shutdown = () => {
    console.log("Shutting down...");
    stopPythonServer();
    server.close(() => process.exit(0));
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);

  ensurePythonServer().catch((err) => {
    console.error("[Python] Failed to start AI server:", err);
  });
}

main().catch((err) => {
  console.error("Fatal startup error:", err);
  process.exit(1);
});
