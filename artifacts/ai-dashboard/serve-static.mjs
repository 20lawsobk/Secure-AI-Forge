// Featherweight production server for the dashboard artifact.
//
// The real application server (api-server artifact, port 8080) already serves
// the built dashboard AND all /api routes. This process exists only because
// the platform requires each runnable artifact to open its own port. It must
// stay tiny: no cluster, no keepalive, no Python spawning — a second full
// api-server here doubles memory + warm-render load and has frozen the VM.
//
// Serves ai-dashboard/dist/public with SPA fallback; proxies /api and
// /uploads to the real server on 8080 in case routing sends them here.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, "dist/public");
const PORT = parseInt(process.env.PORT ?? "3000", 10);
const API_PORT = parseInt(process.env.API_PORT ?? "8080", 10);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".map": "application/json",
  ".txt": "text/plain; charset=utf-8",
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
};

function proxy(req, res) {
  const upstream = http.request(
    { host: "127.0.0.1", port: API_PORT, path: req.url, method: req.method, headers: req.headers },
    (up) => {
      res.writeHead(up.statusCode ?? 502, up.headers);
      up.pipe(res);
    },
  );
  upstream.on("error", () => {
    if (!res.headersSent) res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "upstream unavailable" }));
  });
  req.pipe(upstream);
}

function serveFile(res, filePath, status = 200) {
  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(status, {
    "content-type": MIME[ext] ?? "application/octet-stream",
    "cache-control": ext === ".html" ? "no-cache" : "public, max-age=86400",
  });
  fs.createReadStream(filePath)
    .on("error", () => res.end())
    .pipe(res);
}

const server = http.createServer((req, res) => {
  const url = (req.url ?? "/").split("?")[0];
  if (/^\/(api|uploads)(\/|$)/.test(url)) return proxy(req, res);
  if (url === "/healthz") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end('{"ok":true,"role":"static-dashboard"}');
  }
  // Path traversal guard + static resolution
  const safe = path.normalize(url).replace(/^(\.\.[/\\])+/, "");
  let filePath = path.join(DIST, safe);
  if (!filePath.startsWith(DIST)) filePath = path.join(DIST, "index.html");
  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch {
    stat = null;
  }
  if (!stat || stat.isDirectory()) filePath = path.join(DIST, "index.html"); // SPA fallback
  serveFile(res, filePath);
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`[StaticDashboard] serving ${DIST} on port ${PORT} (api proxy → ${API_PORT})`);
});
