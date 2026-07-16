import path from "path";
import { fileURLToPath } from "url";
import express, { type Express, type Request, type Response } from "express";
import cors from "cors";
import compression from "compression";
import rateLimit from "express-rate-limit";
import { Agent, request as undiciRequest } from "undici";
import router from "./routes";

const app: Express = express();

app.set("trust proxy", 1);

const limiter = rateLimit({
  windowMs: 60_000,
  limit: 300,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: { error: "Rate limit exceeded. Please try again shortly." },
});

app.use(cors());
app.use(compression());
app.use(limiter);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ─── Deployment healthcheck fast-paths ───────────────────────────────────────
// The platform healthchecks GET "/", "/api" and "/uploads" and RESTARTS the VM
// when they fail. These must return 200 immediately and NEVER depend on the
// Python model server (which takes minutes to load) — otherwise every Python
// restart window escalates into a full VM restart loop (observed flapping).
app.get("/healthz", (_req: Request, res: Response) => {
  res.status(200).json({ ok: true });
});
app.get(["/api", "/api/"], (_req: Request, res: Response) => {
  res.status(200).json({ ok: true, service: "api-server" });
});
app.get(["/uploads", "/uploads/"], (_req: Request, res: Response) => {
  res.status(200).json({ ok: true, service: "uploads-proxy" });
});

app.use("/api", router);

// ─── Proxy /uploads/* directly to the Python AI server ──────────────────────
// Uses the same undici keep-alive pool as the API proxy so video file requests
// reuse existing TCP connections instead of opening a new socket each time.

const _MODEL_API_BASE = `http://localhost:${process.env.MODEL_API_PORT || "9878"}`;

const _uploadsPool = new Agent({
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 60_000,
  connections: 8,
  pipelining: 1,
});

app.get("/uploads/*path", async (req: Request, res: Response) => {
  try {
    const upstreamRes = await undiciRequest(`${_MODEL_API_BASE}${req.path}`, {
      method: "GET",
      dispatcher: _uploadsPool,
    });
    if (upstreamRes.statusCode >= 400) {
      res
        .status(upstreamRes.statusCode)
        .send(upstreamRes.statusCode.toString());
      await upstreamRes.body.dump();
      return;
    }
    const contentType =
      (upstreamRes.headers["content-type"] as string | undefined) ??
      "application/octet-stream";
    res.setHeader("Content-Type", contentType);
    const buf = Buffer.from(await upstreamRes.body.arrayBuffer());
    res.send(buf);
  } catch (err) {
    res.status(502).json({ error: "Could not fetch asset from AI server" });
  }
});

// ─── Serve the built React dashboard in production ───────────────────────────
// In CJS output import.meta.url is empty, so fall back to process.cwd() which
// is the monorepo root when Node is launched from there.
if (process.env.NODE_ENV === "production") {
  const dashboardDist = (() => {
    const metaUrl = import.meta?.url;
    if (metaUrl) {
      // ESM path: __dirname = artifacts/api-server/src (dev) or dist (built)
      return path.resolve(
        path.dirname(fileURLToPath(metaUrl)),
        "../../ai-dashboard/dist/public",
      );
    }
    // CJS fallback: process.cwd() is the monorepo root
    return path.resolve(process.cwd(), "artifacts/ai-dashboard/dist/public");
  })();

  app.use(express.static(dashboardDist, { maxAge: "1d" }));

  // SPA fallback — any unmatched GET returns index.html so React Router works.
  // Never 500 here: the deployment healthcheck probes "/" and a 5xx gets the
  // VM restarted. If the dashboard build is missing, serve a minimal 200 page.
  app.get("/*path", (_req: Request, res: Response) => {
    res.sendFile(path.join(dashboardDist, "index.html"), (err) => {
      if (err && !res.headersSent) {
        res
          .status(200)
          .type("html")
          .send("<!doctype html><title>MaxCore</title><p>Server up — dashboard assets unavailable.</p>");
      }
    });
  });
}

// ─── Global error handler — never let an unhandled error become a bare 500 ──
app.use(
  (
    err: Error,
    _req: Request,
    res: Response,
    _next: express.NextFunction,
  ) => {
    console.error("[app] unhandled error:", err?.message || err);
    if (!res.headersSent) {
      res.status(500).json({ error: "internal", detail: String(err?.message || err) });
    }
  },
);

export default app;
