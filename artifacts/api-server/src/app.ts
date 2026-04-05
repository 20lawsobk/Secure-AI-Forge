import path from "path";
import { fileURLToPath } from "url";
import express, { type Express, type Request, type Response } from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import router from "./routes";

const app: Express = express();

const limiter = rateLimit({
  windowMs: 1000,
  limit: 120_000_000,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: { error: "Rate limit exceeded. Please try again shortly." },
});

app.use(cors());
app.use(limiter);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// ─── Proxy /uploads/* directly to the Python AI server ──────────────────────
const _MODEL_API_BASE = `http://localhost:${process.env.MODEL_API_PORT || "9878"}`;

app.get("/uploads/*path", async (req: Request, res: Response) => {
  try {
    const upstream = await fetch(`${_MODEL_API_BASE}${req.path}`);
    if (!upstream.ok) {
      res.status(upstream.status).send(upstream.statusText);
      return;
    }
    const contentType = upstream.headers.get("content-type") || "application/octet-stream";
    res.setHeader("Content-Type", contentType);
    const buf = Buffer.from(await upstream.arrayBuffer());
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
      return path.resolve(path.dirname(fileURLToPath(metaUrl)), "../../ai-dashboard/dist/public");
    }
    // CJS fallback: process.cwd() is the monorepo root
    return path.resolve(process.cwd(), "artifacts/ai-dashboard/dist/public");
  })();

  app.use(express.static(dashboardDist, { maxAge: "1d" }));

  // SPA fallback — any unmatched GET returns index.html so React Router works
  app.get("/*path", (_req: Request, res: Response) => {
    res.sendFile(path.join(dashboardDist, "index.html"));
  });
}

export default app;
