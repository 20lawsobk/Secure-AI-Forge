import { Router, type IRouter, type Request, type Response } from "express";

const router: IRouter = Router();

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;

async function proxyRequest(req: Request, res: Response, path: string) {
  const startTime = Date.now();
  try {
    const url = `${MODEL_API_BASE}${path}`;

    const fetchOptions: RequestInit = {
      method: req.method,
      headers: {
        "Content-Type": "application/json",
        ...(req.headers["x-admin-key"] ? { "X-Admin-Key": req.headers["x-admin-key"] as string } : {}),
        ...(req.headers["x-api-key"] ? { "X-Api-Key": req.headers["x-api-key"] as string } : {}),
      },
    };

    if (req.method !== "GET" && req.method !== "HEAD" && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const response = await fetch(url, fetchOptions);
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (err) {
    const elapsed = Date.now() - startTime;
    console.error(`[Proxy] Error proxying to ${path} (${elapsed}ms):`, err);
    if ((err as any).code === "ECONNREFUSED" || (err as any).cause?.code === "ECONNREFUSED") {
      res.status(503).json({
        error: "AI model server unavailable",
        detail: "The Python AI training server is not running or still initializing.",
      });
    } else {
      res.status(500).json({ error: "Proxy error", detail: String(err) });
    }
  }
}

router.get("/health", async (req, res) => {
  await proxyRequest(req, res, "/health");
});

router.get("/api-keys", async (req, res) => {
  await proxyRequest(req, res, "/api-keys");
});

router.post("/api-keys", async (req, res) => {
  await proxyRequest(req, res, "/api-keys");
});

router.delete("/api-keys/:keyId", async (req, res) => {
  await proxyRequest(req, res, `/api-keys/${req.params.keyId}`);
});

router.post("/api-keys/:keyId/rotate", async (req, res) => {
  await proxyRequest(req, res, `/api-keys/${req.params.keyId}/rotate`);
});

router.get("/model/status", async (req, res) => {
  await proxyRequest(req, res, "/model/status");
});

router.get("/gpu/status", async (req, res) => {
  await proxyRequest(req, res, "/gpu/status");
});

router.get("/gpu/hyper/status", async (req, res) => {
  await proxyRequest(req, res, "/gpu/hyper/status");
});

router.get("/gpu/capabilities", async (req, res) => {
  await proxyRequest(req, res, "/gpu/capabilities");
});

router.get("/training/status", async (req, res) => {
  await proxyRequest(req, res, "/training/status");
});

router.post("/training/start", async (req, res) => {
  await proxyRequest(req, res, "/training/start");
});

router.get("/training/logs", async (req, res) => {
  await proxyRequest(req, res, `/training/logs${req.query.limit ? `?limit=${req.query.limit}` : ""}`);
});

router.post("/training/stop", async (req, res) => {
  await proxyRequest(req, res, "/training/stop");
});

router.get("/training/datasets", async (req, res) => {
  await proxyRequest(req, res, "/training/datasets");
});

router.post("/training/schedule", async (req, res) => {
  await proxyRequest(req, res, "/training/schedule");
});

router.post("/content/generate", async (req, res) => {
  await proxyRequest(req, res, "/content/generate");
});

router.get("/boostsheets", async (req, res) => {
  await proxyRequest(req, res, "/boostsheets");
});

router.get("/dashboard/stats", async (req, res) => {
  await proxyRequest(req, res, "/dashboard/stats");
});

router.get("/storage/status", async (req, res) => {
  await proxyRequest(req, res, "/storage/status");
});

router.post("/storage/feedback", async (req, res) => {
  await proxyRequest(req, res, "/storage/feedback");
});

router.get("/storage/curriculum/:userId", async (req, res) => {
  await proxyRequest(req, res, `/storage/curriculum/${req.params.userId}`);
});

router.get("/storage/datasets", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets");
});

router.post("/storage/datasets/register", async (req, res) => {
  await proxyRequest(req, res, "/storage/datasets/register");
});

router.get("/storage/checkpoints", async (req, res) => {
  await proxyRequest(req, res, "/storage/checkpoints");
});

router.post("/storage/checkpoint/save", async (req, res) => {
  await proxyRequest(req, res, "/storage/checkpoint/save");
});

router.get("/storage/checkpoint/:modelId", async (req, res) => {
  await proxyRequest(req, res, `/storage/checkpoint/${req.params.modelId}`);
});

router.get("/storage/session", async (req, res) => {
  await proxyRequest(req, res, "/storage/session");
});

router.get("/storage/pipeline/status", async (req, res) => {
  await proxyRequest(req, res, "/storage/pipeline/status");
});

router.post("/training/start-from-storage", async (req, res) => {
  await proxyRequest(req, res, "/training/start-from-storage");
});

// ─── Platform API Routes — Main Music Platform Integration ───────────────────

router.post("/platform/social/generate", async (req, res) => {
  await proxyRequest(req, res, "/platform/social/generate");
});

router.post("/platform/social/autopilot", async (req, res) => {
  await proxyRequest(req, res, "/platform/social/autopilot");
});

router.post("/platform/daw/generate", async (req, res) => {
  await proxyRequest(req, res, "/platform/daw/generate");
});

router.post("/platform/distribution/plan", async (req, res) => {
  await proxyRequest(req, res, "/platform/distribution/plan");
});

router.get("/platform/model/info", async (req, res) => {
  await proxyRequest(req, res, "/platform/model/info");
});

router.post("/platform/model/reload", async (req, res) => {
  await proxyRequest(req, res, "/platform/model/reload");
});

// ─── AI Ad System & Autopilot ────────────────────────────────────────────────

router.post("/platform/ads/record", async (req, res) => {
  await proxyRequest(req, res, "/platform/ads/record");
});

router.post("/platform/ads/generate", async (req, res) => {
  await proxyRequest(req, res, "/platform/ads/generate");
});

router.post("/platform/ads/autopilot", async (req, res) => {
  await proxyRequest(req, res, "/platform/ads/autopilot");
});

router.post("/platform/ads/audience", async (req, res) => {
  await proxyRequest(req, res, "/platform/ads/audience");
});

router.get("/platform/ads/performance/:userId", async (req, res) => {
  const query = req.query.platform ? `?platform=${req.query.platform}` : "";
  await proxyRequest(req, res, `/platform/ads/performance/${req.params.userId}${query}`);
});

router.post("/platform/ads/optimize", async (req, res) => {
  await proxyRequest(req, res, "/platform/ads/optimize");
});

export default router;
