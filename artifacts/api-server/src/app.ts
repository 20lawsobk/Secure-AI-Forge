import express, { type Express, type Request, type Response } from "express";
import cors from "cors";
import router from "./routes";

const app: Express = express();

app.use(cors());
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

export default app;
