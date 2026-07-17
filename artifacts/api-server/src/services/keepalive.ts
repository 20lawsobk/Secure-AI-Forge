/**
 * Boot-time model warm-up status service.
 *
 * This module is an OBSERVER of the warm-up state — it does NOT trigger the
 * warm pass itself.  The actual POST /api/warm is fired by python-server.ts
 * (fireWarmPass) after waitForModelReady() succeeds.  This service polls
 * GET /api/warm/status to track that progress and exposes it via
 * getWarmStatus() / waitUntilWarm() so other modules (e.g. the dashboard
 * readiness endpoint) can query warm-up state without coupling to
 * python-server.ts internals.
 *
 * Wire-up: call startWarmService() once in the primary cluster process
 * (index.ts) alongside startKeepalive().  The service is a no-op in worker
 * processes.
 */

import { Agent, request as undiciRequest } from "undici";

const MODEL_API_PORT = process.env.MODEL_API_PORT || "9878";
const MODEL_API_BASE = `http://localhost:${MODEL_API_PORT}`;
const ADMIN_KEY      = process.env.ADMIN_KEY ?? "";

// Dedicated pool — never shares connections with user-traffic proxy.
const _statusPool = new Agent({
  keepAliveTimeout: 60_000,
  keepAliveMaxTimeout: 120_000,
  connections: 2,
  pipelining: 1,
});

// ─── State ───────────────────────────────────────────────────────────────────

export type WarmState = "pending" | "polling" | "warm" | "partial" | "timeout";

let _state: WarmState = "pending";
let _warmAt: string | null = null;
let _subsystems: Record<string, unknown> = {};
let _resolvers: Array<() => void> = [];

function _settle(state: WarmState): void {
  _state = state;
  const cbs = _resolvers.splice(0);
  for (const r of cbs) r();
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

function _makeHeaders(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (ADMIN_KEY) h["X-Admin-Key"] = ADMIN_KEY;
  return h;
}

/**
 * Poll GET /api/warm/status until deep_warm.state is a terminal value
 * (warm | partial | error) or until maxMs elapses.
 * Returns the final state string, or "timeout".
 */
async function _pollWarmStatus(maxMs = 180_000): Promise<string> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    try {
      const { statusCode, body } = await undiciRequest(
        `${MODEL_API_BASE}/api/warm/status`,
        {
          method: "GET",
          dispatcher: _statusPool,
          headers: _makeHeaders(),
          headersTimeout: 0,
          bodyTimeout: 0,
        },
      );
      const raw = await body.text();
      if (statusCode === 200) {
        const parsed = JSON.parse(raw) as {
          deep_warm?: { state?: string; last_warm_at?: string; subsystems?: Record<string, unknown> };
        };
        const dw = parsed.deep_warm ?? {};
        const dwState = dw.state ?? "pending";
        if (dwState === "warm" || dwState === "partial" || dwState === "error") {
          _warmAt = dw.last_warm_at ?? new Date().toISOString();
          _subsystems = dw.subsystems ?? {};
          return dwState;
        }
      }
    } catch {
      // Python not yet up — keep polling
    }
    await new Promise<void>((r) => setTimeout(r, 3_000));
  }
  return "timeout";
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Kick off warm-status polling in the background.
 * Safe to call multiple times — subsequent calls are no-ops once polling starts.
 */
export function startWarmService(): void {
  if (_state !== "pending") return;
  _state = "polling";

  (async () => {
    try {
      console.log("[WarmService] Polling /api/warm/status until deep-warm completes…");
      const finalState = await _pollWarmStatus(180_000);
      const mapped: WarmState =
        finalState === "warm" ? "warm"
        : finalState === "partial" ? "partial"
        : "timeout";
      console.log(`[WarmService] Deep-warm settled: ${mapped}`);
      _settle(mapped);
    } catch (err) {
      console.warn(`[WarmService] Unexpected error while polling warm status: ${err}`);
      _settle("timeout");
    }
  })().catch(() => { _settle("timeout"); });
}

/**
 * Returns a Promise that resolves once the deep-warm pass reaches a terminal
 * state (warm / partial / timeout).  Resolves immediately if already settled.
 */
export function waitUntilWarm(): Promise<void> {
  if (_state !== "pending" && _state !== "polling") {
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    _resolvers.push(resolve);
  });
}

/**
 * Current warm-service snapshot — safe to call at any time from any process.
 */
export function getWarmStatus(): {
  state: WarmState;
  warmAt: string | null;
  subsystems: Record<string, unknown>;
} {
  return { state: _state, warmAt: _warmAt, subsystems: _subsystems };
}
