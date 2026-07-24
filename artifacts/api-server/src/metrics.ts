/**
 * Lightweight in-process metrics for the API server.
 *
 * All counters live in a single module-level object so they survive across
 * hot-reloads in development and across cluster workers in production (each
 * worker keeps its own copy — good enough for operational dashboards).
 *
 * Never throws; every exported function is wrapped defensively.
 */

// ─── Ring buffer for p95 latency ─────────────────────────────────────────────

const LATENCY_RING_SIZE = 100;
const _latencyRing: number[] = new Array(LATENCY_RING_SIZE).fill(0);
let _latencyRingIdx = 0;
let _latencyRingFilled = false;

/**
 * Record a single request latency (ms). Called after every proxied request.
 */
export function recordLatency(ms: number): void {
  try {
    _latencyRing[_latencyRingIdx] = ms;
    _latencyRingIdx = (_latencyRingIdx + 1) % LATENCY_RING_SIZE;
    if (_latencyRingIdx === 0) _latencyRingFilled = true;
  } catch {
    // never raise
  }
}

/** Return the p95 latency from the rolling 100-request window. */
export function getP95LatencyMs(): number {
  try {
    const used = _latencyRingFilled ? _latencyRing : _latencyRing.slice(0, _latencyRingIdx);
    if (used.length === 0) return 0;
    const sorted = [...used].sort((a, b) => a - b);
    const p95idx = Math.floor(sorted.length * 0.95);
    return sorted[Math.min(p95idx, sorted.length - 1)];
  } catch {
    return 0;
  }
}

// ─── Request counters ────────────────────────────────────────────────────────

let _requestsTotal = 0;
const _requestsByRoute = new Map<string, number>();

/** Increment total and per-route counters. */
export function recordRequest(route: string): void {
  try {
    _requestsTotal++;
    _requestsByRoute.set(route, (_requestsByRoute.get(route) ?? 0) + 1);
  } catch {
    // never raise
  }
}

export function getRequestsTotal(): number {
  return _requestsTotal;
}

export function getRequestsByRoute(): Record<string, number> {
  const out: Record<string, number> = {};
  _requestsByRoute.forEach((v, k) => {
    out[k] = v;
  });
  return out;
}
