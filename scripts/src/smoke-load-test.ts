/**
 * MaxBooster Smoke + Load Test
 *
 * Phase 1 — Smoke: hits every significant endpoint once and verifies expected
 *            status codes and key response fields.
 * Phase 2 — Load:  fires N concurrent waves against the read-only endpoints
 *            and reports P50 / P95 / P99 latencies + error rates.
 *
 * Usage:
 *   ADMIN_KEY=mbs_xxx tsx ./src/smoke-load-test.ts [--concurrency=10] [--waves=5] [--timeout=15000]
 *
 * Exits 0 = all smoke checks passed, 1 = one or more failures.
 */

// ─── Config ──────────────────────────────────────────────────────────────────

const API_PORT   = process.env.API_PORT   ?? "8080";
const MODEL_PORT = process.env.MODEL_PORT ?? "9878";
const FE_PORT    = process.env.FE_PORT    ?? "5000";

const API_BASE   = `http://localhost:${API_PORT}`;
const MODEL_BASE = `http://localhost:${MODEL_PORT}`;
const FE_BASE    = `http://localhost:${FE_PORT}`;

const ADMIN_KEY  =
  process.env.ADMIN_KEY ??
  "mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc";

const args        = process.argv.slice(2);
const CONCURRENCY = parseInt(args.find(a => a.startsWith("--concurrency="))?.split("=")[1] ?? "10");
const WAVES       = parseInt(args.find(a => a.startsWith("--waves="))?.split("=")[1] ?? "5");
const TIMEOUT_MS  = parseInt(args.find(a => a.startsWith("--timeout="))?.split("=")[1] ?? "15000");

// ─── Types ────────────────────────────────────────────────────────────────────

interface Endpoint {
  label: string;
  tier: "Frontend" | "API Server" | "Python AI";
  method: "GET" | "POST";
  url: string;
  body?: unknown;
  /** Expected HTTP status code */
  expect: number;
  /** Key that must exist in the JSON response (optional) */
  checkKey?: string;
  /** If true, include in the load-burst phase */
  loadTarget?: boolean;
}

interface SmokeResult {
  endpoint: Endpoint;
  status: number | "ERR";
  latencyMs: number;
  passed: boolean;
  error?: string;
  bodySnippet?: string;
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

const R  = (s: string) => `\x1b[31m${s}\x1b[0m`;
const G  = (s: string) => `\x1b[32m${s}\x1b[0m`;
const Y  = (s: string) => `\x1b[33m${s}\x1b[0m`;
const B  = (s: string) => `\x1b[34m${s}\x1b[0m`;
const C  = (s: string) => `\x1b[36m${s}\x1b[0m`;
const DIM = (s: string) => `\x1b[2m${s}\x1b[0m`;
const BOLD = (s: string) => `\x1b[1m${s}\x1b[0m`;

// ─── Endpoint definitions ─────────────────────────────────────────────────────

const ENDPOINTS: Endpoint[] = [
  // ── Frontend (Vite dev server) ─────────────────────────────────────────────
  {
    label:      "FE root HTML",
    tier:       "Frontend",
    method:     "GET",
    url:        `${FE_BASE}/`,
    expect:     200,
  },

  // ── API Server – infrastructure ────────────────────────────────────────────
  {
    label:      "Healthz",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/healthz`,
    expect:     200,
    checkKey:   "status",
    loadTarget: true,
  },
  {
    label:      "API health (proxy→Python)",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/health`,
    expect:     200,
    checkKey:   "status",
    loadTarget: true,
  },
  {
    label:      "API model status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/model/status`,
    expect:     200,
    checkKey:   "model_loaded",
    loadTarget: true,
  },
  {
    label:      "API GPU status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/gpu/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API GPU hyper status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/gpu/hyper/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API GPU capabilities",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/gpu/capabilities`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API training status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/training/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API training logs",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/training/logs?limit=5`,
    expect:     200,
  },
  {
    label:      "API continuous training status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/training/continuous/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API data puller status",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/training/puller/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "API training datasets",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/training/datasets`,
    expect:     200,
  },
  {
    label:      "API multimodal packs",
    tier:       "API Server",
    method:     "GET",
    url:        `${API_BASE}/api/multimodal/packs`,
    expect:     200,
    loadTarget: true,
  },

  // ── API Server – generation (POST) ─────────────────────────────────────────
  {
    label:      "Content generate (TikTok)",
    tier:       "API Server",
    method:     "POST",
    url:        `${API_BASE}/api/content/generate`,
    body:       { platform: "tiktok", topic: "smoke test drop", tone: "energetic", goal: "growth", include_hashtags: true },
    expect:     200,
    checkKey:   "success",
  },
  {
    label:      "Platform social generate (Instagram)",
    tier:       "API Server",
    method:     "POST",
    url:        `${API_BASE}/api/platform/social/generate`,
    body:       { user_id: "smoke-test-user", platform: "instagram", topic: "new track announcement", tone: "authentic", goal: "growth", num_variants: 1, include_hashtags: true },
    expect:     200,
    checkKey:   "variants",
  },

  // ── Python AI Server – direct ──────────────────────────────────────────────
  {
    label:      "Python /health",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/health`,
    expect:     200,
    checkKey:   "status",
    loadTarget: true,
  },
  {
    label:      "Python /dashboard/stats",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/dashboard/stats`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /model/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/model/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /gpu/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/gpu/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /gpu/hyper/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/gpu/hyper/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /gpu/capabilities",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/gpu/capabilities`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /training/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/training/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /training/continuous/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/training/continuous/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /training/puller/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/training/puller/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /watchdog/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/watchdog/status`,
    expect:     200,
    loadTarget: true,
  },
  {
    label:      "Python /storage/status",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/storage/status`,
    expect:     200,
  },
  {
    label:      "Python /boostsheets",
    tier:       "Python AI",
    method:     "GET",
    url:        `${MODEL_BASE}/boostsheets`,
    expect:     200,
  },
];

// ─── Core request helper ──────────────────────────────────────────────────────

async function hit(ep: Endpoint): Promise<SmokeResult> {
  const t0 = Date.now();
  try {
    const headers: Record<string, string> = {
      "X-Admin-Key": ADMIN_KEY,
      "X-API-Key":   ADMIN_KEY,
    };
    if (ep.body) headers["Content-Type"] = "application/json";

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);

    let res: Response;
    try {
      res = await fetch(ep.url, {
        method:  ep.method,
        headers,
        body:    ep.body ? JSON.stringify(ep.body) : undefined,
        signal:  ctrl.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    const latencyMs = Date.now() - t0;
    let bodySnippet = "";
    let checkPassed = true;

    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      try {
        const json = await res.json() as Record<string, unknown>;
        bodySnippet = JSON.stringify(json).slice(0, 120);
        if (ep.checkKey && !(ep.checkKey in json)) checkPassed = false;
      } catch {
        bodySnippet = "<invalid json>";
        checkPassed = false;
      }
    } else {
      const text = await res.text();
      bodySnippet = text.slice(0, 80).replace(/\n/g, " ");
    }

    const passed = res.status === ep.expect && checkPassed;
    return { endpoint: ep, status: res.status, latencyMs, passed, bodySnippet };
  } catch (err: unknown) {
    const latencyMs = Date.now() - t0;
    const error = err instanceof Error ? err.message : String(err);
    return { endpoint: ep, status: "ERR", latencyMs, passed: false, error };
  }
}

// ─── Pretty printing ──────────────────────────────────────────────────────────

const TIER_COLORS: Record<Endpoint["tier"], (s: string) => string> = {
  "Frontend":  C,
  "API Server": B,
  "Python AI": Y,
};

function col(s: string, w: number): string {
  return s.length >= w ? s.slice(0, w) : s + " ".repeat(w - s.length);
}

function printSmokeTable(results: SmokeResult[]): void {
  const w = { icon: 2, tier: 12, method: 6, label: 42, status: 8, latency: 10, body: 60 };
  const sep = "─".repeat(
    w.icon + w.tier + w.method + w.label + w.status + w.latency + w.body + 14
  );

  console.log("\n" + BOLD("═".repeat(sep.length)));
  console.log(BOLD(" SMOKE TEST RESULTS"));
  console.log(BOLD("═".repeat(sep.length)));

  const hdr = [
    col("", w.icon),
    col("Tier", w.tier),
    col("Method", w.method),
    col("Endpoint", w.label),
    col("HTTP", w.status),
    col("Latency", w.latency),
    col("Response snippet", w.body),
  ].join("  ");
  console.log(DIM(hdr));
  console.log(DIM(sep));

  let lastTier = "";
  for (const r of results) {
    if (r.endpoint.tier !== lastTier) {
      if (lastTier) console.log(DIM("·".repeat(sep.length)));
      lastTier = r.endpoint.tier;
    }

    const icon    = r.passed ? G("✓") : R("✗");
    const tierStr = TIER_COLORS[r.endpoint.tier](col(r.endpoint.tier, w.tier));
    const method  = r.endpoint.method === "POST" ? Y(col(r.endpoint.method, w.method)) : DIM(col(r.endpoint.method, w.method));
    const label   = col(r.endpoint.label, w.label);
    const status  = r.passed
      ? G(col(String(r.status), w.status))
      : R(col(String(r.status), w.status));
    const latency = r.latencyMs < 200 ? G(col(`${r.latencyMs}ms`, w.latency))
                  : r.latencyMs < 1000 ? Y(col(`${r.latencyMs}ms`, w.latency))
                  : R(col(`${r.latencyMs}ms`, w.latency));
    const snippet = DIM(col(r.error ?? r.bodySnippet ?? "", w.body));

    console.log([icon, tierStr, method, label, status, latency, snippet].join("  "));
  }

  console.log(BOLD("═".repeat(sep.length)));
}

// ─── Load burst ───────────────────────────────────────────────────────────────

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)]!;
}

async function runLoadBurst(targets: Endpoint[]): Promise<void> {
  console.log("\n" + BOLD("═".repeat(80)));
  console.log(BOLD(` LOAD BURST  —  ${CONCURRENCY} concurrent × ${WAVES} waves  (${targets.length} endpoints)`));
  console.log(BOLD("═".repeat(80)));

  const allLatencies: number[] = [];
  const errByEndpoint: Map<string, number> = new Map();
  let totalRequests = 0;
  let totalErrors   = 0;

  for (let wave = 1; wave <= WAVES; wave++) {
    process.stdout.write(`  Wave ${wave}/${WAVES}  `);
    const waveStart = Date.now();

    // Build a pool of CONCURRENCY promises, cycling through targets
    const tasks: Promise<SmokeResult>[] = [];
    for (let i = 0; i < CONCURRENCY; i++) {
      const ep = targets[i % targets.length]!;
      tasks.push(hit(ep));
    }

    const results = await Promise.all(tasks);
    const waveMs  = Date.now() - waveStart;

    for (const r of results) {
      totalRequests++;
      allLatencies.push(r.latencyMs);
      if (!r.passed) {
        totalErrors++;
        const k = r.endpoint.label;
        errByEndpoint.set(k, (errByEndpoint.get(k) ?? 0) + 1);
        process.stdout.write(R("✗"));
      } else {
        process.stdout.write(G("·"));
      }
    }

    const waveP50 = percentile([...results.map(r => r.latencyMs)].sort((a, b) => a - b), 50);
    console.log(`  ${DIM(`(${waveMs}ms total, p50=${waveP50}ms)`)}`);
  }

  const sorted = [...allLatencies].sort((a, b) => a - b);
  const p50    = percentile(sorted, 50);
  const p95    = percentile(sorted, 95);
  const p99    = percentile(sorted, 99);
  const errPct = ((totalErrors / totalRequests) * 100).toFixed(1);

  console.log("\n" + BOLD("  Summary"));
  console.log(`  Total requests : ${BOLD(String(totalRequests))}`);
  console.log(`  Errors         : ${totalErrors > 0 ? R(String(totalErrors)) : G("0")}  (${errPct}%)`);
  console.log(`  Latency P50    : ${p50 < 300 ? G(`${p50}ms`) : R(`${p50}ms`)}`);
  console.log(`  Latency P95    : ${p95 < 800 ? G(`${p95}ms`) : R(`${p95}ms`)}`);
  console.log(`  Latency P99    : ${p99 < 2000 ? G(`${p99}ms`) : R(`${p99}ms`)}`);

  if (errByEndpoint.size > 0) {
    console.log("\n" + R("  Failing endpoints:"));
    for (const [label, count] of errByEndpoint) {
      console.log(`    ${R("✗")} ${label} — ${count} error(s)`);
    }
  }

  console.log(BOLD("═".repeat(80)));
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log(BOLD("\n  MaxBooster Smoke + Load Test"));
  console.log(DIM(`  Targets: Frontend:${FE_PORT}  API:${API_PORT}  Python:${MODEL_PORT}`));
  console.log(DIM(`  Admin key prefix: ${ADMIN_KEY.slice(0, 12)}...`));
  console.log(DIM(`  Timeout: ${TIMEOUT_MS}ms  |  Load: concurrency=${CONCURRENCY} waves=${WAVES}\n`));

  // ── Phase 1: sequential smoke ──────────────────────────────────────────────
  console.log(B("  Phase 1 — Smoke checks (sequential)…\n"));
  const smokeResults: SmokeResult[] = [];
  for (const ep of ENDPOINTS) {
    const r = await hit(ep);
    smokeResults.push(r);
    const icon = r.passed ? G("✓") : R("✗");
    process.stdout.write(`  ${icon} ${col(ep.label, 44)} ${r.passed ? G(String(r.status)) : R(String(r.status))}  ${DIM(`${r.latencyMs}ms`)}\n`);
  }

  printSmokeTable(smokeResults);

  const smokePassed  = smokeResults.filter(r => r.passed).length;
  const smokeFailed  = smokeResults.filter(r => !r.passed).length;
  const smokeTotal   = smokeResults.length;

  console.log(`\n  ${BOLD("Smoke")}:  ${G(`${smokePassed}/${smokeTotal} passed`)}  ${smokeFailed > 0 ? R(`${smokeFailed} FAILED`) : ""}`);

  // ── Phase 2: concurrent load burst ────────────────────────────────────────
  const loadTargets = ENDPOINTS.filter(e => e.loadTarget);
  if (loadTargets.length > 0) {
    await runLoadBurst(loadTargets);
  } else {
    console.log(Y("\n  (no load targets defined)"));
  }

  // ── Final verdict ──────────────────────────────────────────────────────────
  console.log();
  if (smokeFailed === 0) {
    console.log(G(BOLD("  ✓ ALL SMOKE CHECKS PASSED")));
  } else {
    console.log(R(BOLD(`  ✗ ${smokeFailed} SMOKE CHECK(S) FAILED`)));
  }
  console.log();

  process.exit(smokeFailed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error(R("\nFatal error:"), err);
  process.exit(1);
});
