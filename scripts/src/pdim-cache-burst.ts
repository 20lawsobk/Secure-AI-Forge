export {};
/**
 * MaxBooster PDIM Cache-Burst Load Test
 *
 * Demonstrates the pdim dedup layer as a scale mechanism:
 *   Phase 1 — Warm: fires one unique payload per slot to seed the pdim cache
 *   Phase 2 — Burst: fires the SAME payloads at extreme concurrency
 *             Cache hits return from pdim (~100ms) instead of inference (~2-5s)
 *             Single-flight: N identical concurrent requests → 1 compute
 *
 * Usage:
 *   ADMIN_KEY=mbs_xxx tsx ./src/pdim-cache-burst.ts [--url=https://...] [--concurrency=500] [--waves=5]
 */

const args        = process.argv.slice(2);
const CONCURRENCY = parseInt(args.find(a => a.startsWith("--concurrency="))?.split("=")[1] ?? "500");
const WAVES       = parseInt(args.find(a => a.startsWith("--waves="))?.split("=")[1] ?? "5");
const TIMEOUT_MS  = parseInt(args.find(a => a.startsWith("--timeout="))?.split("=")[1] ?? "60000");
const EXTERNAL_URL = (args.find(a => a.startsWith("--url="))?.split("=").slice(1).join("=") ?? "").replace(/\/$/, "");

const API_PORT = process.env.API_PORT ?? "8080";
const BASE     = EXTERNAL_URL || `http://localhost:${API_PORT}`;

const ADMIN_KEY =
  process.env.ADMIN_KEY ??
  process.env.AI_SERVER_KEY ??
  process.env.AI_TRAINING_KEY_PROD ??
  "mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc";

// ─── Colour helpers ───────────────────────────────────────────────────────────
const R    = (s: string) => `\x1b[31m${s}\x1b[0m`;
const G    = (s: string) => `\x1b[32m${s}\x1b[0m`;
const Y    = (s: string) => `\x1b[33m${s}\x1b[0m`;
const C    = (s: string) => `\x1b[36m${s}\x1b[0m`;
const DIM  = (s: string) => `\x1b[2m${s}\x1b[0m`;
const BOLD = (s: string) => `\x1b[1m${s}\x1b[0m`;

// ─── Fixed payloads — same hash every time, so pdim dedup cache kicks in ─────
const CACHE_SLOTS = [
  { platform: "tiktok",     topic: "new drop announcement",  tone: "energetic",    goal: "growth"      },
  { platform: "instagram",  topic: "behind the scenes",      tone: "authentic",    goal: "engagement"  },
  { platform: "twitter",    topic: "album release day",      tone: "hype",         goal: "virality"    },
  { platform: "youtube",    topic: "music video premiere",   tone: "cinematic",    goal: "views"       },
  { platform: "tiktok",     topic: "studio session",         tone: "raw",          goal: "connection"  },
  { platform: "instagram",  topic: "tour announcement",      tone: "excited",      goal: "ticket sales"},
  { platform: "twitter",    topic: "collab reveal",          tone: "mysterious",   goal: "buzz"        },
  { platform: "spotify",    topic: "playlist placement",     tone: "grateful",     goal: "streams"     },
];

// ─── Request helper ───────────────────────────────────────────────────────────
interface Result {
  latencyMs: number;
  ok: boolean;
  processingMs?: number;
  cached?: boolean;
  source?: string;
  error?: string;
}

async function fire(payload: object): Promise<Result> {
  const t0 = Date.now();
  try {
    const ctrl  = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${BASE}/api/content/generate`, {
        method:  "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key":  ADMIN_KEY,
          "X-API-Key":    ADMIN_KEY,
        },
        body:   JSON.stringify({ ...payload, include_hashtags: true }),
        signal: ctrl.signal,
      });
    } finally {
      clearTimeout(timer);
    }
    const latencyMs = Date.now() - t0;
    if (!res.ok) return { latencyMs, ok: false, error: `HTTP ${res.status}` };
    const json = await res.json() as Record<string, unknown>;
    return {
      latencyMs,
      ok:           json["success"] === true,
      processingMs: typeof json["processing_time_ms"] === "number" ? json["processing_time_ms"] as number : undefined,
      cached:       json["cached"] === true,
      source:       typeof json["source"] === "string" ? json["source"] as string : undefined,
    };
  } catch (err: unknown) {
    return { latencyMs: Date.now() - t0, ok: false, error: String(err) };
  }
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)]!;
}

// ─── Phase 1: warm the cache ──────────────────────────────────────────────────
async function warmCache(): Promise<void> {
  console.log(C(BOLD("\n  Phase 1 — Cache warm (sequential, one request per slot)\n")));
  for (const [i, payload] of CACHE_SLOTS.entries()) {
    process.stdout.write(`  [${i + 1}/${CACHE_SLOTS.length}] ${payload.platform}/${payload.topic} … `);
    const r = await fire(payload);
    if (r.ok) {
      const tag = r.cached ? G("cache-hit") : Y("computed");
      console.log(`${G("✓")}  ${r.latencyMs}ms  ${DIM(`(${tag}${r.processingMs !== undefined ? `, inference=${r.processingMs}ms` : ""})`)}`);
    } else {
      console.log(`${R("✗")}  ${r.latencyMs}ms  ${R(r.error ?? "failed")}`);
    }
  }
}

// ─── Phase 2: cache-burst ─────────────────────────────────────────────────────
async function cacheBurst(): Promise<void> {
  console.log(C(BOLD(`\n  Phase 2 — PDIM Cache burst  (${CONCURRENCY} concurrent × ${WAVES} waves)\n`)));
  console.log(DIM(`  Every request uses a payload already seeded in pdim.`));
  console.log(DIM(`  Cache hits bypass INFERENCE_GATE entirely → pdim network speed.\n`));

  const allLatencies: number[] = [];
  let totalRequests = 0;
  let totalErrors   = 0;
  let cacheHits     = 0;
  let computeHits   = 0;

  for (let wave = 1; wave <= WAVES; wave++) {
    process.stdout.write(`  Wave ${wave}/${WAVES}  `);
    const waveStart = Date.now();

    const tasks: Promise<Result>[] = [];
    for (let i = 0; i < CONCURRENCY; i++) {
      tasks.push(fire(CACHE_SLOTS[i % CACHE_SLOTS.length]!));
    }

    const results = await Promise.all(tasks);
    const waveMs  = Date.now() - waveStart;

    const waveLats: number[] = [];
    for (const r of results) {
      totalRequests++;
      allLatencies.push(r.latencyMs);
      waveLats.push(r.latencyMs);
      if (!r.ok) {
        totalErrors++;
        process.stdout.write(R("✗"));
      } else {
        process.stdout.write(G("·"));
        if (r.cached) cacheHits++;
        else computeHits++;
      }
    }

    const waveP50 = percentile([...waveLats].sort((a, b) => a - b), 50);
    console.log(`  ${DIM(`(${waveMs}ms total, p50=${waveP50}ms)`)}`);
  }

  const sorted = [...allLatencies].sort((a, b) => a - b);
  const errPct  = ((totalErrors / totalRequests) * 100).toFixed(1);
  const hitRate = totalRequests > 0 ? (((cacheHits) / (cacheHits + computeHits)) * 100).toFixed(1) : "n/a";

  console.log("\n" + BOLD("═".repeat(80)));
  console.log(BOLD("  Summary"));
  console.log(BOLD("═".repeat(80)));
  console.log(`  Total requests   : ${BOLD(String(totalRequests))}`);
  console.log(`  Errors           : ${totalErrors > 0 ? R(String(totalErrors)) : G("0")}  (${errPct}%)`);
  console.log(`  pdim cache hits  : ${G(String(cacheHits))}`);
  console.log(`  Compute (model)  : ${Y(String(computeHits))}`);
  console.log(`  Cache-hit rate   : ${cacheHits > 0 ? G(hitRate + "%") : DIM("0% (no cache hits)")}`);
  console.log(`  Latency P50      : ${percentile(sorted, 50) < 500 ? G(`${percentile(sorted, 50)}ms`) : Y(`${percentile(sorted, 50)}ms`)}`);
  console.log(`  Latency P95      : ${percentile(sorted, 95) < 1500 ? G(`${percentile(sorted, 95)}ms`) : R(`${percentile(sorted, 95)}ms`)}`);
  console.log(`  Latency P99      : ${percentile(sorted, 99) < 3000 ? G(`${percentile(sorted, 99)}ms`) : R(`${percentile(sorted, 99)}ms`)}`);
  console.log(BOLD("═".repeat(80)));

  if (totalErrors === 0) {
    console.log(G(BOLD("\n  ✓ CACHE BURST PASSED — zero errors\n")));
  } else {
    console.log(R(BOLD(`\n  ✗ ${totalErrors} errors across ${totalRequests} requests\n`)));
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  console.log(BOLD("\n  MaxBooster PDIM Cache-Burst Load Test"));
  console.log(DIM(`  Target : ${BASE}`));
  console.log(DIM(`  Key    : ${ADMIN_KEY.slice(0, 12)}...`));
  console.log(DIM(`  Burst  : ${CONCURRENCY} concurrent × ${WAVES} waves`));
  console.log(DIM(`  Slots  : ${CACHE_SLOTS.length} unique payloads (cycled)`));

  await warmCache();
  await cacheBurst();
}

main().catch(err => {
  console.error(R("\nFatal:"), err);
  process.exit(1);
});
