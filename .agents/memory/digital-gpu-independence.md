---
name: DigitalGPU independence from host CPU
description: The DigitalGPU / NumPy SIMD engine is fully self-contained and does not use Replit's CPU resource accounting — throughput scaling math is dedup-ratio-based, not core-count-based.
---

## Rule
The DigitalGPU engine (`ai_model/maxcore/`) is a self-contained compute fabric with its own internal scheduling and memory model. It is 100% independent of the Replit host CPU system — Replit's CPU allocation/limits do not gate its throughput.

**Why:** Confirmed explicitly by the operator. The engine runs NumPy SIMD kernels inside its own allocated memory space using the custom kernel dispatcher (`engine-serve-all-kernels`), not Replit's compute quotas.

**How to apply:**
- Do NOT size optimizations around `os.cpu_count()` as a DigitalGPU throughput ceiling. CPU core math is only relevant for OS-level thread explosion prevention (BLAS thread cap) and I/O-layer concurrency (Node cluster, uvicorn workers).
- The dominant throughput lever at 90M+ scale is **dedup ratio**: `real_computes = total_requests × (1 - cache_hit_rate)`. A 99.99% cache hit rate reduces 90M requests to ~9,000 DigitalGPU calls regardless of host CPU.
- BLAS thread capping (added in 90M optimization session) is still valid to prevent OS thread thrashing, but is not a DigitalGPU performance knob.
- All proxy/cluster/undici optimizations operate entirely outside the DigitalGPU and are pure I/O improvements — they remain fully valid.
- Future GPU-side optimizations should target: dedup key design, cache TTL tuning, cache warm coverage, and the DigitalGPU's own internal batch size / kernel dispatcher settings — not host CPU topology.
