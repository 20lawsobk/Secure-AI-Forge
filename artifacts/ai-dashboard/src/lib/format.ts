export function formatUptime(seconds: number | undefined | null): string {
  if (seconds == null) return "—";
  if (seconds >= 3600) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  }
  return `${Math.floor(seconds)}s`;
}

export function formatOps(ops: number | undefined | null): string {
  if (ops == null) return "—";
  if (ops >= 1e12) return `${(ops / 1e12).toFixed(2)}T`;
  if (ops >= 1e9) return `${(ops / 1e9).toFixed(2)}G`;
  if (ops >= 1e6) return `${(ops / 1e6).toFixed(2)}M`;
  return String(ops);
}

export function formatNumber(n: number | undefined | null): string {
  if (n == null) return "—";
  return n.toLocaleString();
}

export function formatMs(ms: number | undefined | null): string {
  if (ms == null) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}
