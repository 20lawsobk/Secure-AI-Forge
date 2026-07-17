/**
 * Shared mutable server state.
 *
 * Written by python-server.ts (the spawn manager) and read by model-proxy.ts
 * (the request router).  Kept in a separate module so neither side imports
 * the other, which would create a circular dependency.
 *
 * Rule: only python-server.ts calls the setters; everyone else calls the
 * getters.
 */

let _pythonRestarting = false;

/**
 * Mark Python as in the middle of a crash-restart cycle.
 * While true, the proxy hold-queue will park incoming requests instead of
 * returning 503 — they drain automatically when setPythonRestarting(false)
 * is called after the warm-up pass succeeds.
 */
export function setPythonRestarting(v: boolean): void {
  const prev = _pythonRestarting;
  _pythonRestarting = v;
  if (v && !prev) {
    console.log("[ServerState] Python restarting — holding incoming requests");
  } else if (!v && prev) {
    console.log("[ServerState] Python ready — releasing held requests");
  }
}

/**
 * True between a Python crash and the completion of its post-restart warm pass.
 * Proxy functions check this to decide whether to hold or forward requests.
 */
export function isPythonRestarting(): boolean {
  return _pythonRestarting;
}
