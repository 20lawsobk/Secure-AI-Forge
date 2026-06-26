#!/usr/bin/env python3
"""HTTP load / concurrency test for the live MaxCore generation endpoints.

Companion to load_test.py (which exercises the maxcore engine in-process).
This harness hits the REAL FastAPI endpoints (audio / image / text / video)
under concurrency and reports, honestly, TWO independent verdicts per phase:

  * CRASH/BACKPRESSURE  — no connection death (-1) and no real 5xx. A 503 from
    the AdaptiveGate is backpressure WORKING, not a failure, so it is allowed.
  * CORRECTNESS         — the endpoint actually produced real output: a cold
    sync call returns 2xx with a non-empty modality artifact (caption/hook for
    text, an image path for image), and a job endpoint returns a job_id whose
    job polls through to COMPLETED with a real artifact (audio_url / video
    filename / scenes_rendered). A 2xx with an empty/again-stubbed body FAILS.

A phase PASSes only when BOTH verdicts pass, so 503/504 shedding can never be
laundered into a green "correctness" result.

Notes on honesty:
  * Most text/image sync responses do not carry a ``source`` field, so fallback
    detection is done by asserting modality output invariants (real artifact
    present), not by trusting a self-reported ``source``. Where ``source`` IS
    present (completed audio/video job records) it is recorded.
  * Unique (cold) compute on this box is bounded by physical cores; this harness
    measures the real cold rate and does not claim more throughput than the
    hardware can serve. Concurrency bursts validate single-flight/dedup collapse
    and gate-bounded queueing, not infinite parallelism.

Usage:  python3 endpoint_load_test.py [text|image|audio|video|all]
Env:    LOAD_BASE (default http://localhost:9878), ADMIN_KEY / AI_TRAINING_KEY_PROD
"""
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.environ.get("LOAD_BASE", "http://localhost:9878")
KEY = os.environ.get("ADMIN_KEY") or os.environ.get("AI_TRAINING_KEY_PROD")
if not KEY:
    print("FATAL: no ADMIN_KEY / AI_TRAINING_KEY_PROD in env")
    sys.exit(2)

RUN = uuid.uuid4().hex[:8]  # per-run namespace so reruns don't bleed cache

# Modality output invariants: a real (non-stub) response must carry at least
# one of these as a non-empty value.
ART_KEYS = {
    "text": ("caption", "hook", "body", "content", "text", "script", "captions"),
    "image": ("image_url", "url", "path", "file", "filename", "image_path"),
    "audio": ("audio_url", "url", "file", "filename", "path"),
    "video": ("filename", "video_url", "url", "file", "path", "scenes_rendered"),
}


def has_real_output(j, kind):
    """True iff the response carries a non-empty modality artifact."""
    if not isinstance(j, dict):
        return False
    for k in ART_KEYS[kind]:
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and k == "scenes_rendered" and v > 0:
            return True
        if isinstance(v, (list, dict)) and v:
            return True
    return False


def is_2xx(status):
    return 200 <= status < 300


def req(method, path, body=None, timeout=120):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"X-Api-Key": KEY}
    if data is not None:
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            dt = time.time() - t0
            try:
                j = json.loads(raw)
            except Exception:
                j = {"_nonjson": raw[:160].decode("utf-8", "replace")}
            return {"status": resp.status, "dt": dt, "json": j}
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        try:
            j = json.loads(e.read())
        except Exception:
            j = {}
        return {"status": e.code, "dt": dt, "json": j}
    except Exception as e:
        return {"status": -1, "dt": time.time() - t0, "json": {"_err": repr(e)[:160]}}


def gates():
    """Gate/dedup snapshot. Not all front-ends proxy this admin route (e.g.
    the Node api-server allow-list omits it), so return a safe skeleton when
    it is unavailable rather than crashing the phase."""
    j = req("GET", "/api/concurrency/stats", timeout=10)["json"]
    if not isinstance(j, dict) or "inference" not in j:
        skel = {"peak_active": "n/a", "capacity": "n/a", "active": "n/a"}
        return {"inference": dict(skel), "render": dict(skel),
                "dedup_cache": {"hits": 0, "misses": 0, "_unavailable": True}}
    return j


def summarize(results):
    codes, sources, keyset = {}, {}, set()
    cached = 0
    lat = sorted(r["dt"] for r in results)
    for r in results:
        codes[r["status"]] = codes.get(r["status"], 0) + 1
        j = r.get("json") or {}
        if isinstance(j, dict):
            keyset.update(j.keys())
            s = j.get("source")
            if s is not None:
                sources[s] = sources.get(s, 0) + 1
            if j.get("cached") is True:
                cached += 1

    def pct(p):
        if not lat:
            return 0.0
        return round(lat[min(len(lat) - 1, int(len(lat) * p))], 3)

    return {
        "n": len(results),
        "codes": codes,
        "sources": sources or "(no source field)",
        "cached_hits": cached,
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": round(max(lat), 3) if lat else 0.0,
        "resp_keys": sorted(keyset),
    }


def burst(method, path, bodies, workers):
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(req, method, path, b) for b in bodies]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def audit_failures(tag, results, allow_503=True):
    """Crash audit: no -1 (connection death) and no 5xx (real server failure).
    503 from the gate IS backpressure working, not a crash."""
    bad = []
    for r in results:
        st = r["status"]
        if st == -1:
            bad.append(("conn", r["json"]))
        elif st >= 500 and not (st == 503 and allow_503):
            bad.append((st, r["json"]))
    ok = not bad
    print(f"  [{tag}] crash/5xx audit: {'PASS (none)' if ok else 'FAIL'}")
    for code, j in bad[:5]:
        print(f"      !! {code}: {j}")
    return ok


def code_counts(results):
    """Split a burst into accepted (2xx w/ job_id), honest 503 backpressure,
    other-4xx, and hard failures (-1 / non-503 5xx)."""
    accepted = n503 = other4xx = hard = 0
    for r in results:
        st = r["status"]
        if is_2xx(st) and find_job_id(r["json"]):
            accepted += 1
        elif st == 503:
            n503 += 1
        elif 400 <= st < 500:
            other4xx += 1
        elif st == -1 or st >= 500:
            hard += 1
    return accepted, n503, other4xx, hard


def poll_job(poll_path, job_id, budget_s, label):
    deadline = time.time() + budget_s
    last = {}
    while time.time() < deadline:
        r = req("GET", poll_path.format(id=job_id), timeout=15)
        last = r["json"] if isinstance(r["json"], dict) else {}
        st = last.get("status") or last.get("state")
        if st in ("completed", "done", "ready", "finished", "complete"):
            print(f"    {label} job {job_id[:8]} -> COMPLETED in budget "
                  f"(source={last.get('source')}, keys={sorted(last.keys())[:8]})")
            return "completed", last
        if st in ("failed", "error"):
            print(f"    {label} job {job_id[:8]} -> FAILED: {last}")
            return "failed", last
        time.sleep(2)
    print(f"    {label} job {job_id[:8]} -> still '{last.get('status')}' "
          f"(progress={last.get('progress')}) after {budget_s}s budget")
    return "pending", last


def find_job_id(j):
    if not isinstance(j, dict):
        return None
    for k in ("job_id", "jobId", "id", "video_job_id", "audio_job_id"):
        if j.get(k):
            return j[k]
    return None


def verdict(name, crash_ok, correct_ok, note=""):
    print(f"  >> {name} verdict: crash/backpressure={'PASS' if crash_ok else 'FAIL'} "
          f"| correctness={'PASS' if correct_ok else 'FAIL'}{(' — ' + note) if note else ''}")
    return crash_ok and correct_ok


# ─────────────────────────────────────────────────────────────────────────
def phase_text():
    print("\n=== PHASE TEXT (/api/generate/content, /api/generate/text) ===")
    g0 = gates()
    print(f"  gates@start inference={g0['inference']} dedup={g0['dedup_cache']}")

    base = {"platform": "tiktok", "topic": f"summer single {RUN}",
            "tone": "energetic", "artist_name": "Nova"}
    cold = req("POST", "/api/generate/content", base, timeout=90)
    cold_ok = is_2xx(cold["status"]) and has_real_output(cold["json"], "text")
    print(f"  cold compute: status={cold['status']} dt={cold['dt']:.2f}s "
          f"real_output={cold_ok} keys={sorted((cold['json'] or {}).keys())[:8]}")

    # identical-body burst -> dedup / single-flight collapse
    N, W = 64, 64
    ident = burst("POST", "/api/generate/content", [dict(base)] * N, W)
    s = summarize(ident)
    print(f"  identical x{N} (w={W}): {s}")
    ok1 = audit_failures("text-identical", ident)

    # unique-body burst -> queueing + backpressure
    uniq_bodies = [dict(base, topic=f"topic {RUN} {i} {uuid.uuid4().hex[:6]}")
                   for i in range(8)]
    uniq = burst("POST", "/api/generate/content", uniq_bodies, 8)
    s2 = summarize(uniq)
    print(f"  unique x8 (w=8): {s2}")
    ok2 = audit_failures("text-unique", uniq)

    # cache-hit throughput
    t0 = time.time()
    hits = burst("POST", "/api/generate/content", [dict(base)] * 120, 32)
    dt = time.time() - t0
    sh = summarize(hits)
    rps = round(120 / dt, 1) if dt else 0
    print(f"  cache-hit throughput: 120 reqs in {dt:.2f}s = {rps} req/s "
          f"(cached_hits={sh['cached_hits']}, codes={sh['codes']})")
    ok3 = audit_failures("text-cachehit", hits)

    # planner mode (/api/generate/text)
    pl = req("POST", "/api/generate/text",
             {"mode": "content", "platform": "instagram",
              "topic": f"album teaser {RUN}", "tone": "bold"}, timeout=90)
    planner_ok = is_2xx(pl["status"]) and has_real_output(pl["json"], "text")
    print(f"  /api/generate/text mode=content: status={pl['status']} "
          f"dt={pl['dt']:.2f}s real_output={planner_ok} "
          f"keys={sorted((pl['json'] or {}).keys())[:8]}")

    g1 = gates()
    dh = g1["dedup_cache"]["hits"] - g0["dedup_cache"]["hits"]
    collapse = (dh > 0) or (s["cached_hits"] > 0) or (sh["cached_hits"] > 0)
    print(f"  gates@end inference={g1['inference']} dedup_hits_delta={dh} "
          f"peak_active={g1['inference']['peak_active']} collapse_observed={collapse}")

    crash_ok = ok1 and ok2 and ok3
    correct_ok = cold_ok and planner_ok and collapse
    note = "" if collapse else "no dedup/single-flight collapse observed"
    return verdict("TEXT", crash_ok, correct_ok, note)


def phase_image():
    print("\n=== PHASE IMAGE (/api/generate/image) ===")
    g0 = gates()
    print(f"  gates@start render={g0['render']} inference={g0['inference']}")
    base = {"prompt": f"neon portrait of artist Nova {RUN}",
            "aspect_ratio": "9:16", "style": "cinematic"}
    cold = req("POST", "/api/generate/image", base, timeout=120)
    cold_ok = is_2xx(cold["status"]) and has_real_output(cold["json"], "image")
    print(f"  cold compute: status={cold['status']} dt={cold['dt']:.2f}s "
          f"real_output={cold_ok} keys={sorted((cold['json'] or {}).keys())[:10]}")

    # identical burst -> seeded file dedup (skip-if-exists)
    N, W = 12, 12
    ident = burst("POST", "/api/generate/image", [dict(base)] * N, W)
    s = summarize(ident)
    print(f"  identical x{N} (w={W}): {s}")
    ok1 = audit_failures("image-identical", ident)
    ident_real = sum(1 for r in ident if is_2xx(r["status"])
                     and has_real_output(r["json"], "image"))

    # a few unique -> real renders + backpressure
    uniq = burst("POST", "/api/generate/image",
                 [dict(base, prompt=f"{base['prompt']} v{i} {uuid.uuid4().hex[:6]}")
                  for i in range(4)], 4)
    s2 = summarize(uniq)
    print(f"  unique x4 (w=4): {s2}")
    ok2 = audit_failures("image-unique", uniq)

    g1 = gates()
    print(f"  gates@end render={g1['render']} peak_active={g1['render']['peak_active']}")

    crash_ok = ok1 and ok2
    correct_ok = cold_ok and ident_real >= 1
    note = "" if correct_ok else f"no real image artifact (ident_real={ident_real})"
    return verdict("IMAGE", crash_ok, correct_ok, note)


def phase_audio():
    print("\n=== PHASE AUDIO (/api/generate/audio -> /api/audio-job/{id}) ===")
    g0 = gates()
    print(f"  gates@start render={g0['render']}")
    body = {"duration": 8, "intent": f"energetic synth hook {RUN}", "genre": "pop"}
    sub = req("POST", "/api/generate/audio", body, timeout=60)
    print(f"  submit: status={sub['status']} dt={sub['dt']:.2f}s "
          f"resp={ {k: sub['json'].get(k) for k in list((sub['json'] or {}).keys())[:6]} }")
    jid = find_job_id(sub["json"])
    submit_ok = is_2xx(sub["status"]) and bool(jid)
    job_ok = False
    if jid:
        state, last = poll_job("/api/audio-job/{id}", jid, budget_s=60, label="audio")
        job_ok = state == "completed" and has_real_output(last, "audio")

    # concurrency: many submits at once -> all accepted OR honest 503, gate bounds renders
    N, W = 12, 12
    subs = burst("POST", "/api/generate/audio",
                 [dict(body, intent=f"{body['intent']} {i}") for i in range(N)], W)
    s = summarize(subs)
    print(f"  concurrent submits x{N} (w={W}): {s}")
    ok2 = audit_failures("audio-submit", subs)
    accepted, n503, other4xx, hard = code_counts(subs)
    print(f"  submit breakdown: accepted={accepted} 503_backpressure={n503} "
          f"other4xx={other4xx} hard_fail={hard} (of {N})")
    g1 = gates()
    print(f"  gates@end render={g1['render']} peak_active={g1['render']['peak_active']}")

    burst_ok = (hard == 0) and (other4xx == 0) and (accepted + n503 == N) and accepted >= 1
    crash_ok = ok2
    correct_ok = submit_ok and job_ok and burst_ok
    note = ""
    if not correct_ok:
        note = (f"submit_ok={submit_ok} job_completed={job_ok} burst_ok={burst_ok}")
    return verdict("AUDIO", crash_ok, correct_ok, note)


def phase_video():
    print("\n=== PHASE VIDEO (/api/generate-video, /api/video/generate-ai) ===")
    g0 = gates()
    print(f"  gates@start render={g0['render']}")
    body = {"idea": f"neon city night drive {RUN}", "platform": "tiktok",
            "duration": 6, "tone": "energetic"}
    sub = req("POST", "/api/generate-video", body, timeout=60)
    print(f"  submit /api/generate-video: status={sub['status']} dt={sub['dt']:.2f}s "
          f"resp={ {k: sub['json'].get(k) for k in list((sub['json'] or {}).keys())[:6]} }")
    jid = find_job_id(sub["json"])
    submit_ok = is_2xx(sub["status"]) and bool(jid)
    job_ok = False
    if jid:
        state, last = poll_job("/api/video-job/{id}", jid, budget_s=75, label="video")
        job_ok = state == "completed" and has_real_output(last, "video")

    # concurrency: several submits -> accepted OR honest 503, no crash
    N, W = 8, 8
    subs = burst("POST", "/api/generate-video",
                 [dict(body, idea=f"{body['idea']} {i}") for i in range(N)], W)
    s = summarize(subs)
    print(f"  concurrent submits x{N} (w={W}): {s}")
    ok2 = audit_failures("video-submit", subs)
    accepted, n503, other4xx, hard = code_counts(subs)
    print(f"  submit breakdown: accepted={accepted} 503_backpressure={n503} "
          f"other4xx={other4xx} hard_fail={hard} (of {N})")

    # AI variant (requires 'idea') -> must return 2xx + a job handle
    ai = req("POST", "/api/video/generate-ai",
             {"idea": f"summer anthem teaser {RUN}", "platform": "tiktok",
              "tone": "energetic", "goal": "growth"}, timeout=60)
    aj = ai["json"] or {}
    ai_ok = is_2xx(ai["status"]) and bool(find_job_id(aj) or aj.get("poll_url"))
    print(f"  submit /api/video/generate-ai: status={ai['status']} dt={ai['dt']:.2f}s "
          f"job_handle={ai_ok} keys={sorted(aj.keys())[:6]}")

    g1 = gates()
    print(f"  gates@end render={g1['render']} peak_active={g1['render']['peak_active']}")

    burst_ok = (hard == 0) and (other4xx == 0) and (accepted + n503 == N) and accepted >= 1
    crash_ok = ok2
    correct_ok = submit_ok and job_ok and burst_ok and ai_ok
    note = ""
    if not correct_ok:
        note = (f"submit_ok={submit_ok} job_completed={job_ok} "
                f"burst_ok={burst_ok} ai_ok={ai_ok}")
    return verdict("VIDEO", crash_ok, correct_ok, note)


PHASES = {"text": phase_text, "image": phase_image,
          "audio": phase_audio, "video": phase_video}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"MaxCore endpoint load test | base={BASE} run={RUN} target={which}")
    todo = list(PHASES) if which == "all" else [which]
    results = {}
    for name in todo:
        try:
            results[name] = PHASES[name]()
        except Exception as e:
            print(f"  PHASE {name} raised: {e!r}")
            results[name] = False
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:6s}: {'PASS' if v else 'FAIL'}")
    print("OVERALL:", "PASS" if all(results.values()) else "FAIL")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
