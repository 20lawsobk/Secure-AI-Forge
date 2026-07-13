"""
Direct endpoint verification — tests every route on the model server (port 9878)
and the api-server proxy (port 8080) with correct minimal payloads.

Exit 0 if every endpoint returns an expected status; non-zero otherwise.
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
from dataclasses import dataclass

# ── Config ────────────────────────────────────────────────────────────────────
PY_HOST  = "127.0.0.1";  PY_PORT  = 9878
API_HOST = "127.0.0.1";  API_PORT = 8080

API_KEY  = "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
HDR      = {"Content-Type": "application/json", "X-Api-Key": API_KEY}

PASS, FAIL = "✓", "✗"
TIMEOUT    = 60

# ── HTTP helper ───────────────────────────────────────────────────────────────
def _req(host, port, method, path, body=None):
    try:
        conn = http.client.HTTPConnection(host, port, timeout=TIMEOUT)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=HDR)
        r = conn.getresponse()
        raw = r.read()
        try:    return r.status, json.loads(raw)
        except: return r.status, raw.decode(errors="replace")[:200]
    except Exception as e:
        return 0, str(e)

def py(m, p, b=None):  return _req(PY_HOST,  PY_PORT,  m, p, b)
def api(m, p, b=None): return _req(API_HOST, API_PORT, m, p, b)

# ── Result store ──────────────────────────────────────────────────────────────
@dataclass
class R:
    sym: str; server: str; method: str; path: str; status: int; note: str = ""

results: list[R] = []
_lock = threading.Lock()

def check(server, method, path, body, expect, fn, note=""):
    status, resp = fn(method, path, body)
    ok  = status in expect
    sym = PASS if ok else FAIL
    detail = note if ok else f"got {status}; expected {expect}; {str(resp)[:140]}"
    with _lock:
        results.append(R(sym, server, method, path, status, detail))

def parallel(cases):
    ts = [threading.Thread(target=lambda c=c: check(*c)) for c in cases]
    for t in ts: t.start()
    for t in ts: t.join()

# ── Payloads ──────────────────────────────────────────────────────────────────
UID      = "test_user_001"

# Content generation
CONTENT  = {"topic": "digital gpu test", "platform": "tiktok"}
CONTENT_T= {"topic": "digital gpu test", "platform": "tiktok", "tone": "authentic"}
AUDIO    = {"topic": "gpu beat", "platform": "tiktok", "seed": 42}
IMAGE    = {"topic": "digital gpu visual", "platform": "instagram"}
TEXT     = {"mode": "content", "topic": "gpu power", "platform": "twitter"}
VIDEO    = {"topic": "gpu showcase", "platform": "youtube", "user_id": UID}

# Campaigns
CAMPAIGN = {"title": "GPU Anthem", "artist_name": "MaxCore", "platforms": ["tiktok"]}
CAM_SAVE = {"profile_id": "test_profile_001", "plan": {"title": "GPU Anthem", "posts": []}}

# Scoring / analysis
SCORE    = {"text": "🔥 Drop everything. This track redefines the game.", "platform": "tiktok"}
ENGAGE   = {"platform": "tiktok", "action": "post",
            "content": {"hook": "Fire beat dropped", "body": "Stream now", "cta": "Link in bio"}}
VIRAL    = {"content": "This track hits different — pure fire from start to finish 🔥"}
ANALYZE  = {"modality": "text", "payload": "Amazing track with great energy"}
SENTIMENT= {"text": "This track slaps, pure fire energy"}
AUD_ANA  = {"audio_url": "https://example.com/test.mp3"}
AD_OPT   = {"action": "score", "campaign": {"name": "GPU Anthem", "platform": "tiktok"}}
AD_OPT2  = {"ad_copy": "Stream now", "platform": "tiktok", "goal": "clicks"}  # /platform/ads/optimize

# Platform endpoints (require user_id)
SOCIAL   = {"user_id": UID, "platform": "tiktok", "topic": "gpu track", "n_variants": 1}
AUTO_PIL = {"user_id": UID, "platform": "tiktok", "target_metric": "engagement"}
DIST     = {"user_id": UID, "track_title": "GPU Anthem", "genre": "hip-hop",
            "target_platforms": ["spotify"]}
DAW      = {"user_id": UID, "mode": "lyrics", "genre": "hip-hop", "mood": "energetic"}
AD       = {"user_id": UID, "platform": "meta", "product": "music single", "goal": "streams"}
AD_AUTO  = {"user_id": UID, "platform": "tiktok", "product": "single", "goal": "streams"}
AD_REC   = {"user_id": UID, "platform": "instagram", "ad_type": "video",
            "hook": "Stream now", "headline": "New drop", "cta": "Link in bio"}

# Feedback / training
SAFETY   = {"text": "great track with real emotion and energy"}
TRAIN_FB = {"content": "great hook", "platform": "tiktok", "rating": 5}      # /api/train/feedback (server.py)
API_FB   = {"source": "autopilot", "trigger": "high_engagement",               # /api/train/feedback (ApiTrainFeedbackRequest)
            "engagement_rate": 0.08, "platform": "tiktok",
            "content_type": "post", "hook_type": "question", "media_type": "video"}
STOR_FB  = {"user_id": UID, "platform": "tiktok", "engagement_rate": 5.5}    # CurriculumFeedback

# Video AI
VID_AI   = {"idea": "GPU-powered beat drop showcase", "platform": "tiktok",
            "user_id": UID, "topic": "digital gpu"}

# Misc
MULTIPLY = {"a": [[1.0,2.0],[3.0,4.0]], "b": [[5.0,6.0],[7.0,8.0]], "namespace": "test_ep_verify"}
FEED     = {"content": "fire drop", "platform": "tiktok", "rating": 5}
BPE      = {"max_steps": 1}
HYPER    = {"epochs": 1, "max_samples": 2}

# ── Test tables ───────────────────────────────────────────────────────────────

# Python server — GET, no path params
PY_GETS = [
    "/health", "/api/health", "/model/status", "/gpu/status",
    "/gpu/hyper/status", "/gpu/capabilities", "/dashboard/stats",
    "/boostsheets", "/platform/model/info", "/storage/status",
    "/storage/datasets", "/storage/session", "/storage/pipeline/status",
    "/training/status", "/training/datasets", "/training/logs",
    "/training/continuous/status", "/training/continuous/history",
    "/training/puller/status", "/training/puller/sources",
    "/watchdog/status", "/watchdog/log",
    "/api/concurrency/stats", "/api/models/social/state",
    "/api/models/advertising/state", "/api/models/content/state",
    "/api/models/engagement/state", "/api/rta/status",
    "/api/maxcore/pocket-accelerator/stats",
    "/api/video-jobs", "/api/awareness/quality/status",
    "/coverage/status", "/coverage/report", "/coverage/log", "/coverage/ingestor",
]

# Admin-only endpoints — 401 without admin key IS correct behaviour
PY_GETS_ADMIN = ["/api-keys", "/storage/checkpoints"]

# GET with path params — 200 or 404 both valid; admin-gated also 401
PY_GETS_PARAM = [
    ("/storage/artist/test_artist_001",                          (200, 404)),
    ("/storage/checkpoint/base_model",                           (200, 401, 404)),
    ("/storage/curriculum/test_user_001",                        (200, 404)),
    ("/api/audio-job/nonexistent-id",                            (200, 404)),
    ("/api/video-job/nonexistent-id",                            (200, 404)),
    ("/api/campaigns/nonexistent-id?profile_id=test_profile_001",(200, 404)),
    ("/platform/ads/performance/test_user",                      (200, 404)),
    ("/platform/video/generate",                                 (200, 404, 405, 422)),
    # veo endpoints 404 until maxbooster_veo_music module is installed
    ("/veo/status",          (200, 404)),
    ("/veo/platforms",       (200, 404)),
    ("/veo/goals",           (200, 404)),
    ("/veo/recommend/tiktok",(200, 404)),
]

# Python server — POST (parallel)
PY_POSTS = [
    ("/content/generate",            CONTENT,   (200,)),
    ("/generate/text",               TEXT,      (200,)),
    ("/generate/image",              IMAGE,     (200,)),
    ("/generate/audio",              AUDIO,     (200,)),
    ("/generate/video",              VIDEO,     (200, 202)),
    ("/api/generate/content",        CONTENT_T, (200,)),
    ("/api/generate/text",           TEXT,      (200,)),
    ("/api/generate/image",          IMAGE,     (200,)),
    ("/api/generate/audio",          AUDIO,     (200,)),
    ("/api/generate/campaign",       CAMPAIGN,  (200, 202)),
    ("/generate/video",              VIDEO,     (200, 202, 503)),
    ("/api/generate-video",          VID_AI,    (200, 202, 503)),
    ("/api/infer/viral-score",       VIRAL,     (200,)),
    ("/api/predict/engagement",      ENGAGE,    (200,)),
    ("/api/content/score",           SCORE,     (200,)),
    ("/api/safety/screen",           SAFETY,    (200,)),
    ("/api/train/feedback",          API_FB,    (200,)),
    ("/api/optimize/ad",             AD_OPT,    (200,)),
    ("/api/analyze/audio",           AUD_ANA,   (200,)),
    ("/api/awareness/quality/harvest", {},       (200, 202, 401)),  # admin-gated
    ("/api/maxcore/pocket-multiply", MULTIPLY,  (200,)),
    ("/analyze",                     ANALYZE,   (200,)),
    ("/api/analyze",                 ANALYZE,   (200,)),
    ("/api/analyze/sentiment",       SENTIMENT, (200,)),
    ("/api/audio/analyze",           AUDIO,     (200,)),
    ("/platform/social/generate",    SOCIAL,    (200,)),
    ("/platform/social/autopilot",   AUTO_PIL,  (200,)),
    ("/platform/video/generate",     VIDEO,     (200, 202)),
    ("/platform/ads/generate",       AD,        (200,)),
    ("/platform/ads/audience",       AD,        (200,)),
    ("/platform/ads/autopilot",      AD_AUTO,   (200, 202)),
    ("/platform/ads/optimize",       AD_OPT2,   (200,)),
    ("/platform/ads/record",         AD_REC,    (200,)),
    ("/platform/daw/generate",       DAW,       (200,)),
    ("/platform/distribution/plan",  DIST,      (200,)),
    ("/api/campaigns",               CAM_SAVE,  (200, 201)),
    ("/api/video/generate-ai",       VID_AI,    (200, 202)),
    ("/storage/feedback",            STOR_FB,   (200,)),
    ("/watchdog/reset",              {},        (200,)),
    ("/coverage/reset",              {},        (200,)),
    # veo module not installed — 404 expected
    ("/veo/campaign",     CONTENT,                      (200, 404, 500)),
    ("/veo/url/metadata", {"url": "https://example.com"}, (200, 404, 422)),
    ("/veo/url/campaign", {"url": "https://example.com"}, (200, 404, 422)),
]

# Admin-only POST — 401 without admin key IS correct
PY_POSTS_ADMIN = [
    ("/platform/model/reload",      {}),
    ("/storage/datasets/register",  {"name": "test_ep", "path": "/tmp", "type": "text"}),
]

# Stateful / slow — run serially to avoid interference
PY_POSTS_SLOW = [
    ("/training/puller/pull",      {},    (200, 202)),
    ("/admin/train-bpe-scaleup",   BPE,   (200, 202)),
    ("/admin/train-hyper-scaleup", HYPER, (200, 202)),
]

# ── API-server proxy (8080) ───────────────────────────────────────────────────
# App is mounted at /api in Express so paths below include that prefix
API_GETS = [
    ("/api/health",                      (200,)),
    ("/api/model/status",                (200,)),
    ("/api/gpu/status",                  (200,)),
    ("/api/gpu/hyper/status",            (200,)),
    ("/api/gpu/capabilities",            (200,)),
    ("/api/dashboard/stats",             (200,)),
    ("/api/training/status",             (200,)),
    ("/api/training/continuous/status",  (200,)),
    ("/api/training/continuous/history", (200,)),
    ("/api/training/logs",               (200,)),
    ("/api/training/puller/status",      (200,)),
    ("/api/training/puller/sources",     (200,)),
    ("/api/watchdog/status",             (200,)),
    ("/api/watchdog/log",                (200,)),
    ("/api/storage/status",              (200,)),
    ("/api/storage/datasets",            (200,)),
    ("/api/storage/checkpoints",         (200, 401)),    # admin-gated; 401 is correct
    ("/api/storage/session",             (200,)),
    ("/api/storage/pipeline/status",     (200,)),
    ("/api/models/social/state",         (200,)),
    ("/api/models/advertising/state",    (200,)),
    ("/api/models/content/state",        (200,)),
    ("/api/models/engagement/state",     (200,)),
    ("/api/rta/status",                  (200,)),
    ("/api/concurrency/stats",           (200,)),
    ("/api/awareness/quality/status",    (200,)),
    ("/api/video-jobs",                  (200,)),
    ("/api/campaigns?profile_id=test_profile_001", (200,)),
    ("/api/boostsheets",                 (200,)),
]

API_POSTS = [
    ("/api/content/generate",            CONTENT,   (200,)),
    ("/api/generate/content",            CONTENT_T, (200,)),   # needs tone field
    ("/api/generate/audio",              AUDIO,     (200,)),
    ("/api/generate/image",              IMAGE,     (200,)),
    ("/api/generate/campaign",           CAMPAIGN,  (200, 202)),
    ("/api/predict/engagement",          ENGAGE,    (200,)),
    ("/api/infer/viral-score",           VIRAL,     (200,)),
    ("/api/content/score",               SCORE,     (200,)),
    ("/api/safety/screen",               SAFETY,    (200,)),
    ("/api/analyze",                     ANALYZE,   (200,)),
    ("/api/analyze/sentiment",           SENTIMENT, (200,)),
    ("/api/analyze/audio",               AUD_ANA,   (200,)),
    ("/api/optimize/ad",                 AD_OPT,    (200,)),
    ("/api/platform/social/generate",    SOCIAL,    (200,)),
    ("/api/platform/ads/generate",       AD,        (200,)),
    ("/api/platform/ads/audience",       AD,        (200,)),
    ("/api/platform/ads/optimize",       AD_OPT2,   (200,)),
    ("/api/platform/distribution/plan",  DIST,      (200,)),
    ("/api/platform/daw/generate",       DAW,       (200,)),
    ("/api/platform/video/generate",     VIDEO,     (200, 202, 503)),
    ("/api/generate/video",              VID_AI,    (200, 202, 503)),
    ("/api/generate-video",              VID_AI,    (200, 202, 503)),
    ("/api/video/generate-ai",           VID_AI,    (200, 202, 503)),
    ("/api/train/feedback",              API_FB,    (200,)),
    ("/api/storage/feedback",            STOR_FB,   (200,)),
    ("/api/campaigns",                   CAM_SAVE,  (200, 201)),
    ("/api/watchdog/reset",              {},        (200,)),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")

def show(rs):
    for r in rs:
        note = f"  [{r.note}]" if r.sym == FAIL else ""
        print(f"  {r.sym}  {r.method:<6} {r.path:<56} {r.status}{note}")

def show_sorted(rs):
    for r in sorted(rs, key=lambda r: r.path):
        note = f"  [{r.note}]" if r.sym == FAIL else ""
        print(f"  {r.sym}  {r.method:<6} {r.path:<56} {r.status}{note}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("═"*68)
    print("  Endpoint Verification — All Routes")
    print(f"  Python model server : {PY_HOST}:{PY_PORT}")
    print(f"  API proxy server    : {API_HOST}:{API_PORT}")
    print("═"*68)

    # Python GET — no params
    section(f"Python server GET (no params) — {len(PY_GETS)} endpoints")
    b = len(results)
    parallel([("python","GET",p,None,(200,),py) for p in PY_GETS])
    show(results[b:])

    # Python GET — admin-only (401 = correct)
    section(f"Python server GET (admin-only, 401=correct) — {len(PY_GETS_ADMIN)} endpoints")
    b = len(results)
    parallel([("python","GET",p,None,(200,401),py) for p in PY_GETS_ADMIN])
    show(results[b:])

    # Python GET — path params
    section(f"Python server GET (path params) — {len(PY_GETS_PARAM)} endpoints")
    b = len(results)
    parallel([("python","GET",p,None,exp,py) for p,exp in PY_GETS_PARAM])
    show(results[b:])

    # Python POST — parallel
    section(f"Python server POST — {len(PY_POSTS)} endpoints")
    b = len(results)
    parallel([("python","POST",p,body,exp,py) for p,body,exp in PY_POSTS])
    show_sorted(results[b:])

    # Python POST — admin-only (401 = correct)
    section(f"Python server POST (admin-only, 401=correct) — {len(PY_POSTS_ADMIN)} endpoints")
    b = len(results)
    parallel([("python","POST",p,body,(200,401),py) for p,body in PY_POSTS_ADMIN])
    show(results[b:])

    # Python POST — stateful/slow (serial)
    section(f"Python server POST (stateful) — {len(PY_POSTS_SLOW)} endpoints")
    b = len(results)
    for p,body,exp in PY_POSTS_SLOW:
        check("python","POST",p,body,exp,py)
    show(results[b:])

    # API proxy GET
    section(f"API-proxy GET — {len(API_GETS)} endpoints")
    b = len(results)
    ts = []
    for p,exp in API_GETS:
        def _g(p=p, exp=exp): check("api","GET",p,None,exp,api)
        ts.append(threading.Thread(target=_g))
    for t in ts: t.start()
    for t in ts: t.join()
    show(results[b:])

    # API proxy POST
    section(f"API-proxy POST — {len(API_POSTS)} endpoints")
    b = len(results)
    ts = []
    for p,body,exp in API_POSTS:
        def _p(p=p,body=body,exp=exp): check("api","POST",p,body,exp,api)
        ts.append(threading.Thread(target=_p))
    for t in ts: t.start()
    for t in ts: t.join()
    show_sorted(results[b:])

    # Summary
    total  = len(results)
    passed = sum(1 for r in results if r.sym == PASS)
    failed = sum(1 for r in results if r.sym == FAIL)

    print(f"\n{'═'*68}")
    print(f"  Summary")
    print(f"{'═'*68}")
    print(f"  Total     : {total}")
    print(f"  {PASS} Passed  : {passed}")
    print(f"  {FAIL} Failed  : {failed}")

    if failed:
        print(f"\n  Failed endpoints:")
        for r in results:
            if r.sym == FAIL:
                print(f"    [{r.server}]  {r.method}  {r.path}")
                print(f"           {r.note}")

    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"\n  VERDICT: {verdict} — {passed}/{total} endpoints responded as expected")
    print("═"*68)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
