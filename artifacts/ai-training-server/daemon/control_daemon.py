from fastapi import FastAPI, HTTPException, Request
import uvicorn
import subprocess
import os
import threading
import time

DATASET_DIR = os.environ.get("D_DRIVE_DATASET_DIR", r"D:\ai_server\datasets")
LOG_FILE    = os.environ.get("DAEMON_LOG_FILE",    r"D:\ai_server\logs\control.log")
API_KEY     = os.environ.get("CONTROL_DAEMON_API_KEY",
                             "0d044c92899b4694d9339e01ea12c7f0862ce6f005aeb9cbbaefdd7d327b07f3")

app = FastAPI()


def log(msg):
    line = f"[{time.ctime()}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def auth(request: Request):
    key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_command(method: str, source: str, target_dir: str, extra: dict) -> list[str]:
    """Build the correct shell command for each download method."""
    if method == "huggingface":
        cmd = ["huggingface-cli", "download", source,
               "--repo-type", "dataset",
               "--local-dir", target_dir]
        ignore = extra.get("ignore_patterns", [])
        for pat in ignore:
            cmd += ["--ignore-patterns", pat]
        return cmd

    elif method == "http":
        fname = source.split("/")[-1].split("?")[0] or "download"
        out_path = os.path.join(target_dir, fname)
        return ["curl", "-L", "-C", "-", "--retry", "5",
                "--retry-delay", "3", "-o", out_path, source]

    elif method == "ytdlp":
        max_clips = extra.get("max_clips", 200)
        duration  = extra.get("duration", 10)
        clips_dir = os.path.join(target_dir, "clips")
        return ["yt-dlp",
                f"ytsearch{max_clips}:{source}",
                "-o", os.path.join(clips_dir, "%(id)s.%(ext)s"),
                "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]/best",
                "--match-filter", f"duration <= {duration * 3}",
                "--max-downloads", str(max_clips),
                "--ignore-errors",
                "--no-warnings"]

    else:
        # fallback: git lfs clone
        return ["git", "lfs", "clone", source, target_dir]


@app.get("/status")
def status():
    return {"status": "online", "dataset_dir": DATASET_DIR}


@app.post("/start_download")
async def start_download(payload: dict, request: Request):
    auth(request)
    dataset = payload.get("dataset")
    source  = payload.get("repo") or payload.get("source")
    method  = payload.get("method", "huggingface")
    extra   = payload.get("extra", {})
    dest    = payload.get("dest", DATASET_DIR)

    if not dataset or not source:
        raise HTTPException(status_code=400, detail="dataset and repo are required")

    target_dir = os.path.join(dest, dataset)
    os.makedirs(target_dir, exist_ok=True)

    cmd = build_command(method, source, target_dir, extra)
    log(f"[START] {dataset} | method={method} | dest={target_dir}")
    log(f"  CMD: {' '.join(cmd)}")

    def run():
        try:
            result = subprocess.run(cmd, cwd=dest, capture_output=True, text=True)
            if result.returncode == 0:
                log(f"[DONE] {dataset}")
            else:
                log(f"[FAIL] {dataset}: {result.stderr[:300]}")
        except Exception as e:
            log(f"[ERROR] {dataset}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "dataset": dataset, "method": method, "dest": target_dir}


@app.post("/start_training")
async def start_training(payload: dict, request: Request):
    auth(request)
    script = payload.get("script")
    if not script:
        raise HTTPException(status_code=400, detail="script is required")

    log(f"[TRAIN] Starting: {script}")

    def run():
        result = subprocess.run(["python", script], capture_output=True, text=True)
        log(f"[TRAIN] Finished: {script} (code {result.returncode})")
        if result.returncode != 0:
            log(f"  stderr: {result.stderr[:300]}")

    threading.Thread(target=run, daemon=True).start()
    return {"status": "training_started", "script": script}


if __name__ == "__main__":
    log(f"Control Daemon starting — datasets → {DATASET_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=5055)
