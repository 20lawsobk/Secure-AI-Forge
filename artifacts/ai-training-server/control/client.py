import os
import requests
import time

BASE_URL = "https://maxbooster.replit.app"
API_KEY = os.environ.get("CONTROL_DAEMON_API_KEY", "0d044c92899b4694d9339e01ea12c7f0862ce6f005aeb9cbbaefdd7d327b07f3")

def send(endpoint, payload=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for _ in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            return r.json()
        except Exception:
            time.sleep(1)

    return {"error": "connection_failed"}

def start_download(dataset, repo):
    return send("/start_download", {"dataset": dataset, "repo": repo})

def start_training(script):
    return send("/start_training", {"script": script})

def server_status():
    try:
        r = requests.get(f"{BASE_URL}/status")
        return r.json()
    except Exception:
        return {"status": "offline"}
