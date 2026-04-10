import time
import subprocess
import requests

SERVER_URL = "http://127.0.0.1:5055/gpu"

def get_gpu_stats():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        util, mem_used, mem_total = result.strip().split(", ")
        return {
            "utilization": int(util),
            "memory_used": int(mem_used),
            "memory_total": int(mem_total)
        }
    except Exception:
        return {"error": "GPU not found"}

while True:
    stats = get_gpu_stats()
    try:
        requests.post(SERVER_URL, json=stats)
    except Exception:
        pass
    time.sleep(5)
