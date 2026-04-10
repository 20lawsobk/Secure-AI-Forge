import time
import requests

SERVER_URL = "http://127.0.0.1:5055/train"

while True:
    try:
        requests.post(SERVER_URL, json={"tick": True})
    except Exception:
        pass
    time.sleep(60)
