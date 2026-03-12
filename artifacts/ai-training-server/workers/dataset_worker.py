import time
import os
import requests

DATASET_DIR = "D:/ai_server/datasets"
SERVER_URL = "http://127.0.0.1:5055/dataset_event"

def scan():
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files

known = set(scan())

while True:
    current = set(scan())
    new_files = current - known
    if new_files:
        for f in new_files:
            requests.post(SERVER_URL, json={"event": "new_file", "path": f})
    known = current
    time.sleep(3)
