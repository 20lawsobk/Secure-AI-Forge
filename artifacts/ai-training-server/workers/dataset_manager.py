import os
import hashlib
import requests
import subprocess
import time

class DatasetManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def ensure_dir(self, dataset):
        path = os.path.join(self.base_dir, dataset)
        os.makedirs(path, exist_ok=True)
        return path

    def download(self, dataset, repo):
        target = self.ensure_dir(dataset)

        # Resume if partial
        if os.path.exists(os.path.join(target, ".incomplete")):
            return self.resume(dataset, repo)

        with open(os.path.join(target, ".incomplete"), "w") as _f:
            pass

        cmd = ["git", "lfs", "clone", repo, target]
        result = subprocess.run(cmd, shell=False)
        if result.returncode != 0:
            print(f"[DatasetManager] ERROR: git lfs clone failed (exit {result.returncode}) for {dataset}")
            return False

        os.remove(os.path.join(target, ".incomplete"))
        return True

    def resume(self, dataset, repo):
        target = self.ensure_dir(dataset)
        cmd = ["git", "-C", target, "lfs", "fetch", "--all"]
        result = subprocess.run(cmd, shell=False)
        if result.returncode != 0:
            print(f"[DatasetManager] ERROR: git lfs fetch failed (exit {result.returncode}) for {dataset}")
            return False
        return True

    def verify(self, dataset):
        # Optional: add checksum logic
        return True
