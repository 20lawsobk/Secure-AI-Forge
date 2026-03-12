import queue
import threading
import subprocess
import sys

class GPUScheduler:
    def __init__(self):
        self.jobs = queue.Queue()
        self.active = None
        self.last_exit_code = None
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        while True:
            job = self.jobs.get()
            self.active = job
            try:
                cmd = [sys.executable] + job["cmd"].split()[1:] if job["cmd"].startswith("python ") else job["cmd"].split()
                result = subprocess.run(cmd, shell=False)
                self.last_exit_code = result.returncode
                if result.returncode != 0:
                    print(f"[GPUScheduler] WARNING: job exited with code {result.returncode}: {job['cmd']}", flush=True)
            except Exception as e:
                print(f"[GPUScheduler] ERROR running job {job['cmd']}: {e}", flush=True)
                self.last_exit_code = -1
            finally:
                self.active = None

    def submit(self, script):
        self.jobs.put({"cmd": f"python {script}"})
        return True

    def status(self):
        if self.active:
            return {"status": "running", "job": self.active}
        return {"status": "idle", "last_exit_code": self.last_exit_code}
