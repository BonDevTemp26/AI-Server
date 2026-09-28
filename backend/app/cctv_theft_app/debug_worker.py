import time
import sys
from pathlib import Path
import yaml

APP_ROOT = Path('.').resolve()
sys.path.insert(0, str(APP_ROOT))

# dummy _load function
def _load(p):
    return yaml.safe_load(p.read_text())

# Use the app's _LiveWorker
import dashboard.server as server
server.APP_ROOT = APP_ROOT
server.DETECTOR_CFG = APP_ROOT / "configs/detector.yaml"
server.APP_CFG = APP_ROOT / "configs/app.yaml"

worker = server._LiveWorker("id10", "rtsp://bontech:Paris2026%40@212.114.23.247:554/cam/realmonitor?channel=2&subtype=1", detect=True)
time.sleep(10)
print("Worker Status:", worker.status)
if worker.stopped.is_set():
    print("Worker stopped!")
