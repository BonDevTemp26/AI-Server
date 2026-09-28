"""Unified operations dashboard — navbar over the four pipeline stages + review.

    pip install fastapi uvicorn
    python -m dashboard.server            # http://localhost:8080

Pages (one per README_Architec.md phase):

    /            Live Streaming                     add RTSP cameras, watch live MJPEG
    /detection   Per-Frame Detection & Tracking     detector + tracker selection
    /temporal    Temporal Understanding             action-recognition + anomaly model
    /vlm         VLM Verification                   verifier model selection
    /review      Alert Pipeline · Human Review      confirm/reject candidate alerts

Selections are written straight into the runtime configs (``configs/*.yaml``)
with line-level edits that keep the comments intact, so the dashboard and
``python main.py`` always agree. Config changes apply on the next pipeline
start. The review endpoints are the same feedback loop the retraining
pipeline consumes (``data/review/labels.jsonl``).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:                 # `ingest.*` imports for live preview
    sys.path.insert(0, str(APP_ROOT))

APP_CFG = APP_ROOT / "configs/app.yaml"
DETECTOR_CFG = APP_ROOT / "configs/detector.yaml"
TRACKER_CFG = APP_ROOT / "configs/tracker.yaml"
TEMPORAL_CFG = APP_ROOT / "configs/temporal_model.yaml"
VLM_CFG = APP_ROOT / "configs/vlm_verifier.yaml"

PAGES_DIR = Path(__file__).resolve().parent / "pages"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CCTV Theft Detection Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _paths_cfg() -> dict:
    return _load(APP_CFG).get("paths", {})


# ── comment-preserving YAML edits ────────────────────────────────────────────
# yaml.safe_dump would erase the extensive docs in configs/*.yaml, so writes
# are targeted line edits: replace `key: value` in place, append when missing.

def _yaml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))      # double-quoted YAML flow scalar


def _set_top_key(text: str, key: str, value) -> str:
    line = f"{key}: {_yaml_value(value)}"
    new, n = re.subn(rf"(?m)^{re.escape(key)}:[^\n]*$", line, text, count=1)
    if n:
        return new
    return text.rstrip("\n") + f"\n{line}\n"


def _set_nested_key(text: str, parent: str, key: str, value) -> str:
    """Set ``parent:\\n  key: value`` (first match inside the parent block)."""
    m = re.search(rf"(?m)^{re.escape(parent)}:[ \t]*(#[^\n]*)?$", text)
    if not m:
        return text + f"\n{parent}:\n  {key}: {_yaml_value(value)}\n"
    end = re.search(r"(?m)^\S", text[m.end():])            # next top-level key
    block_end = m.end() + (end.start() if end else len(text[m.end():]))
    block = text[m.end():block_end]
    new_block, n = re.subn(rf"(?m)^(\s+){re.escape(key)}:[^\n]*$",
                           rf"\g<1>{key}: {_yaml_value(value)}", block, count=1)
    if not n:
        new_block = f"\n  {key}: {_yaml_value(value)}" + block
    return text[:m.end()] + new_block + text[block_end:]


def _edit_config(path: Path, top: dict | None = None,
                 nested: list[tuple[str, str, object]] | None = None) -> None:
    text = path.read_text()
    for key, value in (top or {}).items():
        text = _set_top_key(text, key, value)
    for parent, key, value in (nested or []):
        text = _set_nested_key(text, parent, key, value)
    path.write_text(text)


def _write_cameras(cameras: list[dict]) -> None:
    """Regenerate the ``cameras:`` block of app.yaml, leaving the rest intact."""
    lines = ["cameras:"]
    for cam in cameras:
        lines += [f"  - id: {_yaml_value(cam['id'])}",
                  f"    source: {_yaml_value(cam['source'])}",
                  f"    loop: {_yaml_value(bool(cam.get('loop', False)))}"]
    block = "\n".join(lines) + "\n\n"
    text = APP_CFG.read_text()
    new, n = re.subn(r"(?ms)^cameras:.*?(?=^\w)", block, text, count=1)
    if not n:
        new = block + text
    APP_CFG.write_text(new)


# ── model catalogs (dropdown options → config values) ────────────────────────

DETECTORS = [
    {"id": "yolo26", "label": "YOLO26", "model": "yolo26s.pt", "runtime": "ready",
     "note": "YOLO26-S — NMS-free, edge-optimised (project default)."},
    {"id": "yolo11", "label": "YOLO11", "model": "yolo11s.pt", "runtime": "ready",
     "note": "YOLO11-S — stable Ultralytics fallback."},
    {"id": "rtdetr", "label": "RT-DETR", "model": "rtdetr-l.pt", "runtime": "ready",
     "note": "RT-DETR-L — transformer detector, runs via Ultralytics."},
    {"id": "rfdetr", "label": "RF-DETR", "model": "rf-detr-base.pth", "runtime": "adapter",
     "note": "Roboflow RF-DETR — needs the `rfdetr` package and an adapter in "
             "detection/ (current backend loads Ultralytics models)."},
    {"id": "dfine", "label": "D-FINE", "model": "dfine-s-coco.pth", "runtime": "adapter",
     "note": "D-FINE — needs a HF-transformers adapter in detection/ "
             "(current backend loads Ultralytics models)."},
]

TRACKERS = [
    {"id": "bytetrack", "label": "ByteTrack", "runtime": "ready",
     "note": "Two-stage association; CCTV-tuned thresholds in tracker.yaml."},
    {"id": "botsort", "label": "BoT-SORT", "runtime": "ready",
     "note": "ByteTrack + camera-motion compensation (optional ReID). "
             "Runs via Ultralytics."},
]

# Extra keys Ultralytics requires when tracker_type is botsort.
BOTSORT_EXTRAS = {"gmc_method": "sparseOptFlow", "proximity_thresh": 0.5,
                  "appearance_thresh": 0.25, "with_reid": False,
                  "model": "osnet_x0_25_msmt17.pt"}

ACTION_MODELS = [
    {"id": "videomaev2", "label": "VideoMAEv2", "mode": "novelty", "runtime": "ready",
     "note": "K710-distilled backbone + per-camera novelty score (mode: novelty). "
             "Flags 'behaviour changed' — pair with VLM verification."},
    {"id": "slowfast", "label": "SlowFast", "mode": None, "runtime": "adapter",
     "note": "Needs a TemporalModel adapter in temporal/ (see temporal/base.py). "
             "Selection is recorded; pipeline keeps its current mode until then."},
    {"id": "x3d", "label": "X3D", "mode": None, "runtime": "adapter",
     "note": "Needs a TemporalModel adapter in temporal/ (see temporal/base.py). "
             "Selection is recorded; pipeline keeps its current mode until then."},
    {"id": "finetuned", "label": "Fine-tuned Model", "mode": "finetuned", "runtime": "ready",
     "note": "Self-describing checkpoint from Fine_tune_dev "
             "(training/train_videomae.py) — real theft-class probabilities."},
]

ANOMALY_MODELS = [
    {"id": "off", "label": "Disabled", "runtime": "ready",
     "note": "Trigger score comes from the action model alone."},
    {"id": "rtfm", "label": "RTFM", "runtime": "ready",
     "note": "RTFM anomaly head over backbone features (temporal/rtfm_infer.py); "
             "needs a trained head checkpoint (rtfm.checkpoint)."},
    {"id": "streamvad", "label": "StreamVAD", "runtime": "adapter",
     "note": "Streaming video anomaly detection — port planned in "
             "temporal/base.py. Selection is recorded; RTFM head stays off."},
]

VLM_MODELS = [
    {"id": "qwen2.5-vl-72b", "label": "Qwen2.5-VL 72B (hosted)", "runtime": "ready",
     "backend": "openai_compatible", "model": "Qwen/Qwen2.5-VL-72B-Instruct",
     "note": "HF Inference Providers router — free monthly credits, needs "
             "HF_TOKEN in .env (project default)."},
    {"id": "qwen3-vl-8b", "label": "Qwen3-VL 8B (hosted)", "runtime": "ready",
     "backend": "openai_compatible", "model": "Qwen/Qwen3-VL-8B-Instruct",
     "note": "Lighter/newer hosted option on the same HF router."},
    {"id": "qwen2.5-vl-7b-local", "label": "Qwen2.5-VL 7B (local)", "runtime": "ready",
     "backend": "hf_local", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
     "quantize_4bit": False, "note": "Fully local, no quota — needs a 16 GB+ GPU."},
    {"id": "qwen2.5-vl-3b-local", "label": "Qwen2.5-VL 3B (local, 4-bit)", "runtime": "ready",
     "backend": "hf_local", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
     "quantize_4bit": True, "note": "Fully local in ~3 GB VRAM via bitsandbytes."},
    {"id": "smolvlm2-500m-local", "label": "SmolVLM2 500M Video (local CPU)", "runtime": "ready",
     "backend": "hf_local", "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
     "quantize_4bit": False, "note": "Video-tuned 500M model — runs on CPU "
                                     "(~1–2 min per clip), zero cost."},
    {"id": "llava-onevision-7b-local", "label": "LLaVA-OneVision 7B (local)", "runtime": "ready",
     "backend": "hf_local", "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
     "quantize_4bit": False, "note": "Open-source alternative family — needs a "
                                     "16 GB+ GPU."},
]


def _match_detector(model: str) -> str:
    for opt in DETECTORS:
        if opt["model"] == model:
            return opt["id"]
    slug = model.lower().replace("-", "").replace("_", "")
    for opt in DETECTORS:
        if opt["id"].replace("-", "") in slug:
            return opt["id"]
    return "custom"


def _checkpoint_info(path_str: str | None) -> dict:
    if not path_str:
        return {"path": None, "found": False}
    p = Path(path_str)
    if not p.is_absolute():
        p = APP_ROOT / p
    return {"path": path_str, "found": p.is_file()}


# ── page routes ──────────────────────────────────────────────────────────────

# NOTE: fixed page names only — route handlers must take no parameters, or
# FastAPI would expose them as query params (→ path traversal on PAGES_DIR).

def _page(name: str) -> HTMLResponse:
    return HTMLResponse((PAGES_DIR / name).read_text())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page_live():
    return _page("live.html")


@app.get("/detection", response_class=HTMLResponse, include_in_schema=False)
def page_detection():
    return _page("detection.html")


@app.get("/temporal", response_class=HTMLResponse, include_in_schema=False)
def page_temporal():
    return _page("temporal.html")


@app.get("/vlm", response_class=HTMLResponse, include_in_schema=False)
def page_vlm():
    return _page("vlm.html")


@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
def page_review():
    return _page("review.html")


# ── Live Streaming — cameras in app.yaml + MJPEG preview ─────────────────────

class CameraRequest(BaseModel):
    source: str                        # rtsp://... | webcam:0 | file path
    id: str | None = None
    loop: bool = False


def _cameras() -> list[dict]:
    return list(_load(APP_CFG).get("cameras") or [])


def _mask_credentials(source: str) -> str:
    return re.sub(r"//([^/@:]+):[^@/]+@", r"//\1:•••@", source)


class _LiveWorker:
    """Grabs frames from one source and fans the latest JPEG out to viewers."""

    MAX_WIDTH = 720
    MAX_PREVIEW_FPS = 8.0

    def __init__(self, cam_id: str, source: str, detect: bool = False):
        self.cam_id, self.source = cam_id, source
        self.detect = detect
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.seq = 0
        self.status = "connecting"
        self.viewers = 0
        self.stopped = threading.Event()
        self._last_publish_ts = 0.0
        self._last_process_ts = 0.0
        name_suffix = "-detect" if detect else ""
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"live-{cam_id}{name_suffix}")
        self._thread.start()

    def _publish(self, jpeg: bytes, status: str = "live") -> None:
        with self.cond:
            self.frame, self.status = jpeg, status
            self.seq += 1
            self.cond.notify_all()

    def _run(self) -> None:
        try:
            import cv2
            from ingest.rtsp_client import FrameSource
        except ImportError as exc:                       # dashboard-only installs
            self.status = f"error: {exc.name} not installed"
            self.stopped.set()
            return
        
        detector = None
        if self.detect:
            try:
                from detection.yolo26_infer import YoloDetectorTracker
                cfg = _load(DETECTOR_CFG)
                
                # Make tracker config path absolute to avoid cwd issues
                tcfg = cfg.get("tracker_config", "bytetrack.yaml")
                if not Path(tcfg).is_absolute() and tcfg not in ("bytetrack.yaml", "botsort.yaml"):
                    cfg["tracker_config"] = str(APP_ROOT / tcfg)
                    
                detector = YoloDetectorTracker(cfg)
            except Exception as exc:
                self.status = f"error loading detector: {exc}"
                self.stopped.set()
                return

        source = self.source
        if "://" not in source and not source.startswith("webcam:") \
                and not Path(source).is_absolute():
            source = str(APP_ROOT / source)              # config paths are app-relative
        name_suffix = "-detect" if self.detect else ""
        src = FrameSource(f"live-{self.cam_id}{name_suffix}", source, loop=True)
        next_t = 0.0
        try:
            for _ts, frame in src.frames():
                if self.stopped.is_set():
                    break
                if not src.is_live:                      # pace file sources to real time
                    now = time.monotonic()
                    if next_t:
                        time.sleep(max(0.0, next_t - now))
                    next_t = max(now, next_t) + 1.0 / src.fps

                now = time.monotonic()
                if detector and (now - self._last_process_ts) < (1.0 / self.MAX_PREVIEW_FPS):
                    # Skip redundant processing when the live worker is already behind,
                    # to keep the MJPEG feed responsive instead of queueing stale frames.
                    continue
                if detector:
                    self._last_process_ts = now
                    detections = detector.process(frame)
                    for d in detections:
                        x1, y1, x2, y2 = map(int, d.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{d.class_name} {d.confidence:.2f}"
                        if d.track_id is not None:
                            label += f" ID:{d.track_id}"
                        cv2.putText(frame, label, (x1, max(10, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                h, w = frame.shape[:2]
                if w > self.MAX_WIDTH:
                    frame = cv2.resize(frame, (self.MAX_WIDTH, int(h * self.MAX_WIDTH / w)))

                now = time.monotonic()
                if (now - self._last_publish_ts) < (1.0 / self.MAX_PREVIEW_FPS):
                    continue
                self._last_publish_ts = now
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._publish(buf.tobytes())
        except Exception as exc:
            self.status = f"error: {exc}"
        finally:
            if self.status.startswith("error"):
                worker_key = f"{self.cam_id}_detect" if self.detect else self.cam_id
                _last_status[worker_key] = self.status   # survives worker cleanup
            self.stopped.set()
            with self.cond:
                self.cond.notify_all()

    def stop(self) -> None:
        self.stopped.set()
        with self.cond:
            self.cond.notify_all()


_workers: dict[str, _LiveWorker] = {}
_workers_lock = threading.Lock()
_last_status: dict[str, str] = {}      # last error per camera (shown while idle)


def _stop_worker(cam_id: str) -> None:
    with _workers_lock:
        worker = _workers.pop(cam_id, None)
    if worker:
        worker.stop()


@app.get("/api/streams")
def list_streams():
    out = []
    for cam in _cameras():
        worker = _workers.get(cam["id"])
        out.append({"id": cam["id"], "source": _mask_credentials(str(cam["source"])),
                    "loop": bool(cam.get("loop", False)),
                    "live": bool(worker and not worker.stopped.is_set()),
                    "status": worker.status if worker
                              else _last_status.get(cam["id"], "idle"),
                    "viewers": worker.viewers if worker else 0})
    return out


@app.post("/api/streams")
def add_stream(req: CameraRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(400, "source is required (rtsp://... | webcam:0 | file)")
    cameras = _cameras()
    cam_id = (req.id or "").strip()
    if not cam_id:
        used = {c["id"] for c in cameras}
        n = 1
        while f"cam{n:02d}" in used:
            n += 1
        cam_id = f"cam{n:02d}"
    elif any(c["id"] == cam_id for c in cameras):
        raise HTTPException(409, f"camera id '{cam_id}' already exists")
    cameras.append({"id": cam_id, "source": source, "loop": req.loop})
    _write_cameras(cameras)
    return {"ok": True, "id": cam_id}


@app.delete("/api/streams/{cam_id}")
def remove_stream(cam_id: str):
    cameras = _cameras()
    remaining = [c for c in cameras if c["id"] != cam_id]
    if len(remaining) == len(cameras):
        raise HTTPException(404, "camera not found")
    _stop_worker(cam_id)
    _last_status.pop(cam_id, None)
    _write_cameras(remaining)
    return {"ok": True}


@app.get("/stream/{cam_id}.mjpg")
def mjpeg_stream(cam_id: str, detect: bool = False):
    cam = next((c for c in _cameras() if c["id"] == cam_id), None)
    if cam is None:
        raise HTTPException(404, "camera not found")
    
    worker_key = f"{cam_id}_detect" if detect else cam_id
    with _workers_lock:
        worker = _workers.get(worker_key)
        if worker is None or worker.stopped.is_set():
            worker = _workers[worker_key] = _LiveWorker(cam_id, str(cam["source"]), detect=detect)
        worker.viewers += 1

    def gen(worker: _LiveWorker):
        last_seq = -1
        try:
            while True:
                with worker.cond:
                    worker.cond.wait_for(
                        lambda: worker.seq != last_seq or worker.stopped.is_set(),
                        timeout=2.0)
                    frame, seq, dead = worker.frame, worker.seq, worker.stopped.is_set()
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(frame)).encode()
                           + b"\r\n\r\n" + frame + b"\r\n")
                if dead:
                    break
        finally:
            with _workers_lock:
                worker.viewers -= 1
                if worker.viewers <= 0:
                    worker.stop()
                    if _workers.get(worker_key) is worker:
                        del _workers[worker_key]

    return StreamingResponse(gen(worker),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ── Per-Frame Detection & Tracking ───────────────────────────────────────────

class DetectionConfigRequest(BaseModel):
    detector: str
    tracker: str


@app.get("/api/config/detection")
def get_detection_config():
    det, trk = _load(DETECTOR_CFG), _load(TRACKER_CFG)
    return {"detector": _match_detector(str(det.get("model", ""))),
            "tracker": trk.get("tracker_type", "bytetrack"),
            "options": {"detectors": DETECTORS, "trackers": TRACKERS},
            "current": {"model": det.get("model"),
                        "fallback_model": det.get("fallback_model"),
                        "imgsz": det.get("imgsz"),
                        "conf_threshold": det.get("conf_threshold"),
                        "tracker_type": trk.get("tracker_type"),
                        "track_buffer": trk.get("track_buffer")}}


@app.post("/api/config/detection")
def set_detection_config(req: DetectionConfigRequest):
    detector = next((d for d in DETECTORS if d["id"] == req.detector), None)
    tracker = next((t for t in TRACKERS if t["id"] == req.tracker), None)
    if detector is None or tracker is None:
        raise HTTPException(400, "unknown detector or tracker id")
    _edit_config(DETECTOR_CFG, top={"model": detector["model"]})
    missing_extras = {} if tracker["id"] != "botsort" else \
        {k: v for k, v in BOTSORT_EXTRAS.items() if k not in _load(TRACKER_CFG)}
    trk_text = _set_top_key(TRACKER_CFG.read_text(), "tracker_type", tracker["id"])
    if missing_extras:                             # keys Ultralytics BoT-SORT needs
        trk_text = (trk_text.rstrip("\n")
                    + "\n\n# BoT-SORT extras (ignored while tracker_type is bytetrack)\n"
                    + "".join(f"{k}: {_yaml_value(v)}\n" for k, v in missing_extras.items()))
    TRACKER_CFG.write_text(trk_text)
    warnings = []
    if detector["runtime"] == "adapter":
        warnings.append(f"{detector['label']} has no runtime adapter yet — "
                        "detection/yolo26_infer.py loads Ultralytics models. "
                        "The selection is saved in detector.yaml.")
    return {"ok": True, "warnings": warnings}


# ── Temporal Understanding ───────────────────────────────────────────────────

class TemporalConfigRequest(BaseModel):
    action: str
    anomaly: str


@app.get("/api/config/temporal")
def get_temporal_config():
    cfg = _load(TEMPORAL_CFG)
    mode = cfg.get("mode", "novelty")
    action = cfg.get("action_recognition") or \
        {"novelty": "videomaev2", "finetuned": "finetuned"}.get(mode, "videomaev2")
    anomaly = cfg.get("anomaly_detection") or \
        ("rtfm" if (cfg.get("rtfm") or {}).get("enabled") else "off")
    return {"action": action, "anomaly": anomaly,
            "options": {"action": ACTION_MODELS, "anomaly": ANOMALY_MODELS},
            "current": {
                "mode": mode,
                "trigger_threshold": (cfg.get("trigger") or {}).get("trigger_threshold"),
                "window": cfg.get("window"),
                "backbone_checkpoint":
                    _checkpoint_info((cfg.get("backbone") or {}).get("checkpoint")),
                "finetuned_checkpoint":
                    _checkpoint_info((cfg.get("finetuned") or {}).get("checkpoint")),
                "rtfm_checkpoint":
                    _checkpoint_info((cfg.get("rtfm") or {}).get("checkpoint")),
                "rtfm_enabled": bool((cfg.get("rtfm") or {}).get("enabled"))}}


@app.post("/api/config/temporal")
def set_temporal_config(req: TemporalConfigRequest):
    action = next((a for a in ACTION_MODELS if a["id"] == req.action), None)
    anomaly = next((a for a in ANOMALY_MODELS if a["id"] == req.anomaly), None)
    if action is None or anomaly is None:
        raise HTTPException(400, "unknown action or anomaly model id")
    top: dict = {"action_recognition": action["id"], "anomaly_detection": anomaly["id"]}
    if action["mode"]:
        top["mode"] = action["mode"]
    _edit_config(TEMPORAL_CFG, top=top,
                 nested=[("rtfm", "enabled", anomaly["id"] == "rtfm")])
    warnings = []
    if not action["mode"]:
        warnings.append(f"{action['label']} has no TemporalModel adapter yet "
                        "(temporal/base.py) — the pipeline keeps its current mode; "
                        "the selection is recorded in temporal_model.yaml.")
    if anomaly["id"] == "streamvad":
        warnings.append("StreamVAD is on the roadmap — the RTFM head stays off; "
                        "the selection is recorded in temporal_model.yaml.")
    if anomaly["id"] == "rtfm":
        cfg = _load(TEMPORAL_CFG)
        if not _checkpoint_info((cfg.get("rtfm") or {}).get("checkpoint"))["found"]:
            warnings.append("RTFM enabled, but its head checkpoint is missing "
                            "(temporal_model.yaml → rtfm.checkpoint).")
    return {"ok": True, "warnings": warnings}


# ── VLM Verification ─────────────────────────────────────────────────────────

class VlmConfigRequest(BaseModel):
    model: str


@app.get("/api/config/vlm")
def get_vlm_config():
    cfg = _load(VLM_CFG)
    backend = cfg.get("backend", "openai_compatible")
    hosted_model = cfg.get("model")
    local_id = (cfg.get("local") or {}).get("model_id")
    selected = "custom"
    for opt in VLM_MODELS:
        if backend == "openai_compatible" and opt["backend"] == backend \
                and opt.get("model") == hosted_model:
            selected = opt["id"]
        if backend == "hf_local" and opt["backend"] == backend \
                and opt.get("model_id") == local_id:
            selected = opt["id"]
    return {"model": selected, "options": VLM_MODELS,
            "current": {"backend": backend, "hosted_model": hosted_model,
                        "endpoint": cfg.get("endpoint"),
                        "local_model_id": local_id,
                        "local_device": (cfg.get("local") or {}).get("device"),
                        "policy": cfg.get("policy")}}


@app.post("/api/config/vlm")
def set_vlm_config(req: VlmConfigRequest):
    opt = next((o for o in VLM_MODELS if o["id"] == req.model), None)
    if opt is None:
        raise HTTPException(400, "unknown VLM model id")
    if opt["backend"] == "openai_compatible":
        _edit_config(VLM_CFG, top={"backend": opt["backend"], "model": opt["model"]})
    else:
        _edit_config(VLM_CFG, top={"backend": opt["backend"]},
                     nested=[("local", "model_id", opt["model_id"]),
                             ("local", "quantize_4bit", opt["quantize_4bit"])])
    warnings = []
    if opt["backend"] == "hf_local":
        warnings.append("Local backend needs: pip install 'transformers>=4.49' "
                        "accelerate" + (" bitsandbytes" if opt["quantize_4bit"] else ""))
    else:
        warnings.append("Hosted backend needs HF_TOKEN in .env "
                        "(huggingface.co/settings/tokens).")
    return {"ok": True, "warnings": warnings}


# ── Alert Pipeline · Human-in-the-loop review (README feedback loop) ─────────

class LabelRequest(BaseModel):
    label: str                          # confirmed_theft | false_positive


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@app.get("/api/reviews")
def list_reviews():
    paths = _paths_cfg()
    review_log = APP_ROOT / paths.get("review_log", "data/review/review.jsonl")
    labels_log = review_log.parent / "labels.jsonl"
    labels = {r["event_id"]: r for r in _read_jsonl(labels_log)}
    items = _read_jsonl(review_log)
    for item in items:
        lbl = labels.get(item["event_id"])
        item["human_label"] = lbl["label"] if lbl else None
        if item.get("clip_path"):
            item["clip_url"] = f"/clips/{Path(item['clip_path']).name}"
    return sorted(items, key=lambda r: r.get("created_at", 0), reverse=True)


@app.post("/api/reviews/{event_id}/label")
def label_review(event_id: str, req: LabelRequest):
    if req.label not in ("confirmed_theft", "false_positive"):
        raise HTTPException(400, "label must be confirmed_theft | false_positive")
    paths = _paths_cfg()
    review_log = APP_ROOT / paths.get("review_log", "data/review/review.jsonl")
    labels_log = review_log.parent / "labels.jsonl"
    labels_log.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_log, "a") as f:
        f.write(json.dumps({"event_id": event_id, "label": req.label,
                            "labeled_at": time.time()}) + "\n")
    return {"ok": True}


@app.get("/clips/{name}")
def serve_clip(name: str):
    clips_dir = (APP_ROOT / _paths_cfg().get("clips_dir", "data/clips")).resolve()
    path = (clips_dir / name).resolve()
    if not path.is_file() or clips_dir not in path.parents:
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
