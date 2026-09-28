from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile, Form, File
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func
import psutil
import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from app.database import get_db
from app.models.all_models import Shop, Camera, DetectionEvent, ModelType, CameraModelAssignment, Model
from app.workers.worker_manager import manager
import psutil
import threading
import time

_cached_cpu_percent = 0.0
def _cpu_poller():
    global _cached_cpu_percent
    # Give psutil an initial meaningless read, then poll every 2 seconds
    psutil.cpu_percent(interval=None)
    while True:
        _cached_cpu_percent = psutil.cpu_percent(interval=2.0)

_poller_thread = threading.Thread(target=_cpu_poller, daemon=True)
_poller_thread.start()

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"]
)

# Shared with worker_api — import via module to stay in-sync
# (both modules use the same process so singleton objects match)
from app.routers.worker_api import motion_manager as _motion_manager
from app.routers.trigger_api import active_cloud_workers


def _rtsp_reachable(rtsp_url: str, timeout: float = 2.5) -> bool:
    """
    Try a TCP connection to the RTSP host:port.
    Returns True if the TCP handshake succeeds (stream is reachable).
    """
    try:
        parsed = urllib.parse.urlparse(rtsp_url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _check_cameras_parallel(cameras, timeout: float = 2.5):
    """
    Check all cameras in parallel (up to 20 threads) so the stats
    endpoint stays fast even with many cameras.
    Returns (online_count, offline_count, per_camera_status dict).
    """
    results = {}
    with ThreadPoolExecutor(max_workers=min(20, len(cameras) or 1)) as ex:
        future_to_id = {
            ex.submit(_rtsp_reachable, cam.rtsp_url, timeout): cam.id
            for cam in cameras
        }
        for future in as_completed(future_to_id):
            cam_id = future_to_id[future]
            try:
                results[cam_id] = future.result()
            except Exception:
                results[cam_id] = False

    online = sum(1 for v in results.values() if v)
    offline = len(results) - online
    return online, offline, results


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    shops_count = db.query(Shop).count()
    cameras_count = db.query(Camera).count()

    from app.utils.settings import get_server_mode
    server_mode = get_server_mode()

    motion_running = set()
    motion_error = set()
    yolo_running = set()

    if server_mode == "cloud":
        try:
            # Safely create a list from items to prevent dictionary changed size errors
            items = list(active_cloud_workers.items())
            dead_keys = [k for k, w in items if not w.is_alive()]
            for k in dead_keys:
                if k in active_cloud_workers:
                    del active_cloud_workers[k]
        except RuntimeError:
            pass # Ignore if it mutates concurrently, it will be cleaned up next time
        
        active_trigger_cams = {cam_id for (cam_id, _), w in list(active_cloud_workers.items()) if w.is_alive()}
        cameras_online = len(active_trigger_cams)
        cameras_error = 0
        cameras_offline = cameras_count - cameras_online
    else:
        # Per-camera state from motion workers: running / error / not started
        all_statuses = _motion_manager.all_statuses()
        motion_running = set(_motion_manager.running_camera_ids())
        motion_error   = set(_motion_manager.error_camera_ids())
        yolo_running   = {cid for cid, w in manager.workers.items() if w.running}

        cameras_online  = len((motion_running | yolo_running))
        cameras_error   = len(motion_error - yolo_running)   # errored and no YOLO worker covering it
        cameras_offline = cameras_count - cameras_online - cameras_error

    today = date.today()
    detections = db.query(
        DetectionEvent.detection_type,
        func.count(DetectionEvent.id)
    ).filter(
        func.date(DetectionEvent.detected_at) == today
    ).group_by(DetectionEvent.detection_type).all()

    detection_counts = {dt: count for dt, count in detections}

    # Motion card = number of cameras currently actively detecting motion
    one_min_ago = datetime.utcnow() - timedelta(minutes=1)
    active_motion_count = db.query(DetectionEvent.camera_id).filter(
        DetectionEvent.detection_type == "motion",
        DetectionEvent.ended_at.is_(None),
        DetectionEvent.detected_at >= one_min_ago
    ).distinct().count()
    detection_counts['Motion'] = active_motion_count
    if 'motion' in detection_counts:
        del detection_counts['motion']

    # Detailed detections based on ModelTypes + 'motion' fallback
    detailed_detections = []
    
    # Get all model types from DB
    model_types = db.query(ModelType).all()
    
    # We always ensure 'motion' is included as a core type
    core_motion_added = False
    
    for mt in model_types:
        if mt.name == 'mog2' or mt.name == 'motion':
            core_motion_added = True
            cameras_running = len(motion_running)
        else:
            cameras_running = db.query(CameraModelAssignment).join(Model).filter(
                Model.type == mt.name,
                CameraModelAssignment.is_running == True
            ).count()
            
        # The database saves motion events as "motion", even if the model type name is "mog2"
        # and it saves human detection events as "human", even if the model type is "yolo_human".
        if mt.name in ("mog2", "motion"):
            actual_type = "motion"
        elif mt.name == "yolo_human":
            actual_type = "human"
        else:
            actual_type = mt.name

        # Get total for today (still available in backend just in case)
        total = db.query(func.count(DetectionEvent.id)).filter(
            DetectionEvent.detection_type == actual_type,
            func.date(DetectionEvent.detected_at) == today
        ).scalar() or 0
        
        # Get detected now (ongoing events)
        active = db.query(DetectionEvent.camera_id).filter(
            DetectionEvent.detection_type == actual_type,
            DetectionEvent.ended_at.is_(None),
            DetectionEvent.detected_at >= one_min_ago
        ).distinct().count()
        
        detailed_detections.append({
            "type": mt.name,
            "display_name": mt.display_name,
            "active_now": active,
            "total_today": total,
            "cameras_running": cameras_running
        })

    if not core_motion_added:
        # Get total for today for generic 'motion'
        mot_total = db.query(func.count(DetectionEvent.id)).filter(
            DetectionEvent.detection_type == 'motion',
            func.date(DetectionEvent.detected_at) == today
        ).scalar() or 0
        
        detailed_detections.insert(0, {
            "type": "motion",
            "display_name": "Motion",
            "active_now": active_motion_count,
            "total_today": mot_total,
            "cameras_running": len(motion_running)
        })
    # Append specific theft detection details from persisted camera state and workers.
    theft_cameras_running = set(
        camera_id for camera_id, worker in manager.theft_workers.items() if worker.running
    )
    theft_cameras_running.update(
        camera_id for camera_id, in db.query(Camera.id).filter(
            Camera.is_active == True,
            Camera.theft_active == True
        ).all()
    )
    theft_total = db.query(func.count(DetectionEvent.id)).filter(
        func.lower(DetectionEvent.detection_type) == 'theft',
        func.date(DetectionEvent.detected_at) == today
    ).scalar() or 0
    theft_active = db.query(DetectionEvent.camera_id).filter(
        func.lower(DetectionEvent.detection_type) == 'theft',
        DetectionEvent.ended_at.is_(None),
        DetectionEvent.detected_at >= one_min_ago
    ).distinct().count()
    
    detailed_detections.append({
        "type": "theft",
        "display_name": "Theft Detection",
        "active_now": theft_active,
        "total_today": theft_total,
        "cameras_running": len(theft_cameras_running)
    })

    cpu_percent = _cached_cpu_percent
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_used_gb = round(ram.used / (1024**3), 2)

    temperature = None
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                temperature = temps['coretemp'][0].current
            elif temps and temps.keys():
                first_key = list(temps.keys())[0]
                temperature = temps[first_key][0].current
    except Exception:
        pass

    def _get_gpu_percent():
        # 1) Try NVIDIA via nvidia-smi (GTX 1660 SUPER)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                val = result.stdout.strip().split("\n")[0].strip()
                return float(val)
        except Exception:
            pass
        # 2) Fallback: AMD sysfs (iGPU)
        try:
            import glob
            gpu_files = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
            if gpu_files:
                with open(gpu_files[0], "r") as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return 0.0

    def _get_gpu_memory():
        """Returns (used_mb, total_mb) for NVIDIA GPU."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                return int(parts[0].strip()), int(parts[1].strip())
        except Exception:
            pass
        return 0, 0

    def _get_npu_percent():
        return 0.0

    gpu_percent = _get_gpu_percent()
    npu_percent = _get_npu_percent()
    gpu_mem_used, gpu_mem_total = _get_gpu_memory()

    from app.utils.settings import get_server_mode
    return {
        "shops_count": shops_count,
        "cameras_count": cameras_count,
        "cameras_online": cameras_online,
        "cameras_offline": cameras_offline,
        "cameras_error": cameras_error,
        "theft_cameras_running": len(theft_cameras_running),
        "theft_videos_today": theft_total,
        "detections": detection_counts,
        "detailed_detections": detailed_detections,
        "server_mode": server_mode,
        "system": {
            "cpu_percent": cpu_percent,
            "gpu_percent": gpu_percent,
            "gpu_mem_used_mb": gpu_mem_used,
            "gpu_mem_total_mb": gpu_mem_total,
            "npu_percent": npu_percent,
            "ram_percent": ram_percent,
            "ram_used_gb": ram_used_gb,
            "temperature": temperature
        }
    }


@router.get("/camera-status")
def get_camera_status(db: Session = Depends(get_db)):
    """
    Per-camera status: online / error / offline.
    Uses motion worker state (stream_ok + error_msg) — fast, no new TCP probe.
    """
    cameras = db.query(Camera).filter(Camera.is_active == True).all()
    all_statuses = _motion_manager.all_statuses()
    result = {}
    for cam in cameras:
        w = all_statuses.get(cam.id, {})
        if w.get("running"):
            state = "online"
        elif w.get("error"):
            state = "error"
        else:
            state = "offline"
        result[cam.id] = {
            "online": state == "online",
            "state": state,
            "error": w.get("error")
        }
    return result


@router.get("/motion-status")
def get_all_camera_motion_status(db: Session = Depends(get_db)):
    """
    Returns EVERY active camera (all shops), each with its current motion
    state — not just cameras that already have a stored detection event.

    For a camera that is currently detecting motion (an open DetectionEvent
    with ended_at IS NULL), we return active=True plus the start time.
    For every other camera we return active=False with no start time,
    so the popup always lists all cameras (e.g. 10/10) instead of only
    the ones that happen to have history.
    """
    cameras = (
        db.query(Camera)
        .filter(Camera.is_active == True)
        .join(Shop, Camera.shop_id == Shop.id)
        .all()
    )

    # Open (ongoing) motion events, most recent first
    one_min_ago = datetime.utcnow() - timedelta(minutes=1)
    open_events = (
        db.query(DetectionEvent)
        .filter(
            DetectionEvent.detection_type == "motion",
            DetectionEvent.ended_at.is_(None),
            DetectionEvent.detected_at >= one_min_ago
        )
        .order_by(DetectionEvent.detected_at.desc())
        .all()
    )
    open_by_camera = {}
    for ev in open_events:
        if ev.camera_id not in open_by_camera:
            open_by_camera[ev.camera_id] = ev

    result = []
    for cam in cameras:
        ev = open_by_camera.get(cam.id)
        result.append({
            "camera_id": cam.id,
            "camera_name": cam.name,
            "shop_id": cam.shop_id,
            "shop_name": cam.shop.name if cam.shop else "-",
            "active": ev is not None,
            "started_at": ev.detected_at if ev else None,
        })

    result.sort(key=lambda r: (r["shop_name"] or "", r["camera_name"] or ""))
    return result


@router.get("/detections")
def get_recent_detections(type: str = None, date_str: str = None, limit: int = 50, active_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(DetectionEvent)

    if type:
        query = query.filter(func.lower(DetectionEvent.detection_type) == type.lower())

    if active_only:
        one_min_ago = datetime.utcnow() - timedelta(minutes=1)
        query = query.filter(
            DetectionEvent.ended_at.is_(None),
            DetectionEvent.detected_at >= one_min_ago
        )

    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            query = query.filter(func.date(DetectionEvent.detected_at) == target_date)
        except ValueError:
            pass

    events = query.order_by(DetectionEvent.detected_at.desc()).limit(limit).all()

    return [
        {
            "id": e.id,
            "camera_name": e.camera_name,
            "shop_name": e.shop_name,
            "detected_at": e.detected_at,
            "ended_at": e.ended_at,
            "detection_type": e.detection_type,
            "metadata": e.metadata_
        } for e in events
    ]

class ReviewUpdate(BaseModel):
    review_state: str

@router.post("/detections/{event_id}/review")
def update_detection_review(event_id: int, payload: ReviewUpdate, db: Session = Depends(get_db)):
    event = db.query(DetectionEvent).filter(DetectionEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    meta = event.metadata_ or {}
    meta["review_state"] = payload.review_state
    
    event.metadata_ = meta
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(event, "metadata_")
    
    db.commit()
    return {"status": "success", "review_state": payload.review_state}

@router.post("/test-video")
async def test_video_pipeline(
    shop_name: str = Form(...),
    camera_name: str = Form(...),
    video: UploadFile = File(...),
):
    # Simulate processing time
    await asyncio.sleep(2)
    # Return dummy detection result
    return {
        "status": "success",
        "score": 0.89,
        "state": "confirmed",
        "message": "Theft behavior detected in dummy pipeline."
    }

from app.utils.settings import get_server_mode, set_server_mode

@router.get("/settings")
def get_dashboard_settings():
    return {"server_mode": get_server_mode()}

class SettingsUpdate(BaseModel):
    server_mode: str

@router.post("/settings")
def update_dashboard_settings(payload: SettingsUpdate, db: Session = Depends(get_db)):
    if payload.server_mode not in ["standalone", "cloud"]:
        raise HTTPException(status_code=400, detail="Invalid server mode")
        
    old_mode = get_server_mode()
    set_server_mode(payload.server_mode)
    
    if old_mode != payload.server_mode:
        active_cameras = db.query(Camera).filter(Camera.is_active == True).all()
        
        if payload.server_mode == "cloud":
            # Stop all continuous workers
            for cam in active_cameras:
                _motion_manager.stop(cam.id)
                manager.stop_theft_worker_manual(cam.id)
                manager.stop_worker(cam.id)
        elif payload.server_mode == "standalone":
            # Start continuous workers
            for cam in active_cameras:
                if getattr(cam, 'motion_active', False):
                    _motion_manager.start(cam.id, cam.rtsp_url, cam.shop_id, cam.name)
                # Theft is triggered by motion, so no continuous start here
                
                running_assignments = db.query(CameraModelAssignment).filter(
                    CameraModelAssignment.camera_id == cam.id,
                    CameraModelAssignment.is_running == True
                ).count()
                if running_assignments > 0:
                    manager.start_worker(cam.id)
                    
    return {"status": "success", "server_mode": payload.server_mode}