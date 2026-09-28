from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.all_models import Shop, Camera
import os
import shutil
import uuid

router = APIRouter(
    prefix="/api/trigger",
    tags=["trigger"]
)

active_cloud_workers = {}

@router.post("/")
async def trigger_camera_event(
    shop_name: str = Form(...),
    camera_name: str = Form(...),
    model_name: str = Form("theft"),
    clip: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Endpoint for local Edge PCs to trigger AI detection on AWS.
    If 'clip' is provided, it processes the uploaded video instantly.
    If 'clip' is None, it opens the live RTSP stream for 30 seconds.
    'model_name' allows targeting a specific YOLO model (e.g. human, vehicle, theft).
    """
    shop = db.query(Shop).filter(Shop.name == shop_name).first()
    if not shop:
        raise HTTPException(status_code=404, detail=f"Shop '{shop_name}' not found")

    camera = db.query(Camera).filter(Camera.shop_id == shop.id, Camera.name == camera_name).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found in shop '{shop_name}'")

    clip_path = None
    if clip:
        # Save the uploaded clip permanently in clips dir so it can be referenced
        upload_dir = "/app/uploads/clips"
        os.makedirs(upload_dir, exist_ok=True)
        clip_filename = f"trigger_{uuid.uuid4()}_{clip.filename}"
        clip_path = os.path.join(upload_dir, clip_filename)
        
        with open(clip_path, "wb") as buffer:
            shutil.copyfileobj(clip.file, buffer)

    # Keyword mapping from edge PC to actual model names
    model_mapping = {
        "theft": "theft",
        "theif": "theft",
        "human": "yolo_human",
        "vehicle": "yolo_vehicle",
        "fire": "yolo_fire",
        "weapon": "yolo_weapon",
        "person": "yolo_human",
        "car": "yolo_vehicle"
    }
    
    target_model = model_mapping.get(model_name.lower(), f"yolo_{model_name.lower()}")

    # Cleanup dead workers
    dead_keys = [k for k, w in active_cloud_workers.items() if not w.is_alive()]
    for k in dead_keys:
        del active_cloud_workers[k]

    worker_key = (camera.id, target_model)
    if worker_key in active_cloud_workers and active_cloud_workers[worker_key].is_alive():
        active_cloud_workers[worker_key].extend_duration(30)
        return {
            "status": "success",
            "message": f"Trigger extended for {camera_name}. Added 30s to {target_model} pipeline.",
            "clip_uploaded": clip is not None,
            "target_model": target_model
        }

    if target_model == "theft":
        # Start the heavy one-shot PipelineOrchestrator EventWorker
        from app.workers.event_worker import EventWorker
        worker = EventWorker(camera_id=camera.id, clip_path=clip_path, duration_s=45)
    else:
        # Start the lightweight one-shot YoloEventWorker
        from app.workers.yolo_event_worker import YoloEventWorker
        worker = YoloEventWorker(camera_id=camera.id, model_type=target_model, clip_path=clip_path, duration_s=45)
        
    worker.daemon = True
    worker.start()
    active_cloud_workers[worker_key] = worker

    return {
        "status": "success",
        "message": f"Trigger received for {camera_name}. AI processing started for model: {target_model}.",
        "clip_uploaded": clip is not None,
        "target_model": target_model
    }
