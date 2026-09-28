from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.all_models import CameraModelAssignment, Camera, Shop, DetectionEvent
from app.workers.worker_manager import manager
from app.workers.motion_worker import MotionWorkerManager
from datetime import datetime

router = APIRouter(prefix="/api/workers", tags=["Workers"])

# Dedicated manager for motion-only (no-model) workers
motion_manager = MotionWorkerManager()

@router.post("/cameras/{camera_id}/theft/start")
def start_camera_theft(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    # Auto-start motion since models depend on it
    motion_manager.start(camera_id, cam.rtsp_url, cam.shop_id, cam.name)
    
    cam.theft_active = True
    cam.motion_active = True
    db.commit()
    return {"message": "Theft worker started and motion auto-enabled"}

@router.post("/cameras/{camera_id}/theft/stop")
def stop_camera_theft(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam:
        cam.theft_active = False
        db.commit()
    return {"message": "Theft worker stopped"}

@router.post("/shops/{shop_id}/theft/start")
def start_shop_theft(shop_id: int, db: Session = Depends(get_db)):
    cameras = db.query(Camera).filter(Camera.shop_id == shop_id).all()
    if not cameras:
        raise HTTPException(status_code=404, detail="No cameras found in shop")
    for cam in cameras:
        motion_manager.start(cam.id, cam.rtsp_url, cam.shop_id, cam.name)
        cam.theft_active = True
        cam.motion_active = True
    db.commit()
    return {"message": f"Theft worker and motion started for all cameras in shop {shop_id}"}

@router.post("/shops/{shop_id}/theft/stop")
def stop_shop_theft(shop_id: int, db: Session = Depends(get_db)):
    cameras = db.query(Camera).filter(Camera.shop_id == shop_id).all()
    for cam in cameras:
        cam.theft_active = False
    db.commit()
    return {"message": f"Theft worker stopped for all cameras in shop {shop_id}"}



@router.post("/cameras/{camera_id}/models/{model_id}/start")
def start_model_on_camera(camera_id: int, model_id: int, db: Session = Depends(get_db)):
    assignment = db.query(CameraModelAssignment).filter(
        CameraModelAssignment.camera_id == camera_id,
        CameraModelAssignment.model_id == model_id
    ).first()

    if not assignment:
        assignment = CameraModelAssignment(
            camera_id=camera_id,
            model_id=model_id,
            is_running=True,
            started_at=datetime.utcnow()
        )
        db.add(assignment)
    else:
        assignment.is_running = True
        assignment.started_at = datetime.utcnow()

    db.commit()

    # Auto-start motion
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam:
        motion_manager.start(camera_id, cam.rtsp_url, cam.shop_id, cam.name)
        cam.motion_active = True
        db.commit()

    manager.start_worker(camera_id)
    return {"message": "Model and motion started on camera"}


@router.post("/cameras/{camera_id}/models/{model_id}/stop")
def stop_model_on_camera(camera_id: int, model_id: int, db: Session = Depends(get_db)):
    assignment = db.query(CameraModelAssignment).filter(
        CameraModelAssignment.camera_id == camera_id,
        CameraModelAssignment.model_id == model_id
    ).first()

    if assignment:
        assignment.is_running = False
        db.commit()

    # If no more running assignments, stop the full worker too
    remaining = db.query(CameraModelAssignment).filter(
        CameraModelAssignment.camera_id == camera_id,
        CameraModelAssignment.is_running == True
    ).count()
    if remaining == 0:
        manager.stop_worker(camera_id)

    return {"message": "Model stopped on camera"}


@router.post("/cameras/{camera_id}/motion/start")
def start_motion_worker(camera_id: int, db: Session = Depends(get_db)):
    """Start a motion-only worker (no YOLO model required)."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    cam.motion_active = True
    db.commit()
    
    motion_manager.start(camera_id, cam.rtsp_url, cam.shop_id, cam.name)
    return {"message": f"Motion worker started for camera {camera_id}"}


@router.post("/cameras/{camera_id}/motion/stop")
def stop_motion_worker(camera_id: int, db: Session = Depends(get_db)):
    """Stop a motion-only worker."""
    # Prevent stopping motion if any model is running
    running_assignments = db.query(CameraModelAssignment).filter(
        CameraModelAssignment.camera_id == camera_id,
        CameraModelAssignment.is_running == True
    ).count()
    
    is_theft_running = False
    if camera_id in manager.theft_workers and manager.theft_workers[camera_id].running:
        is_theft_running = True
        
    if running_assignments > 0 or is_theft_running:
        raise HTTPException(
            status_code=400, 
            detail="Cannot disable motion while AI models are running. Please disable the models first."
        )

    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam:
        cam.motion_active = False
        db.commit()

    motion_manager.stop(camera_id)
    return {"message": f"Motion worker stopped for camera {camera_id}"}


@router.post("/shops/{shop_id}/models/{model_id}/start")
def start_model_on_shop(shop_id: int, model_id: int, db: Session = Depends(get_db)):
    cameras = db.query(Camera).filter(Camera.shop_id == shop_id).all()

    for cam in cameras:
        assignment = db.query(CameraModelAssignment).filter(
            CameraModelAssignment.camera_id == cam.id,
            CameraModelAssignment.model_id == model_id
        ).first()

        if not assignment:
            assignment = CameraModelAssignment(
                camera_id=cam.id,
                model_id=model_id,
                is_running=True,
                started_at=datetime.utcnow()
            )
            db.add(assignment)
        else:
            assignment.is_running = True
            assignment.started_at = datetime.utcnow()

        manager.start_worker(cam.id)
        # Auto-start motion
        motion_manager.start(cam.id, cam.rtsp_url, cam.shop_id, cam.name)

    db.commit()
    return {"message": f"Model and motion started on all cameras in shop {shop_id}"}


@router.post("/shops/{shop_id}/models/{model_id}/stop")
def stop_model_on_shop(shop_id: int, model_id: int, db: Session = Depends(get_db)):
    cameras = db.query(Camera).filter(Camera.shop_id == shop_id).all()
    cam_ids = [c.id for c in cameras]

    assignments = db.query(CameraModelAssignment).filter(
        CameraModelAssignment.camera_id.in_(cam_ids),
        CameraModelAssignment.model_id == model_id
    ).all()

    for assignment in assignments:
        assignment.is_running = False

    db.commit()
    return {"message": f"Model stopped on all cameras in shop {shop_id}"}


@router.get("/cameras/{camera_id}/status")
def get_camera_worker_status(camera_id: int):
    status = manager.get_status(camera_id)
    if not status["running"]:
        # Check motion-only worker
        m = motion_manager.get_status(camera_id)
        if m["running"]:
            return m
    return status

@router.get("/states/all")
def get_all_camera_states(db: Session = Depends(get_db)):
    """Returns the running state of all models, motion, and theft for all cameras."""
    states = {}
    
    # 1. Models from DB
    assignments = db.query(CameraModelAssignment).all()
    for a in assignments:
        if a.camera_id not in states:
            states[a.camera_id] = {}
        states[a.camera_id][f"model_{a.model_id}"] = a.is_running
        
    # 2. Motion worker states
    motion_statuses = motion_manager.all_statuses()
    for cam_id, status in motion_statuses.items():
        if cam_id not in states:
            states[cam_id] = {}
        states[cam_id]["motion"] = status.get("running", False)
        
    # 3. Theft worker states (Now based on DB configuration since it's dynamically triggered)
    cameras = db.query(Camera).all()
    for cam in cameras:
        if cam.id not in states:
            states[cam.id] = {}
        states[cam.id]["theft"] = getattr(cam, 'theft_active', False)
        
    return states
