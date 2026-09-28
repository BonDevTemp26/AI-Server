from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.all_models import Camera
from app.schemas.shop_camera import CameraUpdate, CameraResponse

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

@router.get("/", response_model=List[CameraResponse])
def get_all_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()

@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, camera: CameraUpdate, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    if camera.name is not None:
        # Check name conflict
        existing = db.query(Camera).filter(Camera.shop_id == db_cam.shop_id, Camera.name == camera.name, Camera.id != camera_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Camera with this name already exists in this shop")
        db_cam.name = camera.name
        
    if camera.rtsp_url is not None:
        db_cam.rtsp_url = camera.rtsp_url
    if camera.is_active is not None:
        db_cam.is_active = camera.is_active
        
    db.commit()
    db.refresh(db_cam)
    return db_cam

@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    db.delete(db_cam)
    db.commit()
    return None
