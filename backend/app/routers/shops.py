from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.all_models import Shop, Camera
from app.schemas.shop_camera import ShopCreate, ShopUpdate, ShopResponse, ShopWithCamerasResponse, CameraCreate, CameraResponse

router = APIRouter(prefix="/api/shops", tags=["Shops"])

@router.post("/", response_model=ShopResponse, status_code=status.HTTP_201_CREATED)
def create_shop(shop: ShopCreate, db: Session = Depends(get_db)):
    db_shop = db.query(Shop).filter(Shop.name == shop.name).first()
    if db_shop:
        raise HTTPException(status_code=400, detail="Shop with this name already exists")
    new_shop = Shop(name=shop.name)
    db.add(new_shop)
    db.commit()
    db.refresh(new_shop)
    return new_shop

@router.get("/", response_model=List[ShopResponse])
def get_shops(db: Session = Depends(get_db)):
    return db.query(Shop).all()

@router.put("/{shop_id}", response_model=ShopResponse)
def update_shop(shop_id: int, shop: ShopUpdate, db: Session = Depends(get_db)):
    db_shop = db.query(Shop).filter(Shop.id == shop_id).first()
    if not db_shop:
        raise HTTPException(status_code=404, detail="Shop not found")
    
    # Check for name conflict
    existing = db.query(Shop).filter(Shop.name == shop.name, Shop.id != shop_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Shop with this name already exists")
        
    db_shop.name = shop.name
    db.commit()
    db.refresh(db_shop)
    return db_shop

@router.delete("/{shop_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shop(shop_id: int, db: Session = Depends(get_db)):
    db_shop = db.query(Shop).filter(Shop.id == shop_id).first()
    if not db_shop:
        raise HTTPException(status_code=404, detail="Shop not found")
    
    db.delete(db_shop)
    db.commit()
    return None

@router.post("/{shop_id}/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def add_camera_to_shop(shop_id: int, camera: CameraCreate, db: Session = Depends(get_db)):
    db_shop = db.query(Shop).filter(Shop.id == shop_id).first()
    if not db_shop:
        raise HTTPException(status_code=404, detail="Shop not found")
        
    # Check if camera name already exists in this shop
    db_cam = db.query(Camera).filter(Camera.shop_id == shop_id, Camera.name == camera.name).first()
    if db_cam:
        raise HTTPException(status_code=400, detail="Camera with this name already exists in this shop")
        
    new_camera = Camera(shop_id=shop_id, name=camera.name, rtsp_url=camera.rtsp_url, is_active=camera.is_active)
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    return new_camera

@router.get("/{shop_id}/cameras", response_model=List[CameraResponse])
def get_cameras_for_shop(shop_id: int, db: Session = Depends(get_db)):
    db_shop = db.query(Shop).filter(Shop.id == shop_id).first()
    if not db_shop:
        raise HTTPException(status_code=404, detail="Shop not found")
    return db.query(Camera).filter(Camera.shop_id == shop_id).all()

from datetime import datetime, timedelta
from typing import Optional
from app.models.all_models import DetectionEvent

@router.get("/cameras/{camera_id}/timeline")
def get_camera_timeline(camera_id: int, date: Optional[str] = None, db: Session = Depends(get_db)):
    db_cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    target_date = datetime.strptime(date, "%Y-%m-%d").date() if date else datetime.now().date()
    start_of_day = datetime.combine(target_date, datetime.min.time())
    end_of_day = start_of_day + timedelta(days=1)

    events = db.query(DetectionEvent).filter(
        DetectionEvent.camera_id == camera_id,
        DetectionEvent.detected_at >= start_of_day,
        DetectionEvent.detected_at < end_of_day
    ).all()

    detection_periods = []
    for ev in events:
        end_time = ev.ended_at.isoformat() if ev.ended_at else (ev.detected_at + timedelta(seconds=1)).isoformat()
        detection_periods.append({
            "start": ev.detected_at.isoformat(),
            "end": end_time,
            "type": ev.detection_type,
            "camera_name": ev.camera_name
        })

    active_periods = []
    
    return {
        "active_periods": active_periods,
        "detection_periods": detection_periods
    }
