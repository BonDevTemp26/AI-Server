import json
import os
from datetime import datetime
from app.database import SessionLocal
from app.models.all_models import DetectionEvent

def import_dummy_data():
    db = SessionLocal()
    file_path = "/app/app/cctv_theft_app/data/review/review.jsonl"
    
    if not os.path.exists(file_path):
        print("review.jsonl not found")
        return
        
    from app.models.all_models import Shop, Camera
    
    # Ensure there is a shop and camera
    shop = db.query(Shop).first()
    if not shop:
        shop = Shop(name="Dummy Shop")
        db.add(shop)
        db.commit()
        
    camera = db.query(Camera).filter(Camera.shop_id == shop.id).first()
    if not camera:
        camera = Camera(shop_id=shop.id, name="cam01", rtsp_url="rtsp://dummy")
        db.add(camera)
        db.commit()
        
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Use original clip path logic but point to our uploads dir
            # Dummy data path looks like "data/clips/cam01_...mp4"
            clip_name = os.path.basename(data["clip_path"])
            video_path = f"/api/uploads/clips/{clip_name}"
            
            # Map review state
            review_state = "pending"
            
            meta = {
                "video_path": video_path,
                "review_state": review_state,
                "vlm_reject_score": data.get("score", 0),
                "vlm_content": data.get("vlm_description", ""),
                "dummy_event_id": data.get("event_id")
            }
            
            # See if already exists
            existing = db.query(DetectionEvent).filter(
                DetectionEvent.metadata_['dummy_event_id'].astext == data.get("event_id")
            ).first()
            
            if not existing:
                dt = datetime.fromtimestamp(data["created_at"]) if data.get("created_at") else datetime.utcnow()
                ev = DetectionEvent(
                    camera_id=camera.id,
                    shop_id=shop.id,
                    shop_name=shop.name,
                    camera_name=data.get("camera_id", camera.name),
                    detection_type="theft",
                    confidence=data.get("score", 0.9),
                    detected_at=dt,
                    ended_at=dt,
                    metadata_=meta
                )
                db.add(ev)
                count += 1
                
    db.commit()
    db.close()
    print(f"Imported {count} dummy events.")

if __name__ == "__main__":
    import_dummy_data()
