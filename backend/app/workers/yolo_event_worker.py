import threading
import time
import cv2
import traceback
import os
import uuid
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
from app.database import SessionLocal
from app.models.all_models import DetectionEvent, Camera, Shop, Model

class YoloEventWorker(threading.Thread):
    """
    A one-shot worker that runs a specific YOLO model (like 'yolo_human') 
    on a triggered clip or live stream, then cleanly shuts down.
    """
    def __init__(self, camera_id: int, model_type: str, clip_path: str = None, duration_s: int = 30):
        super().__init__()
        self.camera_id = camera_id
        self.model_type = model_type
        self.clip_path = clip_path
        self.duration_s = duration_s
        self.max_frames = None
        self.fps = 30.0
        self.running = False
        self.error_msg = None

    def extend_duration(self, additional_seconds: int):
        self.duration_s += additional_seconds
        if self.max_frames is not None and not self.clip_path:
            self.max_frames += int(self.fps * additional_seconds)
            print(f"[YOLO-EVENT] Camera {self.camera_id} ({self.model_type}) extended by {additional_seconds}s.", flush=True)

    def run(self):
        self.running = True
        db = SessionLocal()
        cap = None
        video_writer = None
        saved_video_path = None
        
        try:
            camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if not camera:
                self.error_msg = "Camera not found"
                return

            shop = db.query(Shop).filter(Shop.id == camera.shop_id).first()
            shop_name = shop.name if shop else "Unknown"

            # 1. Load the specific Model from DB
            db_model = db.query(Model).filter(Model.type == self.model_type, Model.status == 'active').first()
            if not db_model or not db_model.file_path:
                print(f"[YOLO-EVENT] No active model file found for {self.model_type}")
                self.error_msg = f"Model {self.model_type} not found"
                return
            
            print(f"[YOLO-EVENT] Loading {self.model_type} model from {db_model.file_path}...", flush=True)
            yolo_model = YOLO(db_model.file_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            source = self.clip_path if self.clip_path else camera.rtsp_url
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self.max_frames is None:
                self.max_frames = int(self.fps * self.duration_s) if not self.clip_path else float('inf')
            frame_count = 0
            
            # If we are pulling a live stream, we should record what we see in case of detection
            if not self.clip_path:
                clip_dir = "/app/uploads/clips"
                os.makedirs(clip_dir, exist_ok=True)
                filename = f"trigger_{self.model_type}_{self.camera_id}_{int(time.time())}.mp4"
                abs_path = os.path.join(clip_dir, filename)
                saved_video_path = f"/api/uploads/clips/{filename}"
                
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(abs_path, fourcc, 30.0, (w, h))

            detection_found = False
            max_confidence = 0.0
            
            while self.running and (frame_count < self.max_frames):
                ret, frame = cap.read()
                if not ret:
                    if self.clip_path:
                        break # End of video clip
                    else:
                        time.sleep(2)
                        cap.release()
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                        continue
                
                frame_count += 1
                if video_writer:
                    video_writer.write(frame)
                
                # Only run AI on every 5th frame to save CPU
                if frame_count % 5 != 0:
                    continue
                    
                results = yolo_model.predict(frame, verbose=False, conf=0.5, device=device)
                
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = yolo_model.names[cls_id]
                        
                        # Just looking for basic objects like 'person', 'car', 'weapon'
                        if conf > max_confidence:
                            max_confidence = conf
                        if not detection_found:
                            detection_found = True
                            print(f"[YOLO-EVENT] Detected '{label}' on {camera.name}!")

            # After clip/stream finishes, log event to DB if something was detected
            if detection_found:
                # If we uploaded a clip, use that path. Otherwise use the live recorded path.
                final_video_path = f"/api/uploads/clips/{Path(self.clip_path).name}" if self.clip_path else saved_video_path
                
                ev = DetectionEvent(
                    camera_id=self.camera_id,
                    model_id=db_model.id,
                    shop_id=camera.shop_id,
                    shop_name=shop_name,
                    camera_name=camera.name,
                    detection_type=self.model_type.replace("yolo_", ""),
                    confidence=max_confidence,
                    metadata_={
                        "video_path": final_video_path,
                        "description": f"Trigger-based detection of {self.model_type}"
                    }
                )
                db.add(ev)
                db.commit()
                db.refresh(ev)
                ev.ended_at = datetime.utcnow()
                db.commit()

        except Exception as e:
            self.error_msg = str(e)
            print(f"[YOLO-EVENT] Worker {self.camera_id} crashed: {traceback.format_exc()}")
        finally:
            if cap:
                cap.release()
            if video_writer:
                video_writer.release()
            db.close()
            self.running = False
            
            # Clean up uploaded raw clip if we don't want to save it permanently, 
            # BUT we want to keep it if a detection was found. We'll leave it for now.
            print(f"[YOLO-EVENT] Finished processing {self.model_type} for Camera {self.camera_id}.", flush=True)
