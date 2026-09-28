import threading
import time
import cv2
import traceback
import numpy as np
import os
import collections

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.all_models import DetectionEvent, CameraModelAssignment, Model, Camera, Shop
from app.workers.tracker import HumanTracker
from ultralytics import YOLO

class CameraWorker(threading.Thread):
    def __init__(self, camera_id: int):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.error_msg = None
        self.models_cache = {} 
        self.motion_threshold = 2000 
        
        self.frame_buffer = collections.deque(maxlen=150) # 5 seconds at 30 fps
        self.recording_theft = False
        self.theft_record_frames = 0
        self.theft_video_writer = None
        self.theft_clip_path = None
        self.theft_event_id = None

    def stop(self):
        self.running = False

    def load_active_models(self, db: Session):
        assignments = db.query(CameraModelAssignment).filter(
            CameraModelAssignment.camera_id == self.camera_id,
            CameraModelAssignment.is_running == True
        ).all()
        
        active_model_ids = [a.model_id for a in assignments]
        
        for m_id in list(self.models_cache.keys()):
            if m_id not in active_model_ids:
                del self.models_cache[m_id]
                
        active_db_models = []
        for a in assignments:
            model = db.query(Model).filter(Model.id == a.model_id).first()
            if model and model.file_path and model.status == 'active':
                active_db_models.append(model)
                if model.id not in self.models_cache:
                    try:
                        self.models_cache[model.id] = {
                            "yolo": YOLO(model.file_path),
                            "type": model.type,
                            "db_model": model
                        }
                    except Exception as e:
                        print(f"Failed to load model {model.id}: {e}")
                        
        return active_db_models

    def run(self):
        self.running = True
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if not camera:
                self.error_msg = "Camera not found"
                return
            
            shop = db.query(Shop).filter(Shop.id == camera.shop_id).first()
            shop_name = shop.name if shop else "Unknown"
            
            rtsp_url = camera.rtsp_url
            
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            tracker = HumanTracker()
            
            self.load_active_models(db)
            
            frame_count = 0
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    continue
                    
                frame_count += 1
                self.frame_buffer.append(frame)
                
                if frame_count % 60 == 0:
                    self.load_active_models(db)
                    
                fg_mask = mog2.apply(frame)
                motion_area = np.sum(fg_mask > 200) 
                
                # Skip AI inference only if there is no motion AND no active human tracks.
                # This solves the "static human problem" where stationary humans were lost.
                if motion_area < self.motion_threshold and len(tracker.tracks) == 0:
                    new_tracks, lost_event_ids = tracker.update([])
                    if lost_event_ids:
                        for ev_id in lost_event_ids:
                            ev = db.query(DetectionEvent).filter(DetectionEvent.id == ev_id).first()
                            if ev:
                                ev.ended_at = datetime.utcnow()
                        db.commit()
                    continue
                
                human_detections = []
                
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

                for m_id, m_data in self.models_cache.items():
                    yolo = m_data["yolo"]
                    results = yolo.predict(frame, verbose=False, conf=0.5, device=device)
                    
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = yolo.names[cls_id]
                            xyxy = box.xyxy[0].tolist()
                            
                            if m_data["type"] == "yolo_human" and label == "person":
                                human_detections.append(xyxy)
                            elif m_data["type"] == "yolo_theft" or label == "theft":
                                if not self.recording_theft:
                                    self.recording_theft = True
                                    self.theft_record_frames = 150 # next 5 seconds
                                    clip_dir = "/app/uploads/clips"
                                    os.makedirs(clip_dir, exist_ok=True)
                                    filename = f"theft_{self.camera_id}_{int(time.time())}.mp4"
                                    self.theft_clip_path = os.path.join(clip_dir, filename)
                                    
                                    h, w = frame.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    self.theft_video_writer = cv2.VideoWriter(self.theft_clip_path, fourcc, 30.0, (w, h))
                                    
                                    for f in self.frame_buffer:
                                        self.theft_video_writer.write(f)
                                        
                                    new_event = DetectionEvent(
                                        camera_id=self.camera_id,
                                        model_id=m_id,
                                        shop_id=camera.shop_id,
                                        shop_name=shop_name,
                                        camera_name=camera.name,
                                        detection_type="theft",
                                        confidence=conf,
                                        metadata_={"video_path": f"/uploads/clips/{filename}"}
                                    )
                                    db.add(new_event)
                                    db.commit()
                                    db.refresh(new_event)
                                    self.theft_event_id = new_event.id
                            else:
                                new_event = DetectionEvent(
                                    camera_id=self.camera_id,
                                    model_id=m_id,
                                    shop_id=camera.shop_id,
                                    shop_name=shop_name,
                                    camera_name=camera.name,
                                    detection_type=m_data["type"],
                                    confidence=conf,
                                    ended_at=datetime.utcnow() 
                                )
                                db.add(new_event)
                                db.commit()
                                
                if self.recording_theft:
                    self.theft_video_writer.write(frame)
                    self.theft_record_frames -= 1
                    if self.theft_record_frames <= 0:
                        self.recording_theft = False
                        self.theft_video_writer.release()
                        self.theft_video_writer = None
                        if self.theft_event_id:
                            ev = db.query(DetectionEvent).filter(DetectionEvent.id == self.theft_event_id).first()
                            if ev:
                                ev.ended_at = datetime.utcnow()
                                db.commit()
                            self.theft_event_id = None
                
                new_track_ids, lost_event_ids = tracker.update(human_detections)
                
                for ev_id in lost_event_ids:
                    ev = db.query(DetectionEvent).filter(DetectionEvent.id == ev_id).first()
                    if ev:
                        ev.ended_at = datetime.utcnow()
                
                if new_track_ids:
                    human_model_id = None
                    for m_id, m_data in self.models_cache.items():
                        if m_data["type"] == "yolo_human":
                            human_model_id = m_id
                            break
                            
                    for t_id in new_track_ids:
                        new_event = DetectionEvent(
                            camera_id=self.camera_id,
                            model_id=human_model_id,
                            shop_id=camera.shop_id,
                            shop_name=shop_name,
                            camera_name=camera.name,
                            detection_type="human",
                            confidence=1.0
                        )
                        db.add(new_event)
                        db.commit()
                        db.refresh(new_event)
                        tracker.set_db_event_id(t_id, new_event.id)
                        
        except Exception as e:
            self.error_msg = str(e)
            print(f"Worker {self.camera_id} crashed: {traceback.format_exc()}")
        finally:
            if 'cap' in locals():
                cap.release()
            db.close()
            self.running = False
