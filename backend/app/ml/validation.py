import os
import cv2
import numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.all_models import Model, ModelVersion
import traceback

def validate_model_task(model_id: int, file_path: str):
    db = SessionLocal()
    try:
        model_db = db.query(Model).filter(Model.id == model_id).first()
        if not model_db:
            return

        model_db.status = "validating"
        db.commit()

        # Check extension
        if not file_path.endswith('.pt'):
            model_db.status = "failed"
            db.commit()
            return
            
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            yolo_model = YOLO(file_path)
            
            # Test inference on blank frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = yolo_model.predict(source=blank_frame, imgsz=640, conf=0.5, verbose=False, device=device)
            
            # If we get here without crashing, model is reasonably valid
            model_db.status = "active"
            
            # Create a model version record
            new_version = ModelVersion(model_id=model_id, version_tag=model_db.version, is_current=True)
            db.add(new_version)
            db.commit()
            
        except Exception as e:
            print(f"Validation failed for model {model_id}: {traceback.format_exc()}")
            model_db.status = "failed"
            db.commit()
    finally:
        db.close()
