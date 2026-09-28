from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
from app.database import get_db
from app.models.all_models import Model, CameraModelAssignment, Camera, ModelType, DetectionEvent, ModelVersion
from app.schemas.models import ModelResponse, ModelTypeCreate, ModelTypeResponse
from app.ml.validation import validate_model_task

router = APIRouter(prefix="/api/models", tags=["Models"])

UPLOAD_DIR = "/app/uploads"

@router.post("/upload", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    existing = db.query(Model).filter(Model.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this name already exists")
        
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    new_model = Model(
        name=name,
        type=type,
        version="v1.0",
        file_path=file_path,
        status="pending"
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    
    background_tasks.add_task(validate_model_task, new_model.id, file_path)
    
    return new_model

@router.get("/", response_model=List[ModelResponse])
def get_models(db: Session = Depends(get_db)):
    return db.query(Model).all()

@router.get("/types", response_model=List[ModelTypeResponse])
def get_model_types(db: Session = Depends(get_db)):
    types = db.query(ModelType).all()
    if not types:
        # Defaults
        defaults = [
            ModelType(name="yolo_human", display_name="Human Detection"),
            ModelType(name="yolo_fire", display_name="Fire Detection"),
            ModelType(name="yolo_weapon", display_name="Weapon Detection"),
            ModelType(name="yolo_theft", display_name="Theft Detection"),
            ModelType(name="mog2", display_name="Motion (MOG2)"),
            ModelType(name="custom", display_name="Custom"),
        ]
        for d in defaults:
            db.add(d)
        db.commit()
        types = db.query(ModelType).all()
    return types

@router.post("/types", response_model=ModelTypeResponse, status_code=status.HTTP_201_CREATED)
def create_model_type(type_in: ModelTypeCreate, db: Session = Depends(get_db)):
    existing = db.query(ModelType).filter(ModelType.name == type_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Type already exists")
    new_type = ModelType(name=type_in.name, display_name=type_in.display_name)
    db.add(new_type)
    db.commit()
    db.refresh(new_type)
    return new_type

@router.delete("/types/{type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_type(type_id: int, db: Session = Depends(get_db)):
    db_type = db.query(ModelType).filter(ModelType.id == type_id).first()
    if not db_type:
        raise HTTPException(status_code=404, detail="Type not found")
    db.delete(db_type)
    db.commit()
    return None

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Manually delete dependent records to avoid IntegrityError (foreign key violations)
    db.query(DetectionEvent).filter(DetectionEvent.model_id == model_id).delete(synchronize_session=False)
    db.query(CameraModelAssignment).filter(CameraModelAssignment.model_id == model_id).delete(synchronize_session=False)
    db.query(ModelVersion).filter(ModelVersion.model_id == model_id).delete(synchronize_session=False)
        
    if db_model.file_path and os.path.exists(db_model.file_path):
        try:
            os.remove(db_model.file_path)
        except Exception:
            pass
            
    db.delete(db_model)
    db.commit()
    return None

@router.get("/{model_id}/status")
def get_model_status(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"id": db_model.id, "status": db_model.status}

@router.post("/{model_id}/activate")
def activate_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    db_model.status = "active"
    db.commit()
    return {"message": "Model activated"}

@router.post("/{model_id}/deactivate")
def deactivate_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    db_model.status = "inactive"
    db.commit()
    return {"message": "Model deactivated"}
