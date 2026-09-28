from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base, SessionLocal
from app.models import all_models
from app.routers import shops, cameras, bulk, models, worker_api, dashboard, trigger_api
from app.workers.motion_worker import MotionWorkerManager

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Monitor Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(shops.router)
app.include_router(cameras.router)
app.include_router(bulk.router)
app.include_router(models.router)
app.include_router(worker_api.router)
app.include_router(dashboard.router)
app.include_router(trigger_api.router)

import os
from fastapi.staticfiles import StaticFiles
os.makedirs("/app/uploads", exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory="/app/uploads"), name="uploads")

@app.on_event("startup")
def startup_event():
    from app.utils.settings import get_server_mode
    server_mode = get_server_mode()
    
    if server_mode == "cloud":
        print("Server starting in CLOUD API Mode. Continuous polling workers disabled.")
        return
        
    print("Server starting in STANDALONE Mode. Restoring continuous polling workers...")
    # Automatically restore workers for all active cameras
    db = SessionLocal()
    try:
        from app.routers.worker_api import motion_manager
        from app.workers.worker_manager import manager
        
        active_cameras = db.query(all_models.Camera).filter(all_models.Camera.is_active == True).all()
        for cam in active_cameras:
            # Restore motion worker
            if getattr(cam, 'motion_active', False):
                motion_manager.start(cam.id, cam.rtsp_url, cam.shop_id, cam.name)
            
            # Theft is dynamically triggered by motion, so we don't start a continuous theft worker here.
            

            # Restore standard AI model workers (YOLO)
            running_assignments = db.query(all_models.CameraModelAssignment).filter(
                all_models.CameraModelAssignment.camera_id == cam.id,
                all_models.CameraModelAssignment.is_running == True
            ).count()
            if running_assignments > 0:
                manager.start_worker(cam.id)
                
        print(f"Restored workers for active cameras.")
    except Exception as e:
        print(f"Error restoring workers during startup: {e}")
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "AI Monitor Server API is running"}

