from app.workers.camera_worker import CameraWorker
from app.workers.theft_worker import TheftWorker
from app.database import SessionLocal
from app.models.all_models import CameraModelAssignment, Model

class WorkerManager:
    def __init__(self):
        self.workers = {} 
        self.theft_workers = {}
        import threading
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def _watchdog_loop(self):
        import time
        from app.utils.settings import get_server_mode
        while True:
            time.sleep(60)
            if get_server_mode() == "cloud":
                continue # Do not restart continuous workers in cloud mode
            try:
                all_cams = set(self.workers.keys()).union(set(self.theft_workers.keys()))
                for cam_id in list(all_cams):
                    w = self.workers.get(cam_id)
                    tw = self.theft_workers.get(cam_id)
                    
                    if (w and not w.running) or (tw and not tw.running):
                        print(f"[WATCHDOG] Detected dead worker for Camera {cam_id}. Restarting...", flush=True)
                        if w and not w.running:
                            del self.workers[cam_id]
                        if tw and not tw.running:
                            del self.theft_workers[cam_id]
                        self.start_worker(cam_id)
            except Exception as e:
                print(f"[WATCHDOG] Error: {e}", flush=True)

    def start_worker(self, camera_id: int):
        db = SessionLocal()
        is_theft_assigned = False
        try:
            assignments = db.query(CameraModelAssignment).join(Model).filter(
                CameraModelAssignment.camera_id == camera_id,
                CameraModelAssignment.is_running == True
            ).all()
            for a in assignments:
                if a.model.type in ['yolo_theft', 'theft']:
                    is_theft_assigned = True
        finally:
            db.close()
            
        if camera_id not in self.workers or not self.workers[camera_id].running:
            worker = CameraWorker(camera_id=camera_id)
            worker.daemon = True
            worker.start()
            self.workers[camera_id] = worker
            
        if is_theft_assigned:
            if camera_id not in self.theft_workers or not self.theft_workers[camera_id].running:
                t_worker = TheftWorker(camera_id=camera_id)
                t_worker.daemon = True
                t_worker.start()
                self.theft_workers[camera_id] = t_worker

        return True

    def stop_worker(self, camera_id: int):
        db = SessionLocal()
        is_theft_assigned = False
        any_assigned = False
        try:
            assignments = db.query(CameraModelAssignment).join(Model).filter(
                CameraModelAssignment.camera_id == camera_id,
                CameraModelAssignment.is_running == True
            ).all()
            for a in assignments:
                any_assigned = True
                if a.model.type in ['yolo_theft', 'theft']:
                    is_theft_assigned = True
        finally:
            db.close()
            
        if not any_assigned:
            if camera_id in self.workers:
                self.workers[camera_id].stop()
                self.workers[camera_id].join(timeout=2.0)
                del self.workers[camera_id]
                
        if not is_theft_assigned:
            if camera_id in self.theft_workers:
                self.theft_workers[camera_id].stop()
                self.theft_workers[camera_id].join(timeout=2.0)
                del self.theft_workers[camera_id]
                
        return True

    def start_theft_worker_manual(self, camera_id: int):
        if camera_id not in self.theft_workers or not self.theft_workers[camera_id].running:
            t_worker = TheftWorker(camera_id=camera_id)
            t_worker.daemon = True
            t_worker.start()
            self.theft_workers[camera_id] = t_worker
        return True

    def stop_theft_worker_manual(self, camera_id: int):
        if camera_id in self.theft_workers:
            self.theft_workers[camera_id].stop()
            self.theft_workers[camera_id].join(timeout=2.0)
            del self.theft_workers[camera_id]
        return True

    def get_status(self, camera_id: int):
        status = {"running": False, "error": None}
        if camera_id in self.workers:
            worker = self.workers[camera_id]
            status["running"] = worker.running
            status["error"] = worker.error_msg
            
        if camera_id in self.theft_workers:
            t_worker = self.theft_workers[camera_id]
            if t_worker.running:
                status["running"] = True
            if t_worker.error_msg:
                status["error"] = (status["error"] or "") + " | " + t_worker.error_msg
                
        return status

manager = WorkerManager()
