"""
MotionWorker — lightweight MOG2-based motion detection thread.

States per camera:
  - running=True, error=None       → active, reading frames OK
  - running=False, error=None      → not started / cleanly stopped
  - running=False, error=<msg>     → stream error (RTSP unreachable / bad channel)
"""
import threading
import time
import socket
import urllib.parse
import cv2
import numpy as np
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
from datetime import datetime
from app.database import SessionLocal
from app.models.all_models import DetectionEvent, Shop

MAX_CONSECUTIVE_FAILS = 10   # retry N times before marking error
RECONNECT_WAIT = 5           # seconds between reconnect attempts


def _tcp_reachable(rtsp_url: str, timeout: float = 2.0) -> bool:
    try:
        p = urllib.parse.urlparse(rtsp_url)
        host, port = p.hostname, p.port or 554
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class MotionWorker(threading.Thread):
    MOTION_THRESHOLD = 500  # Tuned for 640x360 resolution
    COOLDOWN_SECONDS = 10

    def __init__(self, camera_id: int, rtsp_url: str, shop_id: int, camera_name: str):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.shop_id = shop_id
        self.camera_name = camera_name
        self.running = False
        self.error_msg = None          # set when stream permanently fails
        self.stream_ok = None          # True/False/None(checking)
        self._last_event_at = None
        self._open_event_id = None

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        self.error_msg = None
        self.stream_ok = None
        try:
            db = SessionLocal()
            try:
                shop = db.query(Shop).filter(Shop.id == self.shop_id).first()
                shop_name = shop.name if shop else "Unknown"
            finally:
                db.close()

            # Quick TCP check before wasting time on OpenCV
            if not _tcp_reachable(self.rtsp_url):
                self.error_msg = "TCP unreachable"
                self.stream_ok = False
                return

            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            mog2 = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=16, detectShadows=False
            )
            consecutive_fails = 0
            frame_count = 0

            while self.running:
                ret, frame = cap.read()
                frame_count += 1

                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                        self.error_msg = f"Stream read failed {consecutive_fails}x in a row"
                        self.stream_ok = False
                        break  # exit loop, thread ends with error state

                    cap.release()
                    time.sleep(RECONNECT_WAIT)
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    continue

                # Successful frame — clear error state
                consecutive_fails = 0
                self.stream_ok = True
                self.error_msg = None

                # Process ~10 FPS (1 out of every 3 frames from a 30 FPS source)
                if frame_count % 3 != 0:
                    continue

                # Resize to 640x360 for much faster CPU processing
                frame = cv2.resize(frame, (640, 360))

                fg_mask = mog2.apply(frame)
                motion_area = int(np.sum(fg_mask > 200))
                # print(f"[TEST] Camera {self.camera_id} moving pixels: {motion_area}")
                now = datetime.utcnow()

                if motion_area >= self.MOTION_THRESHOLD:
                    if (self._last_event_at is None or
                            (now - self._last_event_at).total_seconds() > self.COOLDOWN_SECONDS):
                        if self._open_event_id:
                            db = SessionLocal()
                            try:
                                ev = db.query(DetectionEvent).filter(
                                    DetectionEvent.id == self._open_event_id
                                ).first()
                                if ev and ev.ended_at is None:
                                    ev.ended_at = now
                                    db.commit()
                            finally:
                                db.close()

                        new_ev = DetectionEvent(
                            camera_id=self.camera_id,
                            model_id=None,
                            shop_id=self.shop_id,
                            shop_name=shop_name,
                            camera_name=self.camera_name,
                            detection_type="motion",
                            confidence=min(1.0, motion_area / 50000),
                        )
                        db = SessionLocal()
                        try:
                            db.add(new_ev)
                            db.commit()
                            db.refresh(new_ev)
                            self._open_event_id = new_ev.id
                        finally:
                            db.close()
                        self._last_event_at = now
                        
                        # Trigger Theft Detection (EventWorker) if enabled for this camera
                        from app.models.all_models import Camera
                        db = SessionLocal()
                        try:
                            cam_rec = db.query(Camera).filter(Camera.id == self.camera_id).first()
                            theft_active = bool(cam_rec and getattr(cam_rec, 'theft_active', False))
                        finally:
                            db.close()

                        if theft_active:
                            from app.routers.trigger_api import active_cloud_workers
                            from app.workers.event_worker import EventWorker
                            
                            worker_key = (self.camera_id, "theft")
                            if worker_key in active_cloud_workers and active_cloud_workers[worker_key].is_alive():
                                active_cloud_workers[worker_key].extend_duration(10) # give a small boost if motion continues
                            else:
                                print(f"[MotionWorker] Motion detected on {self.camera_name}! Starting EventWorker for Theft Detection...")
                                worker = EventWorker(camera_id=self.camera_id, clip_path=None, duration_s=45)
                                worker.daemon = True
                                worker.start()
                                active_cloud_workers[worker_key] = worker
                else:
                    if self._open_event_id:
                        db = SessionLocal()
                        try:
                            ev = db.query(DetectionEvent).filter(
                                DetectionEvent.id == self._open_event_id
                            ).first()
                            if ev and ev.ended_at is None:
                                ev.ended_at = now
                                db.commit()
                        finally:
                            db.close()
                        self._open_event_id = None

        except Exception as exc:
            self.error_msg = str(exc)
            self.stream_ok = False
            print(f"[MotionWorker cam={self.camera_id}] crashed: {exc}")
        finally:
            if 'cap' in locals():
                cap.release()
            self.running = False


class MotionWorkerManager:
    """Manages one MotionWorker thread per camera."""

    def __init__(self):
        self._workers: dict[int, MotionWorker] = {}

    def start(self, camera_id: int, rtsp_url: str, shop_id: int, camera_name: str):
        self.stop(camera_id)
        w = MotionWorker(camera_id, rtsp_url, shop_id, camera_name)
        w.start()
        self._workers[camera_id] = w

    def stop(self, camera_id: int):
        w = self._workers.pop(camera_id, None)
        if w:
            w.stop()

    def get_status(self, camera_id: int) -> dict:
        w = self._workers.get(camera_id)
        if w:
            return {
                "running": w.running,
                "error": w.error_msg,
                "stream_ok": w.stream_ok,
            }
        return {"running": False, "error": None, "stream_ok": None}

    def all_statuses(self) -> dict[int, dict]:
        return {
            cid: {
                "running": w.running,
                "error": w.error_msg,
                "stream_ok": w.stream_ok,
            }
            for cid, w in self._workers.items()
        }

    def running_camera_ids(self) -> list[int]:
        return [cid for cid, w in self._workers.items() if w.running]

    def error_camera_ids(self) -> list[int]:
        return [cid for cid, w in self._workers.items()
                if not w.running and w.error_msg]
