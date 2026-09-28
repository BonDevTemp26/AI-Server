"""In-memory rolling frame buffer (30–60 s per camera, README Stage 4).

Frames are JPEG-encoded on append (≈50–150 KB each at 85 quality) so a 60 s /
12.5 fps buffer costs ~40–110 MB per camera — fine for a handful of cameras.
For many real RTSP cameras use ``ingest/segment_muxer.py`` instead (on-disk
ring, no re-encode); both expose the same extract-window contract.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RingBuffer:
    def __init__(self, camera_id: str, duration_s: float = 60.0,
                 store_fps: float = 12.5, jpeg_quality: int = 85):
        self.camera_id = camera_id
        self.duration_s = duration_s
        self.store_interval = 1.0 / store_fps
        self.store_fps = store_fps
        self.jpeg_quality = int(jpeg_quality)
        self._frames: deque[tuple[float, bytes]] = deque()
        self._last_ts = float("-inf")
        self._lock = threading.Lock()

    def append(self, ts: float, frame_bgr: np.ndarray) -> None:
        if ts - self._last_ts + 1e-6 < self.store_interval:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frames.append((ts, buf.tobytes()))
            self._last_ts = ts
            cutoff = ts - self.duration_s
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    @property
    def latest_ts(self) -> float:
        with self._lock:
            return self._frames[-1][0] if self._frames else float("-inf")

    def covers(self, end_ts: float) -> bool:
        return self.latest_ts >= end_ts

    def extract(self, start_ts: float, end_ts: float) -> list[tuple[float, np.ndarray]]:
        """Decoded frames within [start_ts, end_ts]."""
        with self._lock:
            selected = [(t, b) for t, b in self._frames if start_ts <= t <= end_ts]
        return [(t, cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR))
                for t, b in selected]


class BufferManager:
    """Per-camera ring buffers behind one construction point."""

    def __init__(self, duration_s: float = 60.0, store_fps: float = 12.5,
                 jpeg_quality: int = 85):
        self._kw = dict(duration_s=duration_s, store_fps=store_fps,
                        jpeg_quality=jpeg_quality)
        self._buffers: dict[str, RingBuffer] = {}

    def get(self, camera_id: str) -> RingBuffer:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = RingBuffer(camera_id, **self._kw)
        return self._buffers[camera_id]

    def append(self, camera_id: str, ts: float, frame_bgr: np.ndarray) -> None:
        self.get(camera_id).append(ts, frame_bgr)
