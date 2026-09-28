"""RTSP / file / webcam frame source (OpenCV-FFmpeg backend).

Default ingest for development and single-box deployments. Yields
``(stream_ts, frame_bgr)`` and reconnects automatically when an RTSP feed
drops. For multi-camera GPU-batched ingest on Jetson, use
``ingest/gstreamer_pipeline.py`` instead (``runtime.ingest_backend: gstreamer``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _open(source: str) -> cv2.VideoCapture:
    if source.startswith("webcam:"):
        return cv2.VideoCapture(int(source.split(":", 1)[1]))
    if source.startswith("rtsp://"):
        # # TCP transport avoids UDP packet-loss artifacts on wifi/cheap switches
        # os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # STRICT TCP transport avoids UDP packet-loss artifacts on wifi/cheap switches
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # keep latency low
        return cap
    return cv2.VideoCapture(source)


class FrameSource:
    """Iterate frames with stream timestamps; reconnect on live-source failure.

    * File sources: timestamps come from frame index / fps (deterministic, can
      run faster than realtime); ends at EOF unless ``loop=True``.
    * Live sources (rtsp/webcam): timestamps are wall-clock relative to start;
      on read failure the client backs off and reconnects up to
      ``max_reconnects`` times.
    """

    def __init__(self, camera_id: str, source: str, loop: bool = False,
                 reconnect_delay_s: float = 2.0, max_reconnects: int = 0):
        self.camera_id = camera_id
        self.source = source
        self.loop = loop
        self.reconnect_delay_s = reconnect_delay_s
        self.max_reconnects = max_reconnects          # 0 = unlimited
        self.is_live = source.startswith(("rtsp://", "webcam:"))
        self.fps: float = 25.0

    def _log_rtsp_hints(self) -> None:
        """One-time diagnostics for the most common RTSP invocation mistakes."""
        hints = [
            "RTSP open failed — checklist:",
            "  1. Always single-quote the URL on the command line: an unquoted '&' "
            "backgrounds the command and truncates the URL (a '[1] <pid>' line "
            "right after launching is the telltale sign).",
            "  2. Percent-encode special characters in the password: @ → %40, "
            ": → %3A, / → %2F, ? → %3F.",
            "  3. In the ffmpeg log above: 404 = wrong stream path (try subtype=0/1 "
            "on Dahua, /Streaming/Channels/101 on Hikvision); 401 = wrong credentials.",
            "  4. Verify outside the app first:  ffprobe -v error -rtsp_transport tcp "
            "-i '<url>' -show_streams",
        ]
        for h in hints:
            logger.warning("[%s] %s", self.camera_id, h)
        creds = self.source[len("rtsp://"):].rsplit("@", 1)[0]
        if "@" in creds:
            logger.warning("[%s] the URL's user:password section contains a raw '@' — "
                           "encode it as %%40", self.camera_id)

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        reconnects, t0, offset = 0, None, 0.0
        while True:
            cap = _open(self.source)
            if not cap.isOpened():
                cap.release()
                if reconnects == 0 and self.source.startswith("rtsp://"):
                    self._log_rtsp_hints()
                if not self.is_live:
                    raise IOError(f"[{self.camera_id}] cannot open source: {self.source}")
                reconnects += 1
                if self.max_reconnects and reconnects > self.max_reconnects:
                    logger.error("[%s] giving up after %d reconnects", self.camera_id, reconnects)
                    return
                logger.warning("[%s] open failed, retrying in %.1fs", self.camera_id,
                               self.reconnect_delay_s)
                time.sleep(self.reconnect_delay_s)
                continue

            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if self.fps <= 1 or self.fps > 120:
                self.fps = 25.0
            logger.info("[%s] opened %s (%.1f fps)", self.camera_id, self.source, self.fps)
            frame_idx = 0
            if t0 is None:
                t0 = time.monotonic()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = (time.monotonic() - t0) if self.is_live else offset + frame_idx / self.fps
                frame_idx += 1
                yield ts, frame
            cap.release()

            if not self.is_live:
                if not self.loop:
                    return
                offset += frame_idx / self.fps      # keep timestamps monotonic across loops
                continue
            reconnects += 1
            if self.max_reconnects and reconnects > self.max_reconnects:
                logger.error("[%s] stream lost, giving up", self.camera_id)
                return
            logger.warning("[%s] stream dropped, reconnecting", self.camera_id)
            time.sleep(self.reconnect_delay_s)
