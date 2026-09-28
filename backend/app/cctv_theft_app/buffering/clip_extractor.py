"""Snapshot a CandidateEvent's window (T-15s → T+15s) into an mp4 clip.

Reads frames back from the camera's ring buffer and writes an H.264/mp4 file
that Stage 3 (VLM) consumes and Stage 4 attaches to alerts. Falls back to
OpenCV's mp4v codec when the OpenCV build lacks H.264.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2

from buffering.ring_buffer import RingBuffer
from pipeline.events import CandidateEvent

logger = logging.getLogger(__name__)


class ClipExtractor:
    # Use the system ffmpeg (/usr/bin/ffmpeg) which has libx264.
    # The conda ffmpeg (/opt/conda/bin/ffmpeg) only has libopenh264 and shadows
    # the system one in PATH — so we use the absolute path directly.
    FFMPEG = "/usr/bin/ffmpeg"

    def __init__(self, clips_dir: str | Path):
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        import os
        self._have_ffmpeg = os.path.isfile(self.FFMPEG)

    def extract(self, event: CandidateEvent, buffer: RingBuffer) -> Path | None:
        frames = buffer.extract(event.clip_start_ts, event.clip_end_ts)
        if len(frames) < 4:
            logger.warning("[%s] buffer had %d frames for event %s — no clip",
                           event.camera_id, len(frames), event.event_id)
            return None

        out = self.clips_dir / f"{event.camera_id}_{event.event_id}.mp4"
        h, w = frames[0][1].shape[:2]
        span = max(frames[-1][0] - frames[0][0], 1e-3)
        fps = max(1.0, min((len(frames) - 1) / span, 60.0))

        # Try H.264 (avc1) first — required for browser playback.
        # mp4v (MPEG-4 Part 2) is NOT supported by Chrome/Firefox in <video> tags.
        codec_tried = None
        for fourcc_str in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
            if writer.isOpened():
                codec_tried = fourcc_str
                break
            writer.release()

        if not writer.isOpened():
            logger.error("[%s] VideoWriter failed for %s", event.camera_id, out)
            return None

        for _, frame in frames:
            writer.write(frame)
        writer.release()
        logger.debug("[%s] wrote clip with codec=%s", event.camera_id, codec_tried)

        # Re-mux with faststart AND re-encode to H.264 so browsers can play it.
        # Use /usr/bin/ffmpeg (apt) which has libx264.
        # conda's ffmpeg (/opt/conda/bin/ffmpeg) only has libopenh264 — skip it.
        if self._have_ffmpeg:
            tmp = out.with_suffix(".fast.mp4")
            try:
                result = subprocess.run(
                    [self.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(out),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-movflags", "+faststart",
                     str(tmp)],
                    capture_output=True, check=True
                )
                out.unlink()
                tmp.rename(out)
                logger.debug("[%s] H.264 faststart re-encode OK", event.camera_id)
            except subprocess.CalledProcessError as e:
                logger.warning("[%s] ffmpeg H.264 encode failed (%s); keeping original",
                               event.camera_id, e.stderr.decode()[:200] if e.stderr else "")
                if tmp.exists():
                    tmp.unlink()

        logger.info("[%s] clip for %s → %s (%d frames, %.1fs)",
                    event.camera_id, event.event_id, out.name, len(frames), span)
        return out
