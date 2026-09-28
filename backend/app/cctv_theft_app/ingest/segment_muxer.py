"""FFmpeg segment-muxer rolling buffer (the production RTSP path).

Runs one ffmpeg process per camera that continuously records the stream into a
ring of short on-disk segments **without re-encoding** (``-c copy``), exactly
as recommended in README_Architec.md. Snapshotting ``T-15s → T+15s`` then
means concatenating the segments that cover the window and trimming.

Use this instead of the in-memory ring buffer (``buffering/ring_buffer.py``)
when sources are real RTSP cameras: zero CPU for buffering, survives app
restarts, and the extracted clip is the camera's original encoded video.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SegmentMuxer:
    def __init__(self, camera_id: str, rtsp_url: str, out_dir: str | Path,
                 segment_s: int = 5, buffer_duration_s: int = 90):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.segment_s = segment_s
        self.dir = Path(out_dir) / camera_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.num_segments = max(4, buffer_duration_s // segment_s)
        self._proc: subprocess.Popen | None = None
        self._start_wall: float | None = None

    def start(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp", "-i", self.rtsp_url,
            "-c", "copy", "-an",
            "-f", "segment", "-segment_time", str(self.segment_s),
            "-segment_wrap", str(self.num_segments),
            "-reset_timestamps", "1", "-strftime", "0",
            str(self.dir / "seg_%04d.ts"),
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        self._start_wall = time.time()
        logger.info("[%s] segment muxer recording (%d × %ds ring in %s)",
                    self.camera_id, self.num_segments, self.segment_s, self.dir)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _segments_covering(self, start_wall: float, end_wall: float) -> list[Path]:
        """Segments whose [mtime - segment_s, mtime] range overlaps the window."""
        segs = []
        for p in self.dir.glob("seg_*.ts"):
            mtime = p.stat().st_mtime
            if mtime >= start_wall - 1.0 and mtime - self.segment_s <= end_wall + 1.0:
                segs.append((mtime, p))
        return [p for _, p in sorted(segs)]

    def extract_clip(self, start_wall: float, end_wall: float, out_path: str | Path) -> Path:
        """Cut [start_wall, end_wall] (wall-clock seconds) into one mp4."""
        segs = self._segments_covering(start_wall, end_wall)
        if not segs:
            raise FileNotFoundError(f"[{self.camera_id}] no segments cover the window")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        concat = self.dir / f"concat_{int(start_wall)}.txt"
        concat.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
        first_start = segs[0].stat().st_mtime - self.segment_s
        offset = max(0.0, start_wall - first_start)
        duration = max(1.0, end_wall - start_wall)
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "concat", "-safe", "0", "-i", str(concat),
                 "-ss", f"{offset:.2f}", "-t", f"{duration:.2f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an",
                 "-movflags", "+faststart", str(out_path)],
                check=True)
        finally:
            concat.unlink(missing_ok=True)
        logger.info("[%s] extracted %s (%.0fs)", self.camera_id, out_path, duration)
        return out_path

    def state(self) -> dict:
        return {"camera_id": self.camera_id, "running": self.running,
                "segments": len(list(self.dir.glob("seg_*.ts"))),
                "started_at": self._start_wall}


if __name__ == "__main__":   # quick manual check: record 30 s then cut 10 s
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--out-dir", default="data/segments")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    mux = SegmentMuxer("test", args.rtsp, args.out_dir)
    mux.start()
    time.sleep(30)
    now = time.time()
    mux.extract_clip(now - 15, now - 5, "data/clips/segment_test.mp4")
    mux.stop()
    print(json.dumps(mux.state(), indent=2))
