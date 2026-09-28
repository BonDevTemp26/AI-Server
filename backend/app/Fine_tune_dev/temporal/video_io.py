"""Video decoding utilities with graceful backend fallback.

Backend preference: decord (fast random access, x86) → PyAV → OpenCV.
All functions return RGB uint8 numpy arrays. Clips in this pipeline are short
(2–10 s), so the sequential fallback paths stay cheap.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

try:
    import decord
    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_DECORD = False

try:
    import av
    _HAS_AV = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_AV = False

import cv2


def probe(path: str) -> tuple[int, float]:
    """Return ``(num_frames, fps)`` for a video file."""
    if _HAS_DECORD:
        vr = decord.VideoReader(path, num_threads=1)
        return len(vr), float(vr.get_avg_fps())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return n, float(fps)


def read_frames(path: str, indices: list[int] | np.ndarray) -> np.ndarray:
    """Decode the frames at ``indices`` (RGB, shape ``(len(indices), H, W, 3)``).

    Indices may repeat (short clips get padded by repeating the last frame) but
    must be sorted ascending for the sequential backends.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if _HAS_DECORD:
        try:
            vr = decord.VideoReader(path, num_threads=1)
            clipped = np.clip(indices, 0, len(vr) - 1)
            return vr.get_batch(clipped).asnumpy()
        except Exception as exc:
            logger.debug("decord failed on %s (%s); falling back", path, exc)
    if _HAS_AV:
        try:
            return _read_frames_av(path, indices)
        except Exception as exc:
            logger.debug("PyAV failed on %s (%s); falling back to OpenCV", path, exc)
    return _read_frames_cv2(path, indices)


def _read_frames_av(path: str, indices: np.ndarray) -> np.ndarray:
    wanted = set(int(i) for i in indices)
    last_wanted = max(wanted)
    frames: dict[int, np.ndarray] = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
            if i >= last_wanted:
                break
    return _assemble(frames, indices, path)


def _read_frames_cv2(path: str, indices: np.ndarray) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    wanted = set(int(i) for i in indices)
    last_wanted = max(wanted)
    frames: dict[int, np.ndarray] = {}
    i = 0
    while i <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return _assemble(frames, indices, path)


def _assemble(frames: dict[int, np.ndarray], indices: np.ndarray, path: str) -> np.ndarray:
    if not frames:
        raise IOError(f"No decodable frames in {path}")
    max_avail = max(frames)
    out = []
    for idx in indices:
        idx = int(idx)
        while idx not in frames and idx > 0:   # pad short videos with nearest earlier frame
            idx = idx - 1 if idx <= max_avail else max_avail
        out.append(frames.get(idx, frames[max_avail]))
    return np.stack(out)


def iter_stream(path: str) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, frame_rgb)`` sequentially — simulates a live feed
    for offline testing of the streaming inference path."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
