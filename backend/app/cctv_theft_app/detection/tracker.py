"""Detector/tracker factory + mock backend.

ByteTrack itself is configured declaratively in ``configs/tracker.yaml`` and
executed inside Ultralytics (see ``yolo26_infer.YoloDetectorTracker``); this
module provides the common construction point and a dependency-free mock so
the whole pipeline can be integration-tested without GPU, weights, or
ultralytics installed.
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline.events import Detection

logger = logging.getLogger(__name__)


class MockDetectorTracker:
    """Emits one persistent 'person' track per frame — pipeline plumbing tests."""

    def __init__(self, cfg: dict | None = None, device: str = "cpu"):
        self.model_name = "mock"
        self._n = 0

    def process(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._n += 1
        h, w = frame_bgr.shape[:2]
        x = (self._n * 3) % max(1, w - w // 4)          # person drifting across frame
        return [Detection(bbox=(float(x), h * 0.2, float(x + w // 4), h * 0.9),
                          class_id=0, class_name="person", confidence=0.9, track_id=1)]

    def reset(self) -> None:
        self._n = 0


def build_detector(cfg: dict, backend: str = "yolo", device: str = "cuda"):
    """backend: 'yolo' (YOLO26-S + ByteTrack) | 'mock'."""
    if backend == "mock":
        logger.info("Detector backend: MOCK (no real detection)")
        return MockDetectorTracker(cfg, device)
    if backend == "yolo":
        from detection.yolo26_infer import YoloDetectorTracker
        return YoloDetectorTracker(cfg, device)
    raise ValueError(f"Unknown detector backend: {backend!r} (yolo | mock)")
