"""Stage 1 — YOLO26-S detection + ByteTrack tracking (Ultralytics).

One call per frame: ``model.track(persist=True)`` runs the detector and keeps
ByteTrack state inside the Ultralytics model, yielding stable ``track_id``s.
Falls back to YOLO11-S automatically if the installed ultralytics version does
not ship YOLO26 weights yet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.events import Detection

logger = logging.getLogger(__name__)


class YoloDetectorTracker:
    def __init__(self, cfg: dict, device: str = "cuda"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("pip install ultralytics  (or use detector_backend: mock)") from exc

        self.cfg = cfg
        self.device = device
        self.classes = {int(k): v for k, v in cfg["classes_of_interest"].items()}
        self.tracker_config = cfg.get("tracker_config", "bytetrack.yaml")
        self.detect_every_n = max(1, int(cfg.get("detect_every_n", 1)))
        self._frame_count = 0
        self._last: list[Detection] = []

        try:
            self.model = YOLO(cfg["model"])
            self.model_name = cfg["model"]
        except Exception as exc:  # YOLO26 not available in this ultralytics release
            fallback = cfg.get("fallback_model", "yolo11s.pt")
            logger.warning("Could not load %s (%s) — falling back to %s",
                           cfg["model"], exc, fallback)
            self.model = YOLO(fallback)
            self.model_name = fallback
        self._precision = self._precision_kwargs()
        logger.info("Detector ready: %s on %s (tracker: %s)",
                    self.model_name, device, Path(self.tracker_config).name)

    def _precision_kwargs(self) -> dict:
        """fp16 inference flag for the installed ultralytics version.

        Ultralytics ≥8.4 replaced ``half=True`` with ``quantize="fp16"`` (the
        old flag still works but warns on every call). Introspect the default
        config to pass whichever argument this installation understands.
        """
        if not (bool(self.cfg.get("half", True)) and self.device != "cpu"):
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                return {"quantize": "fp16"}
        except Exception:
            pass
        return {"half": True}

    def process(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect + track one frame; returns tracked Detections of interest."""
        self._frame_count += 1
        if (self._frame_count - 1) % self.detect_every_n != 0:
            return self._last                      # coast on the previous tracks

        results = self.model.track(
            frame_bgr,
            persist=True,                          # keep ByteTrack state across calls
            tracker=self.tracker_config,
            conf=float(self.cfg.get("conf_threshold", 0.3)),
            iou=float(self.cfg.get("iou_threshold", 0.5)),
            imgsz=int(self.cfg.get("imgsz", 640)),
            classes=list(self.classes),
            device=self.device,
            verbose=False,
            **self._precision,
        )
        boxes = results[0].boxes
        detections: list[Detection] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids = (boxes.id.cpu().numpy().astype(int)
                   if boxes.id is not None else [None] * len(cls))
            for i in range(len(cls)):
                detections.append(Detection(
                    bbox=tuple(float(v) for v in xyxy[i]),
                    class_id=int(cls[i]),
                    class_name=self.classes.get(int(cls[i]), str(cls[i])),
                    confidence=float(conf[i]),
                    track_id=None if ids[i] is None else int(ids[i]),
                ))
        self._last = detections
        return detections

    def reset(self) -> None:
        """Clear tracker state (call between distinct video sources)."""
        if hasattr(self.model, "predictor") and self.model.predictor:
            trackers = getattr(self.model.predictor, "trackers", None)
            if trackers:
                for t in trackers:
                    t.reset()
        self._frame_count, self._last = 0, []
