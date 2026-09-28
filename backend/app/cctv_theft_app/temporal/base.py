"""Stage 2 core — the TemporalModel interface and sliding-window assembly.

**This is the integration point for the fine-tuned model.** Every temporal
implementation (temporary VideoMAEv2 novelty, the future fine-tuned
classifier, mocks, or a StreamVAD port later) implements :class:`TemporalModel`
and is selected purely by ``mode:`` in ``configs/temporal_model.yaml``:

    novelty    → temporal/videomae_infer.py   (temporary, ships today)
    finetuned  → temporal/finetuned_adapter.py (your model, when ready)
    mock       → temporal/mock_model.py        (integration tests)

The rest of the pipeline only ever sees ``TemporalResult``s.
"""

from __future__ import annotations

import abc
import logging
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.events import TemporalResult

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


class TemporalModel(abc.ABC):
    """Clip window → theft signal. Implementations must be stateless across
    cameras except via explicit per-camera state keyed by ``camera_id``."""

    name: str = "base"

    @abc.abstractmethod
    def process_window(self, camera_id: str, ts: float,
                       clip: np.ndarray) -> TemporalResult:
        """``clip``: float32 ``(3, T, S, S)``, ImageNet-normalized RGB."""

    def reset(self, camera_id: str | None = None) -> None:  # optional override
        pass


def preprocess_frame(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """BGR frame → (3, S, S) float32: short-side resize, center crop, normalize."""
    s = input_size
    h, w = frame_bgr.shape[:2]
    scale = s / min(h, w)
    frame = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]
    frame = frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


@dataclass
class _WindowState:
    frames: deque = field(default_factory=deque)
    last_sample_ts: float = float("-inf")
    last_emit_ts: float = float("-inf")


class WindowAssembler:
    """Subsamples live frames to the model's rate and emits clip windows.

    Frames arrive at camera fps; the model was trained at
    ``clip_fps / sampling_rate`` (default 25/4 = 6.25 fps) on ``num_frames``
    frames (≈2.6 s of context). A window is emitted at most every ``stride_s``.
    """

    def __init__(self, num_frames: int = 16, sampling_rate: int = 4,
                 clip_fps: float = 25.0, stride_s: float = 1.0,
                 input_size: int = 224):
        self.num_frames = num_frames
        self.input_size = input_size
        self.frame_interval_s = sampling_rate / clip_fps
        self.stride_s = stride_s
        self._state: dict[str, _WindowState] = {}

    def push(self, camera_id: str, frame_bgr: np.ndarray,
             ts: float) -> np.ndarray | None:
        """Feed one frame; returns a ``(3, T, S, S)`` clip when a window is due."""
        st = self._state.setdefault(camera_id, _WindowState(
            frames=deque(maxlen=self.num_frames)))
        if ts - st.last_sample_ts + 1e-6 >= self.frame_interval_s:
            st.frames.append(preprocess_frame(frame_bgr, self.input_size))
            st.last_sample_ts = ts
        if len(st.frames) < self.num_frames or ts - st.last_emit_ts < self.stride_s:
            return None
        st.last_emit_ts = ts
        return np.stack(st.frames, axis=1)          # (3, T, S, S)

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._state.clear()
        else:
            self._state.pop(camera_id, None)


def build_temporal_model(cfg: dict, mode: str | None = None,
                         device: str = "cuda") -> TemporalModel:
    """Factory — the only place the pipeline chooses an implementation."""
    mode = mode or cfg.get("mode", "novelty")
    if mode == "mock":
        from temporal.mock_model import MockTemporalModel
        return MockTemporalModel(cfg)
    if mode == "novelty":
        from temporal.videomae_infer import VideoMAEv2NoveltyModel
        return VideoMAEv2NoveltyModel(cfg, device=device)
    if mode == "finetuned":
        from temporal.finetuned_adapter import FinetunedVideoMAEv2Model
        return FinetunedVideoMAEv2Model(cfg, device=device)
    raise ValueError(f"Unknown temporal mode: {mode!r} (novelty | finetuned | mock)")
