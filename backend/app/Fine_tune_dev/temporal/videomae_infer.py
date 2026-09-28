"""Streaming inference for the fine-tuned VideoMAEv2 theft classifier.

Turns per-camera frame streams into per-window action probabilities and feeds
them to the :class:`~temporal.anomaly_scorer.TheftEventAggregator`. Designed to
sit behind Stage 1 (detection/tracking): call ``process_frame`` for every
decoded frame; pass ``persons_present=False`` when the tracker sees nobody so
the expensive temporal model is skipped on empty scenes.

Offline smoke test on a recorded file (simulates a live feed):

    python -m temporal.videomae_infer --video data/raw_clips/sample.mp4 \
        --config configs/temporal_model.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from temporal.anomaly_scorer import TheftCandidateEvent, TheftEventAggregator
from temporal.videomae_model import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
_STD = np.array(IMAGENET_STD, dtype=np.float32) * 255.0


class ClipPreprocessor:
    """Frame → model-ready array: short-side resize, center crop, normalize."""

    def __init__(self, input_size: int = 224):
        self.input_size = input_size

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) RGB uint8 → (3, S, S) float32 normalized."""
        s = self.input_size
        h, w = frame_rgb.shape[:2]
        scale = s / min(h, w)
        frame = cv2.resize(frame_rgb, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_LINEAR)
        h, w = frame.shape[:2]
        top, left = (h - s) // 2, (w - s) // 2
        frame = frame[top:top + s, left:left + s]
        return ((frame.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1)


class VideoMAEv2Classifier:
    """Clip → class probabilities. Backends: fine-tuned torch checkpoint or ONNX.

    * ``backend='torch'``: loads a self-describing checkpoint produced by
      ``training/train_videomae.py`` (``best.pth``).
    * ``backend='onnx'``: loads the export from ``temporal/export_videomae.py``;
      onnxruntime picks TensorRT/CUDA/CPU execution providers in that order.
    """

    def __init__(self, checkpoint: str, backend: str = "torch", device: str = "cuda",
                 class_names: list[str] | None = None):
        self.backend = backend
        self.device = device
        if backend == "torch":
            import torch
            from temporal.videomae_finetune import load_for_inference
            self._torch = torch
            self.model, self.class_names, model_cfg = load_for_inference(checkpoint, device)
            self.num_frames = model_cfg["num_frames"]
            self.input_size = model_cfg["input_size"]
            self._amp_dtype = (torch.bfloat16 if device == "cuda"
                               and torch.cuda.is_bf16_supported() else None)
        elif backend == "onnx":
            import json
            from pathlib import Path

            import onnxruntime as ort
            providers = [p for p in ("VitisAIExecutionProvider", "TensorrtExecutionProvider", "CUDAExecutionProvider",
                                     )
                         if p in ort.get_available_providers()]
            self.session = ort.InferenceSession(checkpoint, providers=providers)
            self._input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape  # (N, 3, T, S, S)
            self.num_frames, self.input_size = int(shape[2]), int(shape[3])
            sidecar = Path(checkpoint).with_suffix(".labels.json")
            if class_names:
                self.class_names = list(class_names)
            elif sidecar.is_file():
                self.class_names = json.loads(sidecar.read_text())["class_names"]
            else:
                raise ValueError(f"Provide class_names or a sidecar file {sidecar}")
            logger.info("ONNX session using providers: %s", self.session.get_providers())
        else:
            raise ValueError(f"backend must be 'torch' or 'onnx', got {backend}")

    def predict(self, clips: np.ndarray) -> np.ndarray:
        """(N, 3, T, S, S) float32 → softmax probabilities (N, num_classes)."""
        clips = np.ascontiguousarray(clips, dtype=np.float32)
        if self.backend == "torch":
            torch = self._torch
            with torch.inference_mode():
                x = torch.from_numpy(clips).to(self.device, non_blocking=True)
                with torch.autocast(self.device, dtype=self._amp_dtype,
                                    enabled=self._amp_dtype is not None):
                    logits = self.model(x)
                return torch.softmax(logits.float(), dim=-1).cpu().numpy()
        logits = self.session.run(None, {self._input_name: clips})[0]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


@dataclass
class _StreamState:
    frames: deque                    # (ts, preprocessed frame) at model sampling fps
    last_sample_ts: float = float("-inf")
    last_infer_ts: float = float("-inf")


class StreamingTemporalAnalyzer:
    """Per-camera sliding-window driver around the classifier + aggregator.

    Frames are subsampled to the fps the model was trained at
    (``clip_fps / sampling_rate``, e.g. 25/4 = 6.25 fps), buffered into
    ``num_frames``-long windows (≈2.6 s of context), and classified every
    ``window_stride_s`` seconds.
    """

    def __init__(self, classifier: VideoMAEv2Classifier,
                 aggregator: TheftEventAggregator, clip_fps: float = 25.0,
                 sampling_rate: int = 4, window_stride_s: float = 1.0):
        self.classifier = classifier
        self.aggregator = aggregator
        self.preproc = ClipPreprocessor(classifier.input_size)
        self.frame_interval_s = sampling_rate / clip_fps
        self.window_stride_s = window_stride_s
        self._streams: dict[str, _StreamState] = {}

    def process_frame(self, camera_id: str, frame_rgb: np.ndarray, ts: float,
                      persons_present: bool = True,
                      anomaly_score: float | None = None
                      ) -> TheftCandidateEvent | None:
        """Feed one live frame; returns a candidate event when one fires."""
        st = self._streams.setdefault(
            camera_id, _StreamState(frames=deque(maxlen=self.classifier.num_frames)))

        if ts - st.last_sample_ts + 1e-6 >= self.frame_interval_s:
            st.frames.append((ts, self.preproc(frame_rgb)))
            st.last_sample_ts = ts

        window_full = len(st.frames) == self.classifier.num_frames
        due = ts - st.last_infer_ts >= self.window_stride_s
        if not (window_full and due and persons_present):
            return None

        st.last_infer_ts = ts
        clip = np.stack([f for _, f in st.frames], axis=1)[None]  # (1, 3, T, S, S)
        probs = self.classifier.predict(clip)[0]
        probs_map = dict(zip(self.classifier.class_names, probs.tolist()))
        return self.aggregator.update(camera_id, ts, probs_map, anomaly_score)

    def last_probs(self, camera_id: str) -> dict[str, float]:
        state = self.aggregator._state.get(camera_id)
        return dict(state.last_probs) if state else {}

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._streams.clear()
        else:
            self._streams.pop(camera_id, None)
        self.aggregator.reset(camera_id)


# ── Offline demo on a recorded file ──────────────────────────────────────────

def main() -> None:
    import yaml
    from pathlib import Path

    from temporal.video_io import iter_stream

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/temporal_model.yaml")
    ap.add_argument("--checkpoint", help="defaults to inference.checkpoint from config")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--camera-id", default="demo_cam")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    icfg, dcfg = cfg["inference"], cfg["data"]

    classifier = VideoMAEv2Classifier(args.checkpoint or icfg["checkpoint"],
                                      backend=args.backend, device=args.device,
                                      class_names=list(dcfg["classes"]))
    aggregator = TheftEventAggregator.from_config(icfg, dcfg)
    analyzer = StreamingTemporalAnalyzer(
        classifier, aggregator, clip_fps=float(dcfg.get("clip_fps", 25)),
        sampling_rate=int(dcfg.get("sampling_rate", 4)),
        window_stride_s=float(icfg.get("window_stride_s", 1.0)))

    events: list[TheftCandidateEvent] = []
    for ts, frame in iter_stream(args.video):
        event = analyzer.process_frame(args.camera_id, frame, ts)
        if event:
            events.append(event)
            print(f"\n🚨 t={ts:6.1f}s  score={event.score:.2f}  action={event.top_action}"
                  f"  clip=[{event.clip_start_ts:.1f}s → {event.clip_end_ts:.1f}s]")
        elif analyzer.last_probs(args.camera_id):
            probs = analyzer.last_probs(args.camera_id)
            top = max(probs, key=probs.get)
            print(f"t={ts:6.1f}s  theft={aggregator.theft_score(probs):.2f}  "
                  f"top={top}:{probs[top]:.2f}", end="\r")

    print(f"\n\n{len(events)} candidate event(s).")
    for e in events:
        print(f"  cam={e.camera_id} t={e.trigger_ts:.1f}s score={e.score:.2f} "
              f"action={e.top_action} clip=[{e.clip_start_ts:.1f}, {e.clip_end_ts:.1f}]s")


if __name__ == "__main__":
    main()
