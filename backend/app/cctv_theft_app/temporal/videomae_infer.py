
"""TEMPORARY Stage-2 model — VideoMAEv2 embedding-novelty scorer.

Chosen as the interim implementation (over StreamVAD / I3D+MIL) because the
final fine-tuned model is also a VideoMAEv2: identical preprocessing, window
geometry, and checkpoint tooling, so cutting over later is a config change.

How it works: the frozen K710-distilled backbone embeds each window; a
per-camera EMA of recent embeddings forms a "normal behavior" baseline, and
the cosine distance of the current window to that baseline becomes the theft
score. This flags behavioral novelty (sudden unusual motion patterns), not
theft semantics — an honest stopgap.

Deploy it with a high trigger threshold and let the VLM verifier carry
precision until ``mode: finetuned`` is ready.

If ``rtfm.enabled``, an RTFM-style anomaly head scores the same embeddings and
is reported separately (fused downstream by the aggregator).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pipeline.events import TemporalResult
from temporal.base import TemporalModel
from temporal.rtfm_infer import RTFMHead

logger = logging.getLogger(__name__)


@dataclass
class _CameraBaseline:
    mean: np.ndarray | None = None
    count: int = 0


class VideoMAEv2NoveltyModel(TemporalModel):
    name = "videomaev2_novelty"

    def __init__(self, cfg: dict, device: str = "cuda"):
        import onnxruntime
        from pathlib import Path

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------
        wcfg = cfg["window"]
        bcfg = cfg["backbone"]
        ncfg = cfg.get("novelty", {})

        self.device = device

        self.ema_alpha = float(
            ncfg.get("ema_alpha", 0.05)
        )

        self.scale = float(
            ncfg.get("scale", 4.0)
        )

        self.warmup_windows = int(
            ncfg.get("warmup_windows", 10)
        )

        self._baselines: dict[str, _CameraBaseline] = {}

        # ---------------------------------------------------------
        # Optional RTFM anomaly head
        # ---------------------------------------------------------
        self.rtfm = None

        # ---------------------------------------------------------
        # VideoMAEv2 ONNX model
        # ---------------------------------------------------------
        onnx_path = str(
            Path(__file__).resolve().parent.parent / "videomaev2.onnx"
        )

        # ---------------------------------------------------------
        # ONNX Runtime execution provider
        #
        # PyTorch ROCm uses the "cuda" API namespace.
        # ONNX Runtime ROCm uses ROCMExecutionProvider.
        #
        # Therefore:
        #
        #     device == "cuda"
        #          ↓
        #     ROCMExecutionProvider
        #          ↓
        #     AMD GPU
        #
        # CPU is kept as fallback.
        # ---------------------------------------------------------
        # if self.device == "cuda":
        #     providers = [
        #         "ROCMExecutionProvider",
        #         "CPUExecutionProvider",
        #     ]
        # else:
        #     providers = [
        #         "CPUExecutionProvider",
        #     ]
        providers = ["CPUExecutionProvider"]

        # ---------------------------------------------------------
        # Create ONNX Runtime session ONCE
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # ONNX Runtime CPU thread control
        # ---------------------------------------------------------
        session_options = onnxruntime.SessionOptions()

        # Prevent excessive CPU thread usage.
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        # Keep graph optimization enabled.
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # ---------------------------------------------------------
        # Create ONNX Runtime session ONCE
        # ---------------------------------------------------------
        self.ort_session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers,
        )

        # ---------------------------------------------------------
        # Log requested and active providers
        # ---------------------------------------------------------
        logger.info(
            "ONNX providers requested: %s",
            providers,
        )

        logger.info(
            "ONNX providers active: %s",
            self.ort_session.get_providers(),
        )

        logger.info(
            "Temporal model ready: %s | device=%s | model=%s",
            self.name,
            self.device,
            onnx_path,
        )

    def _embed(self, clip: np.ndarray) -> np.ndarray:
        """Generate VideoMAEv2 embedding for one temporal clip."""

        # Add batch dimension and ensure contiguous float32 input.
        x = np.ascontiguousarray(
            clip[None]
        ).astype(
            np.float32
        )

        # Get ONNX input name dynamically.
        input_name = self.ort_session.get_inputs()[0].name

        ort_inputs = {
            input_name: x
        }

        # Run ONNX inference.
        feat = self.ort_session.run(
            None,
            ort_inputs,
        )[0]

        return feat[0]

    def process_window(
        self,
        camera_id: str,
        ts: float,
        clip: np.ndarray,
    ) -> TemporalResult:
        """Process one temporal window and calculate novelty score."""

        # ---------------------------------------------------------
        # Generate embedding
        # ---------------------------------------------------------
        emb = self._embed(clip)

        # Normalize current embedding.
        unit = emb / (
            np.linalg.norm(emb) + 1e-8
        )

        # Get/create camera-specific baseline.
        bl = self._baselines.setdefault(
            camera_id,
            _CameraBaseline(),
        )

        # ---------------------------------------------------------
        # First window
        # ---------------------------------------------------------
        if bl.mean is None:
            bl.mean = unit.copy()
            score = 0.0

        # ---------------------------------------------------------
        # Existing baseline
        # ---------------------------------------------------------
        else:
            baseline = bl.mean / (
                np.linalg.norm(bl.mean) + 1e-8
            )

            # Cosine distance.
            novelty = 1.0 - float(
                np.dot(unit, baseline)
            )

            # Scale novelty into [0, 1].
            score = float(
                np.clip(
                    novelty * self.scale,
                    0.0,
                    1.0,
                )
            )

            # During warmup, don't trigger theft.
            if bl.count < self.warmup_windows:
                score = 0.0

            # Update EMA baseline.
            bl.mean = (
                (1.0 - self.ema_alpha) * bl.mean
                + self.ema_alpha * unit
            )

        # Increment number of processed windows.
        bl.count += 1

        # ---------------------------------------------------------
        # Optional RTFM anomaly score
        # ---------------------------------------------------------
        anomaly = (
            self.rtfm.score(emb)
            if self.rtfm is not None
            else None
        )

        # ---------------------------------------------------------
        # Return temporal result
        # ---------------------------------------------------------
        return TemporalResult(
            camera_id=camera_id,
            ts=ts,
            theft_score=score,
            action_probs=None,
            anomaly_score=anomaly,
            model_name=self.name,
        )

    def reset(
        self,
        camera_id: str | None = None,
    ) -> None:
        """Reset camera baseline(s)."""

        if camera_id is None:
            # Reset all camera baselines.
            self._baselines.clear()

            logger.info(
                "Reset all VideoMAEv2 novelty baselines."
            )

        else:
            # Reset one camera baseline.
            self._baselines.pop(
                camera_id,
                None,
            )

            logger.info(
                "Reset VideoMAEv2 novelty baseline for camera %s.",
                camera_id,
            )

