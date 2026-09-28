"""Mock Stage-2 model — scripted theft scores for end-to-end pipeline tests.

Runs the whole application (ingest → windows → trigger → clip → VLM → alerts)
with no GPU, weights, or torch. Script events in configs/temporal_model.yaml:

    mock:
      script:
        - [8.0, 14.0, 0.95]      # stream-seconds 8–14 → theft score 0.95

An empty script emits low deterministic noise (never triggers).
"""

from __future__ import annotations

import logging
import random

import numpy as np

from pipeline.events import TemporalResult
from temporal.base import TemporalModel

logger = logging.getLogger(__name__)


class MockTemporalModel(TemporalModel):
    name = "mock"

    def __init__(self, cfg: dict):
        self.script = [(float(a), float(b), float(s))
                       for a, b, s in (cfg.get("mock", {}).get("script") or [])]
        self._rng = random.Random(1234)
        logger.info("Temporal model ready: MOCK (%d scripted events)", len(self.script))

    def process_window(self, camera_id: str, ts: float, clip: np.ndarray) -> TemporalResult:
        score = next((s for a, b, s in self.script if a <= ts <= b),
                     self._rng.uniform(0.01, 0.08))
        return TemporalResult(camera_id=camera_id, ts=ts, theft_score=score,
                              action_probs={"normal": 1.0 - score, "concealment": score},
                              anomaly_score=None, model_name=self.name)
