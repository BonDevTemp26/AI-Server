"""RTFM-style anomaly head — the "safety net for unseen behaviors".

RTFM (Robust Temporal Feature Magnitude learning) scores snippets by learning
that anomalous snippets have larger, separable feature magnitudes. This module
is the *inference* half: a compact scoring head over VideoMAEv2 window
embeddings, magnitude feature included, matching what
``training/train_rtfm.py`` (Fine_tune_dev roadmap item) will produce.

Until a trained head exists the config ships ``rtfm.enabled: false`` and the
pipeline runs on the action/novelty score alone — the aggregator already
handles ``anomaly_score=None``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RTFMHead:
    """Embedding (C,) → anomaly score in [0, 1]."""

    def __init__(self, embed_dim: int, checkpoint: str, device: str = "cuda"):
        import torch
        import torch.nn as nn
        self._torch = torch

        self.net = nn.Sequential(                 # input: embedding ⊕ ||embedding||
            nn.Linear(embed_dim + 1, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state.get("model", state))
        self.net.eval().to(device)
        self.device = device
        logger.info("RTFM anomaly head loaded from %s", checkpoint)

    @classmethod
    def from_config(cls, rtfm_cfg: dict, embed_dim: int,
                    device: str = "cuda") -> "RTFMHead | None":
        """Returns None (head disabled) unless enabled AND weights exist."""
        if not rtfm_cfg.get("enabled", False):
            return None
        ckpt = Path(rtfm_cfg.get("checkpoint", "models/rtfm_head.pth"))
        if not ckpt.is_file():
            logger.warning("rtfm.enabled=true but %s is missing — anomaly head "
                           "disabled (train it via the Fine_tune_dev training stack)", ckpt)
            return None
        return cls(embed_dim, str(ckpt), device)

    def score(self, embedding) -> float:
        torch = self._torch
        with torch.inference_mode():
            x = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
            x = torch.cat([x, x.norm().unsqueeze(0)]).unsqueeze(0)
            return float(self.net(x).item())
