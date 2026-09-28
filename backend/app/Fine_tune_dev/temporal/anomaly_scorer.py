"""Fuse Stage-2 signals into theft candidate events.

Consumes per-window outputs from the VideoMAEv2 action classifier (and, when
available, an RTFM-style anomaly score), smooths them, applies hysteresis
thresholding with a cooldown, and emits :class:`TheftCandidateEvent`s. Each
event carries the ``T-15s → T+15s`` snapshot window that the rolling buffer
(``buffering/``) must cut and forward to Stage-3 VLM verification — this is
the Stage-2 → Stage-3 contract from README_Architec.md.

Pure Python (no torch/numpy) so it can run anywhere in the pipeline and be
unit-tested without the ML stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass
class TheftCandidateEvent:
    """A flagged window that Stage 3 (VLM) should verify."""
    camera_id: str
    trigger_ts: float                 # stream timestamp (s) when the trigger fired
    score: float                      # smoothed fused theft score at trigger time
    top_action: str                   # most probable non-normal action class
    action_probs: dict[str, float]    # full class distribution at trigger time
    anomaly_score: float | None       # RTFM score if fused, else None
    clip_start_ts: float              # snapshot window for the rolling buffer
    clip_end_ts: float


@dataclass
class _CameraState:
    ema: float = 0.0
    consecutive: int = 0
    armed: bool = True
    last_fire_ts: float = float("-inf")
    last_probs: dict[str, float] = field(default_factory=dict)


class TheftEventAggregator:
    """Per-camera smoothing + hysteresis over fused theft scores.

    Trigger logic per camera:
      1. ``theft = Σ theft_class_weights[c] · p(c)`` from the action classifier.
      2. Fuse with the anomaly score (``max`` or ``weighted``), if provided.
      3. Exponential smoothing: ``ema = α·fused + (1-α)·ema``.
      4. Fire when the EMA stays ≥ ``trigger_threshold`` for ``min_consecutive``
         windows, the camera is armed, and ``cooldown_s`` has elapsed.
      5. Re-arm only after the EMA falls below ``rearm_threshold`` (hysteresis),
         so one long incident produces one event, not a stream of them.
    """

    def __init__(self, class_names: list[str], theft_class_weights: Mapping[str, float],
                 ema_alpha: float = 0.6, trigger_threshold: float = 0.7,
                 rearm_threshold: float = 0.4, min_consecutive: int = 2,
                 cooldown_s: float = 30.0, clip_pre_s: float = 15.0,
                 clip_post_s: float = 15.0, fusion_mode: str = "max",
                 anomaly_weight: float = 0.4,
                 on_event: Callable[[TheftCandidateEvent], None] | None = None):
        unknown = set(theft_class_weights) - set(class_names)
        if unknown:
            raise ValueError(f"theft_class_weights refer to unknown classes: {unknown}")
        if fusion_mode not in ("max", "weighted"):
            raise ValueError(f"fusion_mode must be 'max' or 'weighted', got {fusion_mode}")
        self.class_names = list(class_names)
        self.theft_class_weights = dict(theft_class_weights)
        self.ema_alpha = ema_alpha
        self.trigger_threshold = trigger_threshold
        self.rearm_threshold = rearm_threshold
        self.min_consecutive = max(1, min_consecutive)
        self.cooldown_s = cooldown_s
        self.clip_pre_s = clip_pre_s
        self.clip_post_s = clip_post_s
        self.fusion_mode = fusion_mode
        self.anomaly_weight = anomaly_weight
        self.on_event = on_event
        self._state: dict[str, _CameraState] = {}

    @classmethod
    def from_config(cls, inference_cfg: dict, data_cfg: dict,
                    on_event: Callable[[TheftCandidateEvent], None] | None = None
                    ) -> "TheftEventAggregator":
        fusion = inference_cfg.get("fusion", {})
        return cls(
            class_names=list(data_cfg["classes"]),
            theft_class_weights=data_cfg.get("theft_class_weights", {}),
            ema_alpha=float(inference_cfg.get("ema_alpha", 0.6)),
            trigger_threshold=float(inference_cfg.get("trigger_threshold", 0.7)),
            rearm_threshold=float(inference_cfg.get("rearm_threshold", 0.4)),
            min_consecutive=int(inference_cfg.get("min_consecutive", 2)),
            cooldown_s=float(inference_cfg.get("cooldown_s", 30.0)),
            clip_pre_s=float(inference_cfg.get("clip_pre_s", 15.0)),
            clip_post_s=float(inference_cfg.get("clip_post_s", 15.0)),
            fusion_mode=str(fusion.get("mode", "max")),
            anomaly_weight=float(fusion.get("anomaly_weight", 0.4)),
            on_event=on_event,
        )

    def theft_score(self, probs: Mapping[str, float]) -> float:
        s = sum(w * float(probs.get(c, 0.0)) for c, w in self.theft_class_weights.items())
        return min(max(s, 0.0), 1.0)

    def update(self, camera_id: str, ts: float, probs: Mapping[str, float] | list[float],
               anomaly_score: float | None = None) -> TheftCandidateEvent | None:
        """Feed one classifier window; returns an event if the trigger fires."""
        if not isinstance(probs, Mapping):
            if len(probs) != len(self.class_names):
                raise ValueError(f"expected {len(self.class_names)} probs, got {len(probs)}")
            probs = {c: float(p) for c, p in zip(self.class_names, probs)}

        st = self._state.setdefault(camera_id, _CameraState())
        action = self.theft_score(probs)
        if anomaly_score is None:
            fused = action
        elif self.fusion_mode == "max":
            fused = max(action, float(anomaly_score))
        else:
            fused = (1.0 - self.anomaly_weight) * action + self.anomaly_weight * float(anomaly_score)

        st.ema = self.ema_alpha * fused + (1.0 - self.ema_alpha) * st.ema
        st.last_probs = dict(probs)

        if st.ema >= self.trigger_threshold:
            st.consecutive += 1
        else:
            st.consecutive = 0
            if st.ema < self.rearm_threshold:
                st.armed = True

        should_fire = (st.armed and st.consecutive >= self.min_consecutive
                       and ts - st.last_fire_ts >= self.cooldown_s)
        if not should_fire:
            return None

        st.armed = False
        st.last_fire_ts = ts
        theft_only = {c: probs.get(c, 0.0) for c in self.theft_class_weights}
        event = TheftCandidateEvent(
            camera_id=camera_id,
            trigger_ts=ts,
            score=round(st.ema, 4),
            top_action=max(theft_only, key=theft_only.get),
            action_probs={c: round(float(p), 4) for c, p in probs.items()},
            anomaly_score=None if anomaly_score is None else round(float(anomaly_score), 4),
            clip_start_ts=max(0.0, ts - self.clip_pre_s),
            clip_end_ts=ts + self.clip_post_s,
        )
        logger.info("Theft candidate: cam=%s t=%.1fs score=%.2f action=%s",
                    camera_id, ts, event.score, event.top_action)
        if self.on_event:
            self.on_event(event)
        return event

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._state.clear()
        else:
            self._state.pop(camera_id, None)
