"""Turn per-window TemporalResults into CandidateEvents (Stage 2 → 3 contract).

Per camera: fuse action/novelty score with the optional RTFM anomaly score,
EMA-smooth, then hysteresis-threshold with a cooldown so one incident produces
one event carrying the ``T-15s → T+15s`` snapshot window for the buffer.
Pure Python — unit-testable without the ML stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pipeline.events import CandidateEvent, TemporalResult

logger = logging.getLogger(__name__)


@dataclass
class _CamState:
    ema: float = 0.0
    consecutive: int = 0
    armed: bool = True
    last_fire_ts: float = float("-inf")


class TriggerAggregator:
    def __init__(self, trigger_cfg: dict, fusion_mode: str = "max",
                 anomaly_weight: float = 0.4):
        t = trigger_cfg
        self.ema_alpha = float(t.get("ema_alpha", 0.6))
        self.trigger_threshold = float(t.get("trigger_threshold", 0.7))
        self.rearm_threshold = float(t.get("rearm_threshold", 0.4))
        self.min_consecutive = max(1, int(t.get("min_consecutive", 2)))
        self.cooldown_s = float(t.get("cooldown_s", 30))
        self.clip_pre_s = float(t.get("clip_pre_s", 15))
        self.clip_post_s = float(t.get("clip_post_s", 15))
        if fusion_mode not in ("max", "weighted"):
            raise ValueError(f"fusion_mode must be max|weighted, got {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.anomaly_weight = float(anomaly_weight)
        self._state: dict[str, _CamState] = {}

    @classmethod
    def from_config(cls, temporal_cfg: dict) -> "TriggerAggregator":
        rtfm = temporal_cfg.get("rtfm", {})
        return cls(temporal_cfg.get("trigger", {}),
                   fusion_mode=str(rtfm.get("fusion_mode", "max")),
                   anomaly_weight=float(rtfm.get("anomaly_weight", 0.4)))

    def update(self, result: TemporalResult) -> CandidateEvent | None:
        st = self._state.setdefault(result.camera_id, _CamState())
        fused = result.theft_score
        if result.anomaly_score is not None:
            if self.fusion_mode == "max":
                fused = max(fused, result.anomaly_score)
            else:
                fused = ((1 - self.anomaly_weight) * fused
                         + self.anomaly_weight * result.anomaly_score)

        st.ema = self.ema_alpha * fused + (1 - self.ema_alpha) * st.ema
        if st.ema >= self.trigger_threshold:
            st.consecutive += 1
        else:
            st.consecutive = 0
            if st.ema < self.rearm_threshold:
                st.armed = True

        if not (st.armed and st.consecutive >= self.min_consecutive
                and result.ts - st.last_fire_ts >= self.cooldown_s):
            return None

        st.armed = False
        st.last_fire_ts = result.ts
        probs = result.action_probs or {}
        non_normal = {c: p for c, p in probs.items() if c != "normal"}
        top_action = (max(non_normal, key=non_normal.get) if non_normal
                      else result.model_name or "anomaly")
        event = CandidateEvent(
            camera_id=result.camera_id, trigger_ts=result.ts,
            score=round(st.ema, 4), top_action=top_action,
            action_probs={c: round(p, 4) for c, p in probs.items()},
            anomaly_score=result.anomaly_score,
            clip_start_ts=max(0.0, result.ts - self.clip_pre_s),
            clip_end_ts=result.ts + self.clip_post_s)
        logger.info("🚨 candidate %s: cam=%s t=%.1fs score=%.2f action=%s",
                    event.event_id, event.camera_id, event.trigger_ts,
                    event.score, event.top_action)
        return event

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._state.clear()
        else:
            self._state.pop(camera_id, None)
