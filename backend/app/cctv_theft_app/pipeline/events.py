"""Shared datatypes flowing between the pipeline stages.

Stage 1 (detection)   → FrameResult
Stage 2 (temporal)    → TemporalResult → CandidateEvent
Stage 3 (VLM)         → VerificationResult
Stage 4 (alerts)      → Alert
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Detection:
    """One tracked object in a frame (Stage 1 output)."""
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None               # stable ByteTrack id


@dataclass
class FrameResult:
    camera_id: str
    ts: float                                  # stream time (s from stream start)
    detections: list[Detection] = field(default_factory=list)

    @property
    def person_track_ids(self) -> list[int]:
        return [d.track_id for d in self.detections
                if d.class_name == "person" and d.track_id is not None]

    @property
    def persons_present(self) -> bool:
        return any(d.class_name == "person" for d in self.detections)


@dataclass
class TemporalResult:
    """One classified sliding window (Stage 2 output, pre-aggregation)."""
    camera_id: str
    ts: float
    theft_score: float                          # 0..1 signal used for triggering
    action_probs: dict[str, float] | None = None  # fine-tuned classifier only
    anomaly_score: float | None = None          # RTFM head, if enabled
    model_name: str = ""


@dataclass
class CandidateEvent:
    """A flagged window to snapshot and verify (Stage 2 → Stage 3 contract)."""
    camera_id: str
    trigger_ts: float
    score: float
    top_action: str
    action_probs: dict[str, float]
    anomaly_score: float | None
    clip_start_ts: float                        # T-15s
    clip_end_ts: float                          # T+15s
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)


@dataclass
class VerificationResult:
    """VLM verdict on a candidate clip (Stage 3 output)."""
    verdict: str                                # confirmed | rejected | uncertain
    confidence: float
    description: str
    model: str = ""
    raw_response: str = ""


@dataclass
class Alert:
    """Final, verified alert dispatched by Stage 4."""
    event_id: str
    camera_id: str
    trigger_ts: float
    score: float
    top_action: str
    verdict: str
    vlm_confidence: float
    description: str
    clip_path: str | None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
