"""Stage 2 — Temporal understanding (VideoMAEv2 action recognition).

Public API used by the rest of the pipeline:

    from temporal import (
        VideoMAEv2Classifier,        # clip -> action probabilities
        StreamingTemporalAnalyzer,   # per-camera sliding-window wrapper
        TheftEventAggregator,        # score fusion + hysteresis -> candidate events
        TheftCandidateEvent,
    )
"""

from temporal.anomaly_scorer import TheftCandidateEvent, TheftEventAggregator

__all__ = [
    "ClipPreprocessor",
    "StreamingTemporalAnalyzer",
    "TheftCandidateEvent",
    "TheftEventAggregator",
    "VideoMAEv2Classifier",
]

_LAZY = {"ClipPreprocessor", "StreamingTemporalAnalyzer", "VideoMAEv2Classifier"}


def __getattr__(name):  # keep cv2/torch out of pure-Python imports (PEP 562)
    if name in _LAZY:
        from temporal import videomae_infer

        return getattr(videomae_infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
