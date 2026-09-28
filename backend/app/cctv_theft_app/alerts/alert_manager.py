"""Stage 4 — alert fan-out and persistence.

Every verified alert is (a) appended to the JSONL alert log, (b) appended to
the human-review queue consumed by the dashboard, and (c) dispatched to each
enabled channel. Channel failures are logged, never fatal — a dead webhook
must not stop the detection loop.

New channels (FCM, Twilio, ...) implement ``send(alert, clip_path)`` +
``name`` and get registered in ``_CHANNEL_BUILDERS``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from pipeline.events import Alert, CandidateEvent, VerificationResult

logger = logging.getLogger(__name__)


def _build_mqtt(cfg):
    from alerts.mqtt_publisher import MqttPublisher
    return MqttPublisher(cfg)


def _build_telegram(cfg):
    from alerts.telegram_notifier import TelegramNotifier
    return TelegramNotifier(cfg)


def _build_webhook(cfg):
    from alerts.webhook_dispatcher import WebhookDispatcher
    return WebhookDispatcher(cfg)


_CHANNEL_BUILDERS = {"mqtt": _build_mqtt, "telegram": _build_telegram,
                     "webhook": _build_webhook}


class AlertManager:
    def __init__(self, alerts_cfg: dict, alerts_log: str | Path,
                 review_log: str | Path):
        self.alerts_log = Path(alerts_log)
        self.review_log = Path(review_log)
        for p in (self.alerts_log, self.review_log):
            p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        channels = alerts_cfg.get("channels", {})
        self.console = bool(channels.get("console", {}).get("enabled", True))
        self.notifiers = []
        for name, builder in _CHANNEL_BUILDERS.items():
            ccfg = channels.get(name, {})
            if not ccfg.get("enabled", False):
                continue
            try:
                self.notifiers.append(builder(ccfg))
                logger.info("Alert channel enabled: %s", name)
            except Exception as exc:
                logger.error("Alert channel %s disabled (init failed): %s", name, exc)

    # ── persistence ─────────────────────────────────────────────────────────
    def _append(self, path: Path, record: dict) -> None:
        with self._lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_review_item(self, event: CandidateEvent, result: VerificationResult,
                        clip_path: str | None, alerted: bool) -> None:
        """Every verified candidate goes to the review queue — including VLM
        rejections, which become hard negatives for retraining."""
        self._append(self.review_log, {
            "event_id": event.event_id, "camera_id": event.camera_id,
            "trigger_ts": event.trigger_ts, "score": event.score,
            "top_action": event.top_action, "action_probs": event.action_probs,
            "anomaly_score": event.anomaly_score, "vlm_verdict": result.verdict,
            "vlm_confidence": result.confidence, "vlm_description": result.description,
            "vlm_raw": (result.raw_response or "")[:500],   # debugging weak/odd models
            "clip_path": clip_path, "alerted": alerted,
            "human_label": None, "created_at": time.time(),
        })

    # ── dispatch ────────────────────────────────────────────────────────────
    def dispatch(self, event: CandidateEvent, result: VerificationResult,
                 clip_path: str | None) -> Alert:
        alert = Alert(event_id=event.event_id, camera_id=event.camera_id,
                      trigger_ts=event.trigger_ts, score=event.score,
                      top_action=event.top_action, verdict=result.verdict,
                      vlm_confidence=result.confidence,
                      description=result.description,
                      clip_path=str(clip_path) if clip_path else None)
        self._append(self.alerts_log, alert.to_dict())
        if self.console:
            logger.warning("🚨 ALERT %s | cam=%s action=%s score=%.2f vlm=%s | clip=%s",
                           alert.event_id, alert.camera_id, alert.top_action,
                           alert.score, alert.verdict, alert.clip_path)
        for notifier in self.notifiers:
            try:
                notifier.send(alert, alert.clip_path)
            except Exception as exc:
                logger.error("Channel %s failed for %s: %s",
                             notifier.name, alert.event_id, exc)
        return alert

    def close(self) -> None:
        for n in self.notifiers:
            close = getattr(n, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
