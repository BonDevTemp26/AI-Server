"""Webhook alert channel — Slack / Discord incoming webhooks or a raw JSON POST.

All three styles are free; set the URL in .env → ALERT_WEBHOOK_URL and pick
``style:`` in configs/alerts.yaml. The clip is referenced by path (serve
data/clips/ via the review dashboard for clickable links).
"""

from __future__ import annotations

import logging
import os

from pipeline.events import Alert

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    name = "webhook"

    def __init__(self, cfg: dict):
        self.url = os.environ.get(cfg.get("url_env", "ALERT_WEBHOOK_URL"), "")
        self.style = cfg.get("style", "slack")
        if not self.url:
            raise ValueError("Webhook channel enabled but ALERT_WEBHOOK_URL "
                             "missing from environment")

    def _payload(self, alert: Alert, clip_path: str | None) -> dict:
        text = (f"🚨 *Theft alert* — `{alert.camera_id}` | {alert.top_action} "
                f"(score {alert.score:.2f}, VLM {alert.verdict} "
                f"{alert.vlm_confidence:.2f})\n{alert.description}\n"
                f"clip: `{clip_path}` | event `{alert.event_id}`")
        if self.style == "slack":
            return {"text": text}
        if self.style == "discord":
            return {"content": text.replace("*", "**")}
        return alert.to_dict() | {"clip_path": clip_path}   # generic

    def send(self, alert: Alert, clip_path: str | None) -> None:
        import requests

        resp = requests.post(self.url, json=self._payload(alert, clip_path), timeout=15)
        resp.raise_for_status()
