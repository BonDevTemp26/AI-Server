"""Telegram alert channel (free Bot API; sends the clip as a video message).

Setup: create a bot with @BotFather → put the token in .env
(TELEGRAM_BOT_TOKEN) and the target chat/group id in TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pipeline.events import Alert

logger = logging.getLogger(__name__)


class TelegramNotifier:
    name = "telegram"

    def __init__(self, cfg: dict):
        self.token = os.environ.get(cfg.get("token_env", "TELEGRAM_BOT_TOKEN"), "")
        self.chat_id = os.environ.get(cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
        self.attach_clip = bool(cfg.get("attach_clip", True))
        if not (self.token and self.chat_id):
            raise ValueError("Telegram channel enabled but TELEGRAM_BOT_TOKEN / "
                             "TELEGRAM_CHAT_ID missing from environment")

    def _text(self, alert: Alert) -> str:
        return (f"🚨 THEFT ALERT — {alert.camera_id}\n"
                f"action: {alert.top_action} | score {alert.score:.2f} | "
                f"VLM {alert.verdict} ({alert.vlm_confidence:.2f})\n"
                f"{alert.description}\n"
                f"event {alert.event_id} @ t={alert.trigger_ts:.0f}s")

    def send(self, alert: Alert, clip_path: str | None) -> None:
        import requests

        base = f"https://api.telegram.org/bot{self.token}"
        if self.attach_clip and clip_path and Path(clip_path).is_file():
            with open(clip_path, "rb") as f:
                resp = requests.post(f"{base}/sendVideo", timeout=60,
                                     data={"chat_id": self.chat_id,
                                           "caption": self._text(alert)},
                                     files={"video": f})
        else:
            resp = requests.post(f"{base}/sendMessage", timeout=15,
                                 data={"chat_id": self.chat_id,
                                       "text": self._text(alert)})
        resp.raise_for_status()
