"""MQTT alert channel (free: local mosquitto / EMQX — see docker-compose.yml).

Publishes the alert JSON to a topic for on-prem dashboards / NVR integrations.
The clip itself is referenced by path/URL, not embedded.
"""

from __future__ import annotations

import json
import logging
import os

from pipeline.events import Alert

logger = logging.getLogger(__name__)


class MqttPublisher:
    name = "mqtt"

    def __init__(self, cfg: dict):
        import paho.mqtt.client as mqtt

        self.topic = cfg.get("topic", "cctv/theft/alerts")
        self.qos = int(cfg.get("qos", 1))
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id="cctv-theft-app")
        user = os.environ.get(cfg.get("username_env", ""), "")
        if user:
            self.client.username_pw_set(user, os.environ.get(cfg.get("password_env", ""), ""))
        self.client.connect(cfg.get("host", "localhost"), int(cfg.get("port", 1883)))
        self.client.loop_start()
        logger.info("MQTT channel connected to %s:%s (topic %s)",
                    cfg.get("host"), cfg.get("port"), self.topic)

    def send(self, alert: Alert, clip_path: str | None) -> None:
        payload = alert.to_dict() | {"clip_path": clip_path}
        info = self.client.publish(self.topic, json.dumps(payload), qos=self.qos)
        info.wait_for_publish(timeout=5)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
