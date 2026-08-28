"""MQTT transport for the VLA navigation loop.

Speaks exactly the topic vocabulary the repo already defines in
``MQTT/Test/mqtt_drone_sub.py``:

    in   drone/{id}/telemetry/state      pose the policy is conditioned on
         drone/{id}/telemetry/image_meta camera metadata (optionally a frame URL)
         drone/{id}/telemetry/battery    used only for the low-battery stop
         drone/{id}/status/heartbeat     liveness; no heartbeat -> no commands
         drone/{id}/cmd/ack , cmd/result command feedback
    out  drone/{id}/cmd/vla              the navigation vector
         drone/{id}/cmd/override         {"action": "hold"} on any stop

Publishing is **off** unless explicitly enabled. This talks to a real aircraft;
the default is to render payloads and print them.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

TOPIC_MAP = {
    "state": "telemetry/state",
    "battery": "telemetry/battery",
    "gps": "telemetry/gps",
    "ekf_status": "telemetry/ekf_status",
    "image_meta": "telemetry/image_meta",
    "inference_ctx": "telemetry/inference_ctx",
    "heartbeat": "status/heartbeat",
    "online": "status/online",
    "lwt": "status/lwt",
    "health": "system/health",
    "log": "system/log",
    "ack": "cmd/ack",
    "result": "cmd/result",
    "cmd_vla": "cmd/vla",
    "cmd_mission": "cmd/mission",
    "cmd_override": "cmd/override",
}

# Subscribed by the navigation loop. Command topics are not among them: echoing
# our own commands back into the loop would be a feedback path, not telemetry.
SUBSCRIBE = ["state", "image_meta", "battery", "heartbeat", "online", "lwt",
             "ack", "result"]


@dataclass
class BrokerConfig:
    host: str = "116.111.21.154"
    port: int = 1883
    username: str = "mqtt"
    password: str = ""
    drone_id: str = "drone01"
    client_id: str = ""
    keepalive: int = 60
    # Nothing is published unless this is True. Passing --publish sets it.
    publish_enabled: bool = False


@dataclass
class Telemetry:
    """Last message seen per topic, with the time it arrived."""

    data: dict[str, Any] = field(default_factory=dict)
    at: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
            self.at[key] = time.monotonic()

    def get(self, key: str, default=None):
        with self._lock:
            return self.data.get(key, default)

    def age(self, key: str) -> float:
        """Seconds since this topic last arrived; ``inf`` if never."""
        with self._lock:
            t = self.at.get(key)
        return float("inf") if t is None else time.monotonic() - t


class MqttBridge:
    def __init__(self, cfg: BrokerConfig,
                 on_message: Callable[[str, Any], None] | None = None):
        self.cfg = cfg
        self.telemetry = Telemetry()
        self._on_message = on_message
        self._client = None
        self._connected = threading.Event()
        self.published: list[tuple[str, dict]] = []   # audit trail, dry-run too
        self.n_published = 0

    # ── lifecycle ───────────────────────────────────────────────────────────
    def topic(self, short: str) -> str:
        return f"drone/{self.cfg.drone_id}/{TOPIC_MAP[short]}"

    def connect(self, timeout: float = 10.0) -> None:
        import paho.mqtt.client as mqtt

        cid = self.cfg.client_id or f"vla-nav-{int(time.time())}"
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                                 client_id=cid, protocol=mqtt.MQTTv311)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=cid, protocol=mqtt.MQTTv311)

        client.username_pw_set(self.cfg.username, self.cfg.password)
        client.on_connect = self._on_connect
        client.on_message = self._handle
        client.connect(self.cfg.host, self.cfg.port, keepalive=self.cfg.keepalive)
        client.loop_start()
        self._client = client
        if not self._connected.wait(timeout):
            raise TimeoutError(
                f"no CONNACK from {self.cfg.host}:{self.cfg.port} in {timeout}s")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            return
        for short in SUBSCRIBE:
            client.subscribe(self.topic(short), qos=0)
        self._connected.set()

    def _handle(self, client, userdata, msg):
        short = None
        prefix = f"drone/{self.cfg.drone_id}/"
        if msg.topic.startswith(prefix):
            suffix = msg.topic[len(prefix):]
            for k, v in TOPIC_MAP.items():
                if v == suffix:
                    short = k
                    break
        if short is None:
            return
        text = msg.payload.decode("utf-8", errors="replace")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        self.telemetry.put(short, value)
        if self._on_message:
            self._on_message(short, value)

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    # ── publishing ──────────────────────────────────────────────────────────
    def publish(self, short: str, payload: dict) -> bool:
        """Publish to a command topic. Returns True if it actually went out."""
        topic = self.topic(short)
        self.published.append((topic, payload))
        if not self.cfg.publish_enabled or self._client is None:
            return False
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        info = self._client.publish(topic, body, qos=0)
        self.n_published += 1
        return info.rc == 0

    def is_alive(self, max_heartbeat_age: float = 5.0,
                 max_state_age: float = 2.0) -> tuple[bool, str]:
        """Whether the aircraft looks present enough to accept setpoints."""
        hb = self.telemetry.age("heartbeat")
        st = self.telemetry.age("state")
        if hb > max_heartbeat_age:
            return False, f"heartbeat stale ({hb:.1f}s)"
        if st > max_state_age:
            return False, f"state stale ({st:.1f}s)"
        online = self.telemetry.get("online")
        if isinstance(online, dict) and online.get("online") is False:
            return False, "status/online reports offline"
        return True, "ok"
