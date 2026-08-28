"""A DroneAdapter that flies over the repo's MQTT surface.

Reads pose from ``telemetry/state``, pulls the frame from a configured source,
and publishes the navigation vector to ``cmd/vla``. On any stop -- pilot
takeover, stale telemetry, low battery, loop exit -- it publishes
``cmd/override {"action": "hold"}`` rather than simply going quiet, because a
controller that latches the last setpoint would keep flying.

The ``telemetry/state`` schema is drone-side and not documented in this repo, so
pose fields are resolved by trying a list of candidate dotted paths; override
them with ``--state-path`` when the real schema is known.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vla_nav.adapters.base import DroneAdapter, Observation, to_body_ned, to_enu
from vla_nav.mqtt_bridge import MqttBridge
from vla_nav.navigator import NavVector
from vla_nav.payload import build, hold_payload

# Tried in order until one resolves. Covers the shapes MAVROS / PX4 / DJI
# bridges usually publish.
DEFAULT_STATE_PATHS = {
    "x": ["x", "position.x", "local_position.x", "pose.position.x", "ned.x"],
    "y": ["y", "position.y", "local_position.y", "pose.position.y", "ned.y"],
    "z": ["z", "position.z", "local_position.z", "pose.position.z", "ned.z",
          "altitude", "relative_altitude"],
    "yaw": ["yaw", "yaw_rad", "attitude.yaw", "orientation.yaw", "heading_rad"],
}
DEFAULT_ALT_PATHS = ["relative_altitude", "altitude_agl", "z", "position.z",
                     "local_position.z"]


def dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve(obj, paths: list[str]):
    for p in paths:
        v = dig(obj, p)
        if isinstance(v, (int, float)):
            return float(v)
    return None


@dataclass
class MqttAdapterConfig:
    frame: str = "body_flu"          # or "body_frd" for PX4/MAVLink body-NED
    source_tag: str = "vla.cpp/smolvla-warefly"
    min_battery_pct: float = 20.0
    max_heartbeat_age: float = 5.0
    max_state_age: float = 2.0
    state_paths: dict = field(default_factory=lambda: dict(DEFAULT_STATE_PATHS))
    alt_paths: list = field(default_factory=lambda: list(DEFAULT_ALT_PATHS))
    template: dict | None = None
    valid_for_s: float = 0.5         # how long the drone may honour one vector


class MqttDroneAdapter(DroneAdapter):
    def __init__(self, bridge: MqttBridge, frame_source, instruction: str,
                 cfg: MqttAdapterConfig | None = None):
        self.bridge = bridge
        self.frames = frame_source
        self.instruction = instruction
        self.cfg = cfg or MqttAdapterConfig()
        self.seq = 0
        self.stop_reason = ""
        self._last_state = np.zeros(4, dtype=np.float32)
        self._have_state = False

    # ── in ──────────────────────────────────────────────────────────────────
    def observe(self) -> Observation:
        frame = self.frames.read()
        if frame is None:
            raise RuntimeError("no camera frame available yet")

        st = self.bridge.telemetry.get("state")
        state = self._last_state.copy()
        if isinstance(st, dict):
            vals = [resolve(st, self.cfg.state_paths[k])
                    for k in ("x", "y", "z", "yaw")]
            if all(v is not None for v in vals):
                state = np.array(vals, dtype=np.float32)
                self._last_state = state
                self._have_state = True
        alt = resolve(st, self.cfg.alt_paths) if isinstance(st, dict) else None

        # Offline sources (a recorded episode) carry their own pose; without
        # this an unconnected dry run would feed the policy an all-zero state.
        src_state = getattr(self.frames, "state", None)
        if not self._have_state and callable(src_state):
            s = src_state()
            if s is not None:
                state = np.asarray(s, dtype=np.float32)
                if alt is None:
                    alt = float(state[2])

        return Observation(frame=frame, state=state, altitude=alt, bgr=False,
                           t=time.monotonic())

    @property
    def have_state(self) -> bool:
        return self._have_state

    # ── out ─────────────────────────────────────────────────────────────────
    def send(self, v: NavVector, action=None) -> None:
        conv = to_body_ned if self.cfg.frame == "body_frd" else to_enu
        vec = conv(v)
        act = np.zeros(4, dtype=np.float32) if action is None else np.asarray(action)
        self.seq += 1
        payload = build(
            self.cfg.template or {},
            vector=vec, action=act, instruction=self.instruction,
            seq=self.seq, ts_ms=int(time.time() * 1000),
            frame=self.cfg.frame, source=self.cfg.source_tag,
            valid_for_s=self.cfg.valid_for_s,
        )
        self.bridge.publish("cmd_vla", payload)

    def hold(self, reason: str) -> None:
        self.stop_reason = reason
        self.bridge.publish(
            "cmd_override", hold_payload(self.seq, int(time.time() * 1000), reason))

    def on_stop(self) -> None:
        self.hold(self.stop_reason or "loop exited")

    # ── gate ────────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        alive, why = self.bridge.is_alive(self.cfg.max_heartbeat_age,
                                          self.cfg.max_state_age)
        if not alive:
            self.stop_reason = why
            return False
        batt = self.bridge.telemetry.get("battery")
        if isinstance(batt, dict):
            pct = batt.get("percentage", batt.get("remaining", batt.get("percent")))
            if isinstance(pct, (int, float)):
                pct = pct * 100.0 if pct <= 1.0 else pct
                if pct < self.cfg.min_battery_pct:
                    self.stop_reason = f"battery {pct:.0f}% below floor"
                    return False
        return True

    def close(self) -> None:
        self.frames.close()
