"""The MQTT command path, exercised against a fake broker.

The real broker sits on the drone's network and is not reachable from a
workstation, so the transport is stubbed and everything above it -- telemetry
parsing, the readiness gate, payload rendering, the hold-on-stop behaviour --
is tested for real.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_nav.adapters.mqtt_drone import MqttAdapterConfig, MqttDroneAdapter, resolve
from vla_nav.mqtt_bridge import BrokerConfig, MqttBridge
from vla_nav.navigator import NavVector
from vla_nav.payload import DEFAULT_TEMPLATE


class FakeClient:
    def __init__(self): self.sent = []
    def publish(self, topic, body, qos=0):
        self.sent.append((topic, json.loads(body)))
        class I: rc = 0
        return I()


class FakeFrames:
    def __init__(self): self.n = 0
    def read(self):
        self.n += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)
    def close(self): pass


def make(publish=True):
    b = MqttBridge(BrokerConfig(drone_id="drone01", publish_enabled=publish))
    b._client = FakeClient()
    a = MqttDroneAdapter(b, FakeFrames(), "follow the person",
                         MqttAdapterConfig(template=dict(DEFAULT_TEMPLATE)))
    return b, a


def test_state_paths_resolve_nested_schemas():
    assert resolve({"x": 1.5}, ["x", "position.x"]) == 1.5
    assert resolve({"position": {"x": 2.5}}, ["x", "position.x"]) == 2.5
    assert resolve({"nope": 1}, ["x", "position.x"]) is None
    print("  ok  test_state_paths_resolve_nested_schemas")


def test_observe_reads_pose_from_telemetry():
    b, a = make()
    b.telemetry.put("state", {"position": {"x": 1.0, "y": 2.0, "z": 3.0},
                              "yaw": 0.5})
    obs = a.observe()
    assert np.allclose(obs.state, [1.0, 2.0, 3.0, 0.5]), obs.state
    assert obs.altitude == 3.0
    print("  ok  test_observe_reads_pose_from_telemetry")


def test_topic_and_payload_shape():
    b, a = make()
    a.send(NavVector(1.0, -0.5, 0.25, 0.1), action=np.array([0.1, -0.05, 0.025, 0.01]))
    topic, payload = b._client.sent[-1]
    assert topic == "drone/drone01/cmd/vla", topic
    assert payload["nav_vector"] == {"vx": 1.0, "vy": -0.5, "vz": 0.25,
                                     "yaw_rate": 0.1}, payload["nav_vector"]
    assert payload["frame"] == "body_flu"
    assert payload["cmd_id"] == "vla-1"
    print("  ok  test_topic_and_payload_shape")


def test_frd_conversion_applies_on_send():
    b = MqttBridge(BrokerConfig(publish_enabled=True)); b._client = FakeClient()
    a = MqttDroneAdapter(b, FakeFrames(), "x",
                         MqttAdapterConfig(frame="body_frd",
                                           template=dict(DEFAULT_TEMPLATE)))
    a.send(NavVector(1.0, 2.0, 3.0, 0.5))
    nv = b._client.sent[-1][1]["nav_vector"]
    assert nv == {"vx": 1.0, "vy": -2.0, "vz": -3.0, "yaw_rate": -0.5}, nv
    print("  ok  test_frd_conversion_applies_on_send")


def test_dry_run_publishes_nothing_but_still_records():
    b, a = make(publish=False)
    a.send(NavVector(1.0, 0, 0, 0))
    assert b._client.sent == []          # nothing left the process
    assert len(b.published) == 1         # but the payload was rendered
    print("  ok  test_dry_run_publishes_nothing_but_still_records")


def test_not_ready_without_heartbeat():
    b, a = make()
    assert not a.is_ready() and "heartbeat" in a.stop_reason
    b.telemetry.put("heartbeat", {"ok": True})
    b.telemetry.put("state", {"x": 0, "y": 0, "z": 2, "yaw": 0})
    assert a.is_ready(), a.stop_reason
    print("  ok  test_not_ready_without_heartbeat")


def test_low_battery_stops_commanding():
    b, a = make()
    b.telemetry.put("heartbeat", {"ok": True})
    b.telemetry.put("state", {"x": 0, "y": 0, "z": 2, "yaw": 0})
    b.telemetry.put("battery", {"percentage": 0.11})   # fraction form
    assert not a.is_ready() and "battery" in a.stop_reason, a.stop_reason
    print("  ok  test_low_battery_stops_commanding")


def test_stop_publishes_a_hold_override():
    b, a = make()
    a.on_stop()
    topic, payload = b._client.sent[-1]
    assert topic == "drone/drone01/cmd/override", topic
    assert payload["action"] == "hold"
    print("  ok  test_stop_publishes_a_hold_override")


def test_custom_template_matches_a_foreign_schema():
    """A drone expecting a totally different shape needs no code change."""
    b = MqttBridge(BrokerConfig(publish_enabled=True)); b._client = FakeClient()
    tpl = {"t": "{{ts_ms}}", "id": "{{cmd_id}}",
           "twist": {"linear": ["{{vx}}", "{{vy}}", "{{vz}}"],
                     "angular_z": "{{yaw_rate}}"},
           "note": "vla {{seq}} @ {{frame}}"}
    a = MqttDroneAdapter(b, FakeFrames(), "x", MqttAdapterConfig(template=tpl))
    a.send(NavVector(1.0, 2.0, 3.0, 0.5))
    p = b._client.sent[-1][1]
    assert p["twist"]["linear"] == [1.0, 2.0, 3.0], p
    assert p["twist"]["angular_z"] == 0.5
    assert p["note"] == "vla 1 @ body_flu", p["note"]
    print("  ok  test_custom_template_matches_a_foreign_schema")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)} passed")
