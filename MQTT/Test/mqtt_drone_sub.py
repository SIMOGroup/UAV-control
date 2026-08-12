#!/usr/bin/env python3
"""Subscribe drone MQTT topics from another PC (no ROS required).

Requires: pip install paho-mqtt

Examples:
  python3 mqtt_drone_sub.py
  python3 mqtt_drone_sub.py --drone-id drone01 --pretty
  python3 mqtt_drone_sub.py --only state,battery,heartbeat
  python3 mqtt_drone_sub.py --host 116.111.21.154 -u mqtt -P 'K4CVj6yTUx8DJMaC'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Need paho-mqtt:  pip install paho-mqtt", file=sys.stderr)
    sys.exit(1)


# short name -> MQTT suffix under drone/{id}/
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


def build_topics(drone_id: str, only: list[str] | None) -> list[str]:
    base = f"drone/{drone_id}"
    if not only:
        return [f"{base}/#"]  # everything under this drone
    topics = []
    for name in only:
        key = name.strip().lower()
        if key not in TOPIC_MAP:
            known = ", ".join(sorted(TOPIC_MAP))
            raise SystemExit(f"Unknown topic '{name}'. Known: {known}")
        topics.append(f"{base}/{TOPIC_MAP[key]}")
    return topics


def on_connect(client, userdata, flags, rc, properties=None):
    if rc != 0:
        print(f"[ERR] MQTT connect failed rc={rc}", file=sys.stderr)
        return
    topics = userdata["topics"]
    print(f"[OK] connected → subscribe {topics}")
    for t in topics:
        client.subscribe(t, qos=0)


def on_message(client, userdata, msg):
    pretty = userdata["pretty"]
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    text = msg.payload.decode("utf-8", errors="replace")
    if pretty:
        try:
            obj = json.loads(text)
            text = json.dumps(obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
        print(f"\n[{ts}] {msg.topic}")
        print(text)
    else:
        print(f"[{ts}] {msg.topic} {text}")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="MQTT drone topic subscriber (remote PC)")
    p.add_argument("--host", default="116.111.21.154", help="MQTT broker host")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("-u", "--username", default="mqtt")
    p.add_argument("-P", "--password", default="K4CVj6yTUx8DJMaC")
    p.add_argument("--drone-id", default="drone01")
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated short names (default: all via drone/{id}/#). "
        f"e.g. state,battery,heartbeat. Known: {','.join(sorted(TOPIC_MAP))}",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    p.add_argument("--client-id", default="", help="Optional MQTT client_id")
    args = p.parse_args()

    only = [x for x in args.only.split(",") if x.strip()] if args.only else None
    topics = build_topics(args.drone_id, only)

    userdata = {"topics": topics, "pretty": args.pretty}
    client_id = args.client_id or f"drone-sub-{int(time.time())}"

    # paho v1 / v2 compatible
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    client.user_data_set(userdata)
    client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting {args.host}:{args.port} as {args.username} ...")
    try:
        client.connect(args.host, args.port, keepalive=60)
    except OSError as e:
        print(f"[ERR] cannot connect: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nbye")
        client.disconnect()


if __name__ == "__main__":
    main()
