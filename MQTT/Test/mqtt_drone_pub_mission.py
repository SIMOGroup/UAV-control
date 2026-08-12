#!/usr/bin/env python3
"""Publish one mission command to drone/{id}/cmd/mission (no ROS required).

Requires: pip install paho-mqtt

Examples:
  python3 mqtt_drone_pub_mission.py --payload '{"ts":0,"cmd_id":"mission-1","action":"start"}'
  python3 mqtt_drone_pub_mission.py --payload-file mission.json --drone-id drone01
  python3 mqtt_drone_pub_mission.py --host 116.111.21.154 -u mqtt -P 'K4CVj6yTUx8DJMaC' \\
      --payload '{"ts":0,"cmd_id":"mission-1","waypoints":[]}'

Watch ack/result in another terminal:
  python3 mqtt_drone_sub.py --only ack,result --pretty
"""
from __future__ import annotations

import argparse
import json
import sys
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Need paho-mqtt:  pip install paho-mqtt", file=sys.stderr)
    sys.exit(1)

TOPIC_SUFFIX = "cmd/mission"


def load_payload(args: argparse.Namespace) -> str:
    if bool(args.payload) == bool(args.payload_file):
        raise SystemExit("Provide exactly one of --payload or --payload-file")
    if args.payload_file:
        with open(args.payload_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.payload
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON payload: {e}") from e
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)


def main() -> None:
    p = argparse.ArgumentParser(description="Publish MQTT drone mission command")
    p.add_argument("--host", default="116.111.21.154", help="MQTT broker host")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("-u", "--username", default="mqtt")
    p.add_argument("-P", "--password", default="K4CVj6yTUx8DJMaC")
    p.add_argument("--drone-id", default="drone01")
    p.add_argument("--payload", default="", help="JSON string payload")
    p.add_argument("--payload-file", default="", help="Path to JSON payload file")
    p.add_argument("--client-id", default="", help="Optional MQTT client_id")
    args = p.parse_args()

    payload = load_payload(args)
    topic = f"drone/{args.drone_id}/{TOPIC_SUFFIX}"
    client_id = args.client_id or f"drone-pub-mission-{int(time.time())}"

    client = make_client(client_id)
    client.username_pw_set(args.username, args.password)

    print(f"Connecting {args.host}:{args.port} as {args.username} ...")
    try:
        client.connect(args.host, args.port, keepalive=60)
    except OSError as e:
        print(f"[ERR] cannot connect: {e}", file=sys.stderr)
        sys.exit(1)

    client.loop_start()
    info = client.publish(topic, payload, qos=0)
    info.wait_for_publish(timeout=5)
    client.loop_stop()
    client.disconnect()

    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"[ERR] publish failed rc={info.rc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] published → {topic}")
    print(payload)


if __name__ == "__main__":
    main()
