#!/usr/bin/env python3
"""Closed-loop VLA navigation over MQTT: camera + instruction -> cmd/vla.

    # 1. dry run against a recorded episode -- no broker, no aircraft
    python3 scripts/run_vla_nav.py --camera replay:<uuid> --no-broker

    # 2. dry run against the live broker: real telemetry in, payloads printed
    python3 scripts/run_vla_nav.py --camera meta:url -P "$MQTT_PASSWORD"

    # 3. actually command the aircraft (requires the explicit flag)
    python3 scripts/run_vla_nav.py --camera meta:url -P "$MQTT_PASSWORD" --publish

Publishing is off unless --publish is passed. This drives a real drone: the
default is to render every payload and print it so the schema and the numbers
can be checked against the drone side first.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_nav.adapters.mqtt_drone import (  # noqa: E402
    DEFAULT_STATE_PATHS, MqttAdapterConfig, MqttDroneAdapter,
)
from vla_nav.config import PolicyConfig, SafetyConfig  # noqa: E402
from vla_nav.frames import make_frame_source  # noqa: E402
from vla_nav.mqtt_bridge import BrokerConfig, MqttBridge  # noqa: E402
from vla_nav.navigator import Navigator  # noqa: E402
from vla_nav.payload import load_template  # noqa: E402
from vla_nav.policy import NavPolicy  # noqa: E402

DEFAULT_GGUF = "/mnt/data/thinhld/VLA/vla_models/smolvla_warefly_10fps-bf16.gguf"
WAREFLY_ROOT = Path(os.environ.get(
    "WAREFLY_FPS_ROOT", "/mnt/data/thinhld/dataset/WareFly-VLA/1fps"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    b = p.add_argument_group("broker")
    b.add_argument("--host", default="116.111.21.154")
    b.add_argument("--port", type=int, default=1883)
    b.add_argument("-u", "--username", default="mqtt")
    b.add_argument("-P", "--password", default=os.environ.get("MQTT_PASSWORD", ""))
    b.add_argument("--drone-id", default="drone01")
    b.add_argument("--no-broker", action="store_true",
                   help="Do not connect at all. Requires a replay: camera.")
    b.add_argument("--publish", action="store_true",
                   help="ACTUALLY publish to cmd/vla. Off by default.")

    m = p.add_argument_group("model")
    m.add_argument("--gguf", default=DEFAULT_GGUF)
    m.add_argument("--fps", type=float, default=10.0,
                   help="Rate the checkpoint was TRAINED at. Scales every command.")
    m.add_argument("--instruction", default="follow the person in an orange hard hat")
    m.add_argument("--camera", default="meta:url",
                   help="replay:<uuid> | mqtt:<suffix> | meta:<field> | rtsp://... | 0")

    c = p.add_argument_group("control")
    c.add_argument("--control-hz", type=float, default=10.0)
    c.add_argument("--frame", default="body_flu", choices=["body_flu", "body_frd"],
                   help="body_frd converts to PX4/MAVLink body-NED before sending")
    c.add_argument("--max-speed-xy", type=float, default=1.5)
    c.add_argument("--max-speed-z", type=float, default=0.5)
    c.add_argument("--max-yaw-rate", type=float, default=0.8)
    c.add_argument("--min-altitude", type=float, default=0.8)
    c.add_argument("--max-steps", type=int, default=0)
    c.add_argument("--payload-template", default="",
                   help="JSON file matching the drone-side cmd/vla schema")
    c.add_argument("--state-path", action="append", default=[],
                   metavar="KEY=dotted.path",
                   help="Override a telemetry/state field, e.g. x=local_position.x")
    c.add_argument("--log", default="", help="Write a JSONL flight log here")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_broker and not args.camera.startswith("replay:"):
        raise SystemExit("--no-broker only makes sense with a replay: camera")

    state_paths = {k: list(v) for k, v in DEFAULT_STATE_PATHS.items()}
    for spec in args.state_path:
        key, _, path = spec.partition("=")
        if key not in state_paths or not path:
            raise SystemExit(f"bad --state-path {spec!r}; keys: {list(state_paths)}")
        state_paths[key] = [path]

    broker = BrokerConfig(host=args.host, port=args.port, username=args.username,
                          password=args.password, drone_id=args.drone_id,
                          publish_enabled=args.publish)
    bridge = MqttBridge(broker)
    if not args.no_broker:
        print(f"connecting {args.host}:{args.port} as {args.username} ...")
        bridge.connect()
        print(f"connected; subscribed under drone/{args.drone_id}/")

    frames = make_frame_source(args.camera, bridge=bridge, replay_root=WAREFLY_ROOT)

    print(f"loading {args.gguf}")
    policy = NavPolicy(PolicyConfig(gguf=Path(args.gguf), fps=args.fps))
    policy.set_instruction(args.instruction)
    print(f"  loaded in {policy.load_time:.1f}s  chunk={policy.chunk_size}  "
          f"action_dim={policy.real_action_dim}")

    safety = SafetyConfig(max_speed_xy=args.max_speed_xy,
                          max_speed_z=args.max_speed_z,
                          max_yaw_rate=args.max_yaw_rate,
                          min_altitude=args.min_altitude)
    nav = Navigator(fps=args.fps, safety=safety)
    adapter = MqttDroneAdapter(
        bridge, frames, args.instruction,
        MqttAdapterConfig(frame=args.frame, state_paths=state_paths,
                          template=load_template(args.payload_template or None),
                          valid_for_s=max(2.0 / args.control_hz, 1.0 / args.fps)))

    if args.publish:
        print("\n*** PUBLISH ENABLED -- this will command drone "
              f"{args.drone_id} on {args.host} ***")
        for i in (3, 2, 1):
            print(f"    starting in {i} ...", flush=True)
            time.sleep(1)
    else:
        print("\ndry run: payloads are rendered and printed, nothing is published")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    log = open(args.log, "w", encoding="utf-8") if args.log else None
    period = 1.0 / args.control_hz
    n = 0
    t_start = time.monotonic()
    try:
        while not stop["flag"]:
            if args.max_steps and n >= args.max_steps:
                break
            t0 = time.monotonic()

            if not args.no_broker and not adapter.is_ready():
                print(f"[hold] {adapter.stop_reason}")
                adapter.hold(adapter.stop_reason)
                nav.stop()
                time.sleep(period)
                continue

            try:
                obs = adapter.observe()
            except RuntimeError as e:
                print(f"[wait] {e}")
                time.sleep(period)
                continue

            action = policy.step(obs.frame, obs.state, bgr=obs.bgr)
            nav.set_altitude(obs.altitude)
            nav.submit(action)
            v = nav.tick()
            adapter.send(v, action=action)
            n += 1

            topic, payload = bridge.published[-1]
            print(f"[{n:5d}] v=({v.vx:+.3f},{v.vy:+.3f},{v.vz:+.3f},{v.yaw_rate:+.3f}) "
                  f"{v.status:10s} infer={policy.last_infer_s*1e3:6.1f}ms "
                  f"{'PUB' if broker.publish_enabled else 'dry'} {topic}")
            if log:
                log.write(json.dumps({
                    "n": n, "t": time.time(), "action": action.tolist(),
                    "vector": [v.vx, v.vy, v.vz, v.yaw_rate], "status": v.status,
                    "infer_s": policy.last_infer_s, "payload": payload,
                }) + "\n")
                log.flush()

            step = getattr(frames, "step", None)
            if step is not None and not step():
                print("replay exhausted")
                break

            lag = period - (time.monotonic() - t0)
            if lag > 0:
                time.sleep(lag)
    finally:
        adapter.on_stop()
        print(f"\nheld; {n} steps in {time.monotonic() - t_start:.1f}s, "
              f"{policy.n_net_calls} network calls, "
              f"{bridge.n_published} messages actually published")
        if log:
            log.close()
        bridge.close()
        frames.close()


if __name__ == "__main__":
    main()
