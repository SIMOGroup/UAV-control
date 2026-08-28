"""Fly the policy over a recorded WareFly episode and print the nav vectors.

    python scripts/run_replay.py --episode <uuid> --gguf <path> --fps 1

Nothing is sent to hardware: the ReplayAdapter records commands instead. This is
the end-to-end check that the vla.cpp runtime, preprocessing, navigator and
safety limits all work together before any of it is pointed at an aircraft.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_nav.adapters.replay import SRC_ROOT, ReplayAdapter  # noqa: E402
from vla_nav.config import PolicyConfig, RunConfig, SafetyConfig  # noqa: E402
from vla_nav.runner import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="")
    ap.add_argument("--gguf", default=str(
        Path("/mnt/data/thinhld/VLA/vla_models/smolvla_warefly_1fps-bf16.gguf")))
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--realtime", action="store_true",
                    help="Sleep between ticks. Off by default so the replay is fast.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    uuid = args.episode
    if not uuid:
        uuid = sorted(json.loads((SRC_ROOT / "index.json").read_text()))[0]
    adapter = ReplayAdapter(uuid)
    print(f"episode {uuid}  frames={len(adapter)}")
    # An episode the policy trained on will score far better than a held-out
    # one; say so rather than let the MAE below read as a benchmark result.
    split_path = Path(os.environ.get(
        "WAREFLY_SPLIT", "/mnt/data/thinhld/VLA/dronevla/outputs/canonical_split.json"))
    if split_path.exists():
        split = json.loads(split_path.read_text())
        where = ("TRAIN" if uuid in split.get("train", [])
                 else "held-out VAL" if uuid in split.get("val", []) else "unknown")
        print(f"canonical split: {where}"
              + ("   <- NOT a held-out score" if where == "TRAIN" else ""))
    print(f"instruction: {adapter.instruction!r}")

    cfg = RunConfig(
        policy=PolicyConfig(gguf=Path(args.gguf), fps=args.fps),
        safety=SafetyConfig(),
        control_hz=args.control_hz,
        instruction=adapter.instruction or "follow the person",
    )
    runner = Runner(adapter, cfg)
    print(f"model loaded in {runner.policy.load_time:.1f}s  "
          f"chunk={runner.policy.chunk_size}  action_dim={runner.policy.real_action_dim}")

    gts, preds = [], []
    sleep = None if args.realtime else (lambda _s: None)
    steps = args.max_steps or None

    # step manually so ground truth can be captured alongside each command
    period = 1.0 / cfg.control_hz
    while adapter.is_ready():
        if steps is not None and runner.stats.steps >= steps:
            break
        gts.append(adapter.ground_truth())
        rec = runner.step_once()
        preds.append(rec.action.copy())
        if not adapter.advance():
            break
    adapter.send(runner.nav.stop())

    gt = np.array(gts)
    pr = np.array(preds)
    print(f"\nsteps={len(pr)}  net_calls={runner.policy.n_net_calls} "
          f"(1 per {runner.policy.chunk_size} steps)")
    print(json.dumps(runner.stats.summary(), indent=2))

    print("\nraw policy action vs recorded ground truth (per-dim MAE, metres/rad):")
    for i, d in enumerate(["dx_local", "dy_local", "dz", "dyaw"]):
        print(f"  {d:9s} MAE={np.abs(pr[:, i] - gt[:, i]).mean():.4f}  "
              f"pred_mean={pr[:, i].mean():+.4f}  gt_mean={gt[:, i].mean():+.4f}")

    print("\nfirst 8 navigation vectors actually commanded (body frame, m/s & rad/s):")
    print(f"  {'vx':>8} {'vy':>8} {'vz':>8} {'yaw_rate':>9}  status")
    for v in adapter.sent[:8]:
        print(f"  {v.vx:8.3f} {v.vy:8.3f} {v.vz:8.3f} {v.yaw_rate:9.3f}  {v.status}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "episode": uuid,
            "instruction": adapter.instruction,
            "gguf": args.gguf,
            "fps": args.fps,
            "stats": runner.stats.summary(),
            "commands": [[v.vx, v.vy, v.vz, v.yaw_rate, v.status]
                         for v in adapter.sent],
            "raw_actions": pr.tolist(),
            "ground_truth": gt.tolist(),
        }, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
