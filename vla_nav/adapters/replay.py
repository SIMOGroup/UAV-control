"""A DroneAdapter backed by a recorded WareFly episode.

Lets the whole control path -- preprocessing, vla.cpp, navigator, safety, the
send() call -- run and be inspected without an aircraft. Commands are recorded
instead of sent, so a run can be diffed against the recorded ground-truth track.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from vla_nav.adapters.base import DroneAdapter, Observation
from vla_nav.navigator import NavVector

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC_ROOT = Path(os.environ.get("WAREFLY_FPS_ROOT",
                               "/mnt/data/thinhld/dataset/WareFly-VLA/1fps"))


class ReplayAdapter(DroneAdapter):
    def __init__(self, uuid: str, root: Path | None = None):
        self.root = (root or SRC_ROOT) / uuid
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        with open(self.root / "episode.csv") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows
                     if r["is_last"] != "True"
                     and (self.root / "frames" / r["frame_name"]).exists()]
        self.i = 0
        self.sent: list[NavVector] = []
        index = SRC_ROOT.parent  # index.json lives beside the episode dirs
        self.instruction = ""
        idx_path = (root or SRC_ROOT) / "index.json"
        if idx_path.exists():
            import json
            self.instruction = json.loads(idx_path.read_text())[uuid][
                "language_instruction"]

    def __len__(self) -> int:
        return len(self.rows)

    def observe(self) -> Observation:
        r = self.rows[self.i]
        frame = np.asarray(
            Image.open(self.root / "frames" / r["frame_name"]).convert("RGB"))
        state = np.array([float(r["x"]), float(r["y"]), float(r["z"]),
                          float(r["yaw_rad"])], dtype=np.float32)
        return Observation(frame=frame, state=state, altitude=float(r["z"]),
                           bgr=False, t=float(self.i))

    def ground_truth(self) -> np.ndarray:
        r = self.rows[self.i]
        return np.array([float(r["dx_local"]), float(r["dy_local"]),
                         float(r["dz"]), float(r["dyaw"])], dtype=np.float32)

    def advance(self) -> bool:
        self.i += 1
        return self.i < len(self.rows)

    def send(self, v: NavVector) -> None:
        self.sent.append(v)

    def is_ready(self) -> bool:
        return self.i < len(self.rows)
