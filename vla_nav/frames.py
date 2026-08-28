"""Where the camera frame comes from.

The drone publishes ``telemetry/image_meta``, not the image itself, so the frame
has to be pulled from wherever that metadata points. The URI scheme selects the
source:

    replay:<episode-uuid>   a recorded WareFly episode (no aircraft needed)
    mqtt:<topic-suffix>     JPEG bytes, or base64 in a JSON field, on MQTT
    rtsp://... http://...   anything OpenCV can open (also a bare device index)
    meta:<field>            follow a URL published in telemetry/image_meta

Every source returns HWC uint8 RGB. ``bgr`` is handled at the source, not left
for the policy to guess.
"""
from __future__ import annotations

import abc
import base64
import io
import json
from pathlib import Path

import numpy as np


class FrameSource(abc.ABC):
    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """Latest RGB frame, or None if one is not available yet."""

    def close(self) -> None:
        pass


def _decode(buf: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))


class ReplayFrameSource(FrameSource):
    """Frames from a recorded WareFly episode, advanced by ``step()``."""

    def __init__(self, uuid: str, root: Path):
        import csv
        self.root = root / uuid
        with open(self.root / "episode.csv") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows
                     if r["is_last"] != "True"
                     and (self.root / "frames" / r["frame_name"]).exists()]
        self.i = 0

    def read(self):
        if self.i >= len(self.rows):
            return None
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        p = self.root / "frames" / self.rows[self.i]["frame_name"]
        return np.asarray(Image.open(p).convert("RGB"))

    def state(self):
        """Recorded pose for the current frame, so an offline dry run feeds the
        policy the same state a live flight would rather than zeros."""
        if self.i >= len(self.rows):
            return None
        r = self.rows[self.i]
        return np.array([float(r["x"]), float(r["y"]), float(r["z"]),
                         float(r["yaw_rad"])], dtype=np.float32)

    def ground_truth(self):
        if self.i >= len(self.rows):
            return None
        r = self.rows[self.i]
        return np.array([float(r["dx_local"]), float(r["dy_local"]),
                         float(r["dz"]), float(r["dyaw"])], dtype=np.float32)

    def step(self) -> bool:
        self.i += 1
        return self.i < len(self.rows)


class OpenCVFrameSource(FrameSource):
    """Device index, video file, RTSP or MJPEG-over-HTTP via OpenCV."""

    def __init__(self, uri: str):
        import cv2
        self._cv2 = cv2
        src: object = uri
        if uri.isdigit():
            src = int(uri)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {uri!r}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame[:, :, ::-1].copy()   # BGR -> RGB

    def close(self):
        self.cap.release()


class MqttFrameSource(FrameSource):
    """Frames arriving on an MQTT topic as JPEG bytes or base64 inside JSON."""

    def __init__(self, bridge, topic_suffix: str, b64_field: str = "image"):
        self.bridge = bridge
        self.topic = f"drone/{bridge.cfg.drone_id}/{topic_suffix}"
        self.b64_field = b64_field
        self._latest: np.ndarray | None = None
        client = bridge._client
        if client is None:
            raise RuntimeError("connect the bridge before creating MqttFrameSource")
        client.message_callback_add(self.topic, self._on_frame)
        client.subscribe(self.topic, qos=0)

    def _on_frame(self, client, userdata, msg):
        buf = msg.payload
        if buf[:1] in (b"{", b"["):
            try:
                obj = json.loads(buf.decode("utf-8", errors="replace"))
                b64 = obj.get(self.b64_field)
                if not b64:
                    return
                buf = base64.b64decode(b64)
            except Exception:
                return
        try:
            self._latest = _decode(buf)
        except Exception:
            pass

    def read(self):
        return self._latest


class MetaUrlFrameSource(FrameSource):
    """Fetch whatever URL ``telemetry/image_meta`` most recently advertised."""

    def __init__(self, bridge, field: str = "url", timeout: float = 1.0):
        self.bridge = bridge
        self.field = field
        self.timeout = timeout
        self._last_url: str | None = None

    def read(self):
        meta = self.bridge.telemetry.get("image_meta")
        if not isinstance(meta, dict):
            return None
        url = meta.get(self.field)
        if not url:
            return None
        self._last_url = url
        import urllib.request
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return _decode(r.read())


def make_frame_source(uri: str, *, bridge=None, replay_root: Path | None = None):
    if uri.startswith("replay:"):
        if replay_root is None:
            raise ValueError("replay: source needs replay_root")
        return ReplayFrameSource(uri.split(":", 1)[1], replay_root)
    if uri.startswith("mqtt:"):
        if bridge is None:
            raise ValueError("mqtt: source needs a connected bridge")
        return MqttFrameSource(bridge, uri.split(":", 1)[1])
    if uri.startswith("meta:"):
        if bridge is None:
            raise ValueError("meta: source needs a connected bridge")
        return MetaUrlFrameSource(bridge, uri.split(":", 1)[1] or "url")
    return OpenCVFrameSource(uri)
