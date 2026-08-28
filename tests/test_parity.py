"""Parity checks: the deployment path must match the validated offline path.

Two substitutions are made for the aircraft's benefit -- a torch-free tokenizer
and (when torch is absent) an OpenCV resize. Both are only legitimate if they
reproduce what `dronevla/models/smolvla/inference_vlacpp.py` does, which is the
script that was validated frame-for-frame against LeRobot on the canonical
split. This measures the gap instead of assuming it.

Run in an env that has torch AND transformers (e.g. `groot`), otherwise the
comparison has nothing to compare against and the checks skip.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_nav.config import PolicyConfig

TEXTS = [
    "follow the person in an orange hard hat",
    "approach the worker wearing a yellow vest",
    "stay close to the man in the blue jacket",
]


def test_tokenizer_parity():
    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError:
        print("  skip test_tokenizer_parity (needs transformers)")
        return
    mid = PolicyConfig().tokenizer_id
    hf = AutoTokenizer.from_pretrained(mid)
    fast = Tokenizer.from_pretrained(mid)
    for t in TEXTS:
        text = t + "\n"
        a = hf(text, padding=False, truncation=True, max_length=48).input_ids
        b = fast.encode(text, add_special_tokens=True).ids
        assert a == b, (t, a, b)
    print(f"  ok  test_tokenizer_parity ({len(TEXTS)} instructions, ids identical)")


def _resize_torch(frame, n=256, size=512):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(frame)).resize((n, n))
    t = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear",
                      align_corners=False).squeeze(0)
    return np.ascontiguousarray(t.permute(1, 2, 0).numpy(), dtype=np.float32)


def _resize_cv2(frame, n=256, size=512):
    """The torch-free path: PIL for the downsample, cv2 for the 2x upsample."""
    import cv2
    from PIL import Image
    img = np.array(Image.fromarray(np.ascontiguousarray(frame)).resize((n, n)),
                   dtype=np.float32) / 255.0
    return np.ascontiguousarray(
        cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR),
        dtype=np.float32)


def test_resize_parity_on_real_frames():
    """How far the torch-free fallback drifts from the validated preprocessing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skip test_resize_parity_on_real_frames (no torch to compare against)")
        return
    root = Path("/mnt/data/thinhld/dataset/WareFly-VLA/1fps")
    if not root.exists():
        print("  skip test_resize_parity_on_real_frames (dataset not mounted)")
        return
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    frames = sorted(root.glob("*/frames/frame_00000*.png"))[:5]
    assert frames, "no frames found"
    diffs = []
    for f in frames:
        arr = np.asarray(Image.open(f).convert("RGB"))
        diffs.append(np.abs(_resize_torch(arr) - _resize_cv2(arr)).max())
    worst = float(max(diffs))
    # Only the exact-2x bilinear upsample differs between the two paths, so the
    # gap should be numerical noise. It is asserted, not merely reported: a
    # regression here would silently change what the network sees in flight.
    print(f"  ok  test_resize_parity_on_real_frames "
          f"(max |torch - cv2| = {worst:.4f} over {len(frames)} frames, scale 0..1)")
    assert worst < 1e-4, worst


def test_frame_convention_roundtrip():
    from vla_nav.adapters.base import to_body_ned, to_enu
    from vla_nav.navigator import NavVector
    v = NavVector(1.0, 2.0, 3.0, 0.5)
    assert to_enu(v) == (1.0, 2.0, 3.0, 0.5)
    assert to_body_ned(v) == (1.0, -2.0, -3.0, -0.5)
    print("  ok  test_frame_convention_roundtrip")


def test_payload_renders_typed_values():
    from vla_nav.payload import DEFAULT_TEMPLATE, build
    p = build(DEFAULT_TEMPLATE, vector=(1.0, 0.0, 0.0, 0.0),
              action=np.array([0.1, 0.0, 0.0, 0.0]), instruction="x", seq=1,
              ts_ms=1234, frame="body_flu", source="t", valid_for_s=0.5)
    assert isinstance(p["nav_vector"]["vx"], float), type(p["nav_vector"]["vx"])
    assert isinstance(p["ts"], int) and isinstance(p["seq"], int)
    assert p["cmd_id"] == "vla-1" and p["instruction"] == "x"
    import json
    json.dumps(p)  # must be serialisable as-is
    print("  ok  test_payload_renders_typed_values")


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f()
    print("parity checks done")
