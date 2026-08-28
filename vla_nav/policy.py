"""The vla.cpp side: camera frame + instruction -> 4-DoF action chunk.

This is the deployment twin of
``dronevla/models/smolvla/inference_vlacpp.py``, which is the script that was
validated frame-for-frame against the PyTorch/LeRobot runtime on the canonical
held-out split. The preprocessing here reproduces that path exactly, because any
drift shows up as a *runtime parity* failure rather than a model failure:

  PNG/camera -> RGB -> 256x256 -> /255 -> bilinear upsample to 512x512
  (LeRobot ``resize_with_pad``; 256->512 is an exact 2x so no padding is added)
  -> handed to vla.cpp as PIXEL_F32_RGB_01, since the GGUF config sets
  VISUAL: IDENTITY.

State is the raw ``[x, y, z, yaw]`` zero-padded to ``max_state_dim``; vla.cpp
applies the MEAN_STD normalisation baked into the GGUF and un-normalises the
action itself, so the caller sees metres and radians.

The network emits ``n_suffix`` future steps at once. LeRobot serves them from a
queue and only re-runs the network when the queue empties; we do the same, so
the inference duty cycle on the aircraft matches the one that was benchmarked.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from vla_nav.config import PolicyConfig, VLA_CPP_ROOT


def _import_vla_cpp():
    sys.path.insert(0, str(VLA_CPP_ROOT / "bindings" / "python"))
    os.environ.setdefault("VLA_LIBRARY", str(VLA_CPP_ROOT / "build" / "libvla.so"))
    import vla_cpp  # noqa: E402
    return vla_cpp


def preprocess_frame(frame: np.ndarray, cfg: PolicyConfig,
                     bgr: bool = False) -> np.ndarray:
    """Live camera frame (H, W, 3) uint8 -> the float tensor vla.cpp expects.

    Uses torch when it is importable (bit-identical to the validated offline
    path) and falls back to OpenCV otherwise, so the aircraft does not need a
    torch install. ``tests/test_resize_parity.py`` measures the gap between the
    two on real frames.
    """
    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 camera frame, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {frame.shape}")
    if bgr:
        frame = frame[:, :, ::-1]

    n, size = cfg.resize_first, cfg.image_size
    from PIL import Image

    # The first stage stays on PIL in BOTH paths. It is a ~7x downsample of a
    # 1920x1440 frame, where PIL's antialiased default and cv2's INTER_LINEAR
    # disagree by ~0.2 on a 0..1 scale -- far too much to call a fallback.
    # Only the exact-2x upsample is substituted, and that one agrees to ~1e-7.
    img = np.array(Image.fromarray(np.ascontiguousarray(frame)).resize((n, n)),
                   dtype=np.float32) / 255.0
    if n == size:
        return np.ascontiguousarray(img, dtype=np.float32)

    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(img).permute(2, 0, 1)
        t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear",
                          align_corners=False).squeeze(0)
        return np.ascontiguousarray(t.permute(1, 2, 0).numpy(), dtype=np.float32)
    except ImportError:
        import cv2

        big = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(big, dtype=np.float32)


class NavPolicy:
    """Holds the GGUF model, the tokenized instruction, and the action queue."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        if not Path(self.cfg.gguf).exists():
            raise FileNotFoundError(f"GGUF not found: {self.cfg.gguf}")

        self._vla = _import_vla_cpp()
        t0 = time.perf_counter()
        self.model = self._vla.load(str(self.cfg.gguf))
        self.load_time = time.perf_counter() - t0

        c = self.model.config
        self.max_state_dim = int(c.max_state_dim)
        self.real_action_dim = int(c.real_action_dim)
        self.chunk_size = int(c.n_suffix)

        self._tok = None
        self._tok_kind = ""
        self._tok_cache: dict[str, list[int]] = {}
        self._queue: list[np.ndarray] = []
        self._instruction: str | None = None
        self.n_net_calls = 0
        self.last_infer_s = 0.0

        self._noise = None
        if self.cfg.fixed_noise_seed is not None:
            rng = np.random.default_rng(self.cfg.fixed_noise_seed)
            self._noise = rng.standard_normal(
                int(c.max_action_dim) * self.chunk_size).astype(np.float32)

    # ── language ────────────────────────────────────────────────────────────
    def _load_tokenizer(self):
        """Prefer the torch-free `tokenizers` backend.

        `transformers.AutoTokenizer` drags in torch, which an aircraft has no
        reason to carry. `tokenizers.Tokenizer` produces byte-identical ids for
        this vocabulary -- `tests/test_tokenizer_parity.py` asserts it -- so the
        transformers path is only a fallback.
        """
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(self.cfg.tokenizer_id)
            return ("tokenizers", tok)
        except Exception:
            from transformers import AutoTokenizer
            return ("transformers", AutoTokenizer.from_pretrained(self.cfg.tokenizer_id))

    def tokenize(self, instruction: str) -> list[int]:
        if instruction in self._tok_cache:
            return self._tok_cache[instruction]
        if self._tok is None:
            self._tok_kind, self._tok = self._load_tokenizer()
        # LeRobot's tokenizer step appends a newline then pads to n_lang;
        # vla.cpp pads internally, so only the trailing newline is ours to add.
        text = instruction if instruction.endswith("\n") else instruction + "\n"
        if self._tok_kind == "tokenizers":
            self._tok.enable_truncation(max_length=self.cfg.max_lang_tokens)
            ids = self._tok.encode(text, add_special_tokens=True).ids
        else:
            ids = self._tok(text, padding=False, truncation=True,
                            max_length=self.cfg.max_lang_tokens).input_ids
        self._tok_cache[instruction] = ids
        return ids

    def set_instruction(self, instruction: str) -> None:
        """Change the task. Drops any queued actions from the old instruction."""
        if instruction != self._instruction:
            self._instruction = instruction
            self.tokenize(instruction)
            self.reset()

    def reset(self) -> None:
        """Clear the action queue. Call between episodes / after a takeover."""
        self._queue = []

    # ── inference ───────────────────────────────────────────────────────────
    def step(self, frame: np.ndarray, state_xyz_yaw, *, bgr: bool = False,
             instruction: str | None = None) -> np.ndarray:
        """One control step -> ``[dx_local, dy_local, dz, dyaw]``.

        The returned vector is a *displacement over one model timestep*, in
        metres and radians, in the drone body frame. Converting it to a velocity
        is the navigator's job, not the policy's.
        """
        if instruction is not None:
            self.set_instruction(instruction)
        if self._instruction is None:
            raise RuntimeError("set_instruction() before step()")

        if not self._queue:
            img = preprocess_frame(frame, self.cfg, bgr=bgr)
            state = np.zeros(self.max_state_dim, dtype=np.float32)
            state[:4] = np.asarray(state_xyz_yaw, dtype=np.float32)[:4]
            t0 = time.perf_counter()
            chunk = self.model.predict(
                img, self._tok_cache[self._instruction], state=state,
                noise=self._noise, pixel_format=self._vla.PIXEL_F32_RGB_01)
            self.last_infer_s = time.perf_counter() - t0
            self.n_net_calls += 1
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.size == 0:
                raise RuntimeError(
                    "vla.cpp returned an empty action chunk -- usually the image "
                    f"size ({self.cfg.image_size}) does not match the arch")
            self._queue = [arr[i] for i in range(arr.shape[0])]

        return self._queue.pop(0)[: self.real_action_dim].copy()
