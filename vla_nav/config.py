"""Deployment configuration for the WareFly navigation policy."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

VLA_CPP_ROOT = Path(os.environ.get("VLA_CPP_ROOT", "/mnt/data/thinhld/VLA/vla.cpp"))
MODEL_DIR = Path(os.environ.get("WAREFLY_MODEL_DIR", "/mnt/data/thinhld/VLA/vla_models"))


@dataclass
class PolicyConfig:
    """Which trained checkpoint to fly, and how it was trained.

    ``fps`` is the rate the checkpoint was trained at. It is not a preference:
    the network emits a per-timestep *displacement*, so the velocity it implies
    is ``displacement * fps``. Getting it wrong scales every command.
    """

    gguf: Path = MODEL_DIR / "smolvla_warefly_10fps-bf16.gguf"
    fps: float = 10.0
    tokenizer_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    max_lang_tokens: int = 48
    # SmolVLA's tower resolution. vla.cpp returns an empty action if this does
    # not match the architecture (see vla.cpp CLAUDE.md, VLA_IMG_SIZE).
    image_size: int = 512
    resize_first: int = 256  # LeRobot resizes to 256 before padding up to 512
    # Deterministic flow-matching noise. None => fresh noise per call, which
    # makes two identical frames produce slightly different commands.
    fixed_noise_seed: int | None = 0


@dataclass
class SafetyConfig:
    """Limits applied to every vector before it can reach a flight controller.

    These are deliberately conservative. The policy was trained open-loop on
    recorded flights; it has never seen the states its own mistakes produce.
    """

    max_speed_xy: float = 1.5      # m/s, horizontal
    max_speed_z: float = 0.5       # m/s, vertical
    max_yaw_rate: float = 0.8      # rad/s
    # Below this the command is treated as "hold" rather than crawl.
    deadband_xy: float = 0.02      # m/s
    deadband_z: float = 0.02       # m/s
    deadband_yaw: float = 0.02     # rad/s
    # Largest change allowed between consecutive control ticks, per second.
    slew_xy: float = 2.0           # m/s per second
    slew_z: float = 1.0            # m/s per second
    slew_yaw: float = 2.0          # rad/s per second
    # If no fresh policy output arrives within this many seconds, the navigator
    # commands zero. A stalled inference must not leave the last vector latched.
    watchdog_s: float = 1.0
    # Refuse to command a descent below this altitude (metres, AGL).
    min_altitude: float = 0.8
    max_altitude: float = 20.0


@dataclass
class RunConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control_hz: float = 20.0
    instruction: str = "follow the person in an orange hard hat"
