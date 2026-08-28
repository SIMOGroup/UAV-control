"""Action chunk -> navigation vector, with the limits that make it flyable.

The policy emits a *displacement per model timestep* in the body frame. A flight
controller wants a *velocity setpoint*, usually at a much higher rate than the
policy runs (a 1 FPS checkpoint against a 20 Hz control loop). This module owns
that conversion and everything that has to be true before a learned vector is
allowed to reach an aircraft:

  displacement -> velocity   v = d * fps
  deadband                   crawl commands become hold
  clamp                      per-axis speed ceilings
  slew                       bounded change per control tick
  altitude guard             no descent below the floor, no climb past the roof
  watchdog                   stale policy output decays to zero, never latches

Frame convention (WareFly, and what the adapter must map from):
  dx_local  +forward   (m)      vx  +forward   (m/s)
  dy_local  +left      (m)      vy  +left      (m/s)
  dz        +up        (m)      vz  +up        (m/s)
  dyaw      +CCW       (rad)    yaw_rate +CCW  (rad/s)

PX4/MAVLink body-NED wants forward/right/down, so an adapter targeting it must
negate y and z. ``adapters/base.py`` is where that happens -- never here.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from vla_nav.config import SafetyConfig


@dataclass(frozen=True)
class NavVector:
    """A body-frame velocity setpoint. +forward / +left / +up / +CCW."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    t: float = 0.0
    # Why the vector looks the way it does; carried through so the flight log
    # can distinguish "policy said hold" from "watchdog zeroed it".
    status: str = "ok"

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz, self.yaw_rate], dtype=np.float32)

    def is_zero(self, eps: float = 1e-6) -> bool:
        return bool(np.all(np.abs(self.as_array()) < eps))


ZERO = NavVector(status="zero")


def _deadband(v: float, band: float) -> float:
    return 0.0 if abs(v) < band else v


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


class Navigator:
    """Turns policy displacements into rate-limited body-frame velocities.

    ``submit`` is called whenever a fresh policy output exists (policy rate);
    ``tick`` is called by the control loop (control rate) and always returns a
    vector that is safe to send *right now*.
    """

    def __init__(self, fps: float, safety: SafetyConfig | None = None,
                 clock=time.monotonic):
        if fps <= 0:
            raise ValueError("fps must be positive; it scales every command")
        self.fps = float(fps)
        self.safety = safety or SafetyConfig()
        self._clock = clock
        self._target = ZERO           # what the policy last asked for
        self._current = ZERO          # what we last emitted, for slew limiting
        self._last_submit_t: float | None = None
        # Seeded at construction, not left None: with no baseline the first
        # tick would see dt=0 and hand the target straight through, so the very
        # first setpoint after engaging would be an unlimited step input.
        self._last_tick_t: float = self._clock()
        self._altitude: float | None = None

    # ── inputs ──────────────────────────────────────────────────────────────
    def set_altitude(self, altitude_m: float | None) -> None:
        """Current altitude AGL, used only by the altitude guard."""
        self._altitude = altitude_m

    def submit(self, action: np.ndarray) -> NavVector:
        """Register a fresh policy action ``[dx, dy, dz, dyaw]`` (per timestep)."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 4:
            raise ValueError(f"expected 4-DoF action, got {a.size}")
        if not np.all(np.isfinite(a[:4])):
            # A NaN reaching the flight controller is worse than a stall.
            self._target = replace(ZERO, t=self._clock(), status="nonfinite")
            self._last_submit_t = self._clock()
            return self._target

        s = self.safety
        vx, vy, vz, wz = (float(x) * self.fps for x in a[:4])
        vx = _clamp(_deadband(vx, s.deadband_xy), s.max_speed_xy)
        vy = _clamp(_deadband(vy, s.deadband_xy), s.max_speed_xy)
        vz = _clamp(_deadband(vz, s.deadband_z), s.max_speed_z)
        wz = _clamp(_deadband(wz, s.deadband_yaw), s.max_yaw_rate)

        # Horizontal speed is a magnitude, not two independent axes: clamping
        # per-axis alone lets a diagonal command reach sqrt(2) * max_speed_xy.
        speed = math.hypot(vx, vy)
        if speed > s.max_speed_xy:
            k = s.max_speed_xy / speed
            vx, vy = vx * k, vy * k

        now = self._clock()
        self._last_submit_t = now
        self._target = NavVector(vx, vy, vz, wz, t=now, status="ok")
        return self._target

    # ── output ──────────────────────────────────────────────────────────────
    def tick(self) -> NavVector:
        """The vector to send this control cycle."""
        now = self._clock()
        dt = max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        target = self._target
        status = target.status

        stale = (self._last_submit_t is None
                 or now - self._last_submit_t > self.safety.watchdog_s)
        if stale:
            target = ZERO
            status = "watchdog"

        target = self._guard_altitude(target)
        if target.status != "ok" and status == "ok":
            status = target.status

        self._current = NavVector(
            *self._slew_towards(target, dt), t=now, status=status)
        return self._current

    def _slew_towards(self, target: NavVector, dt: float):
        s = self.safety
        cur, tgt = self._current.as_array(), target.as_array()
        if dt <= 0.0:
            # No time has passed, so no acceleration is justified. Holding the
            # current vector is the safe reading; handing over the target is not.
            return tuple(float(x) for x in cur)
        limits = np.array([s.slew_xy, s.slew_xy, s.slew_z, s.slew_yaw]) * dt
        delta = np.clip(tgt - cur, -limits, limits)
        return tuple(float(x) for x in (cur + delta))

    def _guard_altitude(self, v: NavVector) -> NavVector:
        s = self.safety
        if self._altitude is None:
            return v
        if v.vz < 0.0 and self._altitude <= s.min_altitude:
            return replace(v, vz=0.0, status="alt_floor")
        if v.vz > 0.0 and self._altitude >= s.max_altitude:
            return replace(v, vz=0.0, status="alt_ceiling")
        return v

    def stop(self) -> NavVector:
        """Immediate zero, bypassing the slew limiter. For aborts/takeover."""
        self._target = ZERO
        self._current = replace(ZERO, t=self._clock(), status="stop")
        return self._current
