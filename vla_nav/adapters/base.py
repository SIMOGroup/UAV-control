"""The seam between the learned policy and a particular aircraft stack.

Everything above this file is stack-agnostic: it speaks WareFly's body frame
(+forward / +left / +up / +CCW) in SI units. Everything below it is specific to
one autopilot. Keep the frame conversion here and nowhere else.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from vla_nav.navigator import NavVector


@dataclass
class Observation:
    frame: np.ndarray          # (H, W, 3) uint8
    state: np.ndarray          # [x, y, z, yaw_rad] as the policy was trained on
    altitude: float | None = None
    bgr: bool = False          # OpenCV cameras hand back BGR
    t: float = 0.0


class DroneAdapter(abc.ABC):
    """Implement these five methods to fly the policy on a new stack."""

    @abc.abstractmethod
    def observe(self) -> Observation:
        """Latest camera frame + state. Must not block longer than one tick."""

    @abc.abstractmethod
    def send(self, v: NavVector) -> None:
        """Command a body-frame velocity setpoint. Convert frames HERE."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True only while it is safe to accept learned setpoints.

        Should go False on: loss of offboard/guided mode, pilot takeover, low
        battery, link loss, or a failsafe. The run loop stops commanding the
        moment this is False -- that is the whole point of the method.
        """

    def on_stop(self) -> None:
        """Called once when the loop exits. Hover/land/hand back control."""

    def close(self) -> None:
        pass


def to_body_ned(v: NavVector) -> tuple[float, float, float, float]:
    """WareFly body (+fwd, +left, +up, +CCW) -> PX4/MAVLink body-NED.

    body-NED is (+forward, +right, +down) with yaw positive clockwise, so y, z
    and the yaw rate all flip sign. Getting this wrong flies the aircraft into
    the mirror image of the intended path, which is why it lives in one place.
    """
    return (v.vx, -v.vy, -v.vz, -v.yaw_rate)


def to_enu(v: NavVector) -> tuple[float, float, float, float]:
    """WareFly body -> ROS body-ENU (+forward, +left, +up, +CCW): unchanged."""
    return (v.vx, v.vy, v.vz, v.yaw_rate)
