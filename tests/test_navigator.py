"""Safety behaviour of the navigator. Plain asserts; run with python3 directly."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_nav.config import SafetyConfig
from vla_nav.navigator import Navigator


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def nav(**kw):
    c = Clock()
    s = SafetyConfig(**kw)
    return Navigator(fps=10.0, safety=s, clock=c), c


def test_displacement_to_velocity():
    n, c = nav(slew_xy=1e9, deadband_xy=0.0)
    n.submit(np.array([0.1, 0.0, 0.0, 0.0]))   # 0.1 m per timestep at 10 fps
    c.advance(0.1)
    v = n.tick()
    assert abs(v.vx - 1.0) < 1e-6, v.vx        # -> 1.0 m/s


def test_speed_clamp_is_a_magnitude():
    n, c = nav(max_speed_xy=1.0, slew_xy=1e9, deadband_xy=0.0)
    n.submit(np.array([0.2, 0.2, 0.0, 0.0]))   # 2.0, 2.0 m/s diagonal
    c.advance(0.1)
    v = n.tick()
    speed = (v.vx ** 2 + v.vy ** 2) ** 0.5
    assert speed <= 1.0 + 1e-6, speed          # not sqrt(2) * max


def test_deadband():
    n, c = nav(deadband_xy=0.05, slew_xy=1e9)
    n.submit(np.array([0.001, 0.0, 0.0, 0.0]))  # 0.01 m/s
    c.advance(0.1)
    assert n.tick().vx == 0.0


def test_slew_limits_change():
    n, c = nav(slew_xy=1.0, deadband_xy=0.0, max_speed_xy=10.0)
    n.submit(np.array([0.5, 0.0, 0.0, 0.0]))   # asks for 5 m/s
    c.advance(0.1)
    v = n.tick()
    assert abs(v.vx - 0.1) < 1e-6, v.vx        # 1.0 m/s^2 * 0.1 s


def test_watchdog_zeroes_stale_commands():
    n, c = nav(watchdog_s=0.5, slew_xy=1e9, deadband_xy=0.0)
    n.submit(np.array([0.1, 0.0, 0.0, 0.0]))
    c.advance(0.1); assert n.tick().vx > 0
    c.advance(1.0)
    v = n.tick()
    assert v.vx == 0.0 and v.status == "watchdog"


def test_altitude_floor_blocks_descent_only():
    n, c = nav(min_altitude=1.0, slew_z=1e9, deadband_z=0.0)
    n.set_altitude(0.5)
    n.submit(np.array([0.0, 0.0, -0.1, 0.0]))  # descend
    c.advance(0.1)
    v = n.tick()
    assert v.vz == 0.0 and v.status == "alt_floor"
    n.submit(np.array([0.0, 0.0, 0.1, 0.0]))   # climb is still allowed
    c.advance(0.1)
    assert n.tick().vz > 0


def test_nonfinite_action_never_reaches_the_vehicle():
    n, c = nav(slew_xy=1e9)
    n.submit(np.array([np.nan, 0.0, 0.0, 0.0]))
    c.advance(0.1)
    v = n.tick()
    assert v.is_zero() and v.status == "nonfinite"


def test_stop_bypasses_slew():
    n, c = nav(slew_xy=0.001, deadband_xy=0.0, max_speed_xy=10.0)
    n.submit(np.array([0.5, 0.0, 0.0, 0.0]))
    c.advance(1.0); n.tick()
    assert n.stop().is_zero()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"{len(fns)} passed")
