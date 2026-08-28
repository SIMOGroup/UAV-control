"""The control loop: adapter -> policy -> navigator -> adapter.

Two rates, deliberately decoupled. The policy runs at the rate its checkpoint
was trained at (``PolicyConfig.fps``) and only when its action queue is empty;
the loop ticks at ``control_hz`` and emits a rate-limited setpoint every cycle.
That decoupling is what lets a 1 FPS checkpoint drive a 20 Hz controller
without the aircraft stepping between commands.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vla_nav.adapters.base import DroneAdapter
from vla_nav.config import RunConfig
from vla_nav.navigator import NavVector, Navigator
from vla_nav.policy import NavPolicy


@dataclass
class StepRecord:
    t: float
    action: np.ndarray          # raw policy output, body-frame displacement
    vector: NavVector           # what was actually sent
    infer_s: float
    status: str


@dataclass
class RunStats:
    steps: int = 0
    net_calls: int = 0
    infer_s: list[float] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)

    def summary(self) -> dict:
        a = np.array(self.infer_s) if self.infer_s else np.zeros(1)
        return {
            "steps": self.steps,
            "net_calls": self.net_calls,
            "infer_ms_mean": float(a.mean() * 1e3),
            "infer_ms_p95": float(np.quantile(a, 0.95) * 1e3),
            "infer_ms_max": float(a.max() * 1e3),
        }


class Runner:
    def __init__(self, adapter: DroneAdapter, cfg: RunConfig | None = None,
                 policy: NavPolicy | None = None, clock=time.monotonic):
        self.cfg = cfg or RunConfig()
        self.adapter = adapter
        self.policy = policy or NavPolicy(self.cfg.policy)
        self.nav = Navigator(fps=self.cfg.policy.fps, safety=self.cfg.safety,
                             clock=clock)
        self.policy.set_instruction(self.cfg.instruction)
        self.stats = RunStats()
        self._clock = clock

    def step_once(self) -> StepRecord:
        obs = self.adapter.observe()
        action = self.policy.step(obs.frame, obs.state, bgr=obs.bgr)
        self.nav.set_altitude(obs.altitude)
        self.nav.submit(action)
        v = self.nav.tick()
        self.adapter.send(v)

        rec = StepRecord(t=self._clock(), action=action, vector=v,
                         infer_s=self.policy.last_infer_s, status=v.status)
        self.stats.steps += 1
        self.stats.net_calls = self.policy.n_net_calls
        self.stats.infer_s.append(self.policy.last_infer_s)
        self.stats.records.append(rec)
        return rec

    def run(self, max_steps: int | None = None, sleep=time.sleep) -> RunStats:
        period = 1.0 / self.cfg.control_hz
        try:
            while self.adapter.is_ready():
                if max_steps is not None and self.stats.steps >= max_steps:
                    break
                t0 = self._clock()
                self.step_once()
                # advance() only exists on offline adapters; live ones stream.
                advance = getattr(self.adapter, "advance", None)
                if advance is not None and not advance():
                    break
                lag = period - (self._clock() - t0)
                if lag > 0:
                    sleep(lag)
        finally:
            self.adapter.send(self.nav.stop())
            self.adapter.on_stop()
        return self.stats
