"""SimConfig — thread-safe runtime configuration for NetworkSimulation.

All parameters can be updated live via ``SimConfig.update(**kwargs)`` while the
simulation runs on another thread.  The internal ``threading.Lock`` serialises
concurrent reads/writes so the sim loop never observes a half-updated state.

Usage
-----
::

    from engine.config import SimConfig
    cfg = SimConfig(idm_a_max=2.5, speed_mult=8)
    sim = NetworkSimulation(net, demand=demand, duration=3600, config=cfg)

    # Live update from the HTTP /control handler:
    cfg.update(weather_v0_mult=0.9, demand_scale=1.5)
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ------------------------------------------------------------------
    # IDM driving parameters
    # ------------------------------------------------------------------
    idm_a_max: float = 1.4    # max acceleration  (m/s²)
    idm_b:     float = 2.0    # comfortable decel (m/s²)
    idm_T:     float = 1.5    # desired time headway (s)
    idm_s0:    float = 2.0    # minimum jam gap (m)

    # ------------------------------------------------------------------
    # MOBIL lane-change parameters
    # ------------------------------------------------------------------
    mobil_politeness: float = 0.3   # 0 = selfish, 1 = fully altruistic
    mobil_b_safe:     float = 3.0   # safety decel threshold (m/s²)

    # ------------------------------------------------------------------
    # Signal parameters
    # ------------------------------------------------------------------
    signal_cycle:       float = 90.0   # full cycle length (s)
    signal_green_ratio: float = 0.50   # green fraction of cycle

    # ------------------------------------------------------------------
    # Demand scaling
    # ------------------------------------------------------------------
    demand_scale: float = 1.0   # multiplier on all OD flows

    # ------------------------------------------------------------------
    # Weather multipliers  (applied to IDM params each step)
    # ------------------------------------------------------------------
    weather_v0_mult: float = 1.0   # desired-speed multiplier (rain → 0.9)
    weather_s0_mult: float = 1.0   # min-gap multiplier       (rain → 1.15)
    weather_T_mult:  float = 1.0   # headway multiplier       (rain → 1.15)

    # ------------------------------------------------------------------
    # Performance / simulation speed
    # ------------------------------------------------------------------
    speed_mult: float = 1.0    # wall-clock speed multiplier (1 = real-time)
    use_gpu:    bool  = False  # enable GPU backend (M6)

    # ------------------------------------------------------------------
    # Internal — excluded from comparisons and repr
    # ------------------------------------------------------------------
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        compare=False,
        repr=False,
        hash=False,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, **kwargs) -> None:
        """Thread-safely update one or more config fields.

        Unknown keys and values that cannot be coerced to the field's type
        are silently ignored — the server should never crash on bad input.
        """
        valid = {f.name for f in dataclasses.fields(self) if not f.name.startswith("_")}
        with self._lock:
            for key, value in kwargs.items():
                if key not in valid:
                    continue
                current = getattr(self, key)
                try:
                    setattr(self, key, type(current)(value))
                except (TypeError, ValueError):
                    pass  # coercion failed — leave unchanged

    def snapshot(self) -> dict:
        """Return a plain-dict copy of all public fields (thread-safe)."""
        with self._lock:
            return {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if not f.name.startswith("_")
            }
