"""Milestone 1 — IDM unit tests."""
import math
import random

import pytest

from engine.idm import idm_acceleration


def test_free_road_acceleration():
    """On empty road (s=inf), acceleration approaches a_max × (1 − (v/v0)^delta)."""
    a_max = 1.4
    v0 = 30.0
    delta = 4.0

    # At v=0: expected = a_max * (1 - 0) = a_max
    accel = idm_acceleration(v=0.0, v0=v0, s=1e9, delta_v=0.0, s0=2.0, T=1.5, a=a_max, b=2.0)
    expected = a_max * (1.0 - (0.0 / v0) ** delta)
    assert abs(accel - expected) < 0.01

    # At v=20 m/s
    v = 20.0
    accel = idm_acceleration(v=v, v0=v0, s=1e9, delta_v=0.0, s0=2.0, T=1.5, a=a_max, b=2.0)
    expected = a_max * (1.0 - (v / v0) ** delta)
    assert abs(accel - expected) < 0.01


def test_standstill_braking():
    """When gap s equals s0 and delta_v=0, output is strongly negative."""
    s0 = 2.0
    v = 10.0
    accel = idm_acceleration(v=v, v0=30.0, s=s0, delta_v=0.0,
                              s0=s0, T=1.5, a=1.4, b=2.0)
    # s_star = s0 + v*T = 2 + 15 = 17 >> s0, so (17/2)^2 ≈ 72 dominates
    assert accel < -1.0


def test_desired_speed_equilibrium():
    """At v=v0 and large gap, acceleration is approximately 0."""
    v0 = 30.0
    accel = idm_acceleration(v=v0, v0=v0, s=1e9, delta_v=0.0,
                              s0=2.0, T=1.5, a=1.4, b=2.0)
    assert abs(accel) < 0.01


def test_approaching_leader():
    """With positive delta_v (closing on leader), acceleration decreases vs free road."""
    v, v0, s = 20.0, 30.0, 50.0
    kwargs = dict(v0=v0, s0=2.0, T=1.5, a=1.4, b=2.0)

    accel_free = idm_acceleration(v=v, s=s, delta_v=0.0, **kwargs)
    accel_closing = idm_acceleration(v=v, s=s, delta_v=5.0, **kwargs)

    assert accel_closing < accel_free


def test_no_negative_gap():
    """Function raises ValueError if s <= 0."""
    with pytest.raises(ValueError):
        idm_acceleration(v=10.0, v0=30.0, s=0.0, delta_v=0.0,
                         s0=2.0, T=1.5, a=1.4, b=2.0)
    with pytest.raises(ValueError):
        idm_acceleration(v=10.0, v0=30.0, s=-5.0, delta_v=0.0,
                         s0=2.0, T=1.5, a=1.4, b=2.0)


def test_parameter_bounds():
    """Randomised inputs within valid ranges never produce acceleration > a_max + 0.01."""
    rng = random.Random(42)
    for _ in range(1000):
        v = rng.uniform(0.0, 40.0)
        v0 = rng.uniform(10.0, 50.0)
        s = rng.uniform(0.1, 200.0)
        delta_v = rng.uniform(-10.0, 10.0)
        s0 = rng.uniform(1.0, 5.0)
        T = rng.uniform(0.5, 3.0)
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(1.0, 5.0)

        accel = idm_acceleration(v=v, v0=v0, s=s, delta_v=delta_v,
                                  s0=s0, T=T, a=a, b=b)
        assert accel <= a + 0.01, (
            f"accel {accel:.4f} > a_max {a:.4f} + 0.01 "
            f"(v={v}, v0={v0}, s={s}, delta_v={delta_v})"
        )
