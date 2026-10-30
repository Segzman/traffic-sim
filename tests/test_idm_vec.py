"""Vectorised IDM acceleration tests."""
import numpy as np
import pytest

from engine.idm import idm_acceleration
from engine.idm_vec import idm_acceleration_vec


def test_matches_scalar_reference_for_heterogeneous_vehicles():
    rng = np.random.default_rng(42)
    n = 50

    v = rng.uniform(0.0, 35.0, n)
    v0 = rng.uniform(10.0, 45.0, n)
    s = rng.uniform(0.5, 200.0, n)
    delta_v = rng.uniform(-10.0, 10.0, n)
    s0 = rng.uniform(1.0, 4.0, n)
    T = rng.uniform(0.8, 2.5, n)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(1.0, 5.0, n)

    actual = idm_acceleration_vec(v, v0, s, delta_v, s0, T, a, b)
    expected = np.array([
        idm_acceleration(
            float(v[i]), float(v0[i]), float(s[i]), float(delta_v[i]),
            float(s0[i]), float(T[i]), float(a[i]), float(b[i]),
        )
        for i in range(n)
    ])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_scalar_parameters_broadcast_over_vehicles():
    v = [0.0, 10.0, 20.0]
    actual = idm_acceleration_vec(
        v, 30.0, [1e9, 100.0, 50.0], [0.0, 5.0, -2.0],
        2.0, 1.5, 1.4, 2.0,
    )

    expected = np.array([
        idm_acceleration(0.0, 30.0, 1e9, 0.0, 2.0, 1.5, 1.4, 2.0),
        idm_acceleration(10.0, 30.0, 100.0, 5.0, 2.0, 1.5, 1.4, 2.0),
        idm_acceleration(20.0, 30.0, 50.0, -2.0, 2.0, 1.5, 1.4, 2.0),
    ])

    assert actual.shape == (3,)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_free_road_acceleration_and_equilibrium():
    at_rest = idm_acceleration_vec(
        [0.0], [30.0], [1e9], [0.0], 2.0, 1.5, 1.4, 2.0,
    )
    at_desired_speed = idm_acceleration_vec(
        [30.0], [30.0], [1e9], [0.0], 2.0, 1.5, 1.4, 2.0,
    )

    assert at_rest[0] == pytest.approx(1.4)
    assert at_desired_speed[0] == pytest.approx(0.0, abs=1e-6)


def test_zero_speed_and_zero_gap_are_finite():
    actual = idm_acceleration_vec(
        [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 5.0],
        2.0, 1.5, 1.4, 2.0,
    )

    assert np.all(np.isfinite(actual))
    assert np.all(actual < 0.0)


def test_pulling_away_clamps_desired_gap_at_minimum():
    pulling_away = idm_acceleration_vec(
        [20.0], [30.0], [100.0], [-20.0], 2.0, 1.5, 1.4, 2.0,
    )
    more_pulling_away = idm_acceleration_vec(
        [20.0], [30.0], [100.0], [-100.0], 2.0, 1.5, 1.4, 2.0,
    )
    expected = 1.4 * (1.0 - (20.0 / 30.0) ** 4 - (2.0 / 100.0) ** 2)

    assert pulling_away[0] == pytest.approx(expected)
    assert more_pulling_away[0] == pytest.approx(expected)


def test_per_vehicle_parameters_are_independent():
    aggressive = idm_acceleration_vec(
        [10.0], [40.0], [80.0], [0.0], 1.0, 0.8, 3.0, 4.0,
    )
    cautious = idm_acceleration_vec(
        [10.0], [15.0], [80.0], [0.0], 4.0, 2.5, 0.5, 1.0,
    )

    assert aggressive[0] > cautious[0]
