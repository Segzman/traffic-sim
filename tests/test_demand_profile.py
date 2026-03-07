"""Hourly demand profile tests (Milestone 4)."""
from __future__ import annotations

import pytest
from engine.demand_profile import (
    WEEKDAY_PROFILE,
    WEEKEND_PROFILE,
    profile_multiplier,
    get_profile,
    DayType,
)


def test_weekday_profile_length():
    assert len(WEEKDAY_PROFILE) == 24


def test_weekend_profile_length():
    assert len(WEEKEND_PROFILE) == 24


def test_weekday_morning_peak_is_max():
    """8 AM should be the highest factor on a weekday."""
    peak_hour = WEEKDAY_PROFILE.index(max(WEEKDAY_PROFILE))
    assert peak_hour == 8


def test_weekend_midday_peak():
    """Weekend peak should fall between 10–13."""
    peak_hour = WEEKEND_PROFILE.index(max(WEEKEND_PROFILE))
    assert 10 <= peak_hour <= 13


def test_overnight_low():
    """2–4 AM must have low demand (< 5% of peak)."""
    peak = max(WEEKDAY_PROFILE)
    for h in [2, 3, 4]:
        assert WEEKDAY_PROFILE[h] < 0.05 * peak


def test_profile_multiplier_returns_float():
    v = profile_multiplier(sim_time_s=8 * 3600, day_type=DayType.WEEKDAY)
    assert isinstance(v, float) and v > 0


def test_profile_multiplier_evening_peak():
    am  = profile_multiplier(8  * 3600, DayType.WEEKDAY)
    pm  = profile_multiplier(17 * 3600, DayType.WEEKDAY)
    mid = profile_multiplier(14 * 3600, DayType.WEEKDAY)
    assert am > mid
    assert pm > mid


def test_get_profile_weekday():
    assert get_profile(DayType.WEEKDAY) is WEEKDAY_PROFILE


def test_get_profile_weekend():
    assert get_profile(DayType.WEEKEND) is WEEKEND_PROFILE


def test_spawn_count_peaks_at_rush_hour():
    """Simulation spawns more vehicles at 8 AM than at 3 AM on a weekday."""
    from engine.network import Network, Node, Edge
    from engine.network_simulation import NetworkSimulation

    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=5000.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=2,
                      speed_limit=14.0, geometry=[[0.0, 0.0], [5000.0, 0.0]]))

    demand = {"A": {"B": 200.0}}
    sim = NetworkSimulation(
        network=net,
        demand=demand,
        duration=86400.0,
        seed=1,
        day_type="weekday",
    )
    sim.run()

    am_peak = sum(1 for r in sim.trip_log if 7 * 3600 <= r.entry_time <= 9 * 3600)
    night   = sum(1 for r in sim.trip_log if 2 * 3600 <= r.entry_time <= 4 * 3600)
    assert am_peak > night * 3, f"AM peak={am_peak} night={night}"
