"""Mode split logistic function tests (Milestone 4)."""
from __future__ import annotations

import pytest
from engine.mode_split import mode_split_probs, sample_mode, Mode, ModeSplitConfig


def test_probs_sum_to_one():
    for d in [500, 2000, 5000, 15000]:
        p = mode_split_probs(d)
        assert abs(sum(p.values()) - 1.0) < 1e-6


def test_short_trip_mostly_walk():
    p = mode_split_probs(500)   # 500 m
    assert p[Mode.WALK] > 0.7


def test_long_trip_mostly_car():
    p = mode_split_probs(20000)  # 20 km
    assert p[Mode.CAR] > 0.9


def test_medium_trip_mixed():
    p = mode_split_probs(4000)   # 4 km — expect car > walk
    assert p[Mode.CAR] > p[Mode.WALK]


def test_zero_distance():
    """Zero distance should not raise and walk probability should be ~1."""
    p = mode_split_probs(0)
    assert p[Mode.WALK] == pytest.approx(1.0, abs=0.1)


def test_sample_mode_is_valid():
    mode = sample_mode(3000, seed=42)
    assert mode in (Mode.CAR, Mode.WALK, Mode.BIKE)


def test_sample_mode_distribution():
    """100 samples at 500 m should produce ≥70 walk modes."""
    walks = sum(1 for i in range(100) if sample_mode(500, seed=i) == Mode.WALK)
    assert walks >= 70


def test_mode_split_car_only_config():
    """ModeSplitConfig(force_car=True) should always return CAR."""
    cfg = ModeSplitConfig(force_car=True)
    for d in [100, 1000, 10000]:
        assert sample_mode(d, config=cfg) == Mode.CAR
