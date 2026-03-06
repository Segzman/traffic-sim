"""Milestone 2 — MOBIL lane-change unit tests."""
import pytest

from engine.agents import Vehicle
from engine.idm import idm_acceleration
from engine.mobil import mobil_lane_change


def _veh(lane_id=1, position_s=100.0, speed=20.0, v0=30.0, acceleration=0.0, cooldown=0.0):
    v = Vehicle(id=0, lane_id=lane_id, position_s=position_s, speed=speed, v0=v0,
                s0=2.0, T=1.5, a_max=1.4, b=2.0, acceleration=acceleration,
                lane_change_cooldown=cooldown)
    return v


def test_lane_change_incentive_positive():
    """Vehicle braking behind a slow leader gains > delta_a_thr on free target lane → True."""
    # Set current acceleration to a braking value (slow leader in current lane)
    a_braking = -0.8
    veh = _veh(speed=20.0, acceleration=a_braking)

    # Target lane: free road  → a_c_new ≈ 1.12 m/s²
    # self_gain = 1.12 - (-0.8) = 1.92 >> delta_a_thr = 0.2
    result = mobil_lane_change(
        vehicle=veh,
        follower_current=None,
        leader_target=None,
        follower_target=None,
        delta_a_thr=0.2,
        bias_right=0.1,
        moving_right=False,   # moving left — higher threshold (delta_a_thr + bias = 0.3)
    )
    # gain 1.92 > 0.3 → True
    assert result is True


def test_lane_change_blocked_by_safety():
    """Target follower braking > b_safe → should_change=False regardless of incentive."""
    veh = _veh(position_s=100.0, speed=20.0, acceleration=0.0)

    # Follower extremely close and fast → forced braking >> b_safe
    follower_t = Vehicle(id=2, lane_id=0, position_s=95.1, speed=28.0, v0=30.0,
                         s0=2.0, T=1.5, a_max=1.4, b=2.0, acceleration=0.0)
    # gap_ft = 100 - 95.1 - 4.5 = 0.4 m — nearly zero → a_ft << -4.0

    result = mobil_lane_change(
        vehicle=veh,
        follower_current=None,
        leader_target=None,
        follower_target=follower_t,
        b_safe=4.0,
    )
    assert result is False


def test_keep_right_bias():
    """Equal target-lane conditions: right move (lower threshold) → True, left → False."""
    # Free road in both target lanes.  Set vehicle.acceleration so that
    # net_gain == delta_a_thr exactly (= 0.2), between the two effective thresholds.
    v, v0, s0, T, a_max, b = 20.0, 30.0, 2.0, 1.5, 1.4, 2.0
    a_free = idm_acceleration(v=v, v0=v0, s=1e9, delta_v=0.0, s0=s0, T=T, a=a_max, b=b)
    # net_gain = a_free - a_c_old = delta_a_thr  ⟹  a_c_old = a_free - delta_a_thr
    delta_a_thr = 0.2
    bias = 0.1
    a_c_old = a_free - delta_a_thr

    veh = Vehicle(id=0, lane_id=1, position_s=100.0, speed=v, v0=v0,
                  s0=s0, T=T, a_max=a_max, b=b, acceleration=a_c_old)

    # moving right: effective threshold = 0.2 - 0.1 = 0.1;  gain 0.2 > 0.1 → True
    result_right = mobil_lane_change(
        vehicle=veh, follower_current=None, leader_target=None, follower_target=None,
        delta_a_thr=delta_a_thr, bias_right=bias, moving_right=True,
    )
    # moving left:  effective threshold = 0.2 + 0.1 = 0.3;  gain 0.2 > 0.3 → False
    result_left = mobil_lane_change(
        vehicle=veh, follower_current=None, leader_target=None, follower_target=None,
        delta_a_thr=delta_a_thr, bias_right=bias, moving_right=False,
    )

    assert result_right is True
    assert result_left is False


def test_no_change_when_satisfied():
    """Vehicle already at desired speed with free road → gain ≈ 0 < threshold → False."""
    # At v = v0 = 30 m/s, IDM free-road accel ≈ 0.
    # If vehicle.acceleration is also 0 (same conditions), self_gain = 0 < 0.1 (right threshold)
    veh = _veh(speed=30.0, v0=30.0, acceleration=0.0)

    result = mobil_lane_change(
        vehicle=veh,
        follower_current=None,
        leader_target=None,
        follower_target=None,
        delta_a_thr=0.2,
        bias_right=0.1,
        moving_right=True,   # lower threshold = 0.1; gain ≈ 0 < 0.1 → False
    )
    assert result is False


def test_cooldown_prevents_rapid_changes():
    """Vehicle with lane_change_cooldown > 0 always returns False."""
    veh = _veh(speed=20.0, acceleration=-0.8, cooldown=2.5)  # 2.5 s remaining

    result = mobil_lane_change(
        vehicle=veh,
        follower_current=None,
        leader_target=None,
        follower_target=None,
    )
    assert result is False
