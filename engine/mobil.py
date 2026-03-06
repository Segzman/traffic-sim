"""MOBIL lane-changing model.

MOBIL calls idm_acceleration() internally — it does not re-implement
acceleration logic.  This enforces consistency with the IDM engine.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from engine.idm import idm_acceleration

if TYPE_CHECKING:
    from engine.agents import Vehicle


def mobil_lane_change(
    vehicle: "Vehicle",
    follower_current: "Vehicle | None",
    leader_target: "Vehicle | None",
    follower_target: "Vehicle | None",
    politeness: float = 0.3,
    b_safe: float = 4.0,
    delta_a_thr: float = 0.2,
    bias_right: float = 0.1,
    moving_right: bool = True,
) -> bool:
    """
    MOBIL lane change decision.

    Returns True if the lane change should be executed.

    Parameters
    ----------
    vehicle:          The vehicle considering the lane change.
    follower_current: Vehicle directly behind in the *current* lane (None if none).
    leader_target:    Vehicle directly ahead in the *target* lane (None = free road).
    follower_target:  Vehicle directly behind in the *target* lane (None if none).
    politeness:       Weight on follower welfare (0 = purely selfish).
    b_safe:           Max acceptable deceleration imposed on target follower (m/s²).
    delta_a_thr:      Minimum net acceleration gain needed to change (m/s²).
    bias_right:       Keep-right bias (m/s²); lowers threshold for rightward moves.
    moving_right:     True when moving to a lower lane index (rightward / keep-right).
    """
    # 1. Cooldown check
    if vehicle.lane_change_cooldown > 0.0:
        return False

    # 2. Vehicle's acceleration in the *current* lane (already set by simulation)
    a_c_old = vehicle.acceleration

    # 3. Vehicle's acceleration if it moved to the *target* lane
    if leader_target is None:
        a_c_new = idm_acceleration(
            v=vehicle.speed, v0=vehicle.v0, s=1e9, delta_v=0.0,
            s0=vehicle.s0, T=vehicle.T, a=vehicle.a_max, b=vehicle.b,
        )
    else:
        gap = leader_target.position_s - vehicle.position_s - vehicle.length
        if gap <= 0.0:
            return False        # can't fit
        a_c_new = idm_acceleration(
            v=vehicle.speed, v0=vehicle.v0, s=gap,
            delta_v=vehicle.speed - leader_target.speed,
            s0=vehicle.s0, T=vehicle.T, a=vehicle.a_max, b=vehicle.b,
        )

    # 4. Safety criterion — target follower must not brake harder than b_safe
    a_ft_new = 0.0
    a_ft_old = 0.0
    if follower_target is not None:
        gap_ft = vehicle.position_s - follower_target.position_s - follower_target.length
        if gap_ft <= 0.0:
            return False        # spatial conflict
        a_ft_new = idm_acceleration(
            v=follower_target.speed, v0=follower_target.v0, s=gap_ft,
            delta_v=follower_target.speed - vehicle.speed,
            s0=follower_target.s0, T=follower_target.T,
            a=follower_target.a_max, b=follower_target.b,
        )
        if a_ft_new < -b_safe:
            return False        # safety violated
        a_ft_old = follower_target.acceleration

    # 5. Current follower's gain (vehicle vacates a gap)
    a_fc_gain = 0.0
    if follower_current is not None:
        a_fc_old = follower_current.acceleration
        # Simplified: after vehicle leaves, follower_current faces free road
        a_fc_new = idm_acceleration(
            v=follower_current.speed, v0=follower_current.v0, s=1e9, delta_v=0.0,
            s0=follower_current.s0, T=follower_current.T,
            a=follower_current.a_max, b=follower_current.b,
        )
        a_fc_gain = a_fc_new - a_fc_old

    # 6. Target follower's gain / loss
    a_ft_gain = (a_ft_new - a_ft_old) if follower_target is not None else 0.0

    # 7. MOBIL incentive criterion
    self_gain = a_c_new - a_c_old
    net_gain = self_gain + politeness * (a_fc_gain + a_ft_gain)

    # Moving right lowers the effective threshold (keep-right rule)
    effective_threshold = delta_a_thr - (bias_right if moving_right else -bias_right)

    return net_gain > effective_threshold
