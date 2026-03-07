"""24-hour demand profiles for weekday and weekend (Milestone 4).

Profiles are normalised lookup tables: index = hour of day (0 = midnight),
value = fraction of peak demand.  Peak value is 1.0.
"""
from __future__ import annotations

from enum import Enum


# Weekday: bimodal with peaks at 08:00 and 17:00.
WEEKDAY_PROFILE: list[float] = [
    0.05, 0.02, 0.01, 0.01, 0.02, 0.04,  # 00–05
    0.08, 0.18, 1.00, 0.70, 0.50, 0.45,  # 06–11
    0.60, 0.55, 0.50, 0.55, 0.70, 0.90,  # 12–17
    0.65, 0.45, 0.30, 0.20, 0.12, 0.07,  # 18–23
]

# Weekend: unimodal with peak around 12:00.
WEEKEND_PROFILE: list[float] = [
    0.04, 0.02, 0.01, 0.01, 0.02, 0.03,  # 00–05
    0.05, 0.08, 0.12, 0.20, 0.40, 0.70,  # 06–11
    0.80, 0.75, 0.65, 0.60, 0.55, 0.50,  # 12–17
    0.40, 0.30, 0.20, 0.12, 0.07, 0.05,  # 18–23
]


class DayType(Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"


def get_profile(day_type: DayType) -> list[float]:
    """Return the 24-value profile list for the given day type."""
    if day_type == DayType.WEEKEND:
        return WEEKEND_PROFILE
    return WEEKDAY_PROFILE


def profile_multiplier(
    sim_time_s: float,
    day_type: DayType = DayType.WEEKDAY,
) -> float:
    """Linearly-interpolated demand multiplier for a given clock time.

    Parameters
    ----------
    sim_time_s:
        Seconds since midnight (may exceed 86 400; wraps automatically).
    day_type:
        WEEKDAY or WEEKEND profile to use.

    Returns
    -------
    float
        Demand multiplier in [0.0, 1.0].
    """
    profile = get_profile(day_type)
    h     = (sim_time_s / 3600.0) % 24.0
    lo    = int(h) % 24
    hi    = (lo + 1) % 24
    frac  = h - int(h)
    return profile[lo] * (1.0 - frac) + profile[hi] * frac
