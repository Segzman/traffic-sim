"""Colour scheme and rendering utilities for the traffic simulation visualiser.

These constants and functions mirror the colour logic in ``viz/renderer.js``
so that Python tests can validate the colour scheme without a browser.
"""
from __future__ import annotations

# ------------------------------------------------------------------ #
# Colour constants (must match renderer.js)
# ------------------------------------------------------------------ #

COLOUR_VEHICLE_FREE     = "#00C853"  # speed ≥ 80 % of limit  (green)
COLOUR_VEHICLE_SLOWING  = "#FFD600"  # 40–80 % of limit       (amber)
COLOUR_VEHICLE_CONGESTED = "#D50000" # < 40 % of limit        (red)
COLOUR_VEHICLE_STOPPED  = "#8D6E63"  # essentially stationary (brown)

COLOUR_PEDESTRIAN       = "#2979FF"  # blue

COLOUR_SIGNAL_GREEN     = "#00E676"
COLOUR_SIGNAL_YELLOW    = "#FFEA00"
COLOUR_SIGNAL_RED       = "#FF1744"

COLOUR_ROAD_EDGE        = "#37474F"  # dark slate

# Speed threshold below which a vehicle is considered "stopped" (m/s)
_STOP_THRESHOLD = 0.3


# ------------------------------------------------------------------ #
# Colour functions
# ------------------------------------------------------------------ #

def vehicle_colour(speed: float, speed_limit: float) -> str:
    """Return the hex colour for a vehicle given its current speed.

    Parameters
    ----------
    speed:
        Current vehicle speed (m/s).  Must be ≥ 0.
    speed_limit:
        Edge speed limit (m/s).  Must be > 0.

    Returns
    -------
    str
        Hex colour string (e.g. ``"#00C853"``).
    """
    if speed <= _STOP_THRESHOLD:
        return COLOUR_VEHICLE_STOPPED
    ratio = speed / speed_limit if speed_limit > 0.0 else 0.0
    if ratio >= 0.8:
        return COLOUR_VEHICLE_FREE
    if ratio >= 0.4:
        return COLOUR_VEHICLE_SLOWING
    return COLOUR_VEHICLE_CONGESTED


def signal_colour(state: str) -> str:
    """Return the hex colour for a signal state string.

    Parameters
    ----------
    state:
        One of ``"green"``, ``"yellow"``, or ``"red"``.

    Returns
    -------
    str
        Corresponding hex colour.
    """
    return {
        "green":  COLOUR_SIGNAL_GREEN,
        "yellow": COLOUR_SIGNAL_YELLOW,
        "red":    COLOUR_SIGNAL_RED,
    }.get(state, COLOUR_SIGNAL_RED)
