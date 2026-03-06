"""Milestone 6 — Renderer colour scheme unit tests.

Tests the Python colour utilities in ``viz/__init__.py`` which mirror the
logic in ``viz/renderer.js``.  No browser or canvas required.
"""
import pytest

from viz import (
    vehicle_colour,
    signal_colour,
    COLOUR_VEHICLE_FREE,
    COLOUR_VEHICLE_SLOWING,
    COLOUR_VEHICLE_CONGESTED,
    COLOUR_VEHICLE_STOPPED,
    COLOUR_SIGNAL_GREEN,
    COLOUR_SIGNAL_YELLOW,
    COLOUR_SIGNAL_RED,
    COLOUR_PEDESTRIAN,
)


# ------------------------------------------------------------------ #
# test_renderer_no_exceptions
# ------------------------------------------------------------------ #

def test_renderer_no_exceptions():
    """Colour functions execute without raising for a range of inputs.

    Simulates rendering 10 timesteps of simulation state by calling the
    colour functions for varied vehicle speeds and signal states.
    """
    speed_limit = 13.9  # m/s (~50 km/h)

    # Sweep vehicle speeds from stopped to over-limit (edge case)
    test_speeds = [0.0, 0.1, 0.3, 5.0, 8.0, 11.0, 13.9, 15.0, 30.0]
    for spd in test_speeds:
        colour = vehicle_colour(spd, speed_limit)
        assert isinstance(colour, str), f"vehicle_colour({spd}) should return str"
        assert colour.startswith("#"), (
            f"vehicle_colour({spd}) = '{colour}' should be a hex colour"
        )
        assert len(colour) == 7, (
            f"vehicle_colour({spd}) = '{colour}' should be 7 chars (#RRGGBB)"
        )

    # Sweep signal states
    for state in ("green", "yellow", "red"):
        colour = signal_colour(state)
        assert isinstance(colour, str)
        assert colour.startswith("#")

    # Unknown signal state should still return a valid colour (fallback = red)
    fallback = signal_colour("unknown")
    assert fallback == COLOUR_SIGNAL_RED


# ------------------------------------------------------------------ #
# test_vehicle_colour_by_speed
# ------------------------------------------------------------------ #

def test_vehicle_colour_by_speed():
    """Speed ≥80% of limit → green; 40–80% → amber; <40% → red; ≈0 → brown."""
    limit = 13.9  # m/s

    # Free flow: 90% of speed limit
    colour_free = vehicle_colour(limit * 0.9, limit)
    assert colour_free == COLOUR_VEHICLE_FREE, (
        f"90% speed should be free-flow green; got {colour_free}"
    )

    # Slowing: 60% of speed limit
    colour_slow = vehicle_colour(limit * 0.6, limit)
    assert colour_slow == COLOUR_VEHICLE_SLOWING, (
        f"60% speed should be slowing amber; got {colour_slow}"
    )

    # Congested: 20% of speed limit
    colour_cong = vehicle_colour(limit * 0.2, limit)
    assert colour_cong == COLOUR_VEHICLE_CONGESTED, (
        f"20% speed should be congested red; got {colour_cong}"
    )

    # Stopped: 0 m/s
    colour_stop = vehicle_colour(0.0, limit)
    assert colour_stop == COLOUR_VEHICLE_STOPPED, (
        f"0 m/s should be stopped brown; got {colour_stop}"
    )

    # Boundary: exactly 80% → free flow
    colour_boundary_hi = vehicle_colour(limit * 0.8, limit)
    assert colour_boundary_hi == COLOUR_VEHICLE_FREE, (
        f"Exactly 80% speed should be free-flow; got {colour_boundary_hi}"
    )

    # Boundary: exactly 40% → slowing (not congested)
    colour_boundary_lo = vehicle_colour(limit * 0.4, limit)
    assert colour_boundary_lo == COLOUR_VEHICLE_SLOWING, (
        f"Exactly 40% speed should be slowing; got {colour_boundary_lo}"
    )

    # Pedestrian colour is a distinct blue (sanity check of constant)
    assert COLOUR_PEDESTRIAN == "#2979FF"
