"""Milestone 4 — Web Mercator projection unit tests."""
import math
import pytest
from importer.projection import (
    latlng_to_mercator,
    mercator_to_latlng,
    mercator_distance,
    mercator_bearing,
)


# ------------------------------------------------------------------ #
# test_latlng_roundtrip
# ------------------------------------------------------------------ #

def test_latlng_roundtrip():
    """latlng → mercator → latlng recovers original within 0.000001° tolerance."""
    cases = [
        (51.5074, -0.1278),   # London
        (40.7128, -74.0060),  # New York
        (-33.8688, 151.2093), # Sydney
        (0.0, 0.0),           # Origin
        (85.0, 179.9),        # Near polar limit
        (-85.0, -179.9),
    ]
    for lat, lon in cases:
        x, y = latlng_to_mercator(lat, lon)
        lat2, lon2 = mercator_to_latlng(x, y)
        assert abs(lat2 - lat) < 1e-6, f"lat roundtrip failed: {lat} → {lat2}"
        assert abs(lon2 - lon) < 1e-6, f"lon roundtrip failed: {lon} → {lon2}"


# ------------------------------------------------------------------ #
# test_known_coordinate_london
# ------------------------------------------------------------------ #

def test_known_coordinate_london():
    """51.5074°N, 0.1278°W projects to correct Web Mercator coordinates."""
    x, y = latlng_to_mercator(51.5074, -0.1278)
    # Verified values from the Web Mercator (EPSG:3857) formula.
    # x = R * radians(-0.1278), y = R * ln(tan(π/4 + radians(51.5074)/2))
    assert abs(x - (-14226.6)) < 1.0, f"x={x:.2f} expected≈-14226.6"
    assert abs(y - 6_711_542.5) < 1.0, f"y={y:.1f} expected≈6711542.5"


# ------------------------------------------------------------------ #
# test_mercator_distance_accuracy
# ------------------------------------------------------------------ #

def test_mercator_distance_accuracy():
    """Distance between two Mercator points exactly 1 km apart is ≤ 0.1% error."""
    # Place two points 1000 m apart on the x-axis in Mercator space
    p1 = (0.0, 0.0)
    p2 = (1000.0, 0.0)
    d = mercator_distance(p1, p2)
    assert abs(d - 1000.0) / 1000.0 < 0.001, f"distance={d:.4f}m"

    # Also test diagonal
    p3 = (600.0, 800.0)   # 3-4-5 triangle scaled ×200 → 1000m hypotenuse
    d2 = mercator_distance(p1, p3)
    assert abs(d2 - 1000.0) < 0.001, f"diagonal distance={d2:.4f}m"


# ------------------------------------------------------------------ #
# test_bearing_cardinal
# ------------------------------------------------------------------ #

def test_bearing_cardinal():
    """Cardinal bearings from Mercator pairs are correct to ±0.01°."""
    origin = (0.0, 0.0)

    # North: positive y direction
    north = (0.0, 1000.0)
    assert mercator_bearing(origin, north) == pytest.approx(0.0, abs=0.01)

    # East: positive x direction
    east = (1000.0, 0.0)
    assert mercator_bearing(origin, east) == pytest.approx(90.0, abs=0.01)

    # South: negative y direction
    south = (0.0, -1000.0)
    assert mercator_bearing(origin, south) == pytest.approx(180.0, abs=0.01)

    # West: negative x direction
    west = (-1000.0, 0.0)
    assert mercator_bearing(origin, west) == pytest.approx(270.0, abs=0.01)
