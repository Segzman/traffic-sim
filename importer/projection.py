"""Web Mercator (EPSG:3857) projection utilities."""
from __future__ import annotations
import math

# WGS-84 semi-major axis (used by EPSG:3857)
R_EARTH = 6_378_137.0


def latlng_to_mercator(lat: float, lon: float) -> tuple[float, float]:
    """WGS-84 decimal degrees → Web Mercator (x, y) metres."""
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def mercator_to_latlng(x: float, y: float) -> tuple[float, float]:
    """Web Mercator (x, y) metres → WGS-84 decimal degrees."""
    lon = math.degrees(x / R_EARTH)
    lat = math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2)
    return lat, lon


def mercator_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance in metres between two Web Mercator points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def mercator_bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Bearing in degrees (0 = North, clockwise) from p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    bearing = math.degrees(math.atan2(dx, dy))
    return bearing % 360.0
