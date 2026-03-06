"""Nominatim geocoding — query string → location info + clipped bounding box."""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "TrafficSimPlatform/0.1 (educational)"

# Approximate half-size of the bbox to clip around the geocoded centre.
# ~0.045° lat ≈ 5 km; longitude is corrected for the cosine of latitude.
_HALF_DEG_LAT = 0.045


def geocode(query: str) -> dict:
    """Geocode *query* with Nominatim and return location metadata.

    Parameters
    ----------
    query:
        Free-form place name, e.g. ``"Tokyo, Japan"`` or ``"Vancouver, BC"``.

    Returns
    -------
    dict with keys:

    ``display_name`` : str
        Human-readable place name from Nominatim.
    ``lat``, ``lon`` : float
        WGS-84 centre coordinates.
    ``bbox`` : tuple[float, float, float, float]
        ``(south, west, north, east)`` clipped to roughly ±5 km from centre.

    Raises
    ------
    ValueError
        If Nominatim returns no results.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    })
    url = f"{_NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    with urllib.request.urlopen(req, timeout=10) as resp:
        results = json.loads(resp.read().decode("utf-8"))

    if not results:
        raise ValueError(f"No results found for: {query!r}")

    r = results[0]
    lat = float(r["lat"])
    lon = float(r["lon"])

    # Clip to a fixed radius so we don't import enormous areas.
    # Longitude half-width is wider at low latitudes, narrower near poles.
    south = lat - _HALF_DEG_LAT
    north = lat + _HALF_DEG_LAT
    lon_half = _HALF_DEG_LAT / max(0.05, math.cos(math.radians(lat)))
    west = lon - lon_half
    east = lon + lon_half

    return {
        "display_name": r.get("display_name", query),
        "lat": lat,
        "lon": lon,
        "bbox": (south, west, north, east),
    }
