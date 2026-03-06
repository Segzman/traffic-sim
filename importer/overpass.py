"""Overpass API client with disk caching."""
from __future__ import annotations

import hashlib
import json
import os
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".overpass_cache")
_TIMEOUT = 60          # seconds per request
_MAX_RETRIES = 3
_RETRY_DELAY = 5       # seconds between retries


def build_query(south: float, west: float, north: float, east: float) -> str:
    """Build an Overpass QL query for highway ways in the given bounding box."""
    bbox = f"{south},{west},{north},{east}"
    return (
        f'[out:json][timeout:{_TIMEOUT}];\n'
        f'(\n'
        f'  way["highway"]({bbox});\n'
        f'  node(w);\n'
        f');\n'
        f'out body;\n'
        f'>;\n'
        f'out skel qt;'
    )


def _cache_path(south: float, west: float, north: float, east: float,
                cache_dir: str) -> str:
    key = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"osm_{digest}.json")


def fetch(
    bbox: tuple[float, float, float, float],
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> dict:
    """Fetch OSM data for *bbox* = (south, west, north, east).

    Responses are cached to *cache_dir* to avoid re-fetching during
    development.  Raises ``RuntimeError`` if all retries fail.
    """
    import urllib.request
    import urllib.parse

    south, west, north, east = bbox

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(south, west, north, east, cache_dir)

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)

    query = build_query(south, west, north, east)
    data = urllib.parse.urlencode({"data": query}).encode()

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            result = json.loads(raw)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(result, fh)
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(
        f"Overpass fetch failed after {_MAX_RETRIES} attempts: {last_exc}"
    )
