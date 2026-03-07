"""Weather state and IDM multipliers for Milestone 5.

Live weather is fetched from Open-Meteo (no API key required) and cached
for 15 minutes.  All network errors fall back silently to "clear".
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# IDM multipliers per condition
# ---------------------------------------------------------------------------

WEATHER_MULTIPLIERS: dict[str, dict[str, float]] = {
    "clear":      {"v0": 1.00, "s0": 1.00, "T": 1.00, "a": 1.00},
    "rain":       {"v0": 0.90, "s0": 1.15, "T": 1.15, "a": 0.95},
    "heavy_rain": {"v0": 0.80, "s0": 1.30, "T": 1.30, "a": 0.85},
    "snow":       {"v0": 0.55, "s0": 1.50, "T": 1.50, "a": 0.70},
}


@dataclass
class WeatherState:
    condition: str       # "clear" | "rain" | "heavy_rain" | "snow"
    precip_mm_hr: float  # precipitation rate in mm/h


def apply_weather_multipliers(
    base: dict[str, float],
    weather: WeatherState,
) -> dict[str, float]:
    """Return a copy of *base* IDM params scaled by the weather condition.

    Parameters
    ----------
    base:
        Dict with keys ``v0``, ``s0``, ``T``, ``a`` (and optionally others).
    weather:
        Current weather state.

    Returns
    -------
    dict
        New dict with the same keys; unrecognised keys are passed through
        unchanged.
    """
    mults = WEATHER_MULTIPLIERS.get(weather.condition, WEATHER_MULTIPLIERS["clear"])
    return {k: v * mults.get(k, 1.0) for k, v in base.items()}


# ---------------------------------------------------------------------------
# WMO weather-code → condition mapping
# ---------------------------------------------------------------------------

def _wmo_to_condition(code: int, precip_mm_hr: float) -> str:
    """Map a WMO weather interpretation code + precip rate to a condition str."""
    # Snow codes: 71-77 (snow fall), 85-86 (snow showers)
    if code in range(71, 78) or code in (85, 86):
        return "snow"
    if precip_mm_hr > 4.0:
        return "heavy_rain"
    if precip_mm_hr > 0.1:
        return "rain"
    return "clear"


# ---------------------------------------------------------------------------
# Fetch with 15-minute in-process cache
# ---------------------------------------------------------------------------

_CACHE_TTL_S: float = 900.0   # 15 minutes
_cache_ts: float = -_CACHE_TTL_S    # force miss on first call
_cache_val: WeatherState = WeatherState("clear", 0.0)
_cache_key: tuple[float, float] | None = None


def fetch_weather(lat: float, lon: float) -> WeatherState:
    """Fetch current weather for (*lat*, *lon*) from Open-Meteo.

    Results are cached for 15 minutes.  Any network or parse error returns
    ``WeatherState("clear", 0.0)`` so the simulation degrades gracefully.
    """
    global _cache_ts, _cache_val, _cache_key

    now = time.monotonic()
    if _cache_key == (lat, lon) and now - _cache_ts < _CACHE_TTL_S:
        return _cache_val

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&current=precipitation,weathercode"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        current = data["current"]
        precip   = float(current.get("precipitation", 0.0))
        code     = int(current.get("weathercode", 0))
        condition = _wmo_to_condition(code, precip)
        state = WeatherState(condition, precip)
    except Exception:
        state = WeatherState("clear", 0.0)

    _cache_ts  = now
    _cache_val = state
    _cache_key = (lat, lon)
    return state
