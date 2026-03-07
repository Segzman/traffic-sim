"""Weather fetch, parsing, and IDM multiplier tests (Milestone 5)."""
from __future__ import annotations

import pytest
from engine.weather import apply_weather_multipliers, WeatherState, WEATHER_MULTIPLIERS


# ---------------------------------------------------------------------------
# Multiplier unit tests (no network I/O needed)
# ---------------------------------------------------------------------------

def test_clear_weather_no_change():
    w = WeatherState("clear", 0.0)
    base = dict(v0=30.0, s0=2.0, T=1.5, a=1.4)
    result = apply_weather_multipliers(base, w)
    assert result["v0"] == pytest.approx(30.0)


def test_rain_reduces_v0():
    w = WeatherState("rain", 0.5)
    base = dict(v0=30.0, s0=2.0, T=1.5, a=1.4)
    result = apply_weather_multipliers(base, w)
    assert result["v0"] < 30.0


def test_snow_greatly_reduces_v0():
    w = WeatherState("snow", 2.0)
    base = dict(v0=30.0, s0=2.0, T=1.5, a=1.4)
    result = apply_weather_multipliers(base, w)
    assert result["v0"] < 20.0
    assert result["s0"] > 2.5
    assert result["T"]  > 1.8


def test_weather_multipliers_all_positive():
    base = dict(v0=30.0, s0=2.0, T=1.5, a=1.4)
    for cond in WEATHER_MULTIPLIERS:
        w = WeatherState(cond, 1.0)
        result = apply_weather_multipliers(base, w)
        for k, v in result.items():
            assert v > 0, f"Non-positive param {k} for condition {cond}"


def test_rain_increases_gap():
    w = WeatherState("rain", 0.5)
    base = dict(v0=30.0, s0=2.0, T=1.5, a=1.4)
    result = apply_weather_multipliers(base, w)
    assert result["s0"] > 2.0
    assert result["T"]  > 1.5


# ---------------------------------------------------------------------------
# Simulation speed reduction
# ---------------------------------------------------------------------------

def test_rain_slows_average_speed():
    from engine.network import Network, Node, Edge
    from engine.network_simulation import NetworkSimulation

    def _avg_speed(condition: str) -> float:
        net = Network()
        net.add_node(Node(id="A", x=0.0, y=0.0))
        net.add_node(Node(id="B", x=2000.0, y=0.0))
        net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1,
                          speed_limit=14.0, geometry=[[0.0, 0.0], [2000.0, 0.0]]))
        sim = NetworkSimulation(
            net,
            demand={"A": {"B": 600.0}},
            duration=120.0,
            seed=0,
            weather=WeatherState(condition, 1.0),
        )
        for _ in range(500):
            sim.step()
        speeds = [v.speed for v in sim.vehicles] if sim.vehicles else [0.0]
        return sum(speeds) / len(speeds)

    clear_speed = _avg_speed("clear")
    rain_speed  = _avg_speed("rain")
    assert rain_speed < clear_speed * 0.97, (
        f"Rain should be ≥3% slower than clear (clear={clear_speed:.2f}, rain={rain_speed:.2f})"
    )


# ---------------------------------------------------------------------------
# API fetch — mocked to avoid real network calls
# ---------------------------------------------------------------------------

def test_fetch_weather_network_error_fallback(monkeypatch):
    """On network error, fetch_weather must return clear weather, not raise."""
    import urllib.request

    def _fail(*a, **kw):
        raise OSError("No network")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)

    # Reset module-level cache so the monkeypatch is actually hit
    import engine.weather as _wmod
    _wmod._cache_ts  = -_wmod._CACHE_TTL_S - 1.0
    _wmod._cache_key = None

    from engine.weather import fetch_weather
    w = fetch_weather(48.8566, 2.3522)
    assert w.condition == "clear"
