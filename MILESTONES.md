# Traffic Simulation — Realism Upgrade Milestones

> Implementation plan derived from the ChatGPT research document
> **Build order:** each milestone is self-contained, tested before the next begins.
> Current baseline: IDM · MOBIL · SFM pedestrians · OSM import · CartoDB tiles · bezier roads · OSM signals

---

## Table of Contents

| # | Milestone | Priority | Est. Effort |
|---|-----------|----------|-------------|
| 1 | [Performance Foundation](#milestone-1-performance-foundation) | 🔴 Critical | ~2 days |
| 2 | [Interactive Dashboard](#milestone-2-interactive-dashboard) | 🔴 Critical | ~3 days |
| 3 | [Realistic Spatial Demand](#milestone-3-realistic-spatial-demand) | 🟠 High | ~3 days |
| 4 | [Temporal Demand & Mode Split](#milestone-4-temporal-demand--mode-split) | 🟠 High | ~2 days |
| 5 | [Weather Effects](#milestone-5-weather-effects) | 🟡 Medium | ~1 day |
| 6 | [GPU Acceleration](#milestone-6-gpu-acceleration) | 🟡 Medium | ~2 days |
| 7 | [WebGL Renderer](#milestone-7-webgl-renderer) | 🟢 Optional | ~2 days |
| 8 | [Cross-Platform & Distribution](#milestone-8-cross-platform--distribution) | 🟡 Medium | ~1 day |

---

## Milestone 1 — Performance Foundation

**Goal:** Make the simulation fast enough to run a full simulated day in 1–5 real minutes before any realism features are added on top. Everything else depends on this being solid.

### 1.1 Vectorised IDM (NumPy)

**What:** Rewrite `engine/idm.py` and the per-edge loop in `engine/network_simulation.py` to use NumPy array math instead of Python `for` loops. Expected ~30× speedup on large vehicle counts.

**New file:** `engine/idm_vec.py`
```python
# Vectorised signature:
# idm_acceleration_vec(v, v0, s, delta_v, s0, T, a, b) -> np.ndarray
# All args are 1-D float64 arrays of the same length N.
import numpy as np

def idm_acceleration_vec(v, v0, s, delta_v, s0, T, a, b, delta=4.0):
    s_star = s0 + np.maximum(0.0, v * T + v * delta_v / (2.0 * np.sqrt(a * b)))
    accel  = a * (1.0 - (v / np.maximum(v0, 1e-6)) ** delta - (s_star / np.maximum(s, 1e-3)) ** 2)
    return accel
```

**Changes to `network_simulation.py`:**
- Replace the per-vehicle `for i, veh in enumerate(evs_sorted)` IDM block with a single NumPy call over the edge's vehicle array.
- Build state arrays (`pos`, `spd`, `gap`, `delta_v`) once per edge; apply `idm_acceleration_vec`; write results back to vehicle objects.

**Changes to `network_simulation.py` — spawn queue:**
- Replace `self._spawn_queue: list` with `collections.deque` and `_spawn_queue.pop(0)` → `_spawn_queue.popleft()` (O(1) vs O(n)).

**Remove 300-signal cap:**
- In `run.py` find `osm_signals[:300]` and remove the slice — use all detected signals.

### 1.2 ThreadPoolExecutor (per-edge parallelism)

**What:** After vectorising IDM, divide edges across threads. NumPy releases the GIL during array ops, so threads truly run in parallel.

```python
# In network_simulation.step():
from concurrent.futures import ThreadPoolExecutor
_THREAD_POOL = ThreadPoolExecutor(max_workers=os.cpu_count())

def _process_edge(edge_id):
    evs = self._edge_vehicles[edge_id]
    if not evs: return
    # ... vectorised IDM for this edge ...

with _THREAD_POOL as pool:
    list(pool.map(_process_edge, self._edge_vehicles.keys()))
```

Lane changes and integration remain on the main thread (shared state).

### 1.3 Adaptive Timestep (`dt` scaling)

**What:** Allow `dt` to increase at high speed multipliers. IDM is stable to ~0.5 s. At ≥32× simulation speed raise `dt` to 0.5 s to halve the step count.

**API:** `POST /control {"speed_mult": 32}` → backend sets `sim.dt = 0.5 if mult >= 32 else 0.1`.

### 1.4 Simulation Speed Slider

**What:** Add a `speed_mult` control (1×, 2×, 4×, 8×, 16×, 32×, 64×, 128×, 288×) to the UI that POSTs to `/control`. At 288× a 24 h day runs in ≈5 minutes.

The server's sim thread tightens its sleep interval: `sleep(dt / speed_mult)`.

---

### ✅ Milestone 1 Tests

**File:** `tests/test_performance.py`

```python
"""Performance and correctness tests for vectorised IDM and threading."""
import time, math, numpy as np, pytest

# --- 1. Vectorised IDM correctness ---
from engine.idm import idm_acceleration          # scalar (existing)
from engine.idm_vec import idm_acceleration_vec  # new vectorised

def test_idm_vec_matches_scalar():
    """Vectorised result must match scalar for every element."""
    rng = np.random.default_rng(0)
    N = 1000
    v      = rng.uniform(0, 30, N)
    v0     = np.full(N, 30.0)
    s      = rng.uniform(2, 100, N)
    dv     = rng.uniform(-5, 5, N)
    s0, T, a, b = 2.0, 1.5, 1.4, 2.0
    vec = idm_acceleration_vec(v, v0, s, dv, s0, T, a, b)
    scalar = np.array([idm_acceleration(v[i], v0[i], s[i], dv[i], s0, T, a, b) for i in range(N)])
    np.testing.assert_allclose(vec, scalar, rtol=1e-5)

def test_idm_vec_zero_speed():
    """At v=0, acceleration should be positive (free-flow) when gap is large."""
    v  = np.array([0.0])
    v0 = np.array([30.0])
    s  = np.array([1000.0])
    dv = np.array([0.0])
    a  = idm_acceleration_vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    assert a[0] > 0.0

def test_idm_vec_stop_at_zero_gap():
    """Very small gap should produce large deceleration."""
    v  = np.array([10.0])
    v0 = np.array([30.0])
    s  = np.array([0.1])
    dv = np.array([0.0])
    a  = idm_acceleration_vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    assert a[0] < -1.0

# --- 2. Vectorised IDM throughput ---
def test_idm_vec_throughput():
    """1 million vehicles must complete in < 100 ms."""
    N = 1_000_000
    rng = np.random.default_rng(1)
    v  = rng.uniform(0, 30, N)
    v0 = np.full(N, 30.0)
    s  = rng.uniform(2, 200, N)
    dv = rng.uniform(-2, 2, N)
    t0 = time.perf_counter()
    idm_acceleration_vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"Vectorised IDM too slow: {elapsed:.3f}s for 1 M vehicles"

# --- 3. Spawn queue O(1) ---
from collections import deque
def test_spawn_queue_is_deque():
    """NetworkSimulation._spawn_queue must be a collections.deque."""
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    net = Network()
    net.add_node("A", 0, 0); net.add_node("B", 100, 0)
    net.add_edge("AB", "A", "B", length=100, speed_limit=14, num_lanes=1)
    sim = NetworkSimulation(net, demand={}, duration=10)
    assert isinstance(sim._spawn_queue, deque)

# --- 4. Step-time benchmark (threading) ---
def _make_loaded_sim(n_vehicles=200):
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    net = Network()
    nodes = [(str(i), i * 100, 0) for i in range(20)]
    for nid, x, y in nodes:
        net.add_node(nid, x, y)
    for i in range(19):
        net.add_edge(f"e{i}", str(i), str(i+1), length=100, speed_limit=14, num_lanes=2)
    demand = {"0": {"19": float(n_vehicles * 3600 / 300)}}
    sim = NetworkSimulation(net, demand=demand, duration=300, seed=0)
    # warm up
    for _ in range(50):
        sim.step()
    return sim

def test_step_time_200_vehicles():
    """200 vehicles: one step must complete in < 5 ms."""
    sim = _make_loaded_sim(200)
    t0 = time.perf_counter()
    for _ in range(100):
        sim.step()
    avg = (time.perf_counter() - t0) / 100
    assert avg < 0.005, f"Step too slow: {avg*1000:.2f} ms"

# --- 5. Signal cap removed ---
def test_no_300_signal_cap(monkeypatch, tmp_path):
    """Parser must NOT slice signals to 300; all signals in OSM data are used."""
    import json, importlib
    # Build a fake OSM response with 400 signal nodes
    nodes = [{"type":"node","id":i,"lat":48.8+i*0.0001,"lon":2.3,
              "tags":{"highway":"traffic_signals"}} for i in range(400)]
    osm_data = {"elements": nodes}
    fake_file = tmp_path / "fake.json"
    fake_file.write_text(json.dumps(osm_data))
    # Import parser and check
    from importer.parser import parse_osm
    result = parse_osm(osm_data, bbox=(48.8, 2.3, 48.84, 2.34))
    assert len(result.get("signals", [])) > 300, "Signal cap still active!"

# --- 6. Adaptive dt ---
def test_adaptive_dt():
    """At speed_mult >= 32 the sim dt should be raised to 0.5."""
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    net = Network()
    net.add_node("A", 0, 0); net.add_node("B", 500, 0)
    net.add_edge("AB", "A", "B", length=500, speed_limit=14, num_lanes=1)
    sim = NetworkSimulation(net, demand={}, duration=10)
    sim.set_speed_mult(32)
    assert sim.dt == pytest.approx(0.5, abs=0.01)
    sim.set_speed_mult(1)
    assert sim.dt == pytest.approx(0.1, abs=0.01)
```

**File:** `tests/test_speed_compression.py`

```python
"""24-hour simulation completes within wall-clock time budget."""
import time, pytest

def test_24h_in_under_5_minutes():
    """
    A 24-hour simulation (86400 s) with moderate demand must finish
    in less than 5 minutes of wall time at max speed.
    Skip on slow CI machines.
    """
    pytest.importorskip("numpy")
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation

    net = Network()
    for i in range(10):
        net.add_node(str(i), i * 200, 0)
    for i in range(9):
        net.add_edge(f"e{i}", str(i), str(i+1), length=200, speed_limit=14, num_lanes=2)
    demand = {"0": {"9": 300.0}}  # 300 veh/hr
    sim = NetworkSimulation(net, demand=demand, duration=86400, seed=0)
    sim.set_speed_mult(288)

    t0 = time.perf_counter()
    sim.run()
    elapsed = time.perf_counter() - t0
    assert elapsed < 300, f"24-hour sim took {elapsed:.1f}s — too slow"
```

---

## Milestone 2 — Interactive Dashboard

**Goal:** A collapsible side-panel in the browser with sliders, tooltips, and presets that live-update the running simulation via `POST /control`. All settings persist via `localStorage`.

### 2.1 Extend `POST /control` API

**What:** `run.py` currently handles only `{"action": "..."}`. Extend it to also accept a flat `params` dict:

```json
POST /control
{
  "params": {
    "idm_a_max": 1.8,
    "idm_b": 2.5,
    "idm_T": 1.2,
    "idm_s0": 2.0,
    "mobil_politeness": 0.3,
    "mobil_b_safe": 3.0,
    "signal_cycle": 90,
    "signal_green_ratio": 0.55,
    "demand_scale": 1.5,
    "weather_v0_mult": 0.9,
    "weather_s0_mult": 1.1,
    "speed_mult": 8
  }
}
```

Python backend stores these in a thread-safe `SimConfig` dataclass; `NetworkSimulation.step()` reads from it each tick.

### 2.2 `SimConfig` dataclass

**New file:** `engine/config.py`

```python
import threading
from dataclasses import dataclass, field

@dataclass
class SimConfig:
    # IDM
    idm_a_max: float = 1.4
    idm_b: float = 2.0
    idm_T: float = 1.5
    idm_s0: float = 2.0
    # MOBIL
    mobil_politeness: float = 0.3
    mobil_b_safe: float = 3.0
    # Signals
    signal_cycle: float = 90.0
    signal_green_ratio: float = 0.5
    # Demand
    demand_scale: float = 1.0
    # Weather
    weather_v0_mult: float = 1.0
    weather_s0_mult: float = 1.0
    weather_T_mult: float = 1.0
    # Performance
    speed_mult: float = 1.0
    use_gpu: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, type(getattr(self, k))(v))
```

### 2.3 Dashboard HTML/JS

**Modify:** `viz/index.html` — add a `<div id="dashboard">` panel.

**Features:**
- Collapsible `<details>` sections: **Driving**, **Signals**, **Demand**, **Weather**, **Performance**
- Each parameter: `<label title="[tooltip text]"><span>Label</span><input type="range" ...><output></output></label>`
- Tooltips via native `title` attribute + custom CSS hover overlay for longer descriptions
- `oninput` handler debounced to 150 ms → `fetch('/control', {method:'POST', body: JSON.stringify({params:{...}})})`
- **Presets** toolbar: `Rush Hour`, `Aggressive Drivers`, `Rainy Day`, `Night (low demand)`, `Reset`
- On page load: restore from `localStorage`; push to backend immediately

**Preset definitions (hardcoded in JS):**
```javascript
const PRESETS = {
  "Rush Hour":        { idm_T: 1.0, idm_s0: 1.5, demand_scale: 2.0, signal_green_ratio: 0.6 },
  "Aggressive":       { idm_a_max: 3.0, idm_T: 0.8, mobil_politeness: 0.0 },
  "Rainy Day":        { weather_v0_mult: 0.9, weather_s0_mult: 1.2, weather_T_mult: 1.15 },
  "Night":            { demand_scale: 0.15, idm_T: 1.8 },
  "Reset":            { /* all defaults */ },
};
```

### 2.4 Tooltip definitions

All parameters include a plain-English description shown on hover:

| Parameter | Tooltip |
|-----------|---------|
| `a_max` | Maximum acceleration (m/s²). Higher = snappier vehicles. |
| `b` | Comfortable deceleration (m/s²). Higher = harder braking. |
| `T` | Desired time headway (s). Gap drivers try to maintain. |
| `s0` | Minimum bumper-to-bumper gap (m) at standstill. |
| `politeness` | MOBIL politeness (0=selfish, 1=altruistic). |
| `b_safe` | Safety deceleration threshold for lane change (m/s²). |
| `signal_cycle` | Total signal cycle length (s). |
| `green_ratio` | Fraction of cycle that is green (0–1). |
| `demand_scale` | Demand multiplier (1.0 = baseline, 2.0 = twice as many vehicles). |
| `weather_v0_mult` | Speed limit multiplier in weather (0.9 = 10% slower). |
| `speed_mult` | Simulation speed. 288× = 24 h in ~5 min. |

---

### ✅ Milestone 2 Tests

**File:** `tests/test_dashboard_api.py`

```python
"""Tests for the extended /control API and SimConfig."""
import json, threading, pytest

# --- SimConfig unit tests ---
from engine.config import SimConfig

def test_simconfig_defaults():
    cfg = SimConfig()
    assert cfg.idm_a_max == pytest.approx(1.4)
    assert cfg.idm_T     == pytest.approx(1.5)
    assert cfg.demand_scale == pytest.approx(1.0)

def test_simconfig_update_valid():
    cfg = SimConfig()
    cfg.update(idm_a_max=3.0, idm_T=0.8)
    assert cfg.idm_a_max == pytest.approx(3.0)
    assert cfg.idm_T     == pytest.approx(0.8)

def test_simconfig_ignores_unknown_keys():
    cfg = SimConfig()
    cfg.update(nonexistent_param=999)   # must not raise
    assert not hasattr(cfg, "nonexistent_param")

def test_simconfig_type_coercion():
    cfg = SimConfig()
    cfg.update(idm_a_max="2.5")         # string → float
    assert isinstance(cfg.idm_a_max, float)
    assert cfg.idm_a_max == pytest.approx(2.5)

def test_simconfig_thread_safety():
    cfg = SimConfig()
    errors = []
    def writer():
        for _ in range(1000):
            try: cfg.update(idm_a_max=1.4)
            except Exception as e: errors.append(e)
    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors

# --- /control API integration tests ---
import urllib.request, urllib.error

BASE = "http://127.0.0.1:9999"   # started by conftest.py fixture

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Spin up the run.py HTTP server for API tests."""
    import subprocess, time, sys, os
    proc = subprocess.Popen(
        [sys.executable, "run.py", "--port", "9999", "--headless"],
        cwd="/Users/sekun/Traffic sim",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    yield proc
    proc.terminate()

def _post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(f"{BASE}{path}", data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def test_control_params_idm(server):
    resp = _post("/control", {"params": {"idm_a_max": 2.5}})
    assert resp.get("ok") is True

def test_control_params_demand_scale(server):
    resp = _post("/control", {"params": {"demand_scale": 1.8}})
    assert resp.get("ok") is True

def test_control_params_weather(server):
    resp = _post("/control", {"params": {"weather_v0_mult": 0.85, "weather_s0_mult": 1.2}})
    assert resp.get("ok") is True

def test_control_unknown_params_ignored(server):
    """Unknown param keys must not crash the server."""
    resp = _post("/control", {"params": {"not_a_real_param": 99}})
    assert resp.get("ok") is True

def test_control_speed_mult(server):
    resp = _post("/control", {"params": {"speed_mult": 32}})
    assert resp.get("ok") is True

# --- Config applied in simulation ---
def test_config_affects_acceleration():
    """Higher a_max in SimConfig → vehicles reach target speed faster."""
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    from engine.config import SimConfig

    def _run(a_max):
        net = Network()
        net.add_node("A", 0, 0); net.add_node("B", 1000, 0)
        net.add_edge("AB", "A", "B", length=1000, speed_limit=14, num_lanes=1)
        cfg = SimConfig(idm_a_max=a_max)
        sim = NetworkSimulation(net, demand={"A": {"B": 120.0}}, duration=60, seed=0, config=cfg)
        for _ in range(200):
            sim.step()
        speeds = [v.speed for v in sim.vehicles] if sim.vehicles else [0.0]
        return sum(speeds) / len(speeds)

    avg_slow = _run(0.5)
    avg_fast = _run(3.0)
    assert avg_fast > avg_slow, "Higher a_max should yield higher avg speed"

# --- Preset smoke test ---
def test_presets_valid_keys():
    """All preset parameter keys must exist in SimConfig."""
    from engine.config import SimConfig
    PRESETS = {
        "Rush Hour":  {"idm_T": 1.0, "idm_s0": 1.5, "demand_scale": 2.0},
        "Aggressive": {"idm_a_max": 3.0, "idm_T": 0.8, "mobil_politeness": 0.0},
        "Rainy Day":  {"weather_v0_mult": 0.9, "weather_s0_mult": 1.2},
        "Night":      {"demand_scale": 0.15},
    }
    cfg = SimConfig()
    for name, params in PRESETS.items():
        for key in params:
            assert hasattr(cfg, key), f"Preset '{name}' uses unknown key '{key}'"
```

---

## Milestone 3 — Realistic Spatial Demand

**Goal:** Replace the uniform random OD demand with population-weighted origins (WorldPop) and POI-classified destinations (trip purposes from OSM tags).

### 3.1 WorldPop Raster Integration

**New file:** `engine/worldpop.py`

**What:**
1. Accept a city bounding box and fetch the WorldPop 100 m GeoTIFF for the country (via WOPR REST API or cached local file).
2. Clip the raster to the bbox using `rasterio` windowed read.
3. Build a `scipy.spatial.KDTree` over road network nodes.
4. For each raster cell, snap its centroid to the nearest node; accumulate population as `node_weight[node_id] += population`.
5. Cache the resulting `{node_id: weight}` dict as `{city_slug}_worldpop.pkl` in `.overpass_cache/`.

**API:**
```python
node_weights = load_worldpop_weights(bbox, network, cache_dir=".overpass_cache")
```

### 3.2 Trip Purpose Classification

**New file:** `engine/poi_demand.py`

**What:**
1. Extend `importer/overpass.py` to additionally query POI tags: `amenity`, `office`, `shop`, `building`.
2. Classify each POI node/way into `home | work | school | retail | other` via `PURPOSE_MAP`.
3. Build `{purpose: [node_id, ...]}` lookup.
4. When generating OD pairs: choose origin from `node_weights`, destination from the purpose bucket matching the trip type (sampled from configurable weights: default 40% work, 15% school, 25% retail, 20% other).

**Priority map (applied in order, first match wins):**
```python
PURPOSE_MAP = [
    (("building", "apartments"),  "home"),
    (("building", "residential"), "home"),
    (("building", "house"),       "home"),
    (("building", "office"),      "work"),
    (("building", "commercial"),  "work"),
    (("building", "industrial"),  "work"),
    (("office",   "*"),           "work"),
    (("amenity",  "school"),      "school"),
    (("amenity",  "university"),  "school"),
    (("amenity",  "college"),     "school"),
    (("shop",     "*"),           "retail"),
    (("amenity",  "restaurant"),  "retail"),
    (("amenity",  "cafe"),        "retail"),
]
```

### 3.3 Building Floor Estimation

**New file:** `engine/buildings.py`

**What:** Given an OSM building polygon:
1. Extract `building:levels` → floors.
2. Else `height / 3.0` → floors.
3. Else footprint area heuristic: `floors = max(1, round(area / 200))` (offices) or `max(1, round(area / 100))` (residential).
4. Compute `capacity = floors × footprint_area / sqm_per_person` where `sqm_per_person` is 10 m² (office) or 30 m² (residential).
5. Expose `{node_id: capacity}` for use as destination weights.

---

### ✅ Milestone 3 Tests

**File:** `tests/test_worldpop.py`

```python
"""WorldPop raster loading, snapping, and caching."""
import numpy as np, pytest

@pytest.fixture
def synthetic_raster(tmp_path):
    """Create a tiny synthetic GeoTIFF: 10×10 grid, Paris bbox."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    import numpy as np
    data = np.random.randint(10, 500, (1, 10, 10), dtype=np.uint16)
    transform = from_bounds(2.25, 48.82, 2.42, 48.90, 10, 10)
    path = tmp_path / "pop.tif"
    with rasterio.open(path, 'w', driver='GTiff', height=10, width=10,
                       count=1, dtype='uint16', crs='EPSG:4326',
                       transform=transform) as dst:
        dst.write(data)
    return str(path), data

def _tiny_network():
    from engine.network import Network
    net = Network()
    for i, (lat, lon) in enumerate([(48.83, 2.29), (48.85, 2.34), (48.88, 2.40)]):
        net.add_node(str(i), lon, lat)   # x=lon, y=lat for this test
    net.add_edge("01", "0", "1", length=5000, speed_limit=14, num_lanes=1)
    net.add_edge("12", "1", "2", length=5000, speed_limit=14, num_lanes=1)
    return net

def test_worldpop_weights_sum_approx_raster_total(synthetic_raster, tmp_path):
    """Total node weights must approximate total raster population."""
    from engine.worldpop import load_worldpop_weights
    path, data = synthetic_raster
    net = _tiny_network()
    bbox = (48.82, 2.25, 48.90, 2.42)
    weights = load_worldpop_weights(bbox, net, raster_path=path, cache_dir=str(tmp_path))
    raster_total = int(data.sum())
    weight_total = int(sum(weights.values()))
    # Should be within 5% (some cells outside bbox may be excluded)
    assert abs(weight_total - raster_total) / max(raster_total, 1) < 0.05

def test_worldpop_no_negative_weights(synthetic_raster, tmp_path):
    from engine.worldpop import load_worldpop_weights
    path, _ = synthetic_raster
    net = _tiny_network()
    weights = load_worldpop_weights((48.82, 2.25, 48.90, 2.42), net,
                                    raster_path=path, cache_dir=str(tmp_path))
    assert all(v >= 0 for v in weights.values())

def test_worldpop_all_nodes_in_result(synthetic_raster, tmp_path):
    from engine.worldpop import load_worldpop_weights
    path, _ = synthetic_raster
    net = _tiny_network()
    weights = load_worldpop_weights((48.82, 2.25, 48.90, 2.42), net,
                                    raster_path=path, cache_dir=str(tmp_path))
    for nid in net.nodes:
        assert nid in weights

def test_worldpop_cache_hit(synthetic_raster, tmp_path):
    """Second call must load from cache, not recompute."""
    import time
    from engine.worldpop import load_worldpop_weights
    path, _ = synthetic_raster
    net = _tiny_network()
    kw = dict(raster_path=path, cache_dir=str(tmp_path))
    load_worldpop_weights((48.82, 2.25, 48.90, 2.42), net, **kw)  # warm cache
    t0 = time.perf_counter()
    load_worldpop_weights((48.82, 2.25, 48.90, 2.42), net, **kw)  # from cache
    assert time.perf_counter() - t0 < 0.2   # must be near-instant

def test_worldpop_denser_area_higher_weight(synthetic_raster, tmp_path):
    """Node closest to the densest raster cell gets the highest weight."""
    import rasterio, numpy as np
    from rasterio.transform import from_bounds
    from engine.worldpop import load_worldpop_weights
    path, _ = synthetic_raster
    # Override raster with a single high-density cell at top-left
    data = np.zeros((1, 10, 10), dtype=np.uint16)
    data[0, 0, 0] = 9999
    with rasterio.open(path, 'r+') as dst:
        dst.write(data)
    net = _tiny_network()
    weights = load_worldpop_weights((48.82, 2.25, 48.90, 2.42), net,
                                    raster_path=path, cache_dir=str(tmp_path))
    # Node 0 is nearest to (48.83, 2.29) which is closest to top-left
    assert weights.get("0", 0) > weights.get("2", 0)
```

**File:** `tests/test_poi_demand.py`

```python
"""Trip purpose classification from OSM tags."""
import pytest
from engine.poi_demand import classify_purpose, PURPOSE_MAP

@pytest.mark.parametrize("tags,expected", [
    ({"building": "apartments"},   "home"),
    ({"building": "residential"},  "home"),
    ({"building": "office"},       "work"),
    ({"office": "yes"},            "work"),
    ({"office": "government"},     "work"),
    ({"amenity": "school"},        "school"),
    ({"amenity": "university"},    "school"),
    ({"shop": "supermarket"},      "retail"),
    ({"amenity": "restaurant"},    "retail"),
    ({"natural": "tree"},          "other"),
    ({},                           "other"),
])
def test_classify_purpose(tags, expected):
    assert classify_purpose(tags) == expected

def test_wildcard_office_matches_any_value():
    assert classify_purpose({"office": "lawyer"})    == "work"
    assert classify_purpose({"office": "ngo"})       == "work"

def test_building_priority_over_shop():
    """building=office should win over shop=* (higher in priority list)."""
    result = classify_purpose({"building": "office", "shop": "supermarket"})
    assert result == "work"

def test_purpose_weights_sum_to_1():
    from engine.poi_demand import DEFAULT_PURPOSE_WEIGHTS
    assert abs(sum(DEFAULT_PURPOSE_WEIGHTS.values()) - 1.0) < 1e-6

def test_poi_demand_generator_respects_weights():
    """With 100% work weight, all generated trips must target work nodes."""
    from engine.network import Network
    from engine.poi_demand import POIDemandGenerator
    net = Network()
    for i in range(4):
        net.add_node(str(i), i * 100.0, 0.0)
    for i in range(3):
        net.add_edge(f"e{i}", str(i), str(i+1), length=100, speed_limit=14, num_lanes=1)
    poi_nodes = {"work": ["1", "2", "3"], "home": ["0"]}
    gen = POIDemandGenerator(net, poi_nodes,
                              purpose_weights={"work": 1.0, "home": 0.0,
                                               "school": 0.0, "retail": 0.0, "other": 0.0})
    trips = gen.generate(n=200, seed=0)
    destinations = {t[1] for t in trips}
    assert destinations.issubset({"1", "2", "3"})

def test_buildings_floor_estimation():
    from engine.buildings import estimate_floors, estimate_capacity
    assert estimate_floors({"building:levels": "10"}, footprint_area=500) == 10
    assert estimate_floors({"height": "30"}, footprint_area=500) == pytest.approx(10, abs=1)
    assert estimate_floors({}, footprint_area=2000) >= 1       # heuristic

    # Office: 10 m² per worker
    cap = estimate_capacity({"building": "office"}, floors=5, footprint_area=300)
    assert cap > 0

def test_buildings_no_negative_capacity():
    from engine.buildings import estimate_capacity
    cap = estimate_capacity({"building": "apartments"}, floors=1, footprint_area=50)
    assert cap >= 0
```

---

## Milestone 4 — Temporal Demand & Mode Split

**Goal:** Demand follows a realistic 24-hour weekday/weekend profile, and each generated trip is assigned a mode (car / walk / bike) via a distance-based logistic split.

### 4.1 Hourly Demand Profile

**New file:** `engine/demand_profile.py`

**What:** A 24-value lookup table (normalised). Weekday bimodal (peaks 8 AM, 5 PM). Weekend unimodal (peak 11 AM).

```python
WEEKDAY_PROFILE = [
    0.05, 0.02, 0.01, 0.01, 0.02, 0.04,  # 00–05
    0.08, 0.18, 1.00, 0.70, 0.50, 0.45,  # 06–11
    0.60, 0.55, 0.50, 0.55, 0.70, 0.90,  # 12–17
    0.65, 0.45, 0.30, 0.20, 0.12, 0.07,  # 18–23
]
WEEKEND_PROFILE = [
    0.04, 0.02, 0.01, 0.01, 0.02, 0.03,  # 00–05
    0.05, 0.08, 0.12, 0.20, 0.40, 0.70,  # 06–11
    0.80, 0.75, 0.65, 0.60, 0.55, 0.50,  # 12–17
    0.40, 0.30, 0.20, 0.12, 0.07, 0.05,  # 18–23
]
```

`NetworkSimulation._build_spawn_queue()` multiplies each spawn inter-arrival time by the inverse profile factor at that simulated hour.

### 4.2 Mode Split

**New file:** `engine/mode_split.py`

**What:** Logistic functions of trip distance (metres):
```
P_walk(d) = 1 / (1 + exp(0.004 × (d − 2000)))
P_bike(d) = (1 − P_walk(d)) / (1 + exp(0.0008 × (d − 7000)))
P_car(d)  = 1 − P_walk(d) − P_bike(d)
```

Returns a `Mode` enum: `CAR | WALK | BIKE`. Walk/Bike trips use the pedestrian SFM agent; Bike uses a faster SFM with higher desired speed (~5 m/s).

---

### ✅ Milestone 4 Tests

**File:** `tests/test_demand_profile.py`

```python
"""Hourly demand profile tests."""
import pytest
import numpy as np
from engine.demand_profile import (
    WEEKDAY_PROFILE, WEEKEND_PROFILE, profile_multiplier,
    get_profile, DayType
)

def test_weekday_profile_length():
    assert len(WEEKDAY_PROFILE) == 24

def test_weekend_profile_length():
    assert len(WEEKEND_PROFILE) == 24

def test_weekday_morning_peak_is_max():
    """8 AM should be the highest factor on a weekday."""
    peak_hour = WEEKDAY_PROFILE.index(max(WEEKDAY_PROFILE))
    assert peak_hour == 8

def test_weekend_midday_peak():
    """Weekend peak should fall between 10–13."""
    peak_hour = WEEKEND_PROFILE.index(max(WEEKEND_PROFILE))
    assert 10 <= peak_hour <= 13

def test_overnight_low():
    """2–4 AM must have low demand (< 5% of peak)."""
    peak = max(WEEKDAY_PROFILE)
    for h in [2, 3, 4]:
        assert WEEKDAY_PROFILE[h] < 0.05 * peak

def test_profile_multiplier_returns_float():
    v = profile_multiplier(sim_time_s=8 * 3600, day_type=DayType.WEEKDAY)
    assert isinstance(v, float) and v > 0

def test_profile_multiplier_evening_peak():
    am = profile_multiplier(8 * 3600, DayType.WEEKDAY)
    pm = profile_multiplier(17 * 3600, DayType.WEEKDAY)
    mid = profile_multiplier(14 * 3600, DayType.WEEKDAY)
    assert am > mid
    assert pm > mid

def test_get_profile_weekday():
    assert get_profile(DayType.WEEKDAY) is WEEKDAY_PROFILE

def test_get_profile_weekend():
    assert get_profile(DayType.WEEKEND) is WEEKEND_PROFILE

def test_spawn_count_peaks_at_rush_hour():
    """Simulation spawning more vehicles at 8 AM than 3 AM on a weekday."""
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    net = Network()
    net.add_node("A", 0, 0); net.add_node("B", 5000, 0)
    net.add_edge("AB", "A", "B", length=5000, speed_limit=14, num_lanes=2)
    # Simulate a full 24 h with demand profile active
    demand = {"A": {"B": 200.0}}
    sim = NetworkSimulation(net, demand=demand, duration=86400, seed=1,
                             day_type="weekday")
    sim.run()
    # Check: more trip log entries originate during 7–9 AM than 2–4 AM
    am_peak = sum(1 for r in sim.trip_log if 7*3600 <= r.entry_time <= 9*3600)
    night   = sum(1 for r in sim.trip_log if 2*3600 <= r.entry_time <= 4*3600)
    assert am_peak > night * 3
```

**File:** `tests/test_mode_split.py`

```python
"""Mode split logistic function tests."""
import pytest
from engine.mode_split import mode_split_probs, sample_mode, Mode

def test_probs_sum_to_one():
    for d in [500, 2000, 5000, 15000]:
        p = mode_split_probs(d)
        assert abs(sum(p.values()) - 1.0) < 1e-6

def test_short_trip_mostly_walk():
    p = mode_split_probs(500)    # 500 m
    assert p[Mode.WALK] > 0.7

def test_long_trip_mostly_car():
    p = mode_split_probs(20000)  # 20 km
    assert p[Mode.CAR] > 0.9

def test_medium_trip_mixed():
    p = mode_split_probs(4000)   # 4 km — expect walk < car
    assert p[Mode.CAR] > p[Mode.WALK]

def test_zero_distance():
    """Zero distance should not raise and should return walk."""
    p = mode_split_probs(0)
    assert p[Mode.WALK] == pytest.approx(1.0, abs=0.1)

def test_sample_mode_is_valid():
    mode = sample_mode(3000, seed=42)
    assert mode in (Mode.CAR, Mode.WALK, Mode.BIKE)

def test_sample_mode_distribution():
    """100 samples at 500 m should produce ≥70 walk modes."""
    walks = sum(1 for i in range(100) if sample_mode(500, seed=i) == Mode.WALK)
    assert walks >= 70

def test_mode_split_car_only_config():
    """A custom config with 100% car should always return CAR."""
    from engine.mode_split import ModeSplitConfig
    cfg = ModeSplitConfig(force_car=True)
    for d in [100, 1000, 10000]:
        assert sample_mode(d, config=cfg) == Mode.CAR
```

---

## Milestone 5 — Weather Effects

**Goal:** Fetch live (or manual) weather conditions and automatically adjust IDM parameters, with user override via the dashboard.

### 5.1 Weather Fetch

**New file:** `engine/weather.py`

**What:**
- `fetch_weather(lat, lon) -> WeatherState` — hits `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,weathercode&hourly=precipitation`
- Returns `WeatherState(condition: "clear"|"rain"|"heavy_rain"|"snow", precip_mm_hr: float)`
- Cached for 15 minutes; falls back gracefully on network error.

### 5.2 IDM Multipliers

```python
WEATHER_MULTIPLIERS = {
    "clear":      {"v0": 1.00, "s0": 1.00, "T": 1.00, "a": 1.00},
    "rain":       {"v0": 0.90, "s0": 1.15, "T": 1.15, "a": 0.95},
    "heavy_rain": {"v0": 0.80, "s0": 1.30, "T": 1.30, "a": 0.85},
    "snow":       {"v0": 0.55, "s0": 1.50, "T": 1.50, "a": 0.70},
}
```

Applied inside `NetworkSimulation.step()` to the effective IDM params before each call.

---

### ✅ Milestone 5 Tests

**File:** `tests/test_weather.py`

```python
"""Weather fetch, parsing, and IDM multiplier tests."""
import pytest

# --- Multiplier unit tests (no network needed) ---
from engine.weather import apply_weather_multipliers, WeatherState, WEATHER_MULTIPLIERS

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

# --- Simulation speed reduction ---
def test_rain_slows_average_speed():
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    from engine.weather import WeatherState

    def _avg_speed(condition):
        net = Network()
        net.add_node("A", 0, 0); net.add_node("B", 2000, 0)
        net.add_edge("AB", "A", "B", length=2000, speed_limit=14, num_lanes=1)
        sim = NetworkSimulation(net, demand={"A": {"B": 600.0}}, duration=120,
                                 seed=0, weather=WeatherState(condition, 1.0))
        for _ in range(500):
            sim.step()
        speeds = [v.speed for v in sim.vehicles] if sim.vehicles else [0.0]
        return sum(speeds) / len(speeds)

    clear_speed = _avg_speed("clear")
    rain_speed  = _avg_speed("rain")
    assert rain_speed < clear_speed * 0.97   # at least 3% slower

# --- API fetch (mocked) ---
def test_fetch_weather_network_error_fallback(monkeypatch):
    """On network error, should return clear weather, not raise."""
    import urllib.request
    def _fail(*a, **kw): raise OSError("No network")
    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    from engine.weather import fetch_weather
    w = fetch_weather(48.8566, 2.3522)
    assert w.condition == "clear"    # safe fallback
```

---

## Milestone 6 — GPU Acceleration

**Goal:** Detect hardware (NVIDIA CUDA / Apple Metal / CPU-only) at startup and dispatch IDM computation to the best available backend. Expose a "Performance Mode" selector in the dashboard.

### 6.1 Hardware Detection

**New file:** `engine/compute_backend.py`

```python
# Returns: "cuda" | "metal" | "numpy"
def detect_backend() -> str: ...

# Returns wrapped functions with consistent API:
# idm_batch(v, v0, s, dv, s0, T, a, b) -> array-like
def get_idm_backend(prefer: str = "auto"): ...
```

### 6.2 CuPy Backend (NVIDIA)

```python
import cupy as cp
def idm_cuda(v, v0, s, dv, s0, T, a, b):
    v, v0, s, dv = [cp.asarray(x) for x in (v, v0, s, dv)]
    s_star = s0 + cp.maximum(0, v*T + v*dv / (2*cp.sqrt(a*b)))
    return a * (1 - (v/cp.maximum(v0,1e-6))**4 - (s_star/cp.maximum(s,1e-3))**2)
```

### 6.3 JAX/Metal Backend (Apple Silicon)

```python
import jax.numpy as jnp
@jax.jit
def idm_metal(v, v0, s, dv, s0, T, a, b):
    s_star = s0 + jnp.maximum(0, v*T + v*dv/(2*jnp.sqrt(a*b)))
    return a * (1 - (v/jnp.maximum(v0,1e-6))**4 - (s_star/jnp.maximum(s,1e-3))**2)
```

### 6.4 Power Allocation UI

Dashboard **Performance** panel:
- 🔋 `Battery Saver` — `speed_mult=1`, threads=1, `dt=0.1`
- ⚡ `Balanced` — `speed_mult=8`, threads=`cpu_count//2`
- 🚀 `Max Performance` — `speed_mult=288`, threads=`cpu_count`, GPU if available
- `Custom` — expose individual sliders

---

### ✅ Milestone 6 Tests

**File:** `tests/test_compute_backend.py`

```python
"""GPU / CPU compute backend detection and correctness."""
import numpy as np, pytest
from engine.compute_backend import detect_backend, get_idm_backend

def test_detect_backend_returns_valid_string():
    backend = detect_backend()
    assert backend in ("cuda", "metal", "numpy")

def test_numpy_backend_always_available():
    backend = get_idm_backend(prefer="numpy")
    assert backend is not None

def test_idm_backend_output_matches_numpy():
    """Any backend must produce the same result as the NumPy reference."""
    rng = np.random.default_rng(77)
    N = 500
    v   = rng.uniform(0, 25, N).astype(np.float32)
    v0  = np.full(N, 30.0, np.float32)
    s   = rng.uniform(3, 80, N).astype(np.float32)
    dv  = rng.uniform(-3, 3, N).astype(np.float32)
    ref_fn  = get_idm_backend(prefer="numpy")
    test_fn = get_idm_backend(prefer="auto")
    ref     = np.array(ref_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0))
    result  = np.array(test_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0))
    np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-4)

def test_idm_backend_benchmark_numpy():
    """NumPy backend: 100 k vehicles in < 50 ms."""
    import time
    N = 100_000
    rng = np.random.default_rng(0)
    v  = rng.uniform(0, 25, N)
    v0 = np.full(N, 30.0)
    s  = rng.uniform(3, 80, N)
    dv = np.zeros(N)
    fn = get_idm_backend(prefer="numpy")
    t0 = time.perf_counter()
    for _ in range(10):
        fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    avg = (time.perf_counter() - t0) / 10
    assert avg < 0.05, f"NumPy IDM 100k: {avg*1000:.1f} ms — too slow"

@pytest.mark.skipif(
    detect_backend() not in ("cuda", "metal"),
    reason="GPU not available"
)
def test_gpu_backend_benchmark():
    """GPU backend must be at least 5× faster than NumPy on 100 k vehicles."""
    import time
    N = 100_000
    rng = np.random.default_rng(0)
    v  = rng.uniform(0, 25, N)
    v0 = np.full(N, 30.0)
    s  = rng.uniform(3, 80, N)
    dv = np.zeros(N)
    numpy_fn = get_idm_backend(prefer="numpy")
    gpu_fn   = get_idm_backend(prefer="auto")
    # Warmup
    gpu_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    t0 = time.perf_counter()
    for _ in range(50): numpy_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    numpy_t = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(50): gpu_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    gpu_t = (time.perf_counter() - t0) / 50
    assert numpy_t / gpu_t >= 5, f"GPU speedup only {numpy_t/gpu_t:.1f}×"

def test_graceful_fallback_to_numpy(monkeypatch):
    """If CuPy / JAX fail to import, backend must silently use NumPy."""
    import sys
    # Simulate no cupy, no jax
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "jax",  None)
    import importlib, engine.compute_backend as cb
    importlib.reload(cb)
    backend = cb.detect_backend()
    assert backend == "numpy"

def test_simulation_deterministic_across_backends():
    """CPU and GPU steps must produce same vehicle count after 100 steps."""
    from engine.network import Network
    from engine.network_simulation import NetworkSimulation
    from engine.compute_backend import get_idm_backend

    def _run(backend_name):
        net = Network()
        net.add_node("A", 0, 0); net.add_node("B", 1000, 0)
        net.add_edge("AB", "A", "B", length=1000, speed_limit=14, num_lanes=1)
        sim = NetworkSimulation(net, demand={"A": {"B": 300.0}}, duration=60,
                                 seed=0, compute_backend=backend_name)
        for _ in range(100): sim.step()
        return len(sim.vehicles)

    cpu_count = _run("numpy")
    auto_count = _run("auto")
    assert cpu_count == auto_count
```

---

## Milestone 7 — WebGL Renderer

**Goal:** Replace the Canvas2D vehicle dot loop with a WebGL instanced draw call, targeting ≥60 FPS with 10 k+ simultaneous vehicles.

### 7.1 Architecture

- Keep the tile/road/signal layers on Canvas2D (unchanged).
- Add a second `<canvas id="vehicle-layer">` positioned on top, managed by a WebGL context.
- Each frame: upload vehicle positions + colors to a GPU buffer via `gl.bufferSubData`, then `gl.drawArraysInstanced`.
- Use an instanced point sprite shader (vertex: position + size; fragment: circle with edge antialiasing).

### 7.2 Shader

```glsl
// Vertex shader
attribute vec2 a_position;   // per-instance: [world_x, world_y]
attribute float a_speed;     // for colour mapping
uniform mat3 u_transform;    // world → clip (pan + zoom)
uniform float u_point_size;
varying float v_speed;
void main() {
    vec3 p = u_transform * vec3(a_position, 1.0);
    gl_Position  = vec4(p.xy, 0, 1);
    gl_PointSize = u_point_size;
    v_speed = a_speed;
}

// Fragment shader
varying float v_speed;
void main() {
    vec2 coord = gl_PointCoord - 0.5;
    if (dot(coord, coord) > 0.25) discard;  // circle clip
    float t = clamp(v_speed / 14.0, 0.0, 1.0);
    gl_FragColor = vec4(mix(vec3(1,0.2,0.2), vec3(0.2,1,0.4), t), 1.0);
}
```

---

### ✅ Milestone 7 Tests

**File:** `tests/test_renderer.py` *(extend existing)*

```python
"""WebGL renderer tests (run in headless Chromium via pytest-playwright)."""
import pytest

playwright = pytest.importorskip("playwright.sync_api")

@pytest.fixture(scope="module")
def page(playwright):
    browser = playwright.chromium.launch()
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.goto("http://127.0.0.1:8888")
    yield pg
    browser.close()

def test_vehicle_canvas_exists(page):
    canvas = page.query_selector("#vehicle-layer")
    assert canvas is not None

def test_webgl_context_created(page):
    result = page.evaluate("""() => {
        const c = document.getElementById('vehicle-layer');
        return c && (c.getContext('webgl') !== null || c.getContext('webgl2') !== null);
    }""")
    assert result is True

def test_fps_above_30_with_1000_vehicles(page):
    """Render loop should maintain ≥30 FPS with 1000 mock vehicles."""
    fps = page.evaluate("""() => new Promise(resolve => {
        // inject 1000 fake vehicle positions
        window._testVehicles = Array.from({length:1000}, (_,i) => ({
            x: Math.random()*1000, y: Math.random()*1000, speed: Math.random()*14
        }));
        let frames = 0, t0 = performance.now();
        function count() {
            frames++;
            if (frames < 60) requestAnimationFrame(count);
            else resolve(60000 / (performance.now() - t0));
        }
        requestAnimationFrame(count);
    })""")
    assert fps >= 30, f"FPS too low: {fps:.1f}"

def test_vehicles_disappear_when_none(page):
    """With 0 vehicles the canvas should be clear (no red pixels in centre)."""
    page.evaluate("window._testVehicles = []")
    page.wait_for_timeout(100)
    result = page.evaluate("""() => {
        const c = document.getElementById('vehicle-layer');
        const gl = c.getContext('webgl');
        const px = new Uint8Array(4);
        gl.readPixels(c.width/2, c.height/2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        return px[0];   // red channel
    }""")
    assert result < 10
```

---

## Milestone 8 — Cross-Platform & Distribution

**Goal:** The full project installs and runs correctly on macOS (Intel + Apple Silicon) and Windows 10/11, with a single command.

### 8.1 Dependency Files

**New files:**
- `requirements.txt` — pip packages for Mac/Linux (already exists; extend)
- `environment.yml` — conda-forge environment for all platforms (rasterio, GDAL)
- `environment-cuda.yml` — adds `cupy`, `cudatoolkit`

### 8.2 Path Handling

Replace any string path concatenation with `pathlib.Path`. Audit `importer/overpass.py`, `run.py`, `engine/worldpop.py` for hardcoded `/` separators.

### 8.3 Startup Diagnostics

`run.py` prints to stderr on launch:
```
Platform : macOS arm64  (Apple Silicon)
Backend  : metal (JAX 0.4.x)
Threads  : 10
Cache    : /Users/x/Traffic sim/.overpass_cache
```

### 8.4 Conda Lock Files

Generate per-platform lock files for reproducible installs:
```bash
conda-lock -f environment.yml -p osx-arm64 -p osx-64 -p win-64 -p linux-64
```

---

### ✅ Milestone 8 Tests

**File:** `tests/test_cross_platform.py`

```python
"""Cross-platform path and environment tests."""
import os, sys, pathlib, platform, pytest

def test_pathlib_used_in_overpass(tmp_path):
    """overpass.py cache path construction must work on Windows (no hardcoded /)."""
    import importlib, engine.compute_backend
    from importer.overpass import get_cache_path
    path = get_cache_path("test_key", cache_dir=str(tmp_path))
    assert pathlib.Path(path).parent == tmp_path

def test_cache_dir_creation_cross_platform(tmp_path):
    """Cache directory is created using os.makedirs / pathlib, not shell commands."""
    cache = tmp_path / "sub" / "cache"
    from importer.overpass import ensure_cache_dir
    ensure_cache_dir(str(cache))
    assert cache.is_dir()

def test_no_posix_separators_in_paths(tmp_path):
    """All internal path joins must use os.sep or pathlib."""
    import ast, glob as g
    issues = []
    for pyfile in g.glob("/Users/sekun/Traffic sim/**/*.py", recursive=True):
        if ".git" in pyfile or "__pycache__" in pyfile:
            continue
        src = open(pyfile).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "/" in node.value and ("cache" in node.value or "outputs" in node.value):
                    issues.append(f"{pyfile}:{node.lineno} — literal path with '/'")
    assert not issues, "\n".join(issues[:10])

def test_platform_detection():
    from engine.compute_backend import detect_backend
    backend = detect_backend()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # On Apple Silicon, metal or numpy are the only valid options
        assert backend in ("metal", "numpy")
    elif sys.platform.startswith("win"):
        assert backend in ("cuda", "numpy")

def test_requirements_txt_parseable():
    """requirements.txt must be parseable as pip requirements."""
    import pkg_resources
    path = pathlib.Path("/Users/sekun/Traffic sim/requirements.txt")
    if not path.exists():
        pytest.skip("requirements.txt not found")
    lines = [l.strip() for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for line in lines:
        try:
            list(pkg_resources.parse_requirements(line))
        except Exception as e:
            pytest.fail(f"Bad requirement '{line}': {e}")

def test_environment_yml_exists():
    env = pathlib.Path("/Users/sekun/Traffic sim/environment.yml")
    assert env.exists(), "environment.yml missing for cross-platform conda installs"

def test_http_server_localhost_no_posix_binding():
    """Server must bind to 0.0.0.0 (cross-platform) not a POSIX socket path."""
    import ast
    src = open("/Users/sekun/Traffic sim/run.py").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("/tmp/") or node.value.startswith("/var/"):
                pytest.fail(f"POSIX socket path found in run.py: {node.value!r}")
```

---

## Running the Full Test Suite

```bash
# Install test deps
pip install pytest pytest-benchmark pytest-playwright

# Run all milestone tests
cd "/Users/sekun/Traffic sim"
pytest tests/ -v --tb=short

# Run only a specific milestone
pytest tests/test_performance.py tests/test_speed_compression.py -v

# Run with benchmark output
pytest tests/test_performance.py -v --benchmark-only

# Run cross-platform tests
pytest tests/test_cross_platform.py -v

# Install Playwright browsers (for Milestone 7)
playwright install chromium
pytest tests/test_renderer.py -v
```

---

## Dependency Summary

```
# Core (already in requirements.txt)
numpy>=1.24
scipy>=1.11        # KDTree for WorldPop snapping
requests>=2.31

# Milestone 3 — WorldPop
rasterio>=1.3      # pip or conda-forge
pyproj>=3.5        # CRS reprojection

# Milestone 5 — Weather
# (stdlib urllib only — no extra package)

# Milestone 6 — GPU (optional, install based on platform)
# NVIDIA:  pip install cupy-cuda12x
# Mac M1+: pip install jax-metal  (or: pip install "jax[metal]")

# Milestone 7 — WebGL testing
playwright>=1.40   # only for Milestone 7 renderer tests

# Milestone 8 — Distribution
conda-lock>=2.5    # only needed for lock file generation
```

---

## Commit Tags

Each milestone ends with a tagged commit:

| Tag | Milestone |
|-----|-----------|
| `v1.0` | Baseline (current) |
| `v1.1` | M1 — Performance Foundation |
| `v1.2` | M2 — Interactive Dashboard |
| `v1.3` | M3 — Realistic Spatial Demand |
| `v1.4` | M4 — Temporal Demand & Mode Split |
| `v1.5` | M5 — Weather Effects |
| `v1.6` | M6 — GPU Acceleration |
| `v1.7` | M7 — WebGL Renderer |
| `v1.8` | M8 — Cross-Platform |

> **Build rule:** All tests for milestone N must be green before starting N+1.
