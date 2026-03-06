# Traffic Simulation Platform

A real-time, OSM-powered traffic simulation platform with an interactive browser-based visualisation. Import any city in the world, watch vehicles route through real road networks with accurate signal timing, and tweak simulation parameters live.

![Paris simulation screenshot](outputs/paris_preview.png)

---

## Features

- **Any city, instantly** — type a city name, the platform geocodes it, fetches OpenStreetMap data, and hot-swaps the simulation live in the browser
- **Accurate signal placement** — uses real `highway=traffic_signals` OSM nodes (Paris: 4 600+, London: 4 500+) instead of heuristics
- **IDM car-following** — Intelligent Driver Model with per-vehicle parameters (desired speed, time headway, min gap, acceleration, braking)
- **MOBIL lane changes** — Minimizing Overall Braking Induced by Lane changes with politeness factor, safety checks, and keep-right bias
- **Social Force Model pedestrians** — Helbing & Molnár (1995) repulsion + goal-seeking dynamics
- **Realistic commute demand** — time-of-day AM/PM peak multipliers, Poisson headway spawning, residential → commercial OD matrix
- **CartoDB Dark Matter tiles** — high-quality slippy-map background with retina support, no API key required
- **Smooth bezier roads** — OSM `via_node` geometry rendered with `quadraticCurveTo` for natural-looking curves
- **Detailed signal icons** — 3-light traffic light icons at zoom, compact dot at distance
- **Hot-swap imports** — swap cities without restarting the server; import progress streamed via HTTP polling
- **Zero external dependencies for viz** — plain HTML + Canvas 2D, no npm, no build step
- **Comprehensive test suite** — 12 test modules covering every engine component

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (viz/)                       │
│  index.html  ←→  renderer.js  ←→  Canvas 2D             │
│    CartoDB tiles · bezier roads · vehicles · signals     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (JSON polling ~10 Hz)
┌────────────────────▼────────────────────────────────────┐
│                   run.py  (HTTP server)                   │
│  GET /network   GET /state   GET /import_status          │
│  POST /import   POST /control                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            engine/  (simulation)                  │   │
│  │  NetworkSimulation  ←  Network (graph)            │   │
│  │    IDM · MOBIL · SignalPlan · SFM pedestrians     │   │
│  │    Dijkstra routing · Poisson demand · metrics    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           importer/  (OSM pipeline)               │   │
│  │  geocode → overpass (cached) → parse → infer      │   │
│  │  → NetworkSimulation hot-swap                     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Traffic sim/
├── run.py                  # CLI entry point + HTTP server
├── requirements.txt        # Python dependencies
│
├── engine/                 # Simulation engine (pure Python + NumPy)
│   ├── agents.py           # Vehicle dataclass (IDM + MOBIL parameters)
│   ├── idm.py              # Intelligent Driver Model — pure function
│   ├── mobil.py            # MOBIL lane-change decision model
│   ├── signals.py          # SignalPlan / Phase — signal cycle logic
│   ├── network.py          # Node / Edge / Lane / Network graph
│   ├── network_simulation.py  # Multi-edge routing simulation
│   ├── simulation.py       # Single-road simulation (legacy / tests)
│   ├── pedestrians.py      # Social Force Model pedestrian agents
│   ├── commute.py          # OD demand generation + time-of-day profile
│   └── metrics.py          # TripRecord, MetricsRecorder, BatchMetrics
│
├── importer/               # OSM data pipeline
│   ├── __init__.py         # import_bbox() — full pipeline entry point
│   ├── geocode.py          # Nominatim geocoder (city name → bbox)
│   ├── overpass.py         # Overpass API client + disk cache
│   ├── parser.py           # OSM JSON → nodes / ways / edges
│   ├── inference.py        # Lane count + speed inference (quality flags)
│   └── projection.py       # WGS-84 ↔ Web Mercator (EPSG:3857)
│
├── viz/                    # Browser visualisation (no build step)
│   ├── index.html          # UI shell — cards, controls, search bar
│   └── renderer.js         # Canvas 2D renderer + tile layer
│
├── scenarios/              # Pre-built scenario JSON files
│   ├── oakville_on.json    # Small Canadian city — good for development
│   ├── highway_single_lane.json
│   ├── highway_multilane.json
│   ├── roundabout_vs_signal.json
│   ├── ped_crossing.json
│   └── grid_3x3.json
│
├── tests/                  # pytest test suite
│   ├── test_idm.py
│   ├── test_mobil.py
│   ├── test_signals.py
│   ├── test_simulation.py
│   ├── test_network.py
│   ├── test_routing.py
│   ├── test_metrics.py
│   ├── test_pedestrians.py
│   ├── test_importer.py
│   ├── test_projection.py
│   ├── test_junctions.py
│   └── test_renderer.py
│
├── editor/                 # Scenario editor (standalone)
├── outputs/                # Simulation outputs / exports
└── .overpass_cache/        # Disk-cached Overpass API responses (gitignored)
```

---

## Installation

### Requirements

- Python 3.10+
- Any modern browser (Chrome, Firefox, Safari, Edge)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `requests`, `scipy`, `matplotlib`, `jsonschema`, `pytest`, `pytest-cov`

No npm, no Node.js, no build step required.

---

## Quick Start

### Live simulation with browser visualisation

```bash
python run.py scenarios/oakville_on.json --serve
```

Then open **http://localhost:8765** in your browser.

To import a different city, type its name in the search bar (e.g. "Paris, France") and press Enter.

### Custom port

```bash
python run.py scenarios/oakville_on.json --serve --port 9000
```

### Headless run (metrics only)

```bash
python run.py scenarios/highway_multilane.json
```

### Import a city from the command line

```bash
python run.py --import-bbox 48.85 2.33 48.87 2.36 --out scenarios/paris_centre.json
```

---

## Browser Controls

| Control | Action |
|---|---|
| **Drag** | Pan the map |
| **Scroll / Pinch** | Zoom |
| **Search bar** | Import any city by name |
| **▶ Running / ⏸ Paused** | Toggle simulation |
| **Speed slider** | 0.25× – 8× wall-clock speed |
| **Demand slider** | 0× – 5× traffic demand multiplier |
| **+ / −** | Zoom buttons |
| **⌂** | Fit network to window |

---

## HTTP API

The server exposes a simple JSON API on `http://localhost:8765`.

### `GET /network`

Returns the full road network (re-fetched after each import).

```json
{
  "nodes": { "<id>": { "x": 0.0, "y": 0.0, "junction_type": "signal" } },
  "edges": [ { "id": "...", "from_node": "...", "to_node": "...",
               "num_lanes": 2, "speed_limit": 13.9,
               "geometry": [[x1,y1], [x2,y2]] } ]
}
```

### `GET /state`

Live simulation snapshot (~10 Hz polling).

```json
{
  "clock":    "08:32",
  "sim_time": 5420.0,
  "location": "Paris",
  "vehicles": { "<id>": { "x": 0.0, "y": 0.0, "speed": 12.3,
                           "edge": "way_123_0", "lane": 0 } },
  "pedestrians": { "<id>": { "x": 0.0, "y": 0.0 } },
  "signals":  { "<node_id>": { "state": "green", "phase": 0 } }
}
```

### `GET /import_status`

Progress of an in-flight import.

```json
{ "stage": "fetching", "progress": 0.3, "message": "Downloading OSM data…" }
```

Stages: `idle` → `geocoding` → `fetching` → `parsing` → `building` → `done` / `error`

### `POST /control`

Update runtime controls (JSON body).

```json
{ "paused": false, "speed_mult": 4.0, "demand_mult": 1.5 }
```

All fields optional; omit any you don't want to change.

### `POST /import`

Import a new city by name.

```json
{ "query": "Tokyo, Japan" }
```

Returns `{ "status": "importing" }` immediately; poll `/import_status` for progress.

---

## Simulation Engine

### Intelligent Driver Model (IDM)

Located in `engine/idm.py`. Pure function, safe for NumPy vectorisation.

```
s* = s₀ + max(0, v·T + v·Δv / (2√(a·b)))
a_IDM = a · (1 − (v/v₀)^δ − (s*/s)²)
```

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| Desired speed | v₀ | edge speed limit | Free-road cruising speed (m/s) |
| Min jam gap | s₀ | 2.0 m | Bumper-to-bumper gap at standstill |
| Time headway | T | 1.5 s | Desired following time gap |
| Max acceleration | a | 1.4 m/s² | |
| Comfortable braking | b | 2.0 m/s² | |
| Exponent | δ | 4.0 | Free-road acceleration shape |

### MOBIL Lane Changes

Located in `engine/mobil.py`. Evaluates incentive criterion and safety constraint:

```
net_gain = Δa_self + p · (Δa_fol_current + Δa_fol_target) > threshold
safety:   a_fol_target_after ≥ −b_safe
```

| Parameter | Default | Description |
|---|---|---|
| politeness | 0.3 | Weight on follower welfare (0 = selfish, 1 = altruistic) |
| b_safe | 3.0 m/s² | Max allowed deceleration imposed on target lane follower |
| delta_a_thr | 0.1 m/s² | Min acceleration gain to trigger change |
| bias_right | true | Lower threshold for rightward (lane 0) moves |
| cooldown | 4.0 s | Minimum time between consecutive lane changes |

### Signal Plans

Located in `engine/signals.py`. Each signalised node gets a `SignalPlan` with:
- One or more `Phase` objects (green duration, yellow 3 s, all-red 1 s)
- A per-node `offset` (seconds) for green-wave coordination

Signal nodes are detected from real OSM `highway=traffic_signals` tags:

| City | OSM signal nodes |
|---|---|
| Paris | 4 624 |
| London | 4 523 |
| Shibuya | 2 881 |
| Oakville | 292 |

### Social Force Model Pedestrians

Located in `engine/pedestrians.py`. Reference: Helbing & Molnár (1995).

```
F_total = F_desired + Σ F_ped + Σ F_obstacle

F_desired = (1/τ) · (v₀ · ê_dest − v)          τ = 0.5 s
F_ped     = A · exp(−d_surface / B) · ê_normal   A = 2000 N, B = 0.08 m
```

Vehicles brake hard (deceleration = −b) if pedestrian TTC < 2 s within 20 m.

### Demand Generation

Located in `engine/commute.py`.

1. Nodes classified **residential** (adj. speed ≤ 40 km/h) or **commercial** (> 40 km/h)
2. OD pairs: residential origins → commercial destinations
3. Time-of-day multiplier applied to base demand:

| Period | Multiplier |
|---|---|
| 07:00–09:00 (AM peak) | 3.0× |
| 16:00–19:00 (PM peak) | 2.5× |
| 10:00–15:00 (off-peak) | 0.6× |
| Night (00:00–06:00) | 0.15× |

4. Vehicles spawned via Poisson process: inter-arrival time ~ Exp(1/rate)

---

## OSM Import Pipeline

```
City name
   │
   ▼  importer/geocode.py
Nominatim geocoder → (lat, lon, bbox ±5 km)
   │
   ▼  importer/overpass.py
Overpass API fetch → cache (.overpass_cache/osm_<md5>.json)
   │
   ▼  importer/parser.py
Parse nodes + ways → split at junctions → directed edge segments
Tag merge: out body tags preserved over out skel qt
   │
   ▼  importer/inference.py
Infer lanes + speed (GREEN = OSM tag, AMBER = class default, RED = fallback)
   │
   ▼  run.py  _scenario_from_import()
Build Network graph → detect signal nodes from OSM tags →
generate commute demand → build SignalPlans → NetworkSimulation
```

### Lane inference defaults

| Highway class | Lanes | Speed |
|---|---|---|
| motorway | 3 | 113 km/h |
| trunk | 2 | 80 km/h |
| primary / secondary | 2 | 50 km/h |
| tertiary | 1 | 40 km/h |
| residential | 1 | 30 km/h |
| service | 1 | 15 km/h |
| living_street | 1 | 10 km/h |

---

## Scenario File Format

Scenarios are JSON files in `scenarios/`. Two formats are supported.

### Network scenario (OSM-imported)

```json
{
  "location": "Oakville ON from OSM",
  "duration": 7200.0,
  "warmup":   120.0,
  "network": {
    "nodes": {
      "12345": { "id": "12345", "x": -8887000.0, "y": 5400000.0,
                 "junction_type": "signal", "tags": {"highway": "traffic_signals"} }
    },
    "edges": [
      { "id": "way_67890_0", "from_node": "12345", "to_node": "67890",
        "num_lanes": 2, "speed_limit": 13.9,
        "geometry": [[-8887000, 5400000], [-8886900, 5400050]] }
    ],
    "signal_nodes": ["12345"],
    "demand": { "12345": { "67890": 120.0 } }
  }
}
```

### Road scenario (single-road legacy)

```json
{
  "road": { "length": 500, "num_lanes": 3, "speed_limit": 30.0 },
  "vehicles": { "count": 20, "speed": 25.0 },
  "junction": { "type": "signal", "stop_line": 490 },
  "duration": 300.0
}
```

---

## Visualisation

### Coordinate system

All internal positions use **Web Mercator (EPSG:3857)** metres:
- X increases eastward
- Y increases northward

Canvas screen coordinates:
```
screen_x =  world_x · S + Bx + Px
screen_y = −world_y · S + By + Py
```
where `S` = scale (px/m), `B` = base offset (network centre), `P` = pan offset.

### Map tiles

CartoDB Dark Matter (`dark_nolabels`):
```
https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png
```
- Free, CORS-enabled, no API key required
- Retina tiles (`@2x`) on HiDPI displays
- In-memory LRU cache (1 024 tiles max)
- Zoom level auto-selected: `z = round(log₂(MERC_FULL · S / 256))`

### Road rendering

Roads sorted into 5 tiers by speed limit:

| Tier | Speed | Colour | Width |
|---|---|---|---|
| 0 Motorway | ≥ 20 m/s | amber `#ffab40` | 8 px |
| 1 Primary | ≥ 13 m/s | blue `#448aff` | 5 px |
| 2 Secondary | ≥ 9.5 m/s | grey `#546e7a` | 3 px |
| 3 Residential | ≥ 5.5 m/s | grey `#37474f` | 1.8 px |
| 4 Service | < 5.5 m/s | grey `#263238` | 0.8 px |

Roads drawn with `quadraticCurveTo` smooth bezier curves through via_nodes.

### Vehicle colours

| State | Colour | Condition |
|---|---|---|
| Free flow | `#00e676` green | speed ≥ 80% of limit |
| Slowing | `#ffab00` amber | 40–80% of limit |
| Congested | `#ff1744` red | < 40% of limit |
| Stopped | `#78909c` grey | speed < 0.3 m/s |
| Pedestrian | `#40c4ff` cyan | |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=engine --cov=importer --cov-report=term-missing

# Single module
pytest tests/test_idm.py -v
```

### Test modules

| File | What it tests |
|---|---|
| `test_idm.py` | IDM free-road, following, braking, edge cases |
| `test_mobil.py` | MOBIL incentive, safety, cooldown, bias |
| `test_signals.py` | Phase cycling, offsets, movement states |
| `test_simulation.py` | Single-road end-to-end, junction types |
| `test_network.py` | Graph construction, lane geometry, Dijkstra |
| `test_routing.py` | Multi-hop routing, disconnected graphs |
| `test_metrics.py` | Throughput, delay, queue length, batch CI |
| `test_pedestrians.py` | SFM forces, arrival, vehicle yielding |
| `test_importer.py` | Overpass parsing, tag merge, inference |
| `test_projection.py` | WGS-84 ↔ Mercator round-trips |
| `test_junctions.py` | Signal, roundabout, yield, stop-sign |
| `test_renderer.py` | Coordinate transform, tile math |

---

## Scenarios Included

| File | Description |
|---|---|
| `oakville_on.json` | Oakville, Ontario — small city, 292 OSM signals, good for dev |
| `highway_single_lane.json` | 500 m single-lane highway with signal |
| `highway_multilane.json` | 3-lane highway, signal + MOBIL lane changes |
| `roundabout_vs_signal.json` | Side-by-side comparison of junction types |
| `ped_crossing.json` | Mixed vehicle + pedestrian crossing scenario |
| `grid_3x3.json` | Synthetic 3×3 grid network |

---

## Performance Notes

| City | Nodes | Edges | OSM signals | Import time |
|---|---|---|---|---|
| Oakville, ON | ~8 k | ~9 k | 292 | ~5 s |
| Paris, France | ~97 k | ~105 k | 4 624 | ~45 s |
| London, UK | ~110 k | ~120 k | 4 523 | ~50 s |
| Shibuya, Tokyo | ~45 k | ~50 k | 2 881 | ~25 s |

- Simulation step: O(V + E) where V = active vehicles, E = non-empty edges
- Overpass responses cached to disk; subsequent imports for same bbox are instant
- Visualization culls vehicles to viewport; 60 FPS maintained up to ~2 000 vehicles

---

## Roadmap

Planned features (in priority order):

1. **NumPy-vectorized IDM + thread pool** — batch all vehicle accelerations per-step; ThreadPoolExecutor for per-edge parallelism
2. **Interactive parameter dashboard** — collapsible panel to tweak IDM/MOBIL/signal/demand params live, with hover tooltips explaining each variable
3. **Simulation speed range expansion** — adaptive dt at high speed multipliers (up to 512×), sub-stepping without rendering; simulate a full 24-hour day in 1–5 minutes
4. **User power allocation** — Battery Saver / Balanced / Max Performance modes controlling dt, MOBIL, rendering resolution, tile loading
5. **GPU acceleration (optional)** — CuPy/Numba for CUDA (PC), Metal/MPS for Apple Silicon, WebGL/WebGPU renderer for thousands of vehicles
6. **WorldPop population spawn points** — replace heuristic OD generation with real ~100 m population density raster
7. **Trip purpose classification** — OSM POI tags → work / school / retail / leisure destinations; age-based mode split
8. **Weather effects** — Open-Meteo API integration; IDM parameter modulation (rain → lower v₀, snow → increase s₀ + T)
9. **GTFS transit integration** — bus/train routes as agents; mode split (car vs transit vs walk)
10. **Building capacity model** — floor estimation from OSM `building:levels` / `height`; apartments vs offices from footprint + land use

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add tests for new functionality in `tests/`
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a pull request

---

## Data Attribution

- Road network data: **© OpenStreetMap contributors** (ODbL licence)
  https://www.openstreetmap.org/copyright
- Map tiles: **© CARTO**
  https://carto.com/attributions
- Geocoding: **Nominatim / OpenStreetMap**

---

## References

- Treiber, M., Hennecke, A., & Helbing, D. (2000). *Congested traffic states in empirical observations and microscopic simulations.* Physical Review E, 62(2).
- Kesting, A., Treiber, M., & Helbing, D. (2007). *General lane-changing model MOBIL for car-following models.* Transportation Research Record, 1999(1).
- Helbing, D., & Molnár, P. (1995). *Social force model for pedestrian dynamics.* Physical Review E, 51(5).
- Tsinghua FIB Lab. *WorldCommuting-OD.* https://github.com/tsinghua-fib-lab/WorldCommuting-OD
- WorldPop. *Global High Resolution Population Denominators.* https://hub.worldpop.org/

---

## Licence

MIT — see `LICENSE` for details.
