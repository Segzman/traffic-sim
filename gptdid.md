# GPTDID Report

## Scope
Implemented and validated Milestone 3 behavior updates with focus on:
- Blank startup + stable import flow
- Demand realism (WorldPop/OSM/building weighting)
- Intercity modeling (through + entering + leaving)
- Runtime performance (threaded OD generation)
- Jam safety (no overlap / safer edge entry)

## What Changed

### 1) Startup / import behavior
- Server can start on blank map (`python run.py --serve`) with no default Oakville load.
- Import flow now has stale-job recovery and force-supersede support.
- Frontend retries `POST /import` with `force=true` on 409 conflict.
- Imported scenarios now enforce midnight temporal demand:
  - `temporal_demand=true`
  - `start_hour=0.0`

### 2) Local storage stability
- Moved UI state to scoped/versioned keys per location:
  - `sim_v2_<location_slug>_<param>`
- Added legacy-key purge for old global keys.

### 3) Drivers can enter and leave city
- Added `add_intercity_exchange_demand(...)` in `engine/poi_demand.py`.
- This injects two directional demand families:
  - **Entering city**: boundary -> interior
  - **Leaving city**: interior -> boundary
- Existing through-traffic model remains:
  - boundary -> boundary (`add_intercity_through_demand`)
- Net effect: highways are not only pass-through; they now support city inflow/outflow explicitly.

### 4) Intercity directional weighting from data
- Added surrounding-population exposure loader in `engine/worldpop.py`:
  - `load_worldpop_surrounding_exposure(...)`
- Exposure is directional (`west/east/south/north`) and feeds intercity weighting.

### 5) OD generation performance
- `generate_spatial_demand(...)` now supports threaded generation:
  - `parallel_workers` (defaults to min(8, CPU count))
  - environment override: `M3_PARALLEL_WORKERS`
- Added vectorized destination scoring (NumPy) and capped candidate evaluation (`max_candidate_eval`) for speed.
- Added deterministic post-pass:
  - enforce minimum unique origins
  - enforce minimum unique OD breadth

### 6) Traffic jam safety
- Safer spawn lane entry checks and edge-entry blocking.
- Additional no-overlap clamp after lane-change phase.

## Variables / Calculations Used

### Demand / OD
- `purpose_split`: `{work:0.40, school:0.15, retail:0.25, other:0.20}`
- Gravity destination score:
  - `score_ij = (cap_j^beta) * exp(-gamma_km * d_ij_km)`
  - defaults: `beta=0.9`, `gamma_km=0.12`
- `min_unique_origins`: enforced after stochastic generation.
- `intercity_max_pairs`: through-traffic OD budget.
- `intercity_exchange_pairs`: entering+leaving OD budget.

### Intercity flows
- Through-demand estimate blends:
  - structural lane-km/boundary exposure
  - gateway capacity term
  - optional surrounding-population pull
- Exchange-demand estimate:
  - defaults from through estimate + base peak demand
  - sampled as boundary->interior and interior->boundary routable OD pairs.

### Building capacity weighting
- Capacity proxy:
  - `capacity ~= (footprint_area * floors) / sqm_per_person`
- Uses OSM tags + levels/height heuristics.

### Temporal demand
- Imported-city run starts at midnight with low demand multipliers; avoids initial surge artifact.

## Key Files
- `run.py`
- `engine/poi_demand.py`
- `engine/worldpop.py`
- `engine/network_simulation.py`
- `viz/index.html`
- `tests/test_intercity_demand.py`

## Validation
- Test suite run: **61 passed** (targeted regression + M3/intercity/temporal/jam tests).
- Live verification on localhost:9000:
  - blank startup works
  - import completes
  - midnight vehicle count no longer spikes immediately after import

## References (variables and models)
- WorldPop data/methods: https://www.worldpop.org/
- OSM tagging semantics: https://wiki.openstreetmap.org/wiki/Map_features
- IDM (car-following): https://link.aps.org/doi/10.1103/PhysRevE.62.1805
- MOBIL (lane change): https://akesting.de/download/MOBIL_TRR_2007.pdf
- Traffic Flow Dynamics (Treiber/Kesting): https://traffic-flow-dynamics.org/
- FHWA weather/operations guidance: https://ops.fhwa.dot.gov/weather/roadimpact.htm
- Highway Capacity Manual resources (TRB/FHWA summaries): https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter5.htm

