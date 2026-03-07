# Milestone 3 Calibration Pack

This file records the calibration brief supplied by the user for Milestone 3 implementation.

## Milestone 3 must-haves

- WorldPop 100m raster -> population-weighted origins/spawns.
- OSM POI trip purposes -> `home | work | school | retail` (fallback `other`).
- Building floor/capacity estimation -> destination weighting.

## Core defaults used for implementation

- Purpose split: `work=0.40`, `school=0.15`, `retail=0.25`, `other=0.20`.
- Gravity destination weighting: `beta=0.9`, `gamma_km=0.12`.
- Building defaults:
  - `meters_per_floor=3.3`
  - `office_sqm_per_person=12`
  - `residential_sqm_per_person=25`
- Demand generation baseline: `peak_veh_hr=400`, `max_pairs=60`.

## Purpose mapping (first match wins)

1. `building=apartments|residential|house` -> `home`
2. `building=office|commercial|industrial` or `office=*` -> `work`
3. `amenity=school|university|college` -> `school`
4. `shop=*` or `amenity=restaurant|cafe` -> `retail`
5. otherwise -> `other`

## Notes

- `engine/worldpop.py` supports cached raster-based weighting when `rasterio` is available; otherwise it falls back to deterministic uniform node priors.
- Set `WORLDPOP_RASTER_PATH=/path/to/worldpop_100m.tif` (or `network.worldpop_raster_path` in scenario) to enable true raster-weighted origins in `run.py`.
- `engine/poi_demand.py` combines WorldPop origin weights + POI/building destination buckets.
- `engine/buildings.py` estimates floors and occupancy capacity from OSM tags and footprint area.
