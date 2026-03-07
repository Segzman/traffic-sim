"""Auto-upgrade sparse legacy demand to broader M3 spatial demand."""
from __future__ import annotations

from run import _build_network_sim


def test_build_network_sim_auto_upgrades_sparse_demand():
    scenario = {
        "network": {
            "nodes": [
                {"id": "A", "x": 0.0, "y": 0.0},
                {"id": "B", "x": 200.0, "y": 0.0},
                {"id": "C", "x": 400.0, "y": 0.0},
                {"id": "D", "x": 600.0, "y": 0.0},
                {"id": "E", "x": 800.0, "y": 0.0},
            ],
            "edges": [
                {"id": "AB", "from_node": "A", "to_node": "B", "num_lanes": 1, "speed_limit": 12.0},
                {"id": "BC", "from_node": "B", "to_node": "C", "num_lanes": 1, "speed_limit": 12.0},
                {"id": "CD", "from_node": "C", "to_node": "D", "num_lanes": 1, "speed_limit": 12.0},
                {"id": "DE", "from_node": "D", "to_node": "E", "num_lanes": 1, "speed_limit": 12.0},
            ],
            "bbox": {"south": 43.0, "west": -79.8, "north": 43.2, "east": -79.6},
            "pois": [
                {"id": "p1", "x": 0.0, "y": 0.0, "tags": {"building": "residential"}},
                {"id": "p2", "x": 400.0, "y": 0.0, "tags": {"office": "company"}},
                {"id": "p3", "x": 600.0, "y": 0.0, "tags": {"amenity": "school"}},
                {"id": "p4", "x": 800.0, "y": 0.0, "tags": {"shop": "supermarket"}},
            ],
            "buildings": [
                {"id": "b1", "x": 0.0, "y": 0.0, "footprint_area": 900.0, "tags": {"building": "residential", "building:levels": "6"}},
                {"id": "b2", "x": 400.0, "y": 0.0, "footprint_area": 2500.0, "tags": {"building": "office", "building:levels": "8"}},
                {"id": "b3", "x": 800.0, "y": 0.0, "footprint_area": 1200.0, "tags": {"building": "retail", "building:levels": "2"}},
            ],
        },
        "demand": {"A": {"B": 100.0}},
        "duration": 300.0,
        "seed": 7,
        "auto_m3_demand": True,
        "max_pairs": 40,
        "min_unique_origins": 4,
        "peak_veh_hr": 800.0,
    }

    sim = _build_network_sim(scenario)

    origins = len(sim.demand)
    pairs = sum(len(v) for v in sim.demand.values())

    assert origins >= 4, f"Expected broader M3 origin coverage; got {origins}"
    assert pairs >= 8, f"Expected upgraded OD matrix; got {pairs} pairs"
