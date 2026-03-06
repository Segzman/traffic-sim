"""Milestone 4 — OSM importer tests (uses fixture, no network calls)."""
import json
import os
import pytest

from importer.parser import parse_osm
from importer.inference import infer, QualityFlag
from importer import import_bbox
from engine.simulation import Simulation


# ------------------------------------------------------------------ #
# Fixture loading
# ------------------------------------------------------------------ #

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "osm_small.json")


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _parsed():
    return parse_osm(_load_fixture())


def _enriched():
    return infer(_parsed())


# ------------------------------------------------------------------ #
# test_parse_osm_nodes
# ------------------------------------------------------------------ #

def test_parse_osm_nodes():
    """All OSM nodes are parsed with correct id, lat, lon."""
    parsed = _parsed()
    nodes = parsed["nodes"]

    expected = {
        101: (51.5100, -0.1200),
        102: (51.5110, -0.1200),
        103: (51.5120, -0.1200),
        104: (51.5110, -0.1190),
        105: (51.5110, -0.1210),
    }
    # All expected nodes should be present (they are all referenced by edges)
    for nid, (lat, lon) in expected.items():
        assert nid in nodes, f"Node {nid} missing from parsed output"
        assert abs(nodes[nid]["lat"] - lat) < 1e-7, f"Node {nid} lat wrong"
        assert abs(nodes[nid]["lon"] - lon) < 1e-7, f"Node {nid} lon wrong"
    # Projected x, y should be finite floats
    for node in nodes.values():
        assert isinstance(node["x"], float) and math.isfinite(node["x"])
        assert isinstance(node["y"], float) and math.isfinite(node["y"])


import math   # needed for isfinite check above


# ------------------------------------------------------------------ #
# test_parse_osm_ways
# ------------------------------------------------------------------ #

def test_parse_osm_ways():
    """Highway ways are parsed; non-highway ways are excluded."""
    parsed = _parsed()
    ways = parsed["ways"]
    way_ids = [w["id"] for w in ways]

    assert 201 in way_ids, "primary highway way 201 should be included"
    assert 202 in way_ids, "secondary highway way 202 should be included"
    assert 203 not in way_ids, "building way 203 should be excluded (not a highway)"
    assert len(ways) == 2, f"Expected exactly 2 highway ways; got {len(ways)}"


# ------------------------------------------------------------------ #
# test_way_splitting_at_junctions
# ------------------------------------------------------------------ #

def test_way_splitting_at_junctions():
    """Way 201 (nodes 101→102→103) is split at shared node 102 into two edges."""
    parsed = _parsed()
    edges = parsed["edges"]

    # Way 201 should have been split at node 102 → two edges
    way201_edges = [e for e in edges if e["way_id"] == 201]
    assert len(way201_edges) == 2, (
        f"Way 201 should produce 2 edges after splitting at junction node 102; "
        f"got {len(way201_edges)}: {way201_edges}"
    )

    # The split should be at node 102
    endpoint_nodes = {e["from_node"] for e in way201_edges} | {e["to_node"] for e in way201_edges}
    assert 101 in endpoint_nodes, "Node 101 should be an edge endpoint"
    assert 102 in endpoint_nodes, "Node 102 (junction) should be an edge endpoint"
    assert 103 in endpoint_nodes, "Node 103 should be an edge endpoint"


# ------------------------------------------------------------------ #
# test_inference_lane_count
# ------------------------------------------------------------------ #

def test_inference_lane_count():
    """Way 202 (secondary, no lanes tag) gets AMBER flag and 2 lanes (class default)."""
    enriched = _enriched()
    edges = enriched["edges"]

    way202_edges = [e for e in edges if e["way_id"] == 202]
    assert way202_edges, "No edges from way 202 found"

    for edge in way202_edges:
        assert edge["quality_flags"]["lanes"] == QualityFlag.AMBER.value, (
            f"Way 202 (no lanes tag) should have AMBER lanes flag; "
            f"got {edge['quality_flags']['lanes']}"
        )
        assert edge["num_lanes"] == 2, (
            f"Secondary highway default is 2 lanes; got {edge['num_lanes']}"
        )

    # Way 201 has explicit 'lanes': '2' → GREEN flag
    way201_edges = [e for e in edges if e["way_id"] == 201]
    for edge in way201_edges:
        assert edge["quality_flags"]["lanes"] == QualityFlag.GREEN.value, (
            f"Way 201 has explicit lanes tag → should be GREEN; "
            f"got {edge['quality_flags']['lanes']}"
        )


# ------------------------------------------------------------------ #
# test_inference_speed_limit
# ------------------------------------------------------------------ #

def test_inference_speed_limit():
    """Way 201 (maxspeed tag) → GREEN; Way 202 (no maxspeed) → AMBER."""
    enriched = _enriched()
    edges = enriched["edges"]

    way201_edges = [e for e in edges if e["way_id"] == 201]
    for edge in way201_edges:
        assert edge["quality_flags"]["speed_limit"] == QualityFlag.GREEN.value, (
            "Way 201 has maxspeed tag → speed_limit flag should be GREEN"
        )
        # 50 km/h = 50/3.6 ≈ 13.89 m/s
        assert abs(edge["speed_limit"] - 50 / 3.6) < 0.01, (
            f"speed_limit for 50 km/h should be ≈13.89 m/s; got {edge['speed_limit']:.4f}"
        )

    way202_edges = [e for e in edges if e["way_id"] == 202]
    for edge in way202_edges:
        assert edge["quality_flags"]["speed_limit"] == QualityFlag.AMBER.value, (
            "Way 202 has no maxspeed tag → speed_limit flag should be AMBER"
        )


# ------------------------------------------------------------------ #
# test_quality_flags_in_output
# ------------------------------------------------------------------ #

def test_quality_flags_in_output():
    """All edges in enriched output have quality_flags with 'lanes' and 'speed_limit'."""
    enriched = _enriched()
    edges = enriched["edges"]
    assert edges, "No edges in enriched output"

    for edge in edges:
        assert "quality_flags" in edge, f"Edge {edge['id']} missing quality_flags"
        flags = edge["quality_flags"]
        assert "lanes" in flags, f"Edge {edge['id']} quality_flags missing 'lanes'"
        assert "speed_limit" in flags, f"Edge {edge['id']} quality_flags missing 'speed_limit'"
        valid = {f.value for f in QualityFlag}
        assert flags["lanes"] in valid, f"Invalid lanes flag: {flags['lanes']}"
        assert flags["speed_limit"] in valid, f"Invalid speed_limit flag: {flags['speed_limit']}"


# ------------------------------------------------------------------ #
# test_import_produces_valid_scenario
# ------------------------------------------------------------------ #

def test_import_produces_valid_scenario():
    """import_bbox on fixture data produces a scenario that loads without error."""
    osm_data = _load_fixture()
    scenario = import_bbox(51.509, -0.122, 51.513, -0.118, osm_data=osm_data)

    # Scenario must have required keys
    assert "road" in scenario, "Scenario missing 'road' key"
    assert "vehicles" in scenario, "Scenario missing 'vehicles' key"
    assert "duration" in scenario, "Scenario missing 'duration' key"
    assert "network" in scenario, "Scenario missing 'network' key"

    road = scenario["road"]
    assert road["length"] > 0, "Road length must be positive"
    assert road["num_lanes"] >= 1, "Must have at least 1 lane"
    assert road["speed_limit"] > 0, "Speed limit must be positive"

    # Network should contain edges with quality_flags
    net_edges = scenario["network"]["edges"]
    assert len(net_edges) > 0, "Network must have at least one edge"
    for edge in net_edges:
        assert "quality_flags" in edge

    # Must load and run in simulation engine without exceptions
    sim = Simulation(scenario)
    metrics = sim.run()
    assert isinstance(metrics, dict)
    assert "throughput" in metrics
