"""Milestone 3 — spatial demand (WorldPop + POI + building capacity) tests."""
from __future__ import annotations

import json
import os

from importer.parser import parse_osm
from importer.inference import infer
from importer import import_bbox

from engine.buildings import estimate_building_capacity
from engine.network import Network, Node, Edge
from engine.poi_demand import build_purpose_nodes, generate_spatial_demand
from engine.worldpop import load_worldpop_weights


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "osm_small.json")


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _tiny_network_from_scenario(scenario: dict) -> Network:
    net = Network()
    for nid, nd in scenario["network"]["nodes"].items():
        net.add_node(Node(id=str(nid), x=float(nd["x"]), y=float(nd["y"])))
    for ed in scenario["network"]["edges"]:
        net.add_edge(Edge(
            id=ed["id"],
            from_node=str(ed["from_node"]),
            to_node=str(ed["to_node"]),
            num_lanes=int(ed.get("num_lanes", 1)),
            speed_limit=float(ed.get("speed_limit", 8.3)),
            geometry=[],
            road_type=str(ed.get("road_type", "primary")),
        ))
    return net


def test_parse_osm_emits_pois_and_buildings():
    parsed = parse_osm(_load_fixture())
    assert "pois" in parsed
    assert "buildings" in parsed

    # fixture includes way 203 with building=yes
    assert any(str(p["id"]).startswith("way_203") for p in parsed["pois"])
    assert any(str(b["id"]).startswith("way_203") for b in parsed["buildings"])


def test_worldpop_weights_fallback_contains_all_nodes(tmp_path):
    scenario = import_bbox(51.509, -0.122, 51.513, -0.118, osm_data=_load_fixture())
    net = _tiny_network_from_scenario(scenario)

    weights = load_worldpop_weights(
        bbox=(51.509, -0.122, 51.513, -0.118),
        network=net,
        cache_dir=str(tmp_path),
        raster_path=None,
    )
    assert set(weights.keys()) == set(net.nodes.keys())
    assert all(v > 0.0 for v in weights.values())


def test_building_capacity_positive():
    parsed = parse_osm(_load_fixture())
    building = parsed["buildings"][0]
    cap = estimate_building_capacity(building)
    assert cap > 0.0


def test_generate_spatial_demand_returns_pairs():
    osm = _load_fixture()
    parsed = infer(parse_osm(osm))
    scenario = import_bbox(51.509, -0.122, 51.513, -0.118, osm_data=osm)
    net = _tiny_network_from_scenario(scenario)

    purpose_nodes, _weights = build_purpose_nodes(net, parsed.get("pois"), parsed.get("buildings"))
    assert purpose_nodes["home"], "Expected at least one home node from building tags"

    demand = generate_spatial_demand(
        network=net,
        bbox=(51.509, -0.122, 51.513, -0.118),
        pois=parsed.get("pois"),
        buildings=parsed.get("buildings"),
        seed=7,
        peak_veh_hr=200.0,
        max_pairs=12,
    )

    pair_count = sum(len(v) for v in demand.values())
    flow_total = sum(sum(v.values()) for v in demand.values())
    assert pair_count > 0
    assert flow_total > 0.0
