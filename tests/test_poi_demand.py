"""POI/building demand generation tests."""
import numpy as np
import pytest

from engine.buildings import estimate_building_capacity
from engine.network import Network, Node, Edge
from engine.poi_demand import (
    _classify_purpose,
    _safe_probs,
    add_intercity_through_demand,
    build_purpose_nodes,
    estimate_intercity_through_veh_hr,
    generate_spatial_demand,
)


def _line_network() -> Network:
    net = Network()
    for nid, x in [("A", 0.0), ("B", 500.0), ("C", 1000.0), ("D", 1500.0)]:
        net.add_node(Node(id=nid, x=x, y=0.0))

    for eid, src, dst in [("AB", "A", "B"), ("BA", "B", "A"),
                          ("BC", "B", "C"), ("CB", "C", "B"),
                          ("CD", "C", "D"), ("DC", "D", "C")]:
        net.add_edge(Edge(
            id=eid,
            from_node=src,
            to_node=dst,
            num_lanes=1,
            speed_limit=15.0,
        ))
    return net


def test_classify_purpose_priority_and_building_fallback():
    assert _classify_purpose({"building": "apartments"}) == "home"
    assert _classify_purpose({"building": "office"}) == "work"
    assert _classify_purpose({"office": "yes"}) == "work"
    assert _classify_purpose({"amenity": "school"}) == "school"
    assert _classify_purpose({"amenity": "university"}) == "school"
    assert _classify_purpose({"shop": "bakery"}) == "retail"
    assert _classify_purpose({"building": "office", "shop": "yes"}) == "work"
    assert _classify_purpose({"building": "retail"}) == "retail"
    assert _classify_purpose({"building": "yes"}) == "other"
    assert _classify_purpose({}) == "other"


def test_safe_probs_cleans_invalid_weights():
    probs = _safe_probs(np.array([1.0, np.nan, -2.0, 3.0]))

    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.25)
    assert probs[1] == 0.0
    assert probs[2] == 0.0
    assert probs[3] == pytest.approx(0.75)


def test_safe_probs_uniform_for_zero_or_nan_weights():
    zeros = _safe_probs(np.zeros(4))
    nans = _safe_probs(np.array([np.nan, np.nan]))

    np.testing.assert_allclose(zeros, np.full(4, 0.25))
    np.testing.assert_allclose(nans, np.full(2, 0.5))


def test_build_purpose_nodes_maps_pois_and_building_capacity():
    net = Network()
    for nid, x in [("N0", 0.0), ("N1", 1000.0), ("N2", 2000.0), ("N3", 3000.0)]:
        net.add_node(Node(id=nid, x=x, y=0.0))

    pois = [
        {"x": 0.0, "y": 0.0, "tags": {"building": "apartments"}},
        {"x": 1000.0, "y": 0.0, "tags": {"amenity": "school"}},
        {"x": 2000.0, "y": 0.0, "tags": {"shop": "supermarket"}},
    ]
    building = {
        "x": 3000.0,
        "y": 0.0,
        "tags": {"building": "office", "building:levels": "2"},
        "footprint_area": 1000.0,
    }

    buckets, weights = build_purpose_nodes(net, pois, [building])

    assert buckets["home"] == ["N0"]
    assert buckets["school"] == ["N1"]
    assert buckets["retail"] == ["N2"]
    assert buckets["work"] == ["N3"]
    assert weights["home"]["N0"] == pytest.approx(1.0)
    assert weights["school"]["N1"] == pytest.approx(1.0)
    assert weights["retail"]["N2"] == pytest.approx(1.0)
    assert weights["work"]["N3"] == pytest.approx(
        estimate_building_capacity(building)
    )


def test_build_purpose_nodes_falls_back_to_all_nodes():
    net = Network()
    net.add_node(Node(id="N0", x=0.0, y=0.0))
    net.add_node(Node(id="N1", x=100.0, y=0.0))

    buckets, weights = build_purpose_nodes(net, None, None)

    for purpose in ("home", "work", "school", "retail", "other"):
        assert buckets[purpose] == ["N0", "N1"]
        assert weights[purpose] == {"N0": 1.0, "N1": 1.0}


def test_build_purpose_nodes_ignores_missing_coordinates():
    net = Network()
    net.add_node(Node(id="N0", x=0.0, y=0.0))

    buckets, weights = build_purpose_nodes(
        net,
        [{"y": 0.0, "tags": {"shop": "bakery"}}],
        [{"x": 0.0, "tags": {"building": "office"}}],
    )

    assert buckets["retail"] == ["N0"]
    assert buckets["work"] == ["N0"]
    assert weights["retail"]["N0"] == pytest.approx(1.0)
    assert weights["work"]["N0"] == pytest.approx(1.0)


def test_estimate_intercity_through_veh_hr_edge_cases():
    no_edges = Network()
    no_edges.add_node(Node(id="A", x=0.0, y=0.0))

    assert estimate_intercity_through_veh_hr(
        no_edges, base_peak_veh_hr=1000.0
    ) == 0.0
    assert estimate_intercity_through_veh_hr(
        _line_network(), base_peak_veh_hr=0.0
    ) == 0.0


def test_add_intercity_through_demand_noop_cases():
    net = _line_network()
    demand = {"A": {"B": 12.0}}

    assert add_intercity_through_demand(
        net, demand, through_veh_hr=0.0, max_pairs=10
    ) is demand
    assert add_intercity_through_demand(
        net, demand, through_veh_hr=500.0, max_pairs=0
    ) is demand
    assert demand == {"A": {"B": 12.0}}


def test_generate_spatial_demand_is_reproducible_and_valid(tmp_path):
    net = _line_network()
    kwargs = dict(
        bbox=(-1.0, -1.0, 1.0, 1.0),
        pois=None,
        buildings=None,
        seed=5,
        peak_veh_hr=120.0,
        max_pairs=8,
        min_unique_origins=4,
        worldpop_cache_dir=str(tmp_path),
        intercity_exchange_pairs=0,
        intercity_max_pairs=0,
        parallel_workers=1,
    )

    first = generate_spatial_demand(network=net, **kwargs)
    second = generate_spatial_demand(network=net, **kwargs)

    assert first == second
    assert first
    assert sum(len(dests) for dests in first.values()) <= 8
    assert sum(sum(dests.values()) for dests in first.values()) > 0.0
    assert all(origin in net.nodes for origin in first)
    assert all(
        dest in net.nodes and dest != origin and flow > 0.0
        for origin, dests in first.items()
        for dest, flow in dests.items()
    )


def test_generate_spatial_demand_returns_empty_for_single_node(tmp_path):
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))

    demand = generate_spatial_demand(
        network=net,
        bbox=(-1.0, -1.0, 1.0, 1.0),
        pois=None,
        buildings=None,
        worldpop_cache_dir=str(tmp_path),
    )

    assert demand == {}
