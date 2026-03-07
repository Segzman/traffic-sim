"""Inter-city through-demand injection tests."""
from __future__ import annotations

from engine.network import Network, Node, Edge
from engine.poi_demand import (
    add_intercity_exchange_demand,
    add_intercity_through_demand,
    estimate_intercity_through_veh_hr,
)


def _highway_net() -> Network:
    net = Network()
    net.add_node(Node(id="W", x=0.0, y=0.0))
    net.add_node(Node(id="M", x=5000.0, y=0.0))
    net.add_node(Node(id="E", x=10000.0, y=0.0))

    # Bidirectional motorway corridor.
    net.add_edge(Edge(id="WM", from_node="W", to_node="M", num_lanes=2, speed_limit=30.0,
                      geometry=[[0.0, 0.0], [5000.0, 0.0]], road_type="motorway"))
    net.add_edge(Edge(id="MW", from_node="M", to_node="W", num_lanes=2, speed_limit=30.0,
                      geometry=[[5000.0, 0.0], [0.0, 0.0]], road_type="motorway"))
    net.add_edge(Edge(id="ME", from_node="M", to_node="E", num_lanes=2, speed_limit=30.0,
                      geometry=[[5000.0, 0.0], [10000.0, 0.0]], road_type="motorway"))
    net.add_edge(Edge(id="EM", from_node="E", to_node="M", num_lanes=2, speed_limit=30.0,
                      geometry=[[10000.0, 0.0], [5000.0, 0.0]], road_type="motorway"))
    return net


def test_add_intercity_through_demand_populates_corridor():
    net = _highway_net()
    out = add_intercity_through_demand(
        net,
        demand={},
        seed=1,
        through_veh_hr=600.0,
        max_pairs=20,
    )

    total_flow = sum(sum(v.values()) for v in out.values())
    assert total_flow > 0.0

    # With this tiny corridor, inter-city OD should include west-east and east-west.
    assert "W" in out and "E" in out
    assert out["W"].get("E", 0.0) > 0.0
    assert out["E"].get("W", 0.0) > 0.0


def test_estimate_intercity_through_veh_hr_uses_network_data():
    net = _highway_net()
    est = estimate_intercity_through_veh_hr(net, base_peak_veh_hr=1000.0)
    assert est > 0.0


def test_estimate_intercity_through_veh_hr_increases_with_surrounding_population():
    net = _highway_net()
    base = estimate_intercity_through_veh_hr(net, base_peak_veh_hr=700.0)
    boosted = estimate_intercity_through_veh_hr(
        net,
        base_peak_veh_hr=700.0,
        surrounding_exposure={
            "west": 1_200_000.0,
            "east": 1_100_000.0,
            "south": 300_000.0,
            "north": 250_000.0,
            "total": 2_850_000.0,
        },
    )
    assert boosted >= base


def test_add_intercity_auto_estimate_when_none():
    net = _highway_net()
    out = add_intercity_through_demand(
        net,
        demand={},
        seed=3,
        through_veh_hr=None,
        base_peak_veh_hr=900.0,
        max_pairs=24,
    )
    total_flow = sum(sum(v.values()) for v in out.values())
    assert total_flow > 0.0


def test_add_intercity_exchange_adds_entering_and_leaving_flows():
    net = _highway_net()
    out = add_intercity_exchange_demand(
        net,
        demand={},
        node_weights={"W": 1.0, "M": 10.0, "E": 1.0},
        seed=9,
        inflow_veh_hr=200.0,
        outflow_veh_hr=200.0,
        max_pairs=20,
    )
    assert sum(sum(v.values()) for v in out.values()) > 0.0

    # Expect at least one boundary->interior and one interior->boundary pair.
    has_enter = any((o in {"W", "E"}) and ("M" in dests) for o, dests in out.items())
    has_leave = any((o == "M") and any(d in {"W", "E"} for d in dests) for o, dests in out.items())
    assert has_enter
    assert has_leave
