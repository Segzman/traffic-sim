"""Commute demand generation tests."""
import pytest

from engine.commute import (
    classify_nodes,
    demand_factor,
    generate_commute_demand,
)
from engine.network import Network, Node, Edge


def _commute_network() -> Network:
    net = Network()
    for nid, x in [("R1", 0.0), ("R2", 100.0), ("C1", 200.0), ("C2", 300.0)]:
        net.add_node(Node(id=nid, x=x, y=0.0))
    net.add_node(Node(id="ISO", x=400.0, y=0.0))

    edges = [
        ("R1R2", "R1", "R2", 8.0),
        ("R2R1", "R2", "R1", 8.0),
        ("R2C1", "R2", "C1", 20.0),
        ("C1R2", "C1", "R2", 20.0),
        ("C1C2", "C1", "C2", 20.0),
        ("C2C1", "C2", "C1", 20.0),
    ]
    for eid, src, dst, speed in edges:
        net.add_edge(Edge(
            id=eid,
            from_node=src,
            to_node=dst,
            num_lanes=1,
            speed_limit=speed,
        ))
    return net


def _fast_network() -> Network:
    net = Network()
    for nid, x in [("A", 0.0), ("B", 100.0), ("C", 200.0), ("D", 300.0)]:
        net.add_node(Node(id=nid, x=x, y=0.0))

    for eid, src, dst in [("AB", "A", "B"), ("BA", "B", "A"),
                          ("BC", "B", "C"), ("CB", "C", "B"),
                          ("CD", "C", "D"), ("DC", "D", "C")]:
        net.add_edge(Edge(
            id=eid,
            from_node=src,
            to_node=dst,
            num_lanes=1,
            speed_limit=20.0,
        ))
    return net


def test_classify_nodes_uses_max_adjacent_speed():
    residential, commercial = classify_nodes(_commute_network())

    assert residential == ["R1"]
    assert set(commercial) == {"R2", "C1", "C2"}
    assert "ISO" not in residential
    assert "ISO" not in commercial


def test_classify_nodes_empty_network():
    assert classify_nodes(Network()) == ([], [])


def test_demand_factor_boundaries_and_wrap():
    assert demand_factor(0.0) == pytest.approx(0.15)
    assert demand_factor(6.0) == pytest.approx(1.5)
    assert demand_factor(7.0) == pytest.approx(3.0)
    assert demand_factor(8.99) == pytest.approx(3.0)
    assert demand_factor(9.0) == pytest.approx(1.2)
    assert demand_factor(10.0) == pytest.approx(0.6)
    assert demand_factor(14.99) == pytest.approx(0.6)
    assert demand_factor(15.0) == pytest.approx(1.3)
    assert demand_factor(16.0) == pytest.approx(2.5)
    assert demand_factor(18.99) == pytest.approx(2.5)
    assert demand_factor(19.0) == pytest.approx(1.1)
    assert demand_factor(20.0) == pytest.approx(0.15)
    assert demand_factor(24.0) == pytest.approx(0.15)
    assert demand_factor(25.0) == pytest.approx(0.15)


def test_generate_commute_demand_is_reproducible():
    net = _commute_network()

    first = generate_commute_demand(net, seed=3, peak_veh_hr=300.0, max_pairs=10)
    second = generate_commute_demand(net, seed=3, peak_veh_hr=300.0, max_pairs=10)

    assert first == second
    assert first


def test_generate_commute_demand_uses_residential_and_commercial_sets():
    net = _commute_network()

    demand = generate_commute_demand(net, seed=5, peak_veh_hr=300.0, max_pairs=10)
    total_flow = sum(sum(dests.values()) for dests in demand.values())

    assert set(demand) == {"R1"}
    assert all(dest in {"R2", "C1", "C2"} for dests in demand.values() for dest in dests)
    assert all(flow > 0.0 for dests in demand.values() for flow in dests.values())
    assert 0.5 * 300.0 <= total_flow <= 1.5 * 300.0


def test_generate_commute_demand_respects_max_pairs():
    net = _commute_network()

    demand = generate_commute_demand(net, seed=7, peak_veh_hr=300.0, max_pairs=2)

    assert 0 < sum(len(dests) for dests in demand.values()) <= 2


def test_generate_commute_demand_falls_back_when_classification_is_empty():
    net = _fast_network()

    demand = generate_commute_demand(net, seed=11, peak_veh_hr=200.0, max_pairs=6)

    assert demand
    assert all(origin in net.nodes for origin in demand)
    assert all(
        dest in net.nodes and dest != origin
        for origin, dests in demand.items()
        for dest in dests
    )
    assert sum(sum(dests.values()) for dests in demand.values()) > 0.0


def test_generate_commute_demand_empty_network():
    assert generate_commute_demand(Network()) == {}
