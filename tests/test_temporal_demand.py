"""Temporal demand profile should reduce midnight surge."""
from __future__ import annotations

from engine.network import Network, Node, Edge
from engine.network_simulation import NetworkSimulation


def _net() -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=1000.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1, speed_limit=14.0,
                      geometry=[[0.0, 0.0], [1000.0, 0.0]]))
    return net


def test_midnight_temporal_demand_lower_than_peak_hour():
    net = _net()
    base = {"A": {"B": 3600.0}}

    sim_midnight = NetworkSimulation(
        network=net,
        demand=base,
        duration=3600.0,
        seed=1,
        temporal_demand=True,
        start_hour=0.0,
    )
    sim_peak = NetworkSimulation(
        network=net,
        demand=base,
        duration=3600.0,
        seed=1,
        temporal_demand=True,
        start_hour=8.0,
    )

    n_mid = len(sim_midnight._spawn_queue)
    n_peak = len(sim_peak._spawn_queue)

    assert n_mid < n_peak * 0.08, f"Midnight demand should be much lower ({n_mid} vs {n_peak})"
