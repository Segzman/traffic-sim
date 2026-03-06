"""Milestone 5 — Routing (Dijkstra shortest_path) unit tests."""
import pytest

from engine.network import Network, Node, Edge


# ------------------------------------------------------------------ #
# Helpers to build small test networks
# ------------------------------------------------------------------ #

def _line_network() -> Network:
    """Linear 3-node network: A --(e1)--> B --(e2)--> C."""
    net = Network()
    for nid, x in [("A", 0.0), ("B", 100.0), ("C", 200.0)]:
        net.add_node(Node(id=nid, x=x, y=0.0))
    net.add_edge(Edge(id="e1", from_node="A", to_node="B",
                      num_lanes=1, speed_limit=10.0,
                      geometry=[(0.0, 0.0), (100.0, 0.0)]))
    net.add_edge(Edge(id="e2", from_node="B", to_node="C",
                      num_lanes=1, speed_limit=10.0,
                      geometry=[(100.0, 0.0), (200.0, 0.0)]))
    return net


def _diamond_network() -> Network:
    """Diamond network with a fast upper branch and slow lower branch.

    A --[fast_top, 100 m @ 20 m/s]--> B --[e3, 100 m @ 20 m/s]--> D
    A --[slow_bot, 100 m @  5 m/s]--> C --[e4, 100 m @  5 m/s]--> D
    """
    net = Network()
    for nid, x, y in [
        ("A", 0.0, 0.0), ("B", 100.0, 50.0),
        ("C", 100.0, -50.0), ("D", 200.0, 0.0)
    ]:
        net.add_node(Node(id=nid, x=x, y=y))

    net.add_edge(Edge(id="fast_top", from_node="A", to_node="B",
                      num_lanes=1, speed_limit=20.0,
                      geometry=[(0.0, 0.0), (100.0, 50.0)]))
    net.add_edge(Edge(id="slow_bot", from_node="A", to_node="C",
                      num_lanes=1, speed_limit=5.0,
                      geometry=[(0.0, 0.0), (100.0, -50.0)]))
    net.add_edge(Edge(id="e3", from_node="B", to_node="D",
                      num_lanes=1, speed_limit=20.0,
                      geometry=[(100.0, 50.0), (200.0, 0.0)]))
    net.add_edge(Edge(id="e4", from_node="C", to_node="D",
                      num_lanes=1, speed_limit=5.0,
                      geometry=[(100.0, -50.0), (200.0, 0.0)]))
    return net


# ------------------------------------------------------------------ #
# test_shortest_path_simple
# ------------------------------------------------------------------ #

def test_shortest_path_simple():
    """Dijkstra returns the only path through a linear network."""
    net = _line_network()
    path = net.shortest_path("A", "C")

    assert path == ["e1", "e2"], f"Expected ['e1', 'e2']; got {path}"


def test_shortest_path_same_node():
    """Path from a node to itself is an empty list."""
    net = _line_network()
    path = net.shortest_path("A", "A")
    assert path == [], f"Expected []; got {path}"


def test_shortest_path_no_path():
    """Returns empty list when destination is unreachable."""
    net = _line_network()
    # Add an isolated node with no inbound edges
    net.add_node(Node(id="Z", x=999.0, y=0.0))
    path = net.shortest_path("A", "Z")
    assert path == [], f"Expected []; got {path}"


def test_shortest_path_unknown_node():
    """Raises KeyError for unknown node IDs."""
    net = _line_network()
    with pytest.raises(KeyError):
        net.shortest_path("A", "UNKNOWN")


# ------------------------------------------------------------------ #
# test_shortest_path_avoids_slow_edge
# ------------------------------------------------------------------ #

def test_shortest_path_avoids_slow_edge():
    """Dijkstra with travel_time weight prefers the faster upper path."""
    net = _diamond_network()
    path = net.shortest_path("A", "D", weight="travel_time")

    assert "fast_top" in path, f"Expected fast_top in path; got {path}"
    assert "slow_bot" not in path, f"slow_bot should not be in path; got {path}"


def test_shortest_path_length_weight():
    """With 'length' weight, both paths are the same distance → either valid."""
    net = _diamond_network()
    path = net.shortest_path("A", "D", weight="length")

    # Both paths are approximately equal in Euclidean length
    # (upper: sqrt(100²+50²)*2 ≈ 223 m; lower: same)
    # Dijkstra will pick one; just verify path reaches D
    assert len(path) == 2, f"Expected 2-edge path; got {path}"
    last_edge = net.edges[path[-1]]
    assert last_edge.to_node == "D"


# ------------------------------------------------------------------ #
# test_vehicle_follows_route
# ------------------------------------------------------------------ #

def test_vehicle_follows_route():
    """Vehicle spawned on A→C route advances its route_index as it crosses edges."""
    from engine.network_simulation import NetworkSimulation

    net = _line_network()

    # Override edge lengths for a short test (100 m each, speed 10 m/s → 10 s per edge)
    sim = NetworkSimulation(
        network=net,
        demand={"A": {"C": 7200}},   # high demand → spawn immediately
        duration=30.0,
        seed=1,
    )

    # Step for 5 s — vehicle should have spawned and be partway along first edge
    for _ in range(50):
        sim.step()

    all_vids = [v for v in sim.vehicles]
    assert len(all_vids) > 0 or sim._next_vid > 0, (
        "At least one vehicle should have been spawned"
    )

    # After 25 s (250 steps), vehicles should have crossed both edges
    for _ in range(200):
        sim.step()

    # Trip log should contain completed trips (A→C = two edges, total 200 m / 10 m/s = 20 s)
    assert len(sim.trip_log) > 0, (
        "Vehicles should have completed the A→C route and logged trips"
    )

    # Each trip should have entry_time < exit_time
    for rec in sim.trip_log:
        assert rec.exit_time > rec.entry_time, (
            f"exit_time ({rec.exit_time}) should be > entry_time ({rec.entry_time})"
        )
