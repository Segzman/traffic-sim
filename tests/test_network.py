"""Milestone 2 — network graph and multi-lane integration tests."""
import pytest

from engine.network import Network, Node, Edge, Lane
from engine.agents import Vehicle
from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Network / graph tests
# ---------------------------------------------------------------------------

def test_graph_construction():
    """3 nodes, 4 edges: Network builds correct adjacency list."""
    net = Network()
    for nid, x in [("A", 0), ("B", 100), ("C", 200)]:
        net.add_node(Node(id=nid, x=x, y=0.0))

    for eid, src, dst in [("AB", "A", "B"), ("BA", "B", "A"),
                           ("BC", "B", "C"), ("CB", "C", "B")]:
        net.add_edge(Edge(id=eid, from_node=src, to_node=dst, num_lanes=1, speed_limit=30.0))

    adj = net.adjacency()
    assert set(adj["A"]) == {"AB"}
    assert set(adj["B"]) == {"BA", "BC"}
    assert set(adj["C"]) == {"CB"}
    assert len(net.nodes) == 3
    assert len(net.edges) == 4


def test_lane_offset_geometry():
    """Lane boundary offsets satisfy: lane-0 right = -(total_w/2), lane-N right edge OK."""
    net = Network()
    net.add_node(Node(id="A", x=0, y=0))
    net.add_node(Node(id="B", x=500, y=0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=3, speed_limit=30.0))

    lanes = net.get_lanes_for_edge("AB")
    lane0 = next(la for la in lanes if la.lane_index == 0)
    lane2 = next(la for la in lanes if la.lane_index == 2)

    expected_total_width = 3 * 3.5  # 10.5 m
    half = expected_total_width / 2.0  # 5.25 m

    # Right edge of rightmost lane = -half
    assert abs(net.lane_right_edge_offset(lane0) - (-half)) < 0.1
    # Left edge of leftmost lane = +half
    assert abs(net.lane_left_edge_offset(lane2) - half) < 0.1


def test_multilane_throughput_vs_single():
    """3-lane segment with equivalent per-lane demand produces ≥ 2.5× single-lane throughput."""
    base_count = 20
    road_length = 500.0
    duration = 120.0
    idm = {"v0": 30.0, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0}

    single = Simulation({
        "road": {"length": road_length, "num_lanes": 1, "speed_limit": 30.0},
        "vehicles": {"count": base_count, "initial_speed": 20.0, "idm_params": idm},
        "duration": duration, "seed": 42,
    }).run()

    multi = Simulation({
        "road": {"length": road_length, "num_lanes": 3, "speed_limit": 30.0},
        "vehicles": {"count": base_count * 3, "initial_speed": 20.0, "idm_params": idm},
        "duration": duration, "seed": 42,
    }).run()

    assert multi["throughput"] >= 2.5 * single["throughput"], (
        f"3-lane throughput {multi['throughput']:.3f} < 2.5 × single-lane "
        f"{single['throughput']:.3f}"
    )


def test_heterogeneous_population_variance():
    """500 vehicles with Gaussian v0: sample SD within 10% of specified population SD."""
    target_mean = 30.0
    target_sd = 3.0
    n = 500

    sim = Simulation({
        "road": {"length": 25000.0, "num_lanes": 1, "speed_limit": 35.0},
        "vehicles": {
            "count": n,
            "initial_speed": 20.0,
            "idm_params": {
                "v0": {"mean": target_mean, "sd": target_sd},
                "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0,
            },
        },
        "duration": 0.1,
        "seed": 42,
    })

    v0_values = [v.v0 for v in sim.vehicles]
    mean_v0 = sum(v0_values) / len(v0_values)
    sd_v0 = (sum((x - mean_v0) ** 2 for x in v0_values) / n) ** 0.5

    assert abs(sd_v0 - target_sd) / target_sd < 0.10, (
        f"v0 SD {sd_v0:.3f} more than 10% away from target {target_sd}"
    )
