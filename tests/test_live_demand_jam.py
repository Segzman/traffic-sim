"""Live-mode demand continuity + jam no-overlap safeguards."""
from __future__ import annotations

from engine.agents import Vehicle
from engine.network import Network, Node, Edge
from engine.network_simulation import NetworkSimulation


def _two_node_net(length: float = 500.0, speed: float = 12.0) -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=length, y=0.0))
    net.add_edge(Edge(
        id="AB",
        from_node="A",
        to_node="B",
        num_lanes=1,
        speed_limit=speed,
        geometry=[[0.0, 0.0], [length, 0.0]],
    ))
    return net


def test_continuous_demand_extends_past_duration():
    net = _two_node_net()
    sim = NetworkSimulation(
        network=net,
        demand={"A": {"B": 600.0}},
        duration=60.0,
        seed=1,
        continuous_demand=True,
    )

    # Initial queue should cover well beyond the nominal 60 s duration.
    assert sim._spawn_scheduled_until >= 3600.0

    # Simulate time advancing to 08:00-equivalent horizon and ensure queue extends.
    sim.time = 3700.0
    while sim._spawn_queue and sim._spawn_queue[0][0] <= sim.time:
        sim._spawn_queue.popleft()

    prev_end = sim._spawn_scheduled_until
    sim._extend_spawn_horizon_if_needed()

    assert sim._spawn_scheduled_until > prev_end
    assert sim._spawn_queue, "Expected future spawn events after horizon extension"
    assert sim._spawn_queue[0][0] > sim.time


def test_no_overlap_clamps_jammed_vehicles():
    net = _two_node_net(length=300.0, speed=8.0)
    sim = NetworkSimulation(network=net, demand={}, duration=120.0, seed=0)

    leader = Vehicle(
        id=1,
        lane_id=0,
        position_s=40.0,
        speed=0.5,
        current_edge="AB",
        route=["AB"],
        route_index=0,
        length=4.5,
        s0=2.0,
    )
    follower = Vehicle(
        id=2,
        lane_id=0,
        position_s=39.5,   # overlapping before clamp
        speed=2.0,
        current_edge="AB",
        route=["AB"],
        route_index=0,
        length=4.5,
        s0=2.0,
    )

    sim.vehicles = [leader, follower]
    sim._edge_vehicles["AB"] = [leader, follower]

    sim._enforce_no_overlap()

    min_allowed = leader.position_s - follower.length - follower.s0
    assert follower.position_s <= min_allowed + 1e-9
    assert follower.speed <= leader.speed


def test_edge_transition_blocks_when_next_entry_is_occupied():
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=20.0, y=0.0))
    net.add_node(Node(id="C", x=40.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1, speed_limit=8.0,
                      geometry=[[0.0, 0.0], [20.0, 0.0]]))
    net.add_edge(Edge(id="BC", from_node="B", to_node="C", num_lanes=1, speed_limit=8.0,
                      geometry=[[20.0, 0.0], [40.0, 0.0]]))
    sim = NetworkSimulation(network=net, demand={}, duration=120.0, seed=0)

    blocker = Vehicle(
        id=10,
        lane_id=0,
        position_s=1.0,   # blocks BC entry
        speed=0.0,
        current_edge="BC",
        route=["BC"],
        route_index=0,
        length=4.5,
        s0=2.0,
    )
    mover = Vehicle(
        id=11,
        lane_id=0,
        position_s=19.8,  # crosses AB edge end this step
        speed=5.0,
        acceleration=0.0,
        current_edge="AB",
        route=["AB", "BC"],
        route_index=0,
        length=4.5,
        s0=2.0,
    )

    sim.vehicles = [blocker, mover]
    sim._edge_vehicles["AB"] = [mover]
    sim._edge_vehicles["BC"] = [blocker]

    sim._step_integrate()

    assert mover.current_edge == "AB", "Vehicle should wait on AB when BC entry is blocked"
    assert mover.position_s <= sim.network.edge_length("AB")
    assert mover.speed == 0.0
