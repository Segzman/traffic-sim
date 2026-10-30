"""NetworkSimulation routing, spawning, and runtime tests."""
import random

import pytest

from engine.agents import Vehicle
from engine.config import SimConfig
from engine.network import Network, Node, Edge
from engine.network_simulation import (
    NetworkSimulation,
    _sample_disobey,
    _sample_lognormal,
)
from engine.signals import Phase, SignalPlan
from engine.weather import WeatherState


def _line_network(
    length: float = 100.0,
    speed: float = 10.0,
    num_lanes: int = 1,
) -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=length, y=0.0))
    net.add_edge(Edge(
        id="AB",
        from_node="A",
        to_node="B",
        num_lanes=num_lanes,
        speed_limit=speed,
        geometry=[[0.0, 0.0], [length, 0.0]],
    ))
    return net


def _manual_vehicle(position_s: float = 50.0, speed: float = 10.0) -> Vehicle:
    return Vehicle(
        id=0,
        lane_id=0,
        position_s=position_s,
        speed=speed,
        v0=15.0,
        s0=2.0,
        T=1.5,
        a_max=1.4,
        b=2.0,
        current_edge="AB",
        route=["AB"],
        route_index=0,
    )


def test_set_speed_mult_adapts_dt_and_config():
    cfg = SimConfig()
    sim = NetworkSimulation(
        network=_line_network(),
        demand={},
        duration=10.0,
        config=cfg,
        seed=1,
    )

    assert sim.dt == pytest.approx(0.1)

    sim.set_speed_mult(8.0)
    assert sim.dt == pytest.approx(0.1)
    assert cfg.speed_mult == pytest.approx(8.0)

    sim.set_speed_mult(32.0)
    assert sim.dt == pytest.approx(0.5)
    assert cfg.speed_mult == pytest.approx(32.0)

    sim.set_speed_mult(1.0)
    assert sim.dt == pytest.approx(0.1)


def test_spawn_queue_is_sorted_poisson_flow():
    sim = NetworkSimulation(
        network=_line_network(length=500.0, speed=15.0),
        demand={"A": {"B": 360.0}},
        duration=3600.0,
        seed=42,
    )

    events = list(sim._spawn_queue)

    assert len(events) == pytest.approx(360, rel=0.2)
    assert [event[0] for event in events] == sorted(event[0] for event in events)
    assert all(0.0 <= t < 3600.0 for t, _origin, _dest in events)
    assert {(origin, dest) for _t, origin, dest in events} == {("A", "B")}


def test_demand_multiplier_rebuilds_future_events():
    sim = NetworkSimulation(
        network=_line_network(length=500.0, speed=15.0),
        demand={"A": {"B": 360.0}},
        duration=3600.0,
        seed=7,
    )
    base_count = len(sim._spawn_queue)
    assert base_count > 0

    sim.set_demand_mult(2.0)
    assert len(sim._spawn_queue) > base_count

    sim.set_demand_mult(0.0)
    assert len(sim._spawn_queue) == 0


def test_spawn_vehicle_assigns_route_and_edge_state():
    net = _line_network(length=500.0, speed=15.0)
    sim = NetworkSimulation(network=net, demand={}, duration=60.0, seed=2)

    sim._spawn_vehicle("A", "B")

    assert len(sim.vehicles) == 1
    veh = sim.vehicles[0]
    assert veh.route == ["AB"]
    assert veh.route_index == 0
    assert veh.current_edge == "AB"
    assert veh.position_s == pytest.approx(1.0)
    assert veh.lane_id == 0
    assert veh.speed > 0.0
    assert veh.v0 > 0.0
    assert veh.entry_time == sim.time
    assert veh in sim._edge_vehicles["AB"]


def test_spawn_skips_short_od_pair():
    net = _line_network(length=50.0, speed=10.0)
    sim = NetworkSimulation(network=net, demand={}, duration=60.0, seed=2)

    sim._spawn_vehicle("A", "B")

    assert sim.vehicles == []
    assert sim._next_vid == 0


def test_spawn_skips_unreachable_od_pair():
    net = _line_network(length=500.0, speed=15.0)
    sim = NetworkSimulation(network=net, demand={}, duration=60.0, seed=2)

    sim._spawn_vehicle("B", "A")

    assert sim.vehicles == []
    assert sim._next_vid == 0


def test_entry_blocked_is_lane_specific():
    net = _line_network(length=500.0, speed=15.0, num_lanes=2)
    sim = NetworkSimulation(network=net, demand={}, duration=60.0, seed=0)

    blocker = Vehicle(
        id=1,
        lane_id=0,
        position_s=1.0,
        speed=0.0,
        length=4.5,
        s0=2.0,
        current_edge="AB",
        route=["AB"],
    )
    mover = Vehicle(
        id=2,
        lane_id=1,
        position_s=0.0,
        speed=5.0,
        length=4.5,
        s0=2.0,
        current_edge="AB",
        route=["AB"],
    )
    sim._edge_vehicles["AB"] = [blocker]

    assert sim._entry_blocked("AB", 0, mover) is True
    assert sim._entry_blocked("AB", 1, mover) is False


def test_red_signal_brakes_front_vehicle_near_stop_line():
    net = _line_network(length=100.0, speed=15.0)
    red_plan = SignalPlan(
        node_id="B",
        phases=[Phase(
            green_movements=[],
            green_duration=0.0,
            yellow_duration=0.0,
            all_red_duration=1.0,
        )],
    )
    sim = NetworkSimulation(
        network=net,
        demand={},
        duration=10.0,
        seed=0,
        signal_plans={"B": red_plan},
    )
    veh = _manual_vehicle(position_s=50.0, speed=10.0)
    sim.vehicles = [veh]
    sim._edge_vehicles["AB"] = [veh]

    sim._step_idm_vectorised()

    assert veh.acceleration < 0.0


def test_green_signal_lets_front_vehicle_accelerate():
    net = _line_network(length=100.0, speed=15.0)
    sim = NetworkSimulation(network=net, demand={}, duration=10.0, seed=0)
    veh = _manual_vehicle(position_s=50.0, speed=10.0)
    sim.vehicles = [veh]
    sim._edge_vehicles["AB"] = [veh]

    sim._step_idm_vectorised()

    assert veh.acceleration > 0.0


def test_weather_reduces_idm_acceleration():
    net = _line_network(length=500.0, speed=15.0)
    clear_sim = NetworkSimulation(
        network=net,
        demand={},
        duration=10.0,
        seed=0,
        weather=WeatherState("clear", 0.0),
    )
    snow_sim = NetworkSimulation(
        network=net,
        demand={},
        duration=10.0,
        seed=0,
        weather=WeatherState("snow", 0.0),
    )

    clear_veh = _manual_vehicle(position_s=1.0, speed=0.0)
    snow_veh = _manual_vehicle(position_s=1.0, speed=0.0)
    clear_sim.vehicles = [clear_veh]
    clear_sim._edge_vehicles["AB"] = [clear_veh]
    snow_sim.vehicles = [snow_veh]
    snow_sim._edge_vehicles["AB"] = [snow_veh]

    clear_sim._step_idm_vectorised()
    snow_sim._step_idm_vectorised()

    assert snow_veh.acceleration < clear_veh.acceleration


def test_run_completes_trips_and_reports_metrics():
    net = _line_network(length=100.0, speed=10.0)
    sim = NetworkSimulation(
        network=net,
        demand={"A": {"B": 1800.0}},
        duration=60.0,
        seed=42,
    )

    metrics = sim.run()

    assert metrics["trips_completed"] > 0
    assert metrics["throughput"] == pytest.approx(
        metrics["trips_completed"] / 60.0
    )
    assert len(metrics["trip_log"]) == metrics["trips_completed"]
    assert all(record.delay_s >= 0.0 for record in metrics["trip_log"])
    assert all(
        record.exit_time >= record.entry_time
        for record in metrics["trip_log"]
    )


def test_temporal_demand_peak_has_more_events_than_overnight():
    net = _line_network(length=500.0, speed=15.0)
    peak = NetworkSimulation(
        network=net,
        demand={"A": {"B": 3600.0}},
        duration=3600.0,
        seed=1,
        start_hour=8.0,
        temporal_demand=True,
    )
    overnight = NetworkSimulation(
        network=net,
        demand={"A": {"B": 3600.0}},
        duration=3600.0,
        seed=1,
        start_hour=2.0,
        temporal_demand=True,
    )

    assert len(peak._spawn_queue) > 5 * len(overnight._spawn_queue)


def test_sample_lognormal_respects_bounds_and_mean():
    rng = random.Random(42)
    samples = [
        _sample_lognormal(rng, mean=1.4, cv=0.20, lo=0.56, hi=3.5)
        for _ in range(5000)
    ]

    assert all(0.56 <= value <= 3.5 for value in samples)
    assert sum(samples) / len(samples) == pytest.approx(1.4, rel=0.05)


def test_sample_disobey_is_bounded_and_mostly_polite():
    rng = random.Random(42)
    samples = [_sample_disobey(rng, 1.0) for _ in range(5000)]

    assert all(0.0 <= value <= 1.0 for value in samples)
    assert 0.15 <= sum(samples) / len(samples) <= 0.35
    assert sum(value <= 0.35 for value in samples) / len(samples) > 0.55
    assert _sample_disobey(rng, 0.0) == 0.0
