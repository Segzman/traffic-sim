"""Milestone 1 — simulation integration tests."""
import json

import pytest

from engine.agents import Vehicle
from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(vehicles: list[Vehicle], road_length: float = 10_000.0,
             speed_limit: float = 30.0, seed: int = 42) -> Simulation:
    scenario = {
        "road": {"length": road_length, "speed_limit": speed_limit},
        "duration": 60.0,
        "seed": seed,
    }
    return Simulation(scenario, vehicles=vehicles)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_vehicle_reaches_desired_speed():
    """One vehicle on empty lane reaches 95% of v0 within 30 simulated seconds."""
    v0 = 30.0
    veh = Vehicle(id=0, lane_id=0, position_s=0.0, speed=0.0, v0=v0,
                  a_max=1.4, b=2.0)
    sim = make_sim([veh])

    # Run for 30 seconds
    steps = int(30.0 / sim.dt)
    for _ in range(steps):
        sim.step()

    assert veh.speed >= 0.95 * v0, (
        f"Expected speed >= {0.95 * v0:.1f} m/s after 30s, got {veh.speed:.2f} m/s"
    )


def test_following_vehicle_maintains_gap():
    """Follower's gap to leader never drops below s0 after the initial transient."""
    s0 = 2.0
    leader = Vehicle(id=0, lane_id=0, position_s=200.0, speed=20.0,
                     v0=30.0, s0=s0, T=1.5, a_max=1.4, b=2.0)
    follower = Vehicle(id=1, lane_id=0, position_s=100.0, speed=20.0,
                       v0=30.0, s0=s0, T=1.5, a_max=1.4, b=2.0)
    sim = make_sim([leader, follower])

    transient_steps = 100  # skip first 10 s
    total_steps = 600      # 60 s total

    min_gap = float("inf")
    for step in range(total_steps):
        sim.step()
        if step >= transient_steps:
            gap = leader.position_s - follower.position_s - follower.length
            if gap < min_gap:
                min_gap = gap

    assert min_gap >= s0 - 0.5, (
        f"Minimum gap {min_gap:.3f} m fell below s0 ({s0} m) after transient"
    )


def test_shockwave_propagation():
    """Sudden leader brake produces a rearward speed-reduction wave."""
    n = 10
    spacing = 30.0  # m between vehicle fronts (gap ≈ 25.5 m)
    initial_speed = 25.0

    # Front vehicle at n*spacing, rear vehicle at spacing
    vehicles = [
        Vehicle(id=i, lane_id=0, position_s=(n - i) * spacing,
                speed=initial_speed, v0=25.0, s0=2.0, T=1.5, a_max=1.4, b=2.0)
        for i in range(n)
    ]
    sim = make_sim(vehicles, road_length=50_000.0)

    # Reach approximate steady state
    for _ in range(200):  # 20 s
        sim.step()

    # Record rear-half speed before braking
    sorted_vehs = sorted(sim.vehicles, key=lambda v: v.position_s, reverse=True)
    rear_half = sorted_vehs[n // 2:]
    avg_speed_before = sum(v.speed for v in rear_half) / len(rear_half)

    # Force leader to decelerate (dramatically lower desired speed)
    front_vehicle = sorted_vehs[0]
    front_vehicle.v0 = 1.0

    # Run 30 more seconds — enough for wave to propagate through all 9 gaps
    for _ in range(300):
        sim.step()

    sorted_vehs = sorted(sim.vehicles, key=lambda v: v.position_s, reverse=True)
    rear_half = sorted_vehs[n // 2:]
    avg_speed_after = sum(v.speed for v in rear_half) / len(rear_half)

    assert avg_speed_after < avg_speed_before, (
        f"Rear-half speed did not decrease after leader brake: "
        f"before={avg_speed_before:.2f}, after={avg_speed_after:.2f}"
    )


def test_metric_throughput_nonzero():
    """After 60 s with 10 vehicles, throughput metric > 0."""
    # Place all 10 vehicles in the first half of the road so they will
    # cross the measurement point (90% of road length = 900 m) during the run.
    n = 10
    road_length = 1000.0
    spacing = (road_length * 0.8) / n  # 80 m apart, up to 800 m

    vehicles = [
        Vehicle(id=i, lane_id=0, position_s=(i + 1) * spacing,
                speed=20.0, v0=30.0)
        for i in range(n)
    ]
    scenario = {
        "road": {"length": road_length, "speed_limit": 30.0},
        "duration": 60.0,
        "seed": 42,
    }
    sim = Simulation(scenario, vehicles=vehicles)
    metrics = sim.run()

    assert metrics["throughput"] > 0, (
        f"Expected throughput > 0, got {metrics['throughput']}"
    )


def test_deterministic_with_seed():
    """Two runs with identical seed produce byte-identical metric outputs."""
    scenario = {
        "road": {"length": 1000.0, "speed_limit": 30.0},
        "vehicles": {
            "count": 10,
            "initial_speed": 20.0,
            "idm_params": {"v0": 30.0, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0},
        },
        "duration": 60.0,
        "seed": 42,
    }

    metrics1 = Simulation(scenario).run()
    metrics2 = Simulation(scenario).run()

    assert metrics1 == metrics2, (
        f"Runs produced different metrics:\n  run1={metrics1}\n  run2={metrics2}"
    )
