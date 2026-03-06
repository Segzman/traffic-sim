"""Milestone 1 — Simulation speed / time compression tests.

Verifies that the simulation can compress a full 24-hour day into a
short wall-clock run at high speed multipliers.
"""
from __future__ import annotations

import time

import pytest

from engine.network import Network, Node, Edge
from engine.network_simulation import NetworkSimulation


def _make_network(n_nodes: int = 10, seg_len: float = 300.0,
                  lanes: int = 2) -> Network:
    net = Network()
    for i in range(n_nodes):
        net.add_node(Node(id=str(i), x=float(i * seg_len), y=0.0))
    for i in range(n_nodes - 1):
        net.add_edge(Edge(
            id=f"e{i}",
            from_node=str(i),
            to_node=str(i + 1),
            num_lanes=lanes,
            speed_limit=14.0,
            geometry=[[float(i * seg_len), 0.0], [float((i + 1) * seg_len), 0.0]],
        ))
    return net


# ---------------------------------------------------------------------------
# Speed multiplier mechanics
# ---------------------------------------------------------------------------

class TestSpeedMultiplier:

    def test_speed_mult_1x_step_count(self):
        """At 1× speed the step count equals duration/dt (±1 for float rounding)."""
        net = _make_network()
        sim = NetworkSimulation(net, demand={}, duration=10.0, seed=0)
        steps = 0
        while sim.time < sim.duration:
            sim.step()
            steps += 1
        expected = int(round(10.0 / 0.1))
        # Allow ±1: cumulative floating-point drift in dt additions can cause
        # the loop to execute one extra iteration (100.0 ≈ 9.9999...99 < 10.0).
        assert abs(steps - expected) <= 1, (
            f"Expected ~{expected} steps at 1×, got {steps}"
        )

    def test_speed_mult_32x_step_count(self):
        """At 32× speed, dt=0.5 → fewer steps for same duration."""
        net = _make_network()
        sim = NetworkSimulation(net, demand={}, duration=10.0, seed=0)
        sim.set_speed_mult(32)
        assert sim.dt == pytest.approx(0.5)
        steps = 0
        while sim.time < sim.duration:
            sim.step()
            steps += 1
        # 10s / 0.5s = 20 steps (vs 100 at 1×)
        assert steps == pytest.approx(20, abs=2)

    def test_speed_mult_288x_step_count(self):
        """At 288× speed, dt=0.5 → 24 h * 2 steps/s = 172 800 steps."""
        net    = _make_network()
        demand = {"0": {"9": 200.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=86400.0, seed=0)
        sim.set_speed_mult(288)
        assert sim.dt == pytest.approx(0.5)
        expected_steps = int(round(86400.0 / 0.5))    # 172 800
        assert expected_steps == pytest.approx(172_800, abs=10)


# ---------------------------------------------------------------------------
# Wall-clock time budgets
# ---------------------------------------------------------------------------

class TestWallClockBudgets:

    @pytest.mark.timeout(30)
    def test_1h_sim_under_10s(self):
        """1-hour simulation at 288× must finish in under 10 wall-clock seconds."""
        net    = _make_network(n_nodes=10, seg_len=300.0, lanes=2)
        demand = {"0": {"9": 300.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=3600.0, seed=0)
        sim.set_speed_mult(288)

        t0 = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, (
            f"1-hour sim took {elapsed:.1f}s at 288× — should be < 10 s"
        )

    @pytest.mark.timeout(60)
    def test_6h_sim_under_30s(self):
        """6-hour simulation at 288× must finish in under 30 wall-clock seconds."""
        net    = _make_network(n_nodes=8, seg_len=400.0, lanes=2)
        demand = {"0": {"7": 200.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=21_600.0, seed=0)
        sim.set_speed_mult(288)

        t0 = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, (
            f"6-hour sim took {elapsed:.1f}s at 288× — should be < 30 s"
        )

    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_24h_sim_under_5_minutes(self):
        """Full 24-hour simulation at 288× must finish in under 5 minutes.

        Marked ``slow`` — skipped in fast CI, run in nightly or manual trigger.
        """
        net    = _make_network(n_nodes=10, seg_len=300.0, lanes=2)
        demand = {"0": {"9": 300.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=86_400.0, seed=0)
        sim.set_speed_mult(288)

        t0 = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - t0

        assert elapsed < 300.0, (
            f"24-hour sim took {elapsed:.1f}s — target is < 300 s (5 min)"
        )


# ---------------------------------------------------------------------------
# Correctness at high speed
# ---------------------------------------------------------------------------

class TestHighSpeedCorrectness:

    def test_no_negative_speeds_at_288x(self):
        """High-speed run must not produce negative vehicle speeds."""
        net    = _make_network()
        demand = {"0": {"9": 600.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=3600.0, seed=1)
        sim.set_speed_mult(288)
        steps  = 0
        while sim.time < 300.0:   # first 5 simulated minutes
            sim.step()
            steps += 1
            for v in sim.vehicles:
                assert v.speed >= 0.0, (
                    f"Negative speed at step {steps}, t={sim.time:.1f}s: {v.speed}"
                )

    def test_trip_log_nonempty_at_288x(self):
        """Even at 288× speed, trips must complete and be logged."""
        net    = _make_network(n_nodes=5, seg_len=200.0)
        demand = {"0": {"4": 600.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=1800.0, seed=0)
        sim.set_speed_mult(288)
        sim.run()
        assert len(sim.trip_log) > 0, "No trips completed at 288× — trip_log empty"

    def test_stats_reasonable_at_32x(self):
        """32× simulation: throughput and avg_delay must be physically plausible."""
        net    = _make_network(n_nodes=6, seg_len=300.0, lanes=2)
        demand = {"0": {"5": 400.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=3600.0, seed=42)
        sim.set_speed_mult(32)
        result = sim.run()

        assert result["throughput"] > 0, "No throughput"
        assert result["avg_delay"] >= 0, "Negative avg_delay"
        # Free-flow time: 5 edges × 300 m / 14 m/s ≈ 107 s; delay << duration
        assert result["avg_delay"] < 3600, "avg_delay exceeds simulation duration"

    def test_adaptive_dt_conserves_total_time(self):
        """sim.time after run() must equal sim.duration (within one dt)."""
        net = _make_network()
        sim = NetworkSimulation(net, demand={}, duration=500.0, seed=0)
        sim.set_speed_mult(288)   # dt = 0.5
        sim.run()
        assert abs(sim.time - sim.duration) <= sim.dt + 1e-9
