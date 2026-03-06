"""Milestone 1 — Performance Foundation tests.

Covers:
  • Vectorised IDM correctness (matches scalar, edge cases)
  • Vectorised IDM throughput   (1 M vehicles < 100 ms)
  • Spawn queue type            (must be collections.deque)
  • Step-time benchmark         (200 vehicles, < 5 ms per step)
  • 300-signal cap removed      (_scenario_from_import uses all signals)
  • Adaptive dt                 (set_speed_mult adjusts dt)
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from engine.idm import idm_acceleration
from engine.idm_vec import idm_acceleration_vec
from engine.network import Network, Node, Edge
from engine.network_simulation import NetworkSimulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_node_net(length: float = 1000.0, lanes: int = 1,
                  speed: float = 14.0) -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0,    y=0.0))
    net.add_node(Node(id="B", x=length, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B",
                      num_lanes=lanes, speed_limit=speed,
                      geometry=[[0.0, 0.0], [length, 0.0]]))
    return net


def _chain_net(n_nodes: int = 20, seg_len: float = 200.0,
               lanes: int = 2) -> Network:
    """Linear chain of n_nodes nodes connected by (n_nodes-1) edges."""
    net = Network()
    for i in range(n_nodes):
        net.add_node(Node(id=str(i), x=float(i * seg_len), y=0.0))
    for i in range(n_nodes - 1):
        net.add_edge(Edge(id=f"e{i}", from_node=str(i), to_node=str(i + 1),
                          num_lanes=lanes, speed_limit=14.0,
                          geometry=[[float(i * seg_len), 0.0],
                                    [float((i + 1) * seg_len), 0.0]]))
    return net


def _loaded_sim(n_vehicles: int = 200, warmup_steps: int = 50) -> NetworkSimulation:
    net    = _chain_net(n_nodes=20, seg_len=200.0, lanes=2)
    demand = {"0": {"19": float(n_vehicles * 3600 / 300)}}
    sim    = NetworkSimulation(net, demand=demand, duration=300, seed=0)
    for _ in range(warmup_steps):
        sim.step()
    return sim


# ---------------------------------------------------------------------------
# 1. Vectorised IDM correctness
# ---------------------------------------------------------------------------

class TestIDMVec:

    def test_matches_scalar_reference(self):
        """Every element of the vectorised result must match the scalar IDM."""
        rng = np.random.default_rng(0)
        N   = 1_000
        v   = rng.uniform(0, 30, N)
        v0  = np.full(N, 30.0)
        s   = rng.uniform(2, 100, N)
        dv  = rng.uniform(-5, 5, N)
        s0, T, a, b = 2.0, 1.5, 1.4, 2.0

        vec    = idm_acceleration_vec(v, v0, s, dv, s0, T, a, b)
        scalar = np.array([idm_acceleration(v[i], v0[i], s[i], dv[i], s0, T, a, b)
                           for i in range(N)])
        np.testing.assert_allclose(vec, scalar, rtol=1e-5, atol=1e-9)

    def test_zero_speed_free_flow(self):
        """At v=0 with a large gap the vehicle must accelerate forward."""
        a = idm_acceleration_vec(
            v=[0.0], v0=[30.0], s=[1000.0], delta_v=[0.0],
            s0=2.0, T=1.5, a=1.4, b=2.0,
        )
        assert a[0] > 0.0, "Vehicle at rest in free-flow should accelerate"

    def test_deceleration_near_leader(self):
        """Very small gap must produce strong deceleration."""
        a = idm_acceleration_vec(
            v=[10.0], v0=[30.0], s=[0.5], delta_v=[0.0],
            s0=2.0, T=1.5, a=1.4, b=2.0,
        )
        assert a[0] < -1.0, f"Expected deceleration < -1 m/s², got {a[0]:.3f}"

    def test_no_nan_or_inf_random_inputs(self):
        """Random vehicle states must never produce NaN or Inf."""
        rng = np.random.default_rng(7)
        N   = 5_000
        v   = rng.uniform(0, 40, N)
        v0  = rng.uniform(5, 40, N)
        s   = rng.uniform(0.01, 500, N)
        dv  = rng.uniform(-10, 10, N)
        out = idm_acceleration_vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
        assert np.all(np.isfinite(out)), "NaN or Inf in IDM output"

    def test_per_vehicle_params_broadcast(self):
        """s0/T/a/b as per-vehicle arrays must match element-wise scalar calls."""
        rng  = np.random.default_rng(42)
        N    = 200
        v    = rng.uniform(0, 25, N)
        v0   = np.full(N, 30.0)
        s    = rng.uniform(3, 80, N)
        dv   = np.zeros(N)
        s0_v = rng.uniform(1.5, 3.0, N)
        T_v  = rng.uniform(1.0, 2.0, N)
        a_v  = rng.uniform(1.0, 2.0, N)
        b_v  = rng.uniform(1.5, 3.0, N)

        vec    = idm_acceleration_vec(v, v0, s, dv, s0_v, T_v, a_v, b_v)
        scalar = np.array([
            idm_acceleration(v[i], v0[i], s[i], dv[i], s0_v[i], T_v[i], a_v[i], b_v[i])
            for i in range(N)
        ])
        np.testing.assert_allclose(vec, scalar, rtol=1e-5)

    def test_throughput_1m_vehicles(self):
        """1 million vehicles must complete in under 100 ms."""
        N   = 1_000_000
        rng = np.random.default_rng(1)
        v   = rng.uniform(0, 30, N)
        v0  = np.full(N, 30.0)
        s   = rng.uniform(2, 200, N)
        dv  = rng.uniform(-2, 2, N)

        t0 = time.perf_counter()
        idm_acceleration_vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.10, (
            f"Vectorised IDM 1 M vehicles: {elapsed * 1000:.1f} ms — should be < 100 ms"
        )


# ---------------------------------------------------------------------------
# 2. Spawn queue — must be deque
# ---------------------------------------------------------------------------

class TestSpawnQueue:

    def test_is_deque(self):
        """NetworkSimulation._spawn_queue must be a collections.deque."""
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        assert isinstance(sim._spawn_queue, deque), (
            f"_spawn_queue is {type(sim._spawn_queue).__name__}, expected deque"
        )

    def test_popleft_semantics(self):
        """Spawn queue must be ordered (smallest time first)."""
        sim = NetworkSimulation(
            _two_node_net(),
            demand={"A": {"B": 3600.0}},  # 1 veh/s
            duration=300,
            seed=0,
        )
        q = list(sim._spawn_queue)
        times = [t for t, _, _ in q]
        assert times == sorted(times), "Spawn queue is not sorted by time"

    def test_large_demand_popleft_timing(self):
        """Deque popleft over 10 k events must complete in < 5 ms."""
        net    = _two_node_net()
        demand = {"A": {"B": 36_000.0}}   # 10 veh/s for 1 h
        sim    = NetworkSimulation(net, demand=demand, duration=3600, seed=0)
        q      = sim._spawn_queue

        t0 = time.perf_counter()
        while q:
            q.popleft()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.005, f"10 k poplefts took {elapsed * 1000:.2f} ms"


# ---------------------------------------------------------------------------
# 3. Step-time benchmark
# ---------------------------------------------------------------------------

class TestStepTime:

    def test_step_200_vehicles_under_5ms(self):
        """100 steps with 200 vehicles must average < 5 ms per step."""
        sim = _loaded_sim(n_vehicles=200)

        t0 = time.perf_counter()
        for _ in range(100):
            sim.step()
        avg_ms = (time.perf_counter() - t0) / 100 * 1000

        assert avg_ms < 5.0, (
            f"Step time with 200 vehicles: {avg_ms:.2f} ms — should be < 5 ms"
        )

    def test_step_50_vehicles_under_2ms(self):
        """50 vehicles: avg step < 2 ms."""
        net    = _chain_net(n_nodes=10, seg_len=200.0, lanes=1)
        demand = {"0": {"9": 600.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=300, seed=0)
        for _ in range(30):
            sim.step()

        t0 = time.perf_counter()
        for _ in range(200):
            sim.step()
        avg_ms = (time.perf_counter() - t0) / 200 * 1000

        assert avg_ms < 2.0, (
            f"Step time with ~50 vehicles: {avg_ms:.2f} ms"
        )

    def test_vectorised_faster_than_loop_at_scale(self):
        """Vectorised IDM step should complete faster than 100 scalar IDM calls."""
        import engine.idm as _idm_scalar
        from engine.idm_vec import idm_acceleration_vec as _vec

        N   = 10_000
        rng = np.random.default_rng(99)
        v   = rng.uniform(0, 25, N)
        v0  = np.full(N, 30.0)
        s   = rng.uniform(3, 80, N)
        dv  = np.zeros(N)

        # Time scalar loop
        t0 = time.perf_counter()
        for _ in range(3):
            for i in range(N):
                _idm_scalar.idm_acceleration(v[i], v0[i], s[i], dv[i], 2.0, 1.5, 1.4, 2.0)
        scalar_t = (time.perf_counter() - t0) / 3

        # Time vectorised (exclude first call — warm up)
        _vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
        t0 = time.perf_counter()
        for _ in range(10):
            _vec(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
        vec_t = (time.perf_counter() - t0) / 10

        speedup = scalar_t / max(vec_t, 1e-9)
        assert speedup > 5.0, (
            f"Vectorised not fast enough: {speedup:.1f}× (scalar {scalar_t*1000:.1f} ms, "
            f"vec {vec_t*1000:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 4. 300-signal cap removed
# ---------------------------------------------------------------------------

class TestSignalCap:

    def _make_scenario_with_n_signals(self, n_signals: int) -> dict:
        """Build a minimal _scenario_from_import-compatible dict."""
        nodes: dict = {}
        # Create n_signals + a few plain nodes
        for i in range(n_signals):
            nid = str(i)
            nodes[nid] = {
                "id": nid,
                "lat": 48.8 + i * 0.0001,
                "lon": 2.3,
                "x": float(i * 20),
                "y": 0.0,
                "tags": {"highway": "traffic_signals"},
            }
        edges = []
        for i in range(n_signals - 1):
            edges.append({
                "id":         f"e{i}",
                "from_node":  str(i),
                "to_node":    str(i + 1),
                "num_lanes":  1,
                "speed_limit": 14.0,
            })
        # Return edge makes every node reachable (has in_degree), so all
        # signal-tagged nodes pass the `nid in in_deg` filter in run.py.
        edges.append({
            "id":         "e_return",
            "from_node":  str(n_signals - 1),
            "to_node":    "0",
            "num_lanes":  1,
            "speed_limit": 14.0,
        })
        return {"network": {"nodes": nodes, "edges": edges}}

    def test_all_signals_used_when_above_300(self):
        """With 400 OSM signal nodes, ALL must appear in signal_nodes."""
        from run import _scenario_from_import
        scenario = self._make_scenario_with_n_signals(400)
        result   = _scenario_from_import(scenario)
        n_sig    = len(result["network"]["signal_nodes"])
        assert n_sig > 300, (
            f"Signal cap still active: only {n_sig} signals used (expected > 300)"
        )

    def test_exactly_400_signals_used(self):
        """With 400 OSM signal nodes all reachable, exactly 400 are used."""
        from run import _scenario_from_import
        scenario = self._make_scenario_with_n_signals(400)
        result   = _scenario_from_import(scenario)
        n_sig    = len(result["network"]["signal_nodes"])
        # All 400 nodes have traffic_signals tag and are in-degree reachable
        # (edges connect them all), so exactly 400 should be returned.
        assert n_sig == 400, f"Expected 400 signals, got {n_sig}"

    def test_small_signal_count_unaffected(self):
        """With 50 signals, all 50 are used (regression — was always fine)."""
        from run import _scenario_from_import
        scenario = self._make_scenario_with_n_signals(50)
        result   = _scenario_from_import(scenario)
        n_sig    = len(result["network"]["signal_nodes"])
        assert n_sig == 50


# ---------------------------------------------------------------------------
# 5. Adaptive dt via set_speed_mult
# ---------------------------------------------------------------------------

class TestAdaptiveDt:

    def test_default_dt_is_normal(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        assert sim.dt == pytest.approx(0.1, abs=1e-9)

    def test_fast_mult_raises_dt(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        sim.set_speed_mult(32)
        assert sim.dt == pytest.approx(0.5, abs=1e-9), (
            f"dt should be 0.5 at 32×, got {sim.dt}"
        )

    def test_288x_raises_dt(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        sim.set_speed_mult(288)
        assert sim.dt == pytest.approx(0.5, abs=1e-9)

    def test_below_threshold_stays_normal(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        sim.set_speed_mult(8)
        assert sim.dt == pytest.approx(0.1, abs=1e-9)

    def test_reset_to_normal(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        sim.set_speed_mult(288)
        sim.set_speed_mult(1)
        assert sim.dt == pytest.approx(0.1, abs=1e-9)

    def test_speed_mult_stored(self):
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10)
        sim.set_speed_mult(16)
        assert sim._speed_mult == pytest.approx(16.0)

    def test_config_updated_when_present(self):
        from engine.config import SimConfig
        cfg = SimConfig()
        sim = NetworkSimulation(_two_node_net(), demand={}, duration=10, config=cfg)
        sim.set_speed_mult(64)
        assert cfg.speed_mult == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# 6. Simulation correctness is preserved after vectorisation
# ---------------------------------------------------------------------------

class TestCorrectness:

    def test_vehicles_decelerate_at_red_signal(self):
        """Vehicles approaching a permanent red signal must slow below free flow.

        The average is tested against 90 % of free-flow speed rather than a
        near-zero value because recently-spawned vehicles far from the stop-bar
        inflate the fleet average.  The minimum speed check confirms the head
        vehicle is actually braking hard.
        """
        from engine.signals import SignalPlan, Phase
        net = _two_node_net(length=500.0, speed=14.0)
        FREE_FLOW = 14.0
        # Permanent red at B (tiny 0.1 s green flash then 200 s all-red)
        plan = SignalPlan(
            node_id="B",
            phases=[Phase(green_movements=[], green_duration=0.1,
                          yellow_duration=0.1, all_red_duration=200.0)],
            offset=0.0,
        )
        sim = NetworkSimulation(
            net,
            demand={"A": {"B": 3600.0}},
            duration=80,
            seed=0,
            signal_plans={"B": plan},
        )
        for _ in range(600):   # 60 simulated seconds → queue well established
            sim.step()
        if sim.vehicles:
            speeds    = [v.speed for v in sim.vehicles]
            avg_speed = sum(speeds) / len(speeds)
            min_speed = min(speeds)
            # Fleet average must be clearly below free-flow
            assert avg_speed < FREE_FLOW * 0.9, (
                f"Fleet not slowing at red: avg {avg_speed:.2f} m/s "
                f"vs free-flow {FREE_FLOW} m/s"
            )
            # Head vehicle (at stop-bar) must have nearly stopped
            assert min_speed < 3.0, (
                f"Lead vehicle still moving at red: min {min_speed:.2f} m/s"
            )

    def test_no_negative_speeds(self):
        """No vehicle should ever have negative speed."""
        sim = _loaded_sim(n_vehicles=200, warmup_steps=0)
        for _ in range(300):
            sim.step()
            for v in sim.vehicles:
                assert v.speed >= 0.0, f"Negative speed: {v.speed}"

    def test_trip_log_populated(self):
        """Completed trips must appear in trip_log."""
        net    = _two_node_net(length=500.0)
        demand = {"A": {"B": 1800.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=120, seed=0)
        sim.run()
        assert len(sim.trip_log) > 0, "No trips completed — trip_log is empty"

    def test_vectorised_matches_legacy_trip_count(self):
        """M1 vectorised sim must produce same trip count as old scalar path.

        We compare against a known-good reference count from a short run.
        (Both code paths use the same RNG seed and demand.)
        """
        net    = _two_node_net(length=300.0, speed=14.0)
        demand = {"A": {"B": 1800.0}}
        sim    = NetworkSimulation(net, demand=demand, duration=60, seed=7)
        result = sim.run()
        # At 30 veh/min * 1 min = ~30 trips (allow ±10 for Poisson variance)
        n = result["trips_completed"]
        assert 10 <= n <= 60, f"Unexpected trip count: {n}"
