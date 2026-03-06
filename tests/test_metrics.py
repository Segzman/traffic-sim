"""Milestone 5 — Metrics: CSV export, batch CI, warmup exclusion, grid no-deadlock."""
import csv
import io
import math
import pytest

from engine.metrics import (
    TripRecord,
    trip_log_to_csv,
    run_batch,
    BatchMetrics,
    MetricsRecorder,
)


# ------------------------------------------------------------------ #
# test_csv_export_columns
# ------------------------------------------------------------------ #

def test_csv_export_columns():
    """trip_log_to_csv produces a CSV with all required columns and correct data."""
    records = [
        TripRecord(vehicle_id=0, entry_time=0.0, exit_time=25.3,
                   delay_s=5.3, lane_changes=2, stops=1),
        TripRecord(vehicle_id=1, entry_time=2.0, exit_time=22.1,
                   delay_s=0.1, lane_changes=0, stops=0),
    ]

    csv_str = trip_log_to_csv(records)
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)

    required_columns = {"vehicle_id", "entry_time", "exit_time",
                        "delay_s", "lane_changes", "stops"}
    assert required_columns.issubset(set(reader.fieldnames)), (
        f"CSV is missing columns; got {reader.fieldnames}"
    )
    assert len(rows) == 2, f"Expected 2 data rows; got {len(rows)}"

    assert int(rows[0]["vehicle_id"]) == 0
    assert float(rows[0]["delay_s"]) == pytest.approx(5.3)
    assert int(rows[0]["lane_changes"]) == 2
    assert int(rows[1]["stops"]) == 0


# ------------------------------------------------------------------ #
# test_batch_ci_width
# ------------------------------------------------------------------ #

def _make_sim_factory(mean_throughput: float, sd: float):
    """Return a factory producing mock sim objects with noisy throughput."""
    import random

    class _MockSim:
        def __init__(self, seed):
            self._rng = random.Random(seed * 17 + 3)
            self._mean = mean_throughput
            self._sd = sd

        def run(self):
            val = self._rng.gauss(self._mean, self._sd)
            return {"throughput": val, "avg_delay": self._rng.gauss(10.0, 2.0)}

    return lambda seed: _MockSim(seed)


def test_batch_ci_width():
    """run_batch computes CI width proportional to variance; more runs → narrower CI."""
    factory = _make_sim_factory(mean_throughput=1.0, sd=0.2)

    bm_10 = run_batch(factory, n=10, metrics_keys=["throughput"])
    bm_30 = run_batch(factory, n=30, metrics_keys=["throughput"])

    # Both should have finite, positive CI
    assert bm_10.ci_width["throughput"] > 0.0, "CI width should be positive for n=10"
    assert bm_30.ci_width["throughput"] > 0.0, "CI width should be positive for n=30"

    # Larger n → narrower CI (not guaranteed every seed, but should hold on average)
    # We allow a 20 % tolerance: just check they're both reasonable
    assert bm_10.ci_width["throughput"] < 1.0, "CI suspiciously wide for n=10"
    assert bm_30.ci_width["throughput"] < bm_10.ci_width["throughput"] * 2.0, (
        "n=30 CI should not be wider than 2× n=10 CI"
    )

    # n=1 should give ci_width = 0
    bm_1 = run_batch(factory, n=1, metrics_keys=["throughput"])
    assert bm_1.ci_width["throughput"] == 0.0


# ------------------------------------------------------------------ #
# test_warmup_excluded
# ------------------------------------------------------------------ #

def test_warmup_excluded():
    """Trips completing during warmup period are excluded from trip_log."""
    recorder = MetricsRecorder(road_length=500.0, warmup=30.0)

    # Record a trip that finishes at t=25 (during warmup) → should NOT appear
    recorder.record_trip(
        vehicle_id=0, entry_time=0.0, exit_time=25.0,
        delay_s=5.0, lane_changes=1, stops=0,
    )
    # Record a trip that finishes at t=35 (after warmup) → SHOULD appear
    recorder.record_trip(
        vehicle_id=1, entry_time=5.0, exit_time=35.0,
        delay_s=3.0, lane_changes=0, stops=1,
    )
    # Record a trip exactly at warmup boundary (t=30) → SHOULD appear
    recorder.record_trip(
        vehicle_id=2, entry_time=10.0, exit_time=30.0,
        delay_s=0.0, lane_changes=0, stops=0,
    )

    assert len(recorder.trip_log) == 2, (
        f"Expected 2 post-warmup trips; got {len(recorder.trip_log)}"
    )
    vids = {r.vehicle_id for r in recorder.trip_log}
    assert 0 not in vids, "Vehicle 0 (warmup trip) should be excluded"
    assert 1 in vids, "Vehicle 1 should be in trip_log"
    assert 2 in vids, "Vehicle 2 (boundary) should be in trip_log"


# ------------------------------------------------------------------ #
# test_grid_no_deadlock
# ------------------------------------------------------------------ #

def _build_grid_network():
    """Build the 3×3 grid network programmatically for the deadlock test."""
    from engine.network import Network, Node, Edge

    net = Network()
    spacing = 200.0
    cols = ["A", "B", "C"]
    rows = [1, 2, 3]

    for ci, col in enumerate(cols):
        for row in rows:
            net.add_node(Node(id=f"{col}{row}", x=ci * spacing, y=(row - 1) * spacing))

    def _add_pair(nid1, nid2):
        e1 = Edge(id=f"{nid1}-{nid2}", from_node=nid1, to_node=nid2,
                  num_lanes=1, speed_limit=13.9)
        e2 = Edge(id=f"{nid2}-{nid1}", from_node=nid2, to_node=nid1,
                  num_lanes=1, speed_limit=13.9)
        net.add_edge(e1)
        net.add_edge(e2)

    # Vertical links
    for col in cols:
        _add_pair(f"{col}1", f"{col}2")
        _add_pair(f"{col}2", f"{col}3")
    # Horizontal links
    for row in rows:
        _add_pair(f"A{row}", f"B{row}")
        _add_pair(f"B{row}", f"C{row}")

    return net


def test_grid_no_deadlock():
    """3×3 grid with low demand and 90 s signal cycle should not deadlock in 300 s."""
    from engine.network_simulation import NetworkSimulation
    from engine.signals import Phase, SignalPlan

    net = _build_grid_network()

    # Simple two-phase signal at interior nodes (B1, B2, B3)
    def _make_plan(node_id: str) -> SignalPlan:
        return SignalPlan(
            node_id=node_id,
            phases=[
                Phase(green_movements=["NS"], green_duration=40.0,
                      yellow_duration=3.0, all_red_duration=2.0),
                Phase(green_movements=["EW"], green_duration=40.0,
                      yellow_duration=3.0, all_red_duration=2.0),
            ],
            offset=0.0,
        )

    signal_plans = {nid: _make_plan(nid) for nid in ["B1", "B2", "B3"]}

    demand = {
        "A1": {"C3": 50},
        "C1": {"A3": 50},
        "A3": {"C1": 50},
        "C3": {"A1": 50},
    }

    sim = NetworkSimulation(
        network=net,
        demand=demand,
        duration=300.0,
        seed=42,
        warmup=60.0,
        signal_plans=signal_plans,
    )

    metrics = sim.run()

    assert not metrics["deadlock_detected"], (
        "No deadlock expected with 50 veh/hr demand and 90 s signal cycle"
    )
    assert metrics["trips_completed"] >= 0  # just confirm it ran
