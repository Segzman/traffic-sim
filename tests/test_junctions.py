"""Milestone 3 — Junction behaviour tests (roundabout, stop sign, metrics)."""
import json
import os
import pytest
from engine.simulation import Simulation


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _roundabout_scenario(
    conflict_flow_hr=600, t_min_mean=7.5, t_min_sd=0.0,
    v_speed=0.0, v_pos=1180.0,
    stop_line=1190.0, detection_distance=200.0,
    road_len=1200.0, duration=60.0, seed=0,
):
    return {
        "road": {"length": road_len, "num_lanes": 1, "speed_limit": 13.9},
        "junction": {
            "type": "roundabout",
            "stop_line": stop_line,
            "detection_distance": detection_distance,
            "conflicting_flow_rate": conflict_flow_hr,
            "t_min_mean": t_min_mean,
            "t_min_sd": t_min_sd,
        },
        "vehicles": [
            {"id": 0, "lane_id": 0, "position_s": v_pos, "speed": v_speed,
             "v0": 13.9, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0},
        ],
        "duration": duration,
        "seed": seed,
    }


def _stop_sign_scenario(v_speed=10.0, v_pos=280.0, stop_line=290.0,
                        road_len=300.0, duration=60.0, seed=0):
    return {
        "road": {"length": road_len, "num_lanes": 1, "speed_limit": 13.9},
        "junction": {
            "type": "stop",
            "stop_line": stop_line,
            "detection_distance": 100.0,
            "conflicting_flow_rate": 0.1,   # very low — gaps always large
            "t_min_mean": 2.0,
            "t_min_sd": 0.0,
        },
        "vehicles": [
            {"id": 0, "lane_id": 0, "position_s": v_pos, "speed": v_speed,
             "v0": 13.9, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0},
        ],
        "duration": duration,
        "seed": seed,
    }


# ------------------------------------------------------------------ #
# test_roundabout_gap_acceptance
# ------------------------------------------------------------------ #

def test_roundabout_gap_acceptance():
    """Entering vehicle waits when circulating gap < t_min; proceeds when
    gap is acceptable.

    We test this by running two scenarios:
    - Very high conflict flow (vehicle almost never gets a gap) → vehicle stays stopped.
    - Zero conflict flow (always clear) → vehicle proceeds immediately.
    """
    # --- High conflict flow: conflicts every ~0.5 s, t_min = 5 s → never acceptable
    sc_blocked = _roundabout_scenario(
        conflict_flow_hr=7200,  # 1 conflict every 0.5 s
        t_min_mean=5.0, t_min_sd=0.0,
        v_speed=0.0, v_pos=1185.0,
        stop_line=1190.0, detection_distance=200.0,
        duration=10.0,
    )
    sim_blocked = Simulation(sc_blocked)
    sim_blocked.run()
    # Vehicle should not have crossed the stop line
    assert all(v.position_s < 1190.0 for v in sim_blocked.vehicles), (
        "Vehicle should NOT cross stop line when conflict flow blocks all gaps"
    )

    # --- No conflict flow: t_min = 5 s, flow = 0 → always clear
    sc_free = _roundabout_scenario(
        conflict_flow_hr=0.01,  # effectively zero
        t_min_mean=5.0, t_min_sd=0.0,
        v_speed=0.0, v_pos=1100.0,
        stop_line=1190.0, detection_distance=200.0,
        duration=60.0,
    )
    sim_free = Simulation(sc_free)
    sim_free.run()
    # Vehicle should have accelerated and crossed
    assert any(v.position_s >= 1190.0 for v in sim_free.vehicles), (
        "Vehicle should cross stop line when no conflict flow"
    )


# ------------------------------------------------------------------ #
# test_roundabout_impatience
# ------------------------------------------------------------------ #

def test_roundabout_impatience():
    """After 15 s of waiting, vehicle's effective t_min decreases by ≥ 20%."""
    # Use very high conflict flow so vehicle is blocked for 15+ s
    sc = _roundabout_scenario(
        conflict_flow_hr=7200,  # conflict every 0.5 s → never acceptable for t_min=5
        t_min_mean=5.0, t_min_sd=0.0,
        v_speed=0.0, v_pos=1185.0,
        stop_line=1190.0, detection_distance=200.0,
        duration=20.0,
    )
    sim = Simulation(sc)
    for _ in range(int(15.0 / sim.dt)):
        sim.step()

    veh = sim.vehicles[0]
    original_t_min = sim._veh_state[veh.id]["t_min"]
    effective = sim._effective_t_min(veh.id)
    reduction = (original_t_min - effective) / original_t_min

    assert reduction >= 0.20, (
        f"Expected ≥20% reduction in t_min after 15 s wait; got {reduction:.1%}"
    )


# ------------------------------------------------------------------ #
# test_stop_sign_full_stop
# ------------------------------------------------------------------ #

def test_stop_sign_full_stop():
    """Vehicle must reach speed = 0 before crossing the stop line (no rolling stop)."""
    sc = _stop_sign_scenario(v_speed=10.0, v_pos=260.0, stop_line=290.0,
                             duration=60.0)
    sim = Simulation(sc)

    speed_before_crossing = []
    crossed = False

    for _ in range(int(60.0 / sim.dt)):
        sim.step()
        veh = sim.vehicles[0]
        if not crossed and veh.position_s < 290.0:
            speed_before_crossing.append(veh.speed)
        if veh.position_s >= 290.0:
            crossed = True

    assert len(speed_before_crossing) > 0, "Vehicle never reached the stop zone"
    min_speed = min(speed_before_crossing)
    assert min_speed < 0.05, (
        f"Vehicle did not fully stop before stop line; min speed = {min_speed:.3f} m/s"
    )


# ------------------------------------------------------------------ #
# test_queue_length_metric
# ------------------------------------------------------------------ #

def test_queue_length_metric():
    """10 vehicles queued at red: queue_length metric returns a value between 9 and 11."""
    # Build a scenario with 10 vehicles stationary just behind the stop line,
    # all stuck on red (NS movement, but plan keeps it red initially)
    stop_line = 290.0
    road_len = 300.0
    # Place 10 vehicles within the detection zone, all at rest
    vehicles = [
        {"id": i, "lane_id": 0, "position_s": stop_line - 5.0 - i * 4.6,
         "speed": 0.0, "v0": 13.9, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0}
        for i in range(10)
    ]
    scenario = {
        "road": {"length": road_len, "num_lanes": 1, "speed_limit": 13.9},
        "junction": {
            "type": "signal",
            "stop_line": stop_line,
            "detection_distance": 150.0,
            "movement_id": "NS",
            "plan": {
                "node_id": "J",
                "phases": [
                    # Phase 0: NS green for only 0.1 s (nearly never), phase 1: EW green 100 s
                    {"green_movements": ["NS"],
                     "green_duration": 0.1,
                     "yellow_duration": 0.0,
                     "all_red_duration": 0.0},
                    {"green_movements": ["EW"],
                     "green_duration": 100.0,
                     "yellow_duration": 3.0,
                     "all_red_duration": 1.0},
                ],
                "offset": -0.1,  # immediately in EW phase so NS is red
            },
        },
        "vehicles": vehicles,
        "duration": 5.0,
        "seed": 0,
    }
    sim = Simulation(scenario)
    metrics = sim.run()

    ql = metrics["queue_length"]
    assert 9 <= ql <= 11, f"Expected queue length 9–11 with 10 stopped vehicles; got {ql:.1f}"


# ------------------------------------------------------------------ #
# test_roundabout_vs_signal_delay
# ------------------------------------------------------------------ #

def test_roundabout_vs_signal_delay():
    """Roundabout scenario mean delay is within range reported in literature
    (5–15 s) for ~800 veh/hr balanced demand."""
    scenario_path = os.path.join(
        os.path.dirname(__file__), "..", "scenarios", "roundabout_vs_signal.json"
    )
    with open(scenario_path) as fh:
        cfg = json.load(fh)

    ra_cfg = cfg["roundabout"]
    delays = []
    for run in range(15):
        sc = dict(ra_cfg)
        sc["seed"] = run
        metrics = Simulation(sc).run()
        d = metrics.get("avg_delay", 0.0)
        delays.append(d)

    mean_delay = sum(delays) / len(delays)
    assert 5.0 <= mean_delay <= 15.0, (
        f"Roundabout mean delay {mean_delay:.2f} s outside expected 5–15 s range. "
        f"Individual runs: {[f'{d:.1f}' for d in delays]}"
    )
