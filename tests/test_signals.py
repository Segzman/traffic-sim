"""Milestone 3 — Signal plan unit tests."""
import pytest
from engine.signals import Phase, SignalPlan
from engine.simulation import Simulation


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _two_phase_plan(node_id="J", green=40.0, yellow=3.0, allred=1.0, offset=0.0):
    """Standard 2-phase plan: NS green then EW green."""
    return SignalPlan(
        node_id=node_id,
        phases=[
            Phase(green_movements=["NS"], green_duration=green,
                  yellow_duration=yellow, all_red_duration=allred),
            Phase(green_movements=["EW"], green_duration=green,
                  yellow_duration=yellow, all_red_duration=allred),
        ],
        offset=offset,
    )


def _signal_scenario(phase_green=40.0, phase_yellow=3.0, offset=0.0,
                     v_speed=13.9, v_pos=None, stop_line=390.0,
                     detection_distance=200.0, duration=120.0, movement="NS"):
    """Build a scenario with one vehicle approaching a signalised stop line."""
    road_len = 400.0
    pos = v_pos if v_pos is not None else stop_line - 100.0
    return {
        "road": {"length": road_len, "num_lanes": 1, "speed_limit": 13.9},
        "junction": {
            "type": "signal",
            "stop_line": stop_line,
            "detection_distance": detection_distance,
            "movement_id": movement,
            "plan": {
                "node_id": "J",
                "phases": [
                    {"green_movements": [movement],
                     "green_duration": phase_green,
                     "yellow_duration": phase_yellow,
                     "all_red_duration": 1.0},
                    {"green_movements": ["cross"],
                     "green_duration": phase_green,
                     "yellow_duration": phase_yellow,
                     "all_red_duration": 1.0},
                ],
                "offset": offset,
            },
        },
        "vehicles": [
            {"id": 0, "lane_id": 0, "position_s": pos, "speed": v_speed,
             "v0": 13.9, "s0": 2.0, "T": 1.5, "a_max": 1.4, "b": 2.0},
        ],
        "duration": duration,
        "seed": 0,
    }


# ------------------------------------------------------------------ #
# test_phase_cycling
# ------------------------------------------------------------------ #

def test_phase_cycling():
    """Signal cycles through all phases in correct order; phase durations
    sum to cycle_time."""
    plan = _two_phase_plan()
    expected_cycle = sum(p.total_duration() for p in plan.phases)
    assert plan.cycle_time == pytest.approx(expected_cycle)

    # At t=0: should be in phase 0 (NS green)
    phase0, elapsed0 = plan.current_phase(0.0)
    assert phase0 is plan.phases[0]
    assert elapsed0 == pytest.approx(0.0)

    # After phase 0 total duration: should be in phase 1 (EW green)
    t1 = plan.phases[0].total_duration()
    phase1, elapsed1 = plan.current_phase(t1)
    assert phase1 is plan.phases[1]
    assert elapsed1 == pytest.approx(0.0, abs=1e-9)

    # One full cycle later: back to phase 0
    phase_wrap, elapsed_wrap = plan.current_phase(plan.cycle_time)
    assert phase_wrap is plan.phases[0]
    assert elapsed_wrap == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ #
# test_offset_shifts_phase
# ------------------------------------------------------------------ #

def test_offset_shifts_phase():
    """Two signals with offset = half cycle_time are never in the same
    phase state at the same instant."""
    plan_a = _two_phase_plan(offset=0.0)
    half = plan_a.cycle_time / 2.0   # 45 s offset for 90 s cycle
    plan_b = _two_phase_plan(offset=half)

    # At any sampled time, the two plans should NOT both be in NS green
    for t in range(0, int(plan_a.cycle_time) * 3):
        state_a = plan_a.current_state(float(t), "NS")
        state_b = plan_b.current_state(float(t), "NS")
        # They are exactly half a cycle apart — when A is in phase 0, B is
        # in phase 1 and vice-versa.  They should never both be "green" for NS.
        assert not (state_a == "green" and state_b == "green"), (
            f"Both plans green for NS at t={t}"
        )


# ------------------------------------------------------------------ #
# test_vehicle_stops_on_red
# ------------------------------------------------------------------ #

def test_vehicle_stops_on_red():
    """Vehicle approaching a red signal decelerates and reaches
    speed < 0.5 m/s before the stop line."""
    # Signal starts on phase 1 (EW green) so our NS movement is immediately red
    # Offset = 0 but vehicle's movement = "NS", phase 0 = NS green, phase 1 = EW
    # Start halfway through phase 1 so NS is red for the first ~45 s
    plan_phase_dur = 44.0  # total of one phase (40+3+1)
    initial_time_offset = plan_phase_dur  # skip to EW green phase start → NS=red

    scenario = _signal_scenario(
        phase_green=40.0, phase_yellow=3.0,
        offset=-initial_time_offset,   # shifts plan so NS is red at t=0
        v_speed=13.9, v_pos=250.0,
        stop_line=390.0, detection_distance=200.0,
        duration=60.0,
    )
    sim = Simulation(scenario)
    min_speed_near_stop = float("inf")
    stop_line = 390.0

    for _ in range(int(60.0 / sim.dt)):
        sim.step()
        for v in sim.vehicles:
            if v.position_s < stop_line:
                min_speed_near_stop = min(min_speed_near_stop, v.speed)

    assert min_speed_near_stop < 0.5, (
        f"Vehicle never slowed below 0.5 m/s; min speed near stop = {min_speed_near_stop:.2f}"
    )


# ------------------------------------------------------------------ #
# test_vehicle_clears_on_green
# ------------------------------------------------------------------ #

def test_vehicle_clears_on_green():
    """Vehicle at rest near the stop line accelerates when the phase turns green."""
    # Plan: phase 0 = NS green for 40 s. Vehicle is stationary just behind stop line.
    scenario = _signal_scenario(
        phase_green=40.0, offset=0.0,
        v_speed=0.0, v_pos=380.0,  # 10 m from stop line, at rest
        stop_line=390.0, detection_distance=200.0,
        duration=15.0,  # well within green phase
    )
    sim = Simulation(scenario)
    sim.run()

    # After 15 s of green, vehicle should have crossed the stop line or be moving
    for v in sim.vehicles:
        assert v.speed > 1.0 or v.position_s >= 390.0, (
            f"Vehicle did not accelerate on green: speed={v.speed:.2f}, pos={v.position_s:.1f}"
        )


# ------------------------------------------------------------------ #
# test_yellow_commit_decision
# ------------------------------------------------------------------ #

def test_yellow_commit_decision():
    """Vehicle past commit distance crosses on yellow; vehicle before commit
    distance stops."""
    stop_line = 390.0
    v_speed = 13.9  # m/s
    # Commit distance ≈ v² / (2b) = 13.9² / 4 ≈ 48.3 m
    commit_dist = v_speed ** 2 / (2 * 2.0)  # ~48.3 m

    # ---- Case 1: vehicle is 20 m from stop line (inside commit distance)
    # When yellow starts (offset so vehicle is in yellow phase immediately)
    # phase green=0 means we jump straight to yellow
    yellow_dur = 6.0
    scenario_commits = _signal_scenario(
        phase_green=0.01,  # almost instant green → yellow starts almost immediately
        phase_yellow=yellow_dur, offset=0.0,
        v_speed=v_speed, v_pos=stop_line - 20.0,
        stop_line=stop_line, detection_distance=200.0,
        duration=yellow_dur + 2.0,
    )
    sim_c = Simulation(scenario_commits)
    sim_c.run()
    # Vehicle should have crossed the stop line (committed)
    crossed = any(v.position_s >= stop_line for v in sim_c.vehicles)
    assert crossed, "Vehicle within commit distance should cross on yellow"

    # ---- Case 2: vehicle is 80 m from stop line (outside commit distance)
    scenario_stops = _signal_scenario(
        phase_green=0.01,
        phase_yellow=yellow_dur, offset=0.0,
        v_speed=v_speed, v_pos=stop_line - 80.0,
        stop_line=stop_line, detection_distance=200.0,
        duration=yellow_dur + 5.0,
    )
    sim_s = Simulation(scenario_stops)
    min_speed = float("inf")
    for _ in range(int((yellow_dur + 5.0) / sim_s.dt)):
        sim_s.step()
        for v in sim_s.vehicles:
            if v.position_s < stop_line:
                min_speed = min(min_speed, v.speed)
    assert min_speed < 0.5, (
        f"Vehicle outside commit distance should stop on yellow; min_speed={min_speed:.2f}"
    )
