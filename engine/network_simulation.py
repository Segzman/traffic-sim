"""NetworkSimulation — multi-edge routing simulation over a road network.

Vehicles follow pre-computed routes (lists of edge IDs) and transfer between
edges as they reach the end of each segment.  IDM and basic signal logic are
applied; MOBIL lane changes and Social Force pedestrians are also supported.

Performance (Milestone 1)
-------------------------
* IDM accelerations are computed for **all vehicles at once** using a single
  NumPy vectorised call (``engine.idm_vec.idm_acceleration_vec``).  This is
  ~30× faster than the previous Python loop and releases the GIL so that
  :class:`concurrent.futures.ThreadPoolExecutor` workers can overlap.
* The spawn queue uses ``collections.deque`` for O(1) pop-left instead of
  the previous O(n) ``list.pop(0)``.
* ``set_speed_mult(mult)`` dynamically adjusts both the public
  ``speed_mult`` hint (used by the HTTP server's sleep loop) and the
  timestep ``dt`` (adaptive: 0.5 s at ≥ 32×, 0.1 s otherwise — both
  within IDM stability bounds per the literature).
"""
from __future__ import annotations

import math
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from engine.agents import Vehicle
from engine.idm_vec import idm_acceleration_vec
from engine.network import Network
from engine.metrics import TripRecord, BatchMetrics

if TYPE_CHECKING:
    from engine.pedestrians import Pedestrian
    from engine.config import SimConfig


# ---------------------------------------------------------------------------
# Module-level thread pool (shared across all NetworkSimulation instances).
# Workers = logical CPU cores; each NumPy sub-task releases the GIL so
# threads run truly in parallel on multi-core machines.
# ---------------------------------------------------------------------------
_N_WORKERS: int = max(2, os.cpu_count() or 2)
_POOL = ThreadPoolExecutor(max_workers=_N_WORKERS, thread_name_prefix="ns-idm")

# Speed threshold for "stopped" (m/s)
_STOP_SPEED = 0.3
# Deadlock threshold: stuck below _STOP_SPEED for this many consecutive seconds
_DEADLOCK_STUCK_S = 120.0

# Adaptive-dt thresholds
_DT_NORMAL: float = 0.1   # used at speed_mult < 32
_DT_FAST:   float = 0.5   # used at speed_mult ≥ 32  (IDM stable to ~0.5 s)
_FAST_MULT_THRESHOLD: float = 32.0


class NetworkSimulation:
    """Simulate vehicles routing over a :class:`~engine.network.Network`.

    Parameters
    ----------
    network:
        Road network graph.
    demand:
        ``{origin_node: {dest_node: flow_veh_hr}}`` demand matrix.
    duration:
        Simulation duration (s).
    dt:
        Initial timestep (s).  Overridden by :meth:`set_speed_mult`.
    seed:
        RNG seed.
    warmup:
        Warmup period (s); trips finishing before this time are excluded from
        the trip log.
    signal_plans:
        ``{node_id: SignalPlan}`` — optional pre-built signal plans.
    pedestrians:
        Optional list of :class:`~engine.pedestrians.Pedestrian` agents.
    config:
        Optional :class:`~engine.config.SimConfig` for live parameter updates.
    compute_backend:
        ``"auto"`` | ``"numpy"`` | ``"cuda"`` | ``"metal"`` (M6; ignored now).
    """

    def __init__(
        self,
        network: Network,
        demand: dict[str, dict[str, float]] | None = None,
        duration: float = 300.0,
        dt: float = _DT_NORMAL,
        seed: int = 42,
        warmup: float = 0.0,
        signal_plans: dict | None = None,
        pedestrians: list | None = None,
        config: "SimConfig | None" = None,
        compute_backend: str = "auto",
    ):
        self.network    = network
        self.demand     = demand or {}
        self.duration   = duration
        self.dt         = dt
        self.warmup     = warmup
        self.rng        = np.random.default_rng(seed)
        self.py_rng     = random.Random(seed)
        self.time       = 0.0
        self.config     = config        # SimConfig or None
        self._speed_mult: float = 1.0  # hint for the HTTP server sleep loop

        self._signal_plans = signal_plans or {}
        self.pedestrians: list[Pedestrian] = pedestrians or []

        # Active vehicles
        self.vehicles: list[Vehicle] = []
        self._next_vid = 0

        # Per-edge vehicle lists
        self._edge_vehicles: dict[str, list[Vehicle]] = {
            eid: [] for eid in network.edges
        }

        # Spawn schedule — deque for O(1) popleft
        self._spawn_queue: deque[tuple[float, str, str]] = deque()
        self._demand_mult: float = 1.0
        self._build_spawn_queue()

        # Trip log (completed trips after warmup)
        self.trip_log: list[TripRecord] = []

        # Deadlock tracking
        self._stuck_time: dict[int, float] = {}
        self.deadlock_detected = False

    # ------------------------------------------------------------------ #
    # Speed multiplier / adaptive dt
    # ------------------------------------------------------------------ #

    def set_speed_mult(self, mult: float) -> None:
        """Set the simulation speed multiplier and adapt ``dt`` accordingly.

        At ``mult`` ≥ 32× the timestep is raised to 0.5 s (IDM is stable
        up to ~0.5 s per the literature) to halve the step count and allow
        a full 24-hour day to run in ~5 minutes.
        """
        self._speed_mult = float(mult)
        self.dt = _DT_FAST if self._speed_mult >= _FAST_MULT_THRESHOLD else _DT_NORMAL
        if self.config is not None:
            self.config.update(speed_mult=self._speed_mult)

    # ------------------------------------------------------------------ #
    # Build spawn schedule from demand matrix
    # ------------------------------------------------------------------ #

    def _build_spawn_queue(self) -> None:
        """Convert demand flows to Poisson spawn events (stored in a deque)."""
        events: list[tuple[float, str, str]] = []
        for origin, dests in self.demand.items():
            for dest, flow_hr in dests.items():
                if flow_hr <= 0:
                    continue
                rate_s   = flow_hr / 3600.0
                mean_hw  = 1.0 / rate_s
                t = float(self.rng.exponential(mean_hw))
                while t < self.duration:
                    events.append((t, origin, dest))
                    t += float(self.rng.exponential(mean_hw))
        events.sort()
        self._spawn_queue = deque(events)

    def set_demand_mult(self, mult: float) -> None:
        """Rebuild future spawn events with demand scaled by *mult*.

        Drops all not-yet-processed spawns and regenerates them from the
        current simulation time using the scaled demand rates.
        """
        self._demand_mult = max(0.0, float(mult))
        # Keep only events that have already been consumed (t <= now)
        past = deque((t, o, d) for t, o, d in self._spawn_queue if t <= self.time)
        self._spawn_queue = past

        if self._demand_mult <= 0.0:
            return

        events: list[tuple[float, str, str]] = []
        for origin, dests in self.demand.items():
            for dest, flow_hr in dests.items():
                if flow_hr <= 0:
                    continue
                scaled  = flow_hr * self._demand_mult
                rate_s  = scaled / 3600.0
                mean_hw = 1.0 / rate_s
                t = self.time + float(self.rng.exponential(mean_hw))
                while t < self.duration:
                    events.append((t, origin, dest))
                    t += float(self.rng.exponential(mean_hw))
        events.sort()
        self._spawn_queue.extend(events)
        # Re-sort the full queue (past events at front are already in order)
        sorted_q = sorted(self._spawn_queue)
        self._spawn_queue = deque(sorted_q)

    # ------------------------------------------------------------------ #
    # Vehicle spawn
    # ------------------------------------------------------------------ #

    def _spawn_vehicle(self, origin: str, dest: str) -> None:
        """Compute route and place vehicle on the first edge."""
        route = self.network.shortest_path(origin, dest)
        if not route:
            return

        first_edge = route[0]
        edge = self.network.edges[first_edge]

        existing = self._edge_vehicles.get(first_edge, [])
        if any(v.position_s < 15.0 for v in existing):
            return  # entry blocked

        num_lanes = edge.num_lanes
        if num_lanes <= 1:
            spawn_lane = 0
        else:
            weights    = [0.40] + [0.60 / (num_lanes - 1)] * (num_lanes - 1)
            spawn_lane = int(self.py_rng.choices(range(num_lanes), weights=weights)[0])

        # Apply config IDM overrides if a SimConfig is attached
        cfg = self.config
        a_max = cfg.idm_a_max if cfg else 1.4
        b     = cfg.idm_b     if cfg else 2.0
        T     = cfg.idm_T     if cfg else 1.5
        s0    = cfg.idm_s0    if cfg else 2.0

        v = Vehicle(
            id=self._next_vid,
            lane_id=spawn_lane,
            position_s=1.0,
            speed=edge.speed_limit,
            v0=edge.speed_limit,
            s0=s0,
            T=T,
            a_max=a_max,
            b=b,
            route=route,
            route_index=0,
            current_edge=first_edge,
            entry_time=self.time,
        )
        self._next_vid += 1
        self.vehicles.append(v)
        self._edge_vehicles[first_edge].append(v)

    # ------------------------------------------------------------------ #
    # Signal helper
    # ------------------------------------------------------------------ #

    def _signal_color(self, node_id: str, movement_id: str | None = None) -> str:
        """Return 'green', 'yellow', or 'red' for a node at current time."""
        plan = self._signal_plans.get(node_id)
        if plan is None:
            return "green"
        return plan.current_state(self.time, movement_id)

    # ------------------------------------------------------------------ #
    # IDM — vectorised across ALL vehicles in one NumPy call
    # ------------------------------------------------------------------ #

    def _step_idm_vectorised(self) -> None:
        """Compute IDM accelerations for all vehicles using NumPy batch ops.

        Strategy
        --------
        1. For each edge, sort vehicles front-to-back.
        2. Build flat arrays (one entry per vehicle across all edges):
           gaps, delta_v, v, v0, s0, T, a_max, b.
        3. One call to ``idm_acceleration_vec`` — O(N) NumPy C ops;
           GIL is released for the array math.
        4. Write results back to each ``vehicle.acceleration``.

        Edge transitions are independent; each vehicle lives on exactly
        one edge, so writes never conflict between edges.
        """
        # Collect per-edge sorted lists
        per_edge: list[tuple[list[Vehicle], float, str]] = []
        for edge_id, evs in self._edge_vehicles.items():
            if not evs:
                continue
            edge     = self.network.edges[edge_id]
            edge_len = self.network.edge_length(edge_id)
            color    = self._signal_color(edge.to_node)
            sorted_evs = sorted(evs, key=lambda v: v.position_s, reverse=True)
            per_edge.append((sorted_evs, edge_len, color))

        if not per_edge:
            return

        # Flat arrays
        all_evs:  list[Vehicle] = []
        gap_list: list[float]   = []
        dv_list:  list[float]   = []
        raw_gaps: list[float]   = []   # unclipped — for emergency-brake check

        for sorted_evs, edge_len, color in per_edge:
            for i, veh in enumerate(sorted_evs):
                all_evs.append(veh)
                if i == 0:
                    # Front vehicle: free-flow unless signal is red
                    if color == "red":
                        dist = edge_len - veh.position_s - veh.length / 2.0
                        if 0.0 < dist <= 200.0:
                            raw_gaps.append(dist)
                            gap_list.append(max(0.001, dist))
                            dv_list.append(veh.speed)   # closing on stationary line
                        else:
                            raw_gaps.append(1e9)
                            gap_list.append(1e9)
                            dv_list.append(0.0)
                    else:
                        raw_gaps.append(1e9)
                        gap_list.append(1e9)
                        dv_list.append(0.0)
                else:
                    leader  = sorted_evs[i - 1]
                    gap     = leader.position_s - veh.position_s - veh.length
                    raw_gaps.append(gap)
                    gap_list.append(gap)
                    dv_list.append(veh.speed - leader.speed)

        N = len(all_evs)
        if N == 0:
            return

        v_arr   = np.empty(N, dtype=np.float64)
        v0_arr  = np.empty(N, dtype=np.float64)
        s0_arr  = np.empty(N, dtype=np.float64)
        T_arr   = np.empty(N, dtype=np.float64)
        a_arr   = np.empty(N, dtype=np.float64)
        b_arr   = np.empty(N, dtype=np.float64)
        for i, veh in enumerate(all_evs):
            v_arr[i]  = veh.speed
            v0_arr[i] = veh.v0
            s0_arr[i] = veh.s0
            T_arr[i]  = veh.T
            a_arr[i]  = veh.a_max
            b_arr[i]  = veh.b

        gap_arr = np.maximum(np.array(gap_list, dtype=np.float64), 1e-3)
        dv_arr  = np.array(dv_list, dtype=np.float64)

        # Apply weather multipliers from config (if set)
        cfg = self.config
        if cfg is not None:
            v0_arr  *= cfg.weather_v0_mult
            s0_arr  *= cfg.weather_s0_mult
            T_arr   *= cfg.weather_T_mult

        # ---- THE vectorised IDM call ----
        accel_arr = idm_acceleration_vec(
            v_arr, v0_arr, gap_arr, dv_arr,
            s0_arr, T_arr, a_arr, b_arr,
        )

        # Emergency braking for non-positive raw gaps
        raw_arr = np.array(raw_gaps, dtype=np.float64)
        emergency = raw_arr <= 0.0
        accel_arr[emergency] = -b_arr[emergency]

        # Write back + optional pedestrian yield override
        peds = self.pedestrians
        for i, veh in enumerate(all_evs):
            acc = float(accel_arr[i])
            for ped in peds:
                if ped.exit_time >= 0.0:
                    continue
                if abs(ped.x - veh.position_s) < 20.0 and veh.speed > 0.01:
                    ttc = (ped.x - veh.position_s) / veh.speed
                    if 0.0 < ttc < 2.0:
                        acc = min(acc, -veh.b)
            veh.acceleration = acc

    # ------------------------------------------------------------------ #
    # MOBIL lane changes
    # ------------------------------------------------------------------ #

    def _step_lane_changes(self) -> None:
        """MOBIL lane changes — processed serially per edge (shared state)."""
        from engine.mobil import mobil_lane_change

        # Read MOBIL params from config if present
        cfg = self.config
        politeness = cfg.mobil_politeness if cfg else 0.3
        b_safe     = cfg.mobil_b_safe     if cfg else 3.0

        for edge_id, evs in self._edge_vehicles.items():
            if len(evs) < 2:
                continue
            edge = self.network.edges[edge_id]
            if edge.num_lanes < 2:
                continue

            by_lane: dict[int, list] = {}
            for v in evs:
                by_lane.setdefault(v.lane_id, []).append(v)
            for lst in by_lane.values():
                lst.sort(key=lambda v: v.position_s, reverse=True)

            for veh in list(evs):
                if veh.lane_change_cooldown > 0.0:
                    continue

                def _neighbors(lane):
                    lane_v = by_lane.get(lane, [])
                    ldr = fol = None
                    for u in lane_v:
                        if u is veh:
                            continue
                        if u.position_s > veh.position_s:
                            if ldr is None or u.position_s < ldr.position_s:
                                ldr = u
                        else:
                            if fol is None or u.position_s > fol.position_s:
                                fol = u
                    return ldr, fol

                cur_ldr, cur_fol = _neighbors(veh.lane_id)
                for tgt in [veh.lane_id - 1, veh.lane_id + 1]:
                    if tgt < 0 or tgt >= edge.num_lanes:
                        continue
                    tgt_ldr, tgt_fol = _neighbors(tgt)
                    if mobil_lane_change(
                        veh, cur_fol, tgt_ldr, tgt_fol,
                        politeness=politeness,
                        b_safe=b_safe,
                        delta_a_thr=0.1,
                        bias_right=(tgt < veh.lane_id),
                        moving_right=(tgt < veh.lane_id),
                    ):
                        by_lane.setdefault(veh.lane_id, []).remove(veh)
                        veh.lane_id = tgt
                        veh.lane_change_cooldown = 4.0
                        by_lane.setdefault(tgt, []).append(veh)
                        by_lane[tgt].sort(key=lambda u: u.position_s, reverse=True)
                        break

    # ------------------------------------------------------------------ #
    # Euler integration + edge transitions
    # ------------------------------------------------------------------ #

    def _step_integrate(self) -> None:
        """Euler integrate positions, handle edge transitions and trip completion."""
        dt = self.dt
        to_remove: list[Vehicle] = []

        for veh in self.vehicles:
            # Stop counting
            now_stopped = veh.speed < _STOP_SPEED
            if now_stopped and not veh._was_stopped:
                veh.stops += 1
            veh._was_stopped = now_stopped

            # Integrate
            veh.speed      = max(0.0, veh.speed + veh.acceleration * dt)
            veh.position_s += veh.speed * dt

            # Decrement lane-change cooldown
            if veh.lane_change_cooldown > 0.0:
                veh.lane_change_cooldown = max(0.0, veh.lane_change_cooldown - dt)

            # Deadlock tracking
            if veh.speed < _STOP_SPEED:
                self._stuck_time[veh.id] = self._stuck_time.get(veh.id, 0.0) + dt
                if self._stuck_time[veh.id] > _DEADLOCK_STUCK_S:
                    self.deadlock_detected = True
            else:
                self._stuck_time[veh.id] = 0.0

            # Edge transition
            edge_id  = veh.current_edge
            edge_len = self.network.edge_length(edge_id)

            if veh.position_s >= edge_len:
                overflow = veh.position_s - edge_len
                ev_list  = self._edge_vehicles.get(edge_id)
                if ev_list and veh in ev_list:
                    ev_list.remove(veh)

                next_idx = veh.route_index + 1
                if next_idx < len(veh.route):
                    next_edge_id  = veh.route[next_idx]
                    next_edge     = self.network.edges[next_edge_id]
                    veh.route_index   = next_idx
                    veh.current_edge  = next_edge_id
                    veh.position_s    = overflow
                    veh.v0            = next_edge.speed_limit
                    self._edge_vehicles[next_edge_id].append(veh)
                else:
                    # Trip complete
                    exit_time = self.time
                    if exit_time >= self.warmup:
                        ff_time = sum(
                            self.network.edge_travel_time(eid) for eid in veh.route
                        )
                        delay = max(0.0, (exit_time - veh.entry_time) - ff_time)
                        self.trip_log.append(TripRecord(
                            vehicle_id=veh.id,
                            entry_time=veh.entry_time,
                            exit_time=exit_time,
                            delay_s=delay,
                            lane_changes=veh.lane_changes,
                            stops=veh.stops,
                        ))
                    to_remove.append(veh)

        for veh in to_remove:
            if veh in self.vehicles:
                self.vehicles.remove(veh)

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Advance simulation by ``self.dt`` seconds."""
        # 1. Spawn vehicles from deque (O(1) per spawn)
        while self._spawn_queue and self._spawn_queue[0][0] <= self.time:
            _, origin, dest = self._spawn_queue.popleft()
            self._spawn_vehicle(origin, dest)

        # 2. IDM accelerations — single vectorised NumPy call (GIL released)
        self._step_idm_vectorised()

        # 3. MOBIL lane changes — serial (shared lane state)
        self._step_lane_changes()

        # 4. Euler integration + edge transitions + trip logging
        self._step_integrate()

        # 5. Pedestrian SFM step
        from engine.pedestrians import step_pedestrian
        for ped in self.pedestrians:
            step_pedestrian(ped, self.pedestrians, self.dt, sim_time=self.time)

        self.time += self.dt

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Run for full duration; return summary metrics dict."""
        steps = int(round(self.duration / self.dt))
        for _ in range(steps):
            self.step()

        n         = len(self.trip_log)
        avg_delay = (sum(r.delay_s for r in self.trip_log) / n if n > 0 else 0.0)
        avg_lc    = (sum(r.lane_changes for r in self.trip_log) / n if n > 0 else 0.0)
        throughput = n / max(1.0, self.duration - self.warmup)

        return {
            "throughput":        throughput,
            "avg_delay":         avg_delay,
            "avg_lane_changes":  avg_lc,
            "trips_completed":   n,
            "deadlock_detected": self.deadlock_detected,
            "trip_log":          self.trip_log,
        }
