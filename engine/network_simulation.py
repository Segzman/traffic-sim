"""NetworkSimulation — multi-edge routing simulation over a road network.

Vehicles follow pre-computed routes (lists of edge IDs) and transfer between
edges as they reach the end of each segment.  Per-edge IDM and basic signal
logic are applied via the existing Simulation class.  The simulation also
supports pedestrian agents (Social Force Model) and detects deadlock.
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from engine.agents import Vehicle
from engine.idm import idm_acceleration
from engine.network import Network
from engine.metrics import TripRecord, BatchMetrics

if TYPE_CHECKING:
    from engine.pedestrians import Pedestrian


# Speed threshold for "stopped" (m/s)
_STOP_SPEED = 0.3
# Deadlock threshold: stuck below _STOP_SPEED for this many consecutive seconds
_DEADLOCK_STUCK_S = 120.0


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
        Timestep (s).
    seed:
        RNG seed.
    warmup:
        Warmup period (s); trips finishing before this time are excluded from
        the trip log.
    signal_plans:
        ``{node_id: SignalPlan}`` — optional pre-built signal plans.
    pedestrians:
        Optional list of :class:`~engine.pedestrians.Pedestrian` agents.
    """

    dt = 0.1

    def __init__(
        self,
        network: Network,
        demand: dict[str, dict[str, float]] | None = None,
        duration: float = 300.0,
        seed: int = 42,
        warmup: float = 0.0,
        signal_plans: dict | None = None,
        pedestrians: list | None = None,
    ):
        self.network = network
        self.demand = demand or {}
        self.duration = duration
        self.warmup = warmup
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.time = 0.0

        self._signal_plans = signal_plans or {}
        self.pedestrians: list[Pedestrian] = pedestrians or []

        # Active vehicles: list of Vehicle (with route fields set)
        self.vehicles: list[Vehicle] = []
        self._next_vid = 0

        # Per-edge vehicle lists: edge_id -> [Vehicle, ...]
        self._edge_vehicles: dict[str, list[Vehicle]] = {
            eid: [] for eid in network.edges
        }

        # Spawn schedule: list of (spawn_time, origin_node, dest_node)
        self._spawn_queue: list[tuple[float, str, str]] = []
        self._demand_mult: float = 1.0
        self._build_spawn_queue()

        # Trip log (completed trips after warmup)
        self.trip_log: list[TripRecord] = []

        # Deadlock tracking: vid -> consecutive stuck seconds
        self._stuck_time: dict[int, float] = {}
        self.deadlock_detected = False

    # ------------------------------------------------------------------ #
    # Build spawn schedule from demand matrix
    # ------------------------------------------------------------------ #

    def _build_spawn_queue(self) -> None:
        """Convert demand flows to Poisson spawn events."""
        for origin, dests in self.demand.items():
            for dest, flow_hr in dests.items():
                if flow_hr <= 0:
                    continue
                rate_s = flow_hr / 3600.0
                mean_hw = 1.0 / rate_s
                t = float(self.rng.exponential(mean_hw))
                while t < self.duration:
                    self._spawn_queue.append((t, origin, dest))
                    t += float(self.rng.exponential(mean_hw))
        self._spawn_queue.sort()

    def set_demand_mult(self, mult: float) -> None:
        """Rebuild future spawn events with demand scaled by *mult*.

        Drops all not-yet-processed spawns and regenerates them from the
        current simulation time using the scaled demand rates.
        """
        self._demand_mult = max(0.0, float(mult))
        # Keep only events that have already been consumed (t <= now)
        self._spawn_queue = [(t, o, d) for t, o, d in self._spawn_queue
                             if t <= self.time]
        if self._demand_mult <= 0.0:
            return
        for origin, dests in self.demand.items():
            for dest, flow_hr in dests.items():
                if flow_hr <= 0:
                    continue
                scaled = flow_hr * self._demand_mult
                rate_s = scaled / 3600.0
                mean_hw = 1.0 / rate_s
                t = self.time + float(self.rng.exponential(mean_hw))
                while t < self.duration:
                    self._spawn_queue.append((t, origin, dest))
                    t += float(self.rng.exponential(mean_hw))
        self._spawn_queue.sort()

    # ------------------------------------------------------------------ #
    # Vehicle spawn
    # ------------------------------------------------------------------ #

    def _spawn_vehicle(self, origin: str, dest: str) -> None:
        """Compute route and place vehicle on the first edge."""
        route = self.network.shortest_path(origin, dest)
        if not route:
            return  # no path — skip

        first_edge = route[0]
        edge = self.network.edges[first_edge]
        edge_len = self.network.edge_length(first_edge)

        # Check first edge clear at entry
        existing = self._edge_vehicles.get(first_edge, [])
        if any(v.position_s < 15.0 for v in existing):
            return  # entry blocked — skip this spawn

        # Assign lane: rightmost (lane 0) biased, but allow outer lanes on
        # multi-lane edges so vehicles spread visibly across the carriageway.
        num_lanes = edge.num_lanes
        if num_lanes <= 1:
            spawn_lane = 0
        else:
            # Weight: lane 0 (right/slow) gets 40%, remaining share equally
            weights = [0.40] + [0.60 / (num_lanes - 1)] * (num_lanes - 1)
            spawn_lane = int(self.py_rng.choices(range(num_lanes), weights=weights)[0])

        v = Vehicle(
            id=self._next_vid,
            lane_id=spawn_lane,
            position_s=1.0,
            speed=edge.speed_limit,
            v0=edge.speed_limit,
            s0=2.0,
            T=1.5,
            a_max=1.4,
            b=2.0,
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
    # Main step
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Advance simulation by dt."""
        # --- Spawn vehicles from queue ---
        while self._spawn_queue and self._spawn_queue[0][0] <= self.time:
            _, origin, dest = self._spawn_queue.pop(0)
            self._spawn_vehicle(origin, dest)

        # --- IDM acceleration per edge ---
        for edge_id, evs in self._edge_vehicles.items():
            if not evs:
                continue
            edge = self.network.edges[edge_id]
            edge_len = self.network.edge_length(edge_id)

            # Sort front-to-back
            evs_sorted = sorted(evs, key=lambda v: v.position_s, reverse=True)

            # Check if node at end of edge is red
            to_node = edge.to_node
            color = self._signal_color(to_node)
            stop_line = edge_len

            for i, veh in enumerate(evs_sorted):
                if i == 0:
                    # Front vehicle: free-flow or stop at red
                    if color == "red":
                        dist = stop_line - veh.position_s - veh.length / 2.0
                        if 0.0 < dist <= 200.0:
                            accel = idm_acceleration(
                                v=veh.speed, v0=veh.v0, s=max(0.001, dist),
                                delta_v=veh.speed,
                                s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                            )
                        else:
                            accel = idm_acceleration(
                                v=veh.speed, v0=veh.v0, s=1e9, delta_v=0.0,
                                s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                            )
                    else:
                        accel = idm_acceleration(
                            v=veh.speed, v0=veh.v0, s=1e9, delta_v=0.0,
                            s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                        )
                else:
                    leader = evs_sorted[i - 1]
                    gap = leader.position_s - veh.position_s - veh.length
                    if gap <= 0.0:
                        accel = -veh.b
                    else:
                        accel = idm_acceleration(
                            v=veh.speed, v0=veh.v0, s=gap,
                            delta_v=veh.speed - leader.speed,
                            s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                        )

                # Vehicle-pedestrian yield: brake hard if ped TTC < 2 s
                for ped in self.pedestrians:
                    if ped.exit_time >= 0.0:
                        continue
                    # Rough proximity check (within 20 m of vehicle)
                    if abs(ped.x - veh.position_s) < 20.0:
                        if veh.speed > 0.01:
                            ttc = (ped.x - veh.position_s) / veh.speed
                            if 0.0 < ttc < 2.0:
                                accel = min(accel, -veh.b)

                veh.acceleration = accel

        # --- MOBIL lane changes (per edge, multi-lane only) ---
        from engine.mobil import mobil_lane_change
        for edge_id, evs in self._edge_vehicles.items():
            if len(evs) < 2:
                continue
            edge = self.network.edges[edge_id]
            if edge.num_lanes < 2:
                continue
            # Build per-lane sorted lists for this edge
            by_lane: dict[int, list] = {}
            for v in evs:
                by_lane.setdefault(v.lane_id, []).append(v)
            for lst in by_lane.values():
                lst.sort(key=lambda v: v.position_s, reverse=True)

            for veh in list(evs):
                if veh.lane_change_cooldown > 0.0:
                    continue  # cooldown decremented in the outer loop below

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
                changed = False
                for tgt in [veh.lane_id - 1, veh.lane_id + 1]:
                    if tgt < 0 or tgt >= edge.num_lanes:
                        continue
                    tgt_ldr, tgt_fol = _neighbors(tgt)
                    if mobil_lane_change(
                        veh, cur_fol, tgt_ldr, tgt_fol,
                        politeness=0.3, b_safe=3.0,
                        delta_a_thr=0.1, bias_right=(tgt < veh.lane_id),
                        moving_right=(tgt < veh.lane_id),
                    ):
                        by_lane.setdefault(veh.lane_id, []).remove(veh)
                        veh.lane_id = tgt
                        veh.lane_change_cooldown = 4.0
                        by_lane.setdefault(tgt, []).append(veh)
                        by_lane[tgt].sort(key=lambda u: u.position_s, reverse=True)
                        changed = True
                        break

        # Decrement cooldowns for vehicles not processed above
        for veh in self.vehicles:
            if veh.lane_change_cooldown > 0.0:
                veh.lane_change_cooldown = max(0.0, veh.lane_change_cooldown - self.dt)

        # --- Euler integration + stop counting + edge transitions ---
        to_remove: list[Vehicle] = []

        for veh in self.vehicles:
            # Count stops
            prev_stopped = veh._was_stopped
            now_stopped = veh.speed < _STOP_SPEED
            if now_stopped and not prev_stopped:
                veh.stops += 1
            veh._was_stopped = now_stopped

            # Integrate
            veh.speed = max(0.0, veh.speed + veh.acceleration * self.dt)
            veh.position_s += veh.speed * self.dt

            # Deadlock tracking
            if veh.speed < _STOP_SPEED:
                self._stuck_time[veh.id] = self._stuck_time.get(veh.id, 0.0) + self.dt
                if self._stuck_time[veh.id] > _DEADLOCK_STUCK_S:
                    self.deadlock_detected = True
            else:
                self._stuck_time[veh.id] = 0.0

            # Edge transition
            edge_id = veh.current_edge
            edge_len = self.network.edge_length(edge_id)

            if veh.position_s >= edge_len:
                overflow = veh.position_s - edge_len
                # Remove from current edge
                if edge_id in self._edge_vehicles and veh in self._edge_vehicles[edge_id]:
                    self._edge_vehicles[edge_id].remove(veh)

                next_idx = veh.route_index + 1
                if next_idx < len(veh.route):
                    # Advance to next edge
                    next_edge_id = veh.route[next_idx]
                    next_edge = self.network.edges[next_edge_id]
                    veh.route_index = next_idx
                    veh.current_edge = next_edge_id
                    veh.position_s = overflow
                    veh.v0 = next_edge.speed_limit
                    self._edge_vehicles[next_edge_id].append(veh)
                else:
                    # Trip complete
                    exit_time = self.time
                    if exit_time >= self.warmup:
                        # Compute delay vs free-flow
                        ff_time = sum(
                            self.network.edge_travel_time(eid)
                            for eid in veh.route
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

        # --- Pedestrian step ---
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

        n = len(self.trip_log)
        avg_delay = (
            sum(r.delay_s for r in self.trip_log) / n if n > 0 else 0.0
        )
        avg_lane_changes = (
            sum(r.lane_changes for r in self.trip_log) / n if n > 0 else 0.0
        )
        throughput = n / max(1.0, self.duration - self.warmup)

        return {
            "throughput": throughput,
            "avg_delay": avg_delay,
            "avg_lane_changes": avg_lane_changes,
            "trips_completed": n,
            "deadlock_detected": self.deadlock_detected,
            "trip_log": self.trip_log,
        }
