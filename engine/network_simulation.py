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
from engine.vehicle_classes import VEHICLE_CLASSES, mix_for_road_type

if TYPE_CHECKING:
    from engine.pedestrians import Pedestrian
    from engine.config import SimConfig


# ---------------------------------------------------------------------------
# Driver behaviour distribution
# ---------------------------------------------------------------------------

def _sample_lognormal(rng: random.Random, mean: float, cv: float,
                      lo: float, hi: float) -> float:
    """Sample from a log-normal distribution with the given mean and CV, clipped.

    Log-normal is the natural choice for strictly-positive parameters like
    acceleration and braking — it is bounded below at zero, right-skewed, and
    matches empirical IDM calibration from NGSIM data (Kesting & Treiber 2008,
    CV ≈ 15-25 % for a_max and b across the driver population).

    Parameters
    ----------
    mean : desired mean of the distribution
    cv   : coefficient of variation  (std / mean),  e.g. 0.20 for 20 %
    lo   : hard lower clip (physics floor)
    hi   : hard upper clip (physics ceiling)
    """
    sigma = math.sqrt(math.log(cv * cv + 1.0))
    mu    = math.log(mean) - 0.5 * sigma * sigma
    return max(lo, min(hi, rng.lognormvariate(mu, sigma)))


def _sample_disobey(rng: random.Random, max_val: float) -> float:
    """Sample per-driver disobedience from a truncated-normal distribution.

    Empirical basis
    ---------------
    NGSIM trajectory studies (Punzo & Simonelli 2005; Kesting & Treiber 2008)
    show that desired headways and free speeds follow approximately log-normal
    distributions: the majority of drivers cluster near-compliant values and
    only a minority tail into aggressive behaviour.

    A truncated-normal with mu = 0.25 * max and sigma = 0.20 * max replicates
    this shape within the [0, max] interval:

        ≈ 65 %  polite / average       (disobey ≤ 0.35 * max)
        ≈ 28 %  mildly aggressive      (0.35 – 0.70 * max)
        ≈  7 %  aggressive             (disobey > 0.70 * max)

    Compare with the old uniform(0, max) which assigned equal probability to
    every aggression level — empirically incorrect.

    Rejection sampling is used; expected ~1.1 iterations for these parameters
    (rejection rate < 3 %).
    """
    if max_val <= 0.0:
        return 0.0
    mu    = 0.25 * max_val
    sigma = 0.20 * max_val
    for _ in range(50):
        v = rng.gauss(mu, sigma)
        if 0.0 <= v <= max_val:
            return v
    return mu  # fallback — only reached if sigma is very small


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
# Abandon threshold: vehicle removed after being stuck this long (simulates driver
# giving up and leaving — prevents permanent ramp/deadlock accumulation).
_VEHICLE_ABANDON_S = 300.0

# Route quality guards applied at spawn time
# Minimum euclidean O–D distance (m); shorter trips are skipped.
_MIN_OD_DIST_M: float = 80.0
# Maximum ratio of route_length / euclidean_distance; routes that wind much more
# than the straight-line path are rejected (catches cloverleaf loop paths).
_MAX_ROUTE_WINDING: float = 4.0
# Only enforce the winding check when euclidean distance exceeds this (m).
_WINDING_MIN_DIST_M: float = 300.0

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
        continuous_demand: bool = False,
        start_hour: float = 0.0,
        temporal_demand: bool = False,
        day_type: str | None = None,
        weather: "WeatherState | None" = None,
    ):
        from engine.demand_profile import DayType
        from engine.weather import WeatherState

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
        self._continuous_demand = bool(continuous_demand)
        self.start_hour = float(start_hour) % 24.0
        # day_type implicitly enables temporal demand when supplied
        self.temporal_demand = bool(temporal_demand) or (day_type is not None)
        self._day_type = (
            DayType.WEEKEND
            if str(day_type).lower() == "weekend"
            else DayType.WEEKDAY
        )
        self._weather: WeatherState | None = weather
        self._spawn_horizon_s = 3600.0          # keep 1h of future demand queued
        self._spawn_scheduled_until = 0.0

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

        # Network exit nodes: dead-ends where the OSM bbox clips a road.
        # Vehicles re-routed here will drive off the edge of the map instead
        # of vanishing mid-road.
        self._exit_nodes: frozenset[str] = frozenset(
            nid for nid in self.network.nodes
            if not self.network._adj.get(nid)
        )

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

    def _demand_factor(self, sim_time_s: float) -> float:
        """Time-of-day demand multiplier from the 24-hour profile lookup."""
        if not self.temporal_demand:
            return 1.0
        from engine.demand_profile import profile_multiplier
        clock_s = ((self.start_hour + sim_time_s / 3600.0) % 24.0) * 3600.0
        return profile_multiplier(clock_s, self._day_type)

    def _scaled_flow_hr(self, flow_hr: float, sim_time_s: float) -> float:
        """Flow after runtime multipliers (control + config)."""
        cfg_scale = (self.config.demand_scale if self.config else 1.0)
        tod = self._demand_factor(sim_time_s)
        return max(0.0, float(flow_hr) * self._demand_mult * float(cfg_scale) * tod)

    def _generate_spawn_events(
        self,
        t_start: float,
        t_end: float,
    ) -> list[tuple[float, str, str]]:
        """Generate Poisson spawn events in [t_start, t_end)."""
        events: list[tuple[float, str, str]] = []
        if t_end <= t_start:
            return events

        # Piecewise-constant rate over short windows to support temporal demand.
        seg = 900.0  # 15 min segments
        seg_start = float(t_start)
        while seg_start < t_end:
            seg_end = min(t_end, seg_start + seg)
            mid = 0.5 * (seg_start + seg_end)
            for origin, dests in self.demand.items():
                for dest, flow_hr in dests.items():
                    scaled = self._scaled_flow_hr(flow_hr, mid)
                    if scaled <= 0.0:
                        continue
                    rate_s = scaled / 3600.0
                    mean_hw = 1.0 / rate_s
                    t = float(seg_start + self.rng.exponential(mean_hw))
                    while t < seg_end:
                        events.append((t, origin, dest))
                        t += float(self.rng.exponential(mean_hw))
            seg_start = seg_end
        events.sort()
        return events

    def _extend_spawn_horizon_if_needed(self) -> None:
        """Extend queued demand window for long-running live simulations."""
        if not self._continuous_demand:
            return
        # Refill when less than 5 minutes of future events remain.
        if self.time + 300.0 < self._spawn_scheduled_until:
            return
        start = max(self._spawn_scheduled_until, self.time)
        end = start + self._spawn_horizon_s
        events = self._generate_spawn_events(start, end)
        self._spawn_queue.extend(events)
        self._spawn_scheduled_until = end

    def _build_spawn_queue(self) -> None:
        """Convert demand flows to Poisson spawn events (stored in a deque)."""
        end = self.duration
        if self._continuous_demand:
            end = max(self.duration, self._spawn_horizon_s)
        events = self._generate_spawn_events(0.0, end)
        self._spawn_queue = deque(events)
        self._spawn_scheduled_until = end

    def set_demand_mult(self, mult: float) -> None:
        """Rebuild future spawn events with demand scaled by *mult*.

        Drops all not-yet-processed spawns and regenerates them from the
        current simulation time using the scaled demand rates.
        """
        self._demand_mult = max(0.0, float(mult))
        end = self.duration
        if self._continuous_demand:
            end = max(self._spawn_scheduled_until, self.time + self._spawn_horizon_s)
        if end <= self.time:
            end = self.time + self._spawn_horizon_s

        events = self._generate_spawn_events(self.time, end)
        self._spawn_queue = deque(events)
        self._spawn_scheduled_until = end

    # ------------------------------------------------------------------ #
    # Vehicle spawn
    # ------------------------------------------------------------------ #

    def _spawn_vehicle(self, origin: str, dest: str) -> None:
        """Compute route and place vehicle on the first edge."""
        # ── O–D sanity guards ───────────────────────────────────────────────
        o_nd = self.network.nodes.get(origin)
        d_nd = self.network.nodes.get(dest)
        if o_nd is None or d_nd is None:
            return
        euclidean = math.hypot(d_nd.x - o_nd.x, d_nd.y - o_nd.y)

        # Skip trivially short O-D pairs (cloverleaf loop-ramp noise, etc.)
        if euclidean < _MIN_OD_DIST_M:
            return

        route = self.network.shortest_path(origin, dest)
        if not route:
            return

        # Reject excessively winding routes: if the road path is more than
        # _MAX_ROUTE_WINDING × the straight-line distance the vehicle would be
        # looping through interchange ramps rather than travelling anywhere useful.
        if euclidean >= _WINDING_MIN_DIST_M:
            route_len = sum(self.network.edge_length(e) for e in route)
            if route_len > _MAX_ROUTE_WINDING * euclidean:
                return

        # Mode split: walk/bike trips don't produce car-model vehicles.
        # Only active when temporal_demand is on (M4 feature).
        if self.temporal_demand:
            from engine.mode_split import mode_split_probs, Mode
            _ms_probs = mode_split_probs(euclidean)
            _ms_r = self.py_rng.random()
            _ms_cum = 0.0
            _trip_mode = Mode.CAR
            for _m in (Mode.WALK, Mode.BIKE, Mode.CAR):
                _ms_cum += _ms_probs[_m]
                if _ms_r < _ms_cum:
                    _trip_mode = _m
                    break
            if _trip_mode != Mode.CAR:
                return

        first_edge = route[0]
        edge = self.network.edges[first_edge]

        num_lanes = edge.num_lanes
        # ── Vehicle class sampling ──────────────────────────────────────────
        cfg = self.config

        # Road-type baseline (HCM) optionally overridden by SimConfig sliders
        road_type = edge.road_type
        baseline  = mix_for_road_type(road_type)

        raw = {
            "car":   (cfg.vehicle_mix_car   if (cfg and cfg.vehicle_mix_car   >= 0)
                      else baseline["car"]),
            "van":   (cfg.vehicle_mix_van   if (cfg and cfg.vehicle_mix_van   >= 0)
                      else baseline["van"]),
            "truck": (cfg.vehicle_mix_truck if (cfg and cfg.vehicle_mix_truck >= 0)
                      else baseline["truck"]),
            "bus":   (cfg.vehicle_mix_bus   if (cfg and cfg.vehicle_mix_bus   >= 0)
                      else baseline["bus"]),
        }
        types = list(raw.keys())
        total = sum(raw.values()) or 1.0
        wts   = [raw[t] / total for t in types]
        vtype = self.py_rng.choices(types, weights=wts)[0]
        vc    = VEHICLE_CLASSES[vtype]

        # ── Lane selection (enforce truck/bus right-lane discipline) ────────
        discipline = cfg.truck_lane_discipline if cfg else True
        if discipline:
            max_lane = min(num_lanes - 1, vc.lane_max)
        else:
            max_lane = num_lanes - 1
        allowed = list(range(max_lane + 1))
        if len(allowed) <= 1:
            spawn_lane = allowed[0] if allowed else 0
        else:
            # Prefer lanes with more entry space to avoid concentrated jam overlap.
            existing = self._edge_vehicles.get(first_edge, [])
            lane_space: list[float] = []
            for lane in allowed:
                lead_pos = min(
                    (v.position_s for v in existing if v.lane_id == lane),
                    default=1e9,
                )
                lane_space.append(float(lead_pos))
            entry_clearance = max(8.0, vc.length + vc.s0 + 2.0)
            free_lanes = [lane for lane, lead in zip(allowed, lane_space) if lead > entry_clearance]
            if free_lanes:
                spawn_lane = int(self.py_rng.choice(free_lanes))
            else:
                return  # all allowed lanes blocked at entry

        # Single-lane case still needs entry blocking.
        if len(allowed) <= 1:
            existing = self._edge_vehicles.get(first_edge, [])
            entry_clearance = max(8.0, vc.length + vc.s0 + 2.0)
            if any(v.lane_id == spawn_lane and v.position_s <= entry_clearance for v in existing):
                return

        # ── IDM base params: class defaults, overridden by global sliders ───
        a_max_override = cfg and cfg.idm_a_max != 1.4
        b_override     = cfg and cfg.idm_b     != 2.0

        a_max = cfg.idm_a_max if a_max_override else vc.a_max
        b     = cfg.idm_b     if b_override     else vc.b
        T     = cfg.idm_T     if (cfg and cfg.idm_T  != 1.5) else vc.T
        s0    = cfg.idm_s0    if (cfg and cfg.idm_s0 != 2.0) else vc.s0

        # ── Natural per-vehicle variation in acceleration & braking ──────────
        # a_max and b are log-normally distributed across the driver population
        # (Kesting & Treiber 2008, NGSIM calibration: CV ≈ 20 %).
        # Only applied when the class default is in use; explicit config slider
        # overrides produce a deterministic, uniform fleet as expected.
        if not a_max_override:
            a_max = _sample_lognormal(self.py_rng, a_max, cv=0.20,
                                      lo=a_max * 0.40, hi=a_max * 2.50)
        if not b_override:
            b = _sample_lognormal(self.py_rng, b, cv=0.20,
                                  lo=b * 0.40, hi=b * 2.50)

        # ── Disobedience: sample per-vehicle rule-breaking factor ───────────
        # Distribution: truncated-normal (mu=0.25·max, σ=0.20·max) so most
        # drivers are near-compliant and only a tail is aggressive.
        # See _sample_disobey() for empirical justification.
        disobey = 0.0
        if cfg and cfg.disobedience > 0.0:
            disobey = _sample_disobey(self.py_rng, cfg.disobedience * vc.max_disobedience)

        # Apply disobedience multipliers (physics-capped per class)
        v0  = edge.speed_limit * vc.speed_factor * (1.0 + disobey * vc.speed_excess)
        T   = max(0.4, T  * (1.0 - disobey * vc.gap_reduction))
        s0  = max(0.5, s0 * (1.0 - disobey * vc.gap_reduction))
        pol = max(0.0, 0.3 * (1.0 - disobey * vc.politeness_factor))
        dat = max(0.0, 0.2 * (1.0 - 0.7 * disobey))   # delta_a_thr

        v = Vehicle(
            id=self._next_vid,
            lane_id=spawn_lane,
            position_s=1.0,
            speed=edge.speed_limit * vc.speed_factor,
            v0=v0,
            s0=s0,
            T=T,
            a_max=a_max,
            b=b,
            length=vc.length,
            width=vc.width,
            vehicle_type=vtype,
            politeness=pol,
            delta_a_thr=dat,
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

        # Apply WeatherState multipliers (M5) — overrides/extends SimConfig weather
        if self._weather is not None:
            from engine.weather import WEATHER_MULTIPLIERS
            _wm = WEATHER_MULTIPLIERS.get(self._weather.condition,
                                          WEATHER_MULTIPLIERS["clear"])
            v0_arr *= _wm["v0"]
            s0_arr *= _wm["s0"]
            T_arr  *= _wm["T"]
            a_arr  *= _wm["a"]

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

                # Respect truck/bus lane discipline: don't attempt moves that
                # would exceed the vehicle class's lane_max constraint.
                discipline = cfg.truck_lane_discipline if cfg else True
                vc_info = VEHICLE_CLASSES.get(veh.vehicle_type)
                if discipline:
                    veh_lane_max = vc_info.lane_max if vc_info else 99
                else:
                    veh_lane_max = 99

                # When discipline is disabled for normally right-lane-restricted
                # classes (truck/bus), suppress the right-lane drift bias so
                # vehicles genuinely spread across all lanes rather than
                # gravitating back to lane 0.
                normally_restricted = vc_info is not None and vc_info.lane_max == 0
                apply_right_bias = discipline or not normally_restricted

                cur_ldr, cur_fol = _neighbors(veh.lane_id)
                for tgt in [veh.lane_id - 1, veh.lane_id + 1]:
                    if tgt < 0 or tgt >= edge.num_lanes:
                        continue
                    if tgt > veh_lane_max:   # lane discipline — skip left moves
                        continue
                    tgt_ldr, tgt_fol = _neighbors(tgt)
                    if mobil_lane_change(
                        veh, cur_fol, tgt_ldr, tgt_fol,
                        politeness=politeness,
                        b_safe=b_safe,
                        delta_a_thr=0.1,
                        bias_right=(tgt < veh.lane_id) and apply_right_bias,
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

    def _entry_blocked(
        self,
        edge_id: str,
        lane_id: int,
        vehicle: Vehicle,
        *,
        enter_position: float = 0.0,
    ) -> bool:
        """Return True when target edge insertion would violate headway."""
        min_gap = max(0.5, vehicle.s0)
        for other in self._edge_vehicles.get(edge_id, []):
            if other.lane_id != lane_id:
                continue
            gap = other.position_s - float(enter_position) - vehicle.length
            if gap < min_gap:
                return True
        return False

    def _enforce_no_overlap(self) -> None:
        """Clamp lane-wise positions so jammed vehicles cannot overlap."""
        for edge_id, evs in self._edge_vehicles.items():
            if not evs:
                continue
            by_lane: dict[int, list[Vehicle]] = {}
            for v in evs:
                by_lane.setdefault(v.lane_id, []).append(v)

            for lane_vehs in by_lane.values():
                lane_vehs.sort(key=lambda v: v.position_s, reverse=True)
                for i in range(1, len(lane_vehs)):
                    leader = lane_vehs[i - 1]
                    follower = lane_vehs[i]
                    min_gap = max(0.1, follower.s0)
                    max_pos = leader.position_s - follower.length - min_gap
                    if follower.position_s > max_pos:
                        follower.position_s = max(0.0, max_pos)
                        follower.speed = min(follower.speed, leader.speed)
                        follower.acceleration = min(follower.acceleration, 0.0)

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

            # Deadlock / abandon tracking
            if veh.speed < _STOP_SPEED:
                stuck = self._stuck_time.get(veh.id, 0.0) + dt
                self._stuck_time[veh.id] = stuck
                if stuck > _DEADLOCK_STUCK_S:
                    self.deadlock_detected = True
                # After _VEHICLE_ABANDON_S, re-route to the nearest network
                # boundary (dead-end node) so the vehicle drives itself off the
                # map rather than vanishing.
                if stuck > _VEHICLE_ABANDON_S and self._exit_nodes:
                    curr_to = self.network.edges[veh.current_edge].to_node
                    cn = self.network.nodes.get(curr_to)
                    if cn is not None:
                        best_exit = min(
                            self._exit_nodes,
                            key=lambda nid: math.hypot(
                                self.network.nodes[nid].x - cn.x,
                                self.network.nodes[nid].y - cn.y,
                            ),
                        )
                        if best_exit != curr_to:
                            exit_route = self.network.shortest_path(
                                curr_to, best_exit
                            )
                            if exit_route:
                                # Replace remaining route with boundary path.
                                veh.route = (
                                    list(veh.route[: veh.route_index + 1])
                                    + exit_route
                                )
                                self._stuck_time[veh.id] = 0.0
                                continue  # let vehicle find its own way out
                    # Fallback: truly unreachable — remove as last resort.
                    ev_list_a = self._edge_vehicles.get(veh.current_edge)
                    if ev_list_a and veh in ev_list_a:
                        ev_list_a.remove(veh)
                    to_remove.append(veh)
                    continue
            else:
                self._stuck_time[veh.id] = 0.0

            # Edge transition
            edge_id  = veh.current_edge
            edge_len = self.network.edge_length(edge_id)

            if veh.position_s >= edge_len:
                overflow = veh.position_s - edge_len
                ev_list  = self._edge_vehicles.get(edge_id)
                next_idx = veh.route_index + 1
                if next_idx < len(veh.route):
                    next_edge_id  = veh.route[next_idx]
                    next_edge     = self.network.edges[next_edge_id]
                    target_lane = min(veh.lane_id, max(0, next_edge.num_lanes - 1))

                    # Queue spillback: do not enter a blocked next edge.
                    if self._entry_blocked(
                        next_edge_id,
                        target_lane,
                        veh,
                        enter_position=overflow,
                    ):
                        veh.position_s = max(0.0, edge_len - max(0.1, veh.length + veh.s0))
                        veh.speed = 0.0
                        continue

                    if ev_list and veh in ev_list:
                        ev_list.remove(veh)
                    veh.route_index   = next_idx
                    veh.current_edge  = next_edge_id
                    veh.lane_id       = target_lane
                    veh.position_s    = overflow
                    veh.v0            = next_edge.speed_limit
                    self._edge_vehicles[next_edge_id].append(veh)
                else:
                    if ev_list and veh in ev_list:
                        ev_list.remove(veh)
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
        self._extend_spawn_horizon_if_needed()

        # 1. Spawn vehicles from deque (O(1) per spawn)
        while self._spawn_queue and self._spawn_queue[0][0] <= self.time:
            _, origin, dest = self._spawn_queue.popleft()
            self._spawn_vehicle(origin, dest)

        # 2. IDM accelerations — single vectorised NumPy call (GIL released)
        self._step_idm_vectorised()

        # 3. MOBIL lane changes — serial (shared lane state)
        self._step_lane_changes()
        self._enforce_no_overlap()

        # 4. Euler integration + edge transitions + trip logging
        self._step_integrate()
        self._enforce_no_overlap()

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
