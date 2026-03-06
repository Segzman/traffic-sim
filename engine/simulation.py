"""Multi-lane timestep loop — Euler integration at 0.1 s."""
from __future__ import annotations

import math
import numpy as np

from engine.idm import idm_acceleration
from engine.agents import Vehicle
from engine.metrics import MetricsRecorder


class Simulation:
    dt = 0.1          # timestep in seconds
    COOLDOWN = 3.0    # default lane-change cooldown (s)

    def __init__(self, scenario: dict, vehicles: list[Vehicle] | None = None):
        road = scenario["road"]
        self.road_length: float = road["length"]
        self.speed_limit: float = road["speed_limit"]
        self.num_lanes: int = road.get("num_lanes", 1)
        self.duration: float = scenario.get("duration", 60.0)

        seed = scenario.get("seed", 42)
        self.rng = np.random.default_rng(seed)

        if vehicles is not None:
            self.vehicles = vehicles
        else:
            self.vehicles = self._init_vehicles(scenario)

        # ---------- Junction setup ----------
        self._junction_cfg: dict | None = scenario.get("junction")
        self._signal_plan = None
        self._conflict_times: list[float] = []
        self._veh_state: dict[int, dict] = {}   # vid -> per-vehicle junction state
        self._init_junction(scenario)

        # ---------- Spawner setup ----------
        self._spawner_cfg: dict | None = scenario.get("spawner")
        self._spawner_next_t: float = 0.0
        self._next_vid: int = max((v.id for v in self.vehicles), default=-1) + 1
        if self._spawner_cfg:
            # First spawn at a random time within first headway (avoids bunching)
            rate_s = self._spawner_cfg.get("arrival_rate_veh_hr", 200) / 3600.0
            self._spawner_next_t = float(self.rng.uniform(0.0, 1.0 / rate_s))

        jcfg = self._junction_cfg or {}
        self.metrics = MetricsRecorder(
            self.road_length,
            num_lanes=self.num_lanes,
            stop_line=jcfg.get("stop_line", self.road_length),
            detection_distance=jcfg.get("detection_distance", 100.0),
        )
        self.time = 0.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _sample(self, param) -> float:
        """Return a float from a fixed value or {'mean': m, 'sd': s} dict."""
        if isinstance(param, dict):
            return float(self.rng.normal(param["mean"], param["sd"]))
        return float(param)

    def _init_vehicles(self, scenario: dict) -> list[Vehicle]:
        vconfig = scenario.get("vehicles", {})

        if isinstance(vconfig, list):
            return [Vehicle(**v) if isinstance(v, dict) else v for v in vconfig]

        # Generate from count + parameters
        count = vconfig.get("count", 10)
        initial_speed = vconfig.get("initial_speed", 20.0)
        params = vconfig.get("idm_params", {})

        # Distribute vehicles evenly across lanes
        vehicles: list[Vehicle] = []
        vid = 0
        for lane_id in range(self.num_lanes):
            n_in_lane = count // self.num_lanes + (1 if lane_id < count % self.num_lanes else 0)
            spacing = self.road_length / (n_in_lane + 1)
            for j in range(n_in_lane):
                v = Vehicle(
                    id=vid,
                    lane_id=lane_id,
                    position_s=(j + 1) * spacing,
                    speed=initial_speed,
                    v0=max(5.0, self._sample(params.get("v0", 30.0))),
                    s0=float(params.get("s0", 2.0)),
                    T=float(params.get("T", 1.5)),
                    a_max=float(params.get("a_max", 1.4)),
                    b=float(params.get("b", 2.0)),
                )
                vehicles.append(v)
                vid += 1
        return vehicles

    def _init_junction(self, scenario: dict) -> None:
        """Parse junction config and pre-generate conflict arrival times."""
        jcfg = self._junction_cfg
        if jcfg is None:
            return

        jtype = jcfg.get("type", "uncontrolled")

        # Build SignalPlan from JSON config
        if jtype == "signal":
            from engine.signals import Phase, SignalPlan
            plan_cfg = jcfg.get("plan", {})
            phases = [
                Phase(
                    green_movements=p["green_movements"],
                    green_duration=float(p["green_duration"]),
                    yellow_duration=float(p.get("yellow_duration", 3.0)),
                    all_red_duration=float(p.get("all_red_duration", 1.0)),
                )
                for p in plan_cfg.get("phases", [])
            ]
            self._signal_plan = SignalPlan(
                node_id=plan_cfg.get("node_id", "J"),
                phases=phases,
                offset=float(plan_cfg.get("offset", 0.0)),
            )

        # Pre-generate Poisson conflict stream for roundabout/yield/stop
        if jtype in ("roundabout", "yield", "stop"):
            flow_hr = float(jcfg.get("conflicting_flow_rate", 400.0))
            flow_per_s = flow_hr / 3600.0
            mean_headway = 1.0 / flow_per_s  # seconds between conflicts
            # Generate conflicts for duration + 60s buffer
            total = self.duration + 60.0
            t = self.rng.exponential(mean_headway)  # first conflict
            while t < total:
                self._conflict_times.append(t)
                t += self.rng.exponential(mean_headway)

        # Initialise per-vehicle state for junction vehicles
        t_min_mean = float(jcfg.get("t_min_mean", 5.0))
        t_min_sd = float(jcfg.get("t_min_sd", 0.5))
        for v in self.vehicles:
            self._get_veh_state(v.id, t_min_mean, t_min_sd)

    def _get_veh_state(self, vid: int,
                       t_min_mean: float = 5.0,
                       t_min_sd: float = 0.5) -> dict:
        """Get or initialise per-vehicle junction state."""
        if vid not in self._veh_state:
            t_min = max(1.0, self.rng.normal(t_min_mean, t_min_sd)
                        if t_min_sd > 0 else t_min_mean)
            self._veh_state[vid] = {
                "t_min": float(t_min),
                "wait_time": 0.0,       # seconds spent waiting at junction
                "has_stopped": False,   # for stop-sign: has vehicle fully stopped?
            }
        return self._veh_state[vid]

    def _effective_t_min(self, vid: int) -> float:
        """Return impatience-adjusted gap acceptance threshold."""
        state = self._veh_state.get(vid, {})
        t_min = state.get("t_min", 5.0)
        wait_time = state.get("wait_time", 0.0)
        # Linearly decrease t_min; minimum 30% of original
        factor = max(0.3, 1.0 - wait_time / 45.0)
        return t_min * factor

    # ------------------------------------------------------------------
    # Junction override helpers
    # ------------------------------------------------------------------

    def _stop_at_line_accel(self, vehicle: Vehicle, dist_to_stop: float) -> float:
        """IDM deceleration targeting the stop line as a stationary leader."""
        s = max(0.001, dist_to_stop)
        return idm_acceleration(
            v=vehicle.speed, v0=vehicle.v0, s=s,
            delta_v=vehicle.speed,   # closing on a stationary target
            s0=vehicle.s0, T=vehicle.T, a=vehicle.a_max, b=vehicle.b,
        )

    def _time_to_next_conflict(self, sim_time: float) -> float:
        """Seconds until next conflict vehicle arrives (Poisson stream)."""
        future = [t for t in self._conflict_times if t > sim_time]
        if not future:
            return 1e9  # no more conflicts
        return min(future) - sim_time

    def _signal_override(self, vehicle: Vehicle, dist_to_stop: float,
                         sim_time: float) -> float | None:
        """Return decel or None (proceed) based on signal state."""
        if self._signal_plan is None:
            return None
        movement_id = self._junction_cfg.get("movement_id")
        state = self._signal_plan.current_state(sim_time, movement_id)
        if state == "green":
            return None
        if state == "yellow":
            # Commit distance: braking distance from current speed
            stopping_dist = vehicle.speed ** 2 / (2.0 * vehicle.b)
            if dist_to_stop <= stopping_dist:
                return None  # committed – cross on yellow
            return self._stop_at_line_accel(vehicle, dist_to_stop)
        # Red
        return self._stop_at_line_accel(vehicle, dist_to_stop)

    def _gap_acceptance_override(self, vehicle: Vehicle, dist_to_stop: float,
                                 sim_time: float) -> float | None:
        """Return decel or None based on gap acceptance (roundabout/yield)."""
        eff_t_min = self._effective_t_min(vehicle.id)
        time_to_next = self._time_to_next_conflict(sim_time)
        if time_to_next > eff_t_min:
            return None  # acceptable gap – proceed
        return self._stop_at_line_accel(vehicle, dist_to_stop)

    def _stop_sign_override(self, vehicle: Vehicle, dist_to_stop: float,
                            sim_time: float) -> float | None:
        """Return decel or None for stop-sign logic."""
        vstate = self._get_veh_state(vehicle.id)
        if not vstate["has_stopped"]:
            if vehicle.speed < 0.05:
                vstate["has_stopped"] = True
                # Now proceed to gap acceptance below
            else:
                return self._stop_at_line_accel(vehicle, dist_to_stop)
        # After full stop: gap acceptance with impatience
        return self._gap_acceptance_override(vehicle, dist_to_stop, sim_time)

    def _junction_override(self, vehicle: Vehicle,
                           sim_time: float) -> float | None:
        """Compute junction acceleration override for the front vehicle.

        Returns None if no override (IDM remains in effect),
        or a float override acceleration value.
        Only applied to the vehicle nearest the stop line (front of queue).
        """
        if self._junction_cfg is None:
            return None

        jtype = self._junction_cfg.get("type", "uncontrolled")
        if jtype == "uncontrolled":
            return None

        stop_line = self._junction_cfg.get("stop_line", self.road_length)
        dist_to_stop = stop_line - vehicle.position_s - vehicle.length / 2.0

        if dist_to_stop <= 0.0:
            return None  # vehicle has cleared the stop line

        detection_dist = self._junction_cfg.get("detection_distance", 100.0)
        if dist_to_stop > detection_dist:
            return None  # vehicle is outside detection zone

        if jtype == "signal":
            return self._signal_override(vehicle, dist_to_stop, sim_time)
        if jtype in ("roundabout", "yield"):
            return self._gap_acceptance_override(vehicle, dist_to_stop, sim_time)
        if jtype == "stop":
            return self._stop_sign_override(vehicle, dist_to_stop, sim_time)
        return None

    def _front_junction_vehicle(self) -> Vehicle | None:
        """Vehicle closest to (but not past) the stop line."""
        if self._junction_cfg is None:
            return None
        stop_line = self._junction_cfg.get("stop_line", self.road_length)
        candidates = [v for v in self.vehicles if v.position_s < stop_line]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.position_s)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _by_lane(self) -> dict[int, list[Vehicle]]:
        """Group vehicles by lane_id; each list is sorted front-to-back."""
        groups: dict[int, list[Vehicle]] = {}
        for v in self.vehicles:
            groups.setdefault(v.lane_id, []).append(v)
        for lst in groups.values():
            lst.sort(key=lambda v: v.position_s, reverse=True)
        return groups

    def _neighbors_in_lane(
        self,
        by_lane: dict[int, list[Vehicle]],
        vehicle: Vehicle,
        lane_id: int,
    ) -> tuple[Vehicle | None, Vehicle | None]:
        """Return (leader, follower) in lane_id relative to vehicle.position_s."""
        lane_vehs = by_lane.get(lane_id, [])
        leader: Vehicle | None = None
        follower: Vehicle | None = None
        for v in lane_vehs:
            if v.id == vehicle.id:
                continue
            if v.position_s > vehicle.position_s:
                if leader is None or v.position_s < leader.position_s:
                    leader = v
            else:
                if follower is None or v.position_s > follower.position_s:
                    follower = v
        return leader, follower

    def _follower_in_lane(
        self,
        by_lane: dict[int, list[Vehicle]],
        vehicle: Vehicle,
        lane_id: int,
    ) -> Vehicle | None:
        """Closest vehicle behind `vehicle` in `lane_id` (excluding itself)."""
        _, follower = self._neighbors_in_lane(by_lane, vehicle, lane_id)
        return follower

    # ------------------------------------------------------------------
    # MOBIL lane-change phase
    # ------------------------------------------------------------------

    def _execute_mobil(self, by_lane: dict[int, list[Vehicle]]) -> None:
        from engine.mobil import mobil_lane_change

        proposals: list[tuple[float, Vehicle, int, bool]] = []

        for veh in self.vehicles:
            if veh.lane_change_cooldown > 0.0:
                continue

            accepted = False
            if veh.lane_id > 0:
                tgt = veh.lane_id - 1
                ldr_t, flw_t = self._neighbors_in_lane(by_lane, veh, tgt)
                flw_c = self._follower_in_lane(by_lane, veh, veh.lane_id)
                if mobil_lane_change(
                    veh, flw_c, ldr_t, flw_t,
                    politeness=veh.politeness, b_safe=veh.b_safe,
                    delta_a_thr=veh.delta_a_thr, bias_right=veh.bias_right,
                    moving_right=True,
                ):
                    proposals.append((veh.position_s, veh, tgt, True))
                    accepted = True

            if not accepted and veh.lane_id < self.num_lanes - 1:
                tgt = veh.lane_id + 1
                ldr_t, flw_t = self._neighbors_in_lane(by_lane, veh, tgt)
                flw_c = self._follower_in_lane(by_lane, veh, veh.lane_id)
                if mobil_lane_change(
                    veh, flw_c, ldr_t, flw_t,
                    politeness=veh.politeness, b_safe=veh.b_safe,
                    delta_a_thr=veh.delta_a_thr, bias_right=veh.bias_right,
                    moving_right=False,
                ):
                    proposals.append((veh.position_s, veh, tgt, False))

        proposals.sort(key=lambda x: x[0], reverse=True)
        new_positions_by_lane: dict[int, list[float]] = {}

        for _, veh, tgt_lane, _ in proposals:
            conflicting = new_positions_by_lane.get(tgt_lane, [])
            if any(abs(p - veh.position_s) < veh.length * 2 for p in conflicting):
                continue
            veh.lane_id = tgt_lane
            veh.lane_change_cooldown = self.COOLDOWN
            new_positions_by_lane.setdefault(tgt_lane, []).append(veh.position_s)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Spawner / vehicle removal
    # ------------------------------------------------------------------

    def _spawn_and_remove(self) -> None:
        """Generate vehicles at a fixed rate; remove vehicles past stop line."""
        sp = self._spawner_cfg
        if sp is None:
            return  # no spawner — keep all vehicles in the list

        # Only remove vehicles that have cleared the junction when a spawner is
        # active (so new arrivals don't pile up behind departed vehicles)
        sl = (self._junction_cfg or {}).get("stop_line", self.road_length)
        self.vehicles = [v for v in self.vehicles if v.position_s <= sl + 20.0]

        rate_s = sp.get("arrival_rate_veh_hr", 200) / 3600.0
        if rate_s <= 0 or self.time < self._spawner_next_t:
            return

        # Check entry is clear (no vehicle within 15 m of position 0)
        if any(v.position_s < 15.0 for v in self.vehicles):
            # Delay spawn by one step but keep target time
            return

        v0 = float(sp.get("v0", self.speed_limit))
        new_v = Vehicle(
            id=self._next_vid, lane_id=0, position_s=1.0, speed=v0,
            v0=v0, s0=2.0, T=1.5, a_max=1.4, b=2.0,
        )
        self.vehicles.append(new_v)
        jcfg = self._junction_cfg or {}
        self._get_veh_state(
            self._next_vid,
            float(jcfg.get("t_min_mean", 5.0)),
            float(jcfg.get("t_min_sd", 0.5)),
        )
        self._next_vid += 1
        self._spawner_next_t += 1.0 / rate_s

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance simulation by one timestep (dt = 0.1 s)."""
        # Spawn new vehicles / remove completed ones
        self._spawn_and_remove()

        # Decrement lane-change cooldowns
        for v in self.vehicles:
            if v.lane_change_cooldown > 0.0:
                v.lane_change_cooldown = max(0.0, v.lane_change_cooldown - self.dt)

        # Group vehicles by lane (sorted front to back)
        by_lane = self._by_lane()

        # Identify the front vehicle for junction priority
        front_veh = self._front_junction_vehicle()

        # Compute IDM accelerations per lane
        for lane_id, lane_vehs in by_lane.items():
            for i, veh in enumerate(lane_vehs):
                if i == 0:
                    accel = idm_acceleration(
                        v=veh.speed, v0=veh.v0, s=1e9, delta_v=0.0,
                        s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                    )
                else:
                    leader = lane_vehs[i - 1]
                    gap = leader.position_s - veh.position_s - veh.length
                    if gap <= 0.0:
                        accel = -veh.b
                    else:
                        accel = idm_acceleration(
                            v=veh.speed, v0=veh.v0, s=gap,
                            delta_v=veh.speed - leader.speed,
                            s0=veh.s0, T=veh.T, a=veh.a_max, b=veh.b,
                        )
                veh.acceleration = accel

        # Junction override — only for front vehicle at the stop line
        if front_veh is not None:
            override = self._junction_override(front_veh, self.time)
            if override is not None:
                front_veh.acceleration = override
                # Update wait time for this vehicle
                vstate = self._veh_state.get(front_veh.id)
                if vstate is not None:
                    if front_veh.speed < 1.0:
                        vstate["wait_time"] += self.dt
            else:
                # Vehicle is free to move — reset wait timer
                vstate = self._veh_state.get(front_veh.id)
                if vstate is not None:
                    vstate["wait_time"] = 0.0

        # MOBIL lane changes (multi-lane only)
        if self.num_lanes > 1:
            self._execute_mobil(by_lane)

        # Euler integration (simultaneous)
        for v in self.vehicles:
            v.speed = max(0.0, v.speed + v.acceleration * self.dt)
            v.position_s += v.speed * self.dt

        self.time += self.dt
        self.metrics.record(self.vehicles, self.time)

    def run(self) -> dict:
        """Run simulation for full duration; return metrics summary."""
        steps = int(round(self.duration / self.dt))
        for _ in range(steps):
            self.step()
        return self.metrics.summary()
