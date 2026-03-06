"""Throughput, average speed, density, lane-utilisation, queue length, delay.

M5 additions
------------
* TripRecord dataclass
* BatchMetrics — run N simulations, compute mean ± 95 % CI
* TripLog — CSV export of individual trip data
* warmup support in MetricsRecorder
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from engine.agents import Vehicle

# Speed threshold below which a vehicle is considered queued / waiting (m/s)
QUEUE_SPEED_THRESHOLD = 2.0
# Speed below which we start counting "stopped" time toward delay (m/s)
DELAY_SPEED_THRESHOLD = 1.0


# ------------------------------------------------------------------ #
# TripRecord
# ------------------------------------------------------------------ #

@dataclass
class TripRecord:
    """Data for a single completed vehicle trip."""
    vehicle_id: int
    entry_time: float
    exit_time: float
    delay_s: float
    lane_changes: int
    stops: int

    # CSV column order
    _COLUMNS = ("vehicle_id", "entry_time", "exit_time", "delay_s",
                "lane_changes", "stops")


# ------------------------------------------------------------------ #
# CSV export
# ------------------------------------------------------------------ #

def trip_log_to_csv(records: list[TripRecord]) -> str:
    """Serialise a list of TripRecord objects to a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(TripRecord._COLUMNS)
    for r in records:
        writer.writerow([
            r.vehicle_id, r.entry_time, r.exit_time,
            r.delay_s, r.lane_changes, r.stops,
        ])
    return buf.getvalue()


def write_trip_csv(records: list[TripRecord], path: str) -> None:
    """Write trip log CSV to *path*."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(trip_log_to_csv(records))


# ------------------------------------------------------------------ #
# Batch mode
# ------------------------------------------------------------------ #

@dataclass
class BatchMetrics:
    """Mean ± 95 % CI for a batch of simulation runs."""
    n: int
    mean: dict[str, float]
    ci_width: dict[str, float]   # half-width of the 95 % CI for each metric

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"BatchMetrics(n={self.n})"]
        for k in sorted(self.mean):
            lines.append(f"  {k}: {self.mean[k]:.4f} ± {self.ci_width[k]:.4f}")
        return "\n".join(lines)


def run_batch(
    sim_factory: Callable[[int], object],
    n: int = 10,
    metrics_keys: list[str] | None = None,
) -> BatchMetrics:
    """Run *n* independent simulations and compute mean ± 95 % CI.

    Parameters
    ----------
    sim_factory:
        Callable that accepts an integer seed and returns a simulation
        object with a ``.run() -> dict`` method.
    n:
        Number of replications.
    metrics_keys:
        Which keys to aggregate.  If None, all numeric keys from the
        first run are used.

    Returns
    -------
    BatchMetrics
    """
    results: list[dict] = []
    for i in range(n):
        sim = sim_factory(i)
        r = sim.run()
        results.append(r)

    if metrics_keys is None:
        metrics_keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]

    # Compute mean and 95 % CI (t-distribution approximated by 1.96 * std / sqrt(n))
    mean: dict[str, float] = {}
    ci_width: dict[str, float] = {}

    for key in metrics_keys:
        vals = [float(r.get(key, 0.0)) for r in results]
        m = sum(vals) / n
        if n > 1:
            var = sum((v - m) ** 2 for v in vals) / (n - 1)
            se = math.sqrt(var / n)
            # Use t-critical value ≈ 1.96 for large n, exact for small n via lookup
            t_crit = _t_critical(n - 1)
            hw = t_crit * se
        else:
            hw = 0.0
        mean[key] = m
        ci_width[key] = hw

    return BatchMetrics(n=n, mean=mean, ci_width=ci_width)


# t-distribution critical values for 95 % CI (two-tailed, α=0.025)
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 100: 1.984,
}


def _t_critical(df: int) -> float:
    """Two-tailed t critical value at α=0.025 for *df* degrees of freedom."""
    if df in _T_TABLE:
        return _T_TABLE[df]
    # For large df use 1.96
    closest = max(k for k in _T_TABLE if k <= df)
    return _T_TABLE[closest]


# ------------------------------------------------------------------ #
# MetricsRecorder (enhanced with warmup support)
# ------------------------------------------------------------------ #

class MetricsRecorder:
    """Records per-step metrics and produces a summary.

    Parameters
    ----------
    warmup:
        Warmup period (s).  Observations before this time are discarded.
    """

    def __init__(self, road_length: float, measurement_point: float = None,
                 num_lanes: int = 1,
                 stop_line: float = None,
                 detection_distance: float = 100.0,
                 warmup: float = 0.0):
        self.road_length = road_length
        self.measurement_point = (
            measurement_point if measurement_point is not None else road_length * 0.9
        )
        self.num_lanes = num_lanes
        self.stop_line = stop_line if stop_line is not None else road_length
        self.detection_distance = detection_distance
        self.warmup = warmup

        # Throughput
        self._throughput_count = 0
        self._prev_positions: dict[int, float] = {}

        # Speed
        self._speed_sum = 0.0
        self._speed_count = 0

        # Density
        self._density_sum = 0.0
        self._density_steps = 0

        # Per-lane utilisation
        self._lane_count_sum: dict[int, float] = {}
        self._lane_steps = 0

        # Queue length (vehicles with speed < threshold within queue zone)
        self._queue_sum = 0
        self._queue_steps = 0
        self._queue_zone_start = self.stop_line - self.detection_distance

        # Per-vehicle delay tracking
        self._vehicle_delay_accumulator: dict[int, float] = {}
        self._vehicle_crossed: set[int] = set()
        self._completed_delays: list[float] = []

        # Trip log
        self.trip_log: list[TripRecord] = []

        self._total_time = 0.0

    def record(self, vehicles: list[Vehicle], time: float) -> None:
        self._total_time = time
        dt = 0.1  # expected simulation timestep

        # Skip recording during warmup
        in_warmup = time < self.warmup

        # Throughput: vehicles crossing the measurement point (count even in warmup
        # to keep _prev_positions consistent, but only increment counter post-warmup)
        for v in vehicles:
            prev = self._prev_positions.get(v.id)
            if prev is not None and prev < self.measurement_point <= v.position_s:
                if not in_warmup:
                    self._throughput_count += 1
            self._prev_positions[v.id] = v.position_s

        if in_warmup:
            return  # skip all other accumulation during warmup

        # Average speed
        if vehicles:
            self._speed_sum += sum(v.speed for v in vehicles)
            self._speed_count += len(vehicles)

        # Density
        active = sum(1 for v in vehicles if 0.0 <= v.position_s <= self.road_length)
        self._density_sum += active / self.road_length
        self._density_steps += 1

        # Per-lane utilisation
        if vehicles:
            lane_counts: dict[int, int] = {}
            for v in vehicles:
                lane_counts[v.lane_id] = lane_counts.get(v.lane_id, 0) + 1
            n = len(vehicles)
            for lid, cnt in lane_counts.items():
                self._lane_count_sum[lid] = self._lane_count_sum.get(lid, 0.0) + cnt / n
        self._lane_steps += 1

        # Queue length
        queue_count = 0
        for v in vehicles:
            in_zone = self._queue_zone_start <= v.position_s < self.stop_line
            if in_zone and v.speed < QUEUE_SPEED_THRESHOLD:
                queue_count += 1
        self._queue_sum += queue_count
        self._queue_steps += 1

        # Delay
        for v in vehicles:
            if v.id not in self._vehicle_crossed and v.position_s >= self.stop_line:
                self._vehicle_crossed.add(v.id)
                if v.id in self._vehicle_delay_accumulator:
                    self._completed_delays.append(self._vehicle_delay_accumulator[v.id])
                continue

            if v.id in self._vehicle_crossed:
                continue

            in_zone = self._queue_zone_start <= v.position_s < self.stop_line
            if in_zone and v.speed < DELAY_SPEED_THRESHOLD:
                acc = self._vehicle_delay_accumulator.get(v.id, 0.0)
                self._vehicle_delay_accumulator[v.id] = acc + dt

    def record_trip(
        self,
        vehicle_id: int,
        entry_time: float,
        exit_time: float,
        delay_s: float,
        lane_changes: int,
        stops: int,
    ) -> None:
        """Add a completed trip to the log (only if exit_time >= warmup)."""
        if exit_time >= self.warmup:
            self.trip_log.append(TripRecord(
                vehicle_id=vehicle_id,
                entry_time=entry_time,
                exit_time=exit_time,
                delay_s=delay_s,
                lane_changes=lane_changes,
                stops=stops,
            ))

    def queue_length(self) -> float:
        """Mean queue length (vehicles) during the simulation."""
        if self._queue_steps == 0:
            return 0.0
        return self._queue_sum / self._queue_steps

    def summary(self) -> dict:
        throughput = (
            self._throughput_count / self._total_time if self._total_time > 0 else 0.0
        )
        avg_speed = (
            self._speed_sum / self._speed_count if self._speed_count > 0 else 0.0
        )
        density = (
            self._density_sum / self._density_steps if self._density_steps > 0 else 0.0
        )
        lane_utilisation: dict[int, float] = {}
        if self._lane_steps > 0:
            for lid, total in self._lane_count_sum.items():
                lane_utilisation[lid] = total / self._lane_steps

        avg_delay = (
            sum(self._completed_delays) / len(self._completed_delays)
            if self._completed_delays else 0.0
        )

        return {
            "throughput": throughput,
            "avg_speed": avg_speed,
            "density": density,
            "lane_utilisation": lane_utilisation,
            "queue_length": self.queue_length(),
            "avg_delay": avg_delay,
        }
