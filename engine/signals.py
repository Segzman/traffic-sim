"""Signal plans and phase cycling for signalised intersections."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Phase:
    """One stage in a signal cycle."""
    green_movements: list   # edge / movement IDs with right-of-way
    green_duration: float   # seconds
    yellow_duration: float = 3.0
    all_red_duration: float = 1.0

    def total_duration(self) -> float:
        """Total time for this phase (green + yellow + all-red)."""
        return self.green_duration + self.yellow_duration + self.all_red_duration


@dataclass
class SignalPlan:
    """Full signal plan for a junction node."""
    node_id: str
    phases: list        # list[Phase]
    offset: float = 0.0  # seconds offset from global clock

    @property
    def cycle_time(self) -> float:
        """Sum of all phase total durations."""
        return sum(p.total_duration() for p in self.phases)

    def current_phase(self, sim_time: float):
        """Return (Phase, elapsed_in_phase) for the given simulation time.

        elapsed_in_phase is seconds elapsed since this phase started
        (counting from the beginning of the green sub-phase).
        """
        t = (sim_time - self.offset) % self.cycle_time
        if t < 0:
            t += self.cycle_time
        elapsed_so_far = 0.0
        for phase in self.phases:
            d = phase.total_duration()
            if t < elapsed_so_far + d:
                return phase, t - elapsed_so_far
            elapsed_so_far += d
        # Fallback – should not be reached with valid input
        return self.phases[-1], self.phases[-1].total_duration()

    def current_state(self, sim_time: float, movement_id: str = None) -> str:
        """Return 'green', 'yellow', or 'red' for movement_id at sim_time.

        If movement_id is None the state of the active phase is returned.
        A movement is green/yellow only when it is listed in the current
        phase's green_movements; otherwise it is always 'red'.
        """
        phase, elapsed = self.current_phase(sim_time)

        # Determine sub-phase state for the current phase
        if elapsed < phase.green_duration:
            sub_state = "green"
        elif elapsed < phase.green_duration + phase.yellow_duration:
            sub_state = "yellow"
        else:
            sub_state = "red"

        if movement_id is None:
            return sub_state

        # Movement-specific: green/yellow only when in this phase's movements
        if movement_id in phase.green_movements:
            return sub_state
        return "red"
