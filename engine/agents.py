"""Vehicle dataclass with IDM and MOBIL parameter fields."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vehicle:
    id: int
    lane_id: int
    position_s: float       # arc length along lane (m)
    speed: float            # m/s
    length: float = 4.5     # m  (bumper-to-bumper; set at spawn from VehicleClass)
    width:  float = 1.8     # m  (set at spawn from VehicleClass)
    vehicle_type: str = "car"   # "car" | "van" | "truck" | "bus"
    # IDM params
    v0: float = 30.0        # desired speed (m/s)
    s0: float = 2.0         # minimum jam gap (m)
    T: float = 1.5          # time headway (s)
    a_max: float = 1.4      # max acceleration (m/s²)
    b: float = 2.0          # comfortable braking (m/s²)
    # Derived each step
    acceleration: float = 0.0
    # MOBIL lane-change params
    politeness: float = 0.3     # politeness factor p
    b_safe: float = 4.0         # max deceleration imposed on target follower (m/s²)
    delta_a_thr: float = 0.2    # minimum acceleration gain threshold (m/s²)
    bias_right: float = 0.1     # keep-right bias (m/s²)
    # Lane-change state
    lane_change_cooldown: float = 0.0   # seconds remaining before next change
    # Routing (M5)
    route: list = field(default_factory=list)   # ordered list of edge IDs
    route_index: int = 0                        # index into route of current edge
    current_edge: str = ""                      # edge ID currently being traversed
    # Trip statistics (M5)
    entry_time: float = 0.0     # simulation time when vehicle entered network
    lane_changes: int = 0       # total lane changes executed
    stops: int = 0              # number of times vehicle came to a full stop
    _was_stopped: bool = False  # internal: tracks stop-to-start transitions
