"""Vehicle class definitions with physical dimensions and IDM/MOBIL defaults.

Data sources
------------
* Physical dimensions  : FHWA Vehicle Classification Guide (2013) + SAE J1366
* IDM parameter ranges : Treiber & Kesting "Traffic Flow Dynamics" (2013) §11.3
* Fleet mix proportions: Highway Capacity Manual 7e Table 3-1 (2022) by road type

Road-type baseline mixes are keyed by OSM ``highway`` tag values and follow
HCM urban-arterial / freeway proportions for North American roads.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# VehicleClass definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleClass:
    """Physical and behavioural profile for a road vehicle class.

    All IDM params are *class defaults*; global SimConfig sliders override them
    (they act as fleet-wide adjustments while the class values are per-type baselines).
    """
    type:         str     # "car" | "van" | "truck" | "bus"

    # ── Physical dimensions ─────────────────────────────────────────────────
    length:       float   # metres  (bumper-to-bumper)
    width:        float   # metres

    # ── IDM longitudinal parameters ─────────────────────────────────────────
    a_max:        float   # m/s²  max acceleration
    b:            float   # m/s²  comfortable braking
    T:            float   # s     desired time-headway
    s0:           float   # m     minimum jam gap

    # ── Free-flow speed ──────────────────────────────────────────────────────
    speed_factor: float   # multiplier on posted speed limit → desired speed v0

    # ── Default spawn proportion ─────────────────────────────────────────────
    proportion:   float   # fraction in [0,1]; used when no road-type baseline exists

    # ── Lane discipline ──────────────────────────────────────────────────────
    lane_max:     int     # highest 0-indexed lane permitted (99 = no restriction)

    # ── Disobedience caps ────────────────────────────────────────────────────
    # Maximum rule-breaking at global disobedience=1.0.
    # Physics (braking distance, mass) caps how reckless each class can be.
    max_disobedience:   float   # [0,1]  maximum personal disobedience factor
    speed_excess:       float   # max v0 factor above posted limit  (e.g. 0.40 → 40 %)
    gap_reduction:      float   # max T and s0 reduction fraction   (e.g. 0.50 → -50 %)
    politeness_factor:  float   # politeness × (1 - disobey × this) floored at 0


# ---------------------------------------------------------------------------
# Class catalogue
# ---------------------------------------------------------------------------

VEHICLE_CLASSES: dict[str, VehicleClass] = {
    # ── Passenger car (sedan / hatchback / SUV) ─────────────────────────────
    # FHWA Class 2-3 | L=4.5 m, W=1.8 m
    # IDM: calibrated urban driver (HCM 7e free-flow headway T=1.5 s, b=1.8 m/s²)
    "car": VehicleClass(
        type="car",
        length=4.5,  width=1.8,
        a_max=1.6,   b=1.8,
        T=1.5,       s0=2.0,
        speed_factor=1.00,
        proportion=0.80,
        lane_max=99,                      # can use any lane
        max_disobedience=1.00,
        speed_excess=0.40,                # up to 40 % over limit
        gap_reduction=0.50,               # headway/gap down to 50 %
        politeness_factor=0.80,
    ),

    # ── Delivery van / light commercial / minivan ────────────────────────────
    # FHWA Class 3-4 | L=5.5 m, W=2.1 m
    "van": VehicleClass(
        type="van",
        length=5.5,  width=2.1,
        a_max=1.2,   b=2.0,
        T=1.4,       s0=2.5,
        speed_factor=0.95,
        proportion=0.10,
        lane_max=99,                      # can use any lane
        max_disobedience=0.80,
        speed_excess=0.25,
        gap_reduction=0.30,
        politeness_factor=0.50,
    ),

    # ── Heavy goods truck (single-unit or articulated) ───────────────────────
    # FHWA Class 8-10 | L=12 m, W=2.5 m
    # Higher s0 and T: braking distance much longer than a car
    "truck": VehicleClass(
        type="truck",
        length=12.0, width=2.5,
        a_max=0.8,   b=1.5,
        T=1.8,       s0=4.0,
        speed_factor=0.85,
        proportion=0.07,
        lane_max=0,                       # rightmost lane only (default)
        max_disobedience=0.30,
        speed_excess=0.10,                # physics limits speeding
        gap_reduction=0.10,
        politeness_factor=0.20,
    ),

    # ── Transit bus (full-size, fixed-route) ─────────────────────────────────
    # L=14 m, W=2.5 m; lower desired speed (stops along route)
    "bus": VehicleClass(
        type="bus",
        length=14.0, width=2.5,
        a_max=0.9,   b=1.5,
        T=1.6,       s0=4.0,
        speed_factor=0.90,
        proportion=0.03,
        lane_max=0,                       # rightmost lane only (default)
        max_disobedience=0.20,
        speed_excess=0.05,
        gap_reduction=0.05,
        politeness_factor=0.10,
    ),
}


# ---------------------------------------------------------------------------
# Road-type baseline mixes  (HCM 7e Table 3-1, urban North America)
# ---------------------------------------------------------------------------

# OSM highway tags → {type: proportion}
_ROAD_MIX: dict[str, dict[str, float]] = {
    # High-speed divided highway — more freight, fewer buses
    # Truck % calibrated to FHWA empirical range 8-15 % for urban freeways
    "motorway":        {"car": 0.75, "van": 0.10, "truck": 0.12, "bus": 0.03},
    "motorway_link":   {"car": 0.75, "van": 0.10, "truck": 0.12, "bus": 0.03},
    "trunk":           {"car": 0.75, "van": 0.10, "truck": 0.12, "bus": 0.03},
    "trunk_link":      {"car": 0.75, "van": 0.10, "truck": 0.12, "bus": 0.03},

    # Urban arterial — standard mix
    # Truck % calibrated to HCM 7e arterial range 2-5 %
    "primary":         {"car": 0.83, "van": 0.10, "truck": 0.04, "bus": 0.03},
    "primary_link":    {"car": 0.83, "van": 0.10, "truck": 0.04, "bus": 0.03},
    "secondary":       {"car": 0.83, "van": 0.10, "truck": 0.04, "bus": 0.03},
    "secondary_link":  {"car": 0.83, "van": 0.10, "truck": 0.04, "bus": 0.03},

    # Collector / local — mostly private cars
    "tertiary":        {"car": 0.85, "van": 0.12, "truck": 0.02, "bus": 0.01},
    "tertiary_link":   {"car": 0.85, "van": 0.12, "truck": 0.02, "bus": 0.01},
    "residential":     {"car": 0.87, "van": 0.12, "truck": 0.01, "bus": 0.00},
    "unclassified":    {"car": 0.87, "van": 0.12, "truck": 0.01, "bus": 0.00},

    # Service / private road — virtually no heavy vehicles
    "service":         {"car": 0.90, "van": 0.09, "truck": 0.01, "bus": 0.00},
    "living_street":   {"car": 0.92, "van": 0.08, "truck": 0.00, "bus": 0.00},
    "road":            {"car": 0.87, "van": 0.12, "truck": 0.01, "bus": 0.00},
}

_DEFAULT_MIX: dict[str, float] = {"car": 0.80, "van": 0.10, "truck": 0.07, "bus": 0.03}


def mix_for_road_type(road_type: str) -> dict[str, float]:
    """Return a normalised type→proportion dict for the given OSM highway class.

    Falls back to the generic urban-arterial mix for unknown road types.
    """
    raw = _ROAD_MIX.get(road_type, _DEFAULT_MIX)
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}
