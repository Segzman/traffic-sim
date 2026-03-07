"""Building floor and capacity estimation helpers for M3 demand."""
from __future__ import annotations

import math


_DEFAULT_METERS_PER_FLOOR = 3.3
_DEFAULT_DIVISORS = {
    "office": 500.0,
    "residential": 350.0,
}
_DEFAULT_SQM_PER_PERSON = {
    "office": 12.0,
    "residential": 25.0,
}


def _as_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def infer_building_use(tags: dict) -> str:
    """Infer coarse building use: residential|office|other."""
    b = str(tags.get("building", "")).lower()
    if b in {"apartments", "residential", "house", "detached", "semidetached_house"}:
        return "residential"
    if b in {"office", "commercial", "industrial", "retail"}:
        return "office"
    if "office" in tags:
        return "office"
    return "residential" if tags.get("addr:housenumber") else "other"


def estimate_floors(
    tags: dict,
    footprint_area: float,
    meters_per_floor: float = _DEFAULT_METERS_PER_FLOOR,
) -> int:
    """Estimate floors using levels -> height -> footprint heuristic."""
    levels = _as_float(tags.get("building:levels"))
    if levels is not None:
        return max(1, int(round(levels)))

    height = _as_float(tags.get("height"))
    if height is not None and height > 0:
        return max(1, int(round(height / max(1e-6, meters_per_floor))))

    b_use = infer_building_use(tags)
    if b_use == "office":
        div = _DEFAULT_DIVISORS["office"]
    else:
        div = _DEFAULT_DIVISORS["residential"]
    return max(1, int(round(max(1.0, footprint_area) / div)))


def estimate_building_capacity(
    building: dict,
    meters_per_floor: float = _DEFAULT_METERS_PER_FLOOR,
    office_sqm_per_person: float = _DEFAULT_SQM_PER_PERSON["office"],
    residential_sqm_per_person: float = _DEFAULT_SQM_PER_PERSON["residential"],
) -> float:
    """Estimate people capacity from OSM building tags + footprint area."""
    tags = building.get("tags", {})
    area = max(0.0, float(building.get("footprint_area", 0.0)))
    floors = estimate_floors(tags, area, meters_per_floor=meters_per_floor)
    gross = max(1.0, area * floors)

    b_use = infer_building_use(tags)
    if b_use == "office":
        sqm_per_person = max(1.0, office_sqm_per_person)
    else:
        sqm_per_person = max(1.0, residential_sqm_per_person)

    return gross / sqm_per_person


def classify_building_purpose(tags: dict) -> str:
    """Map building tags to destination purpose bucket."""
    b = str(tags.get("building", "")).lower()
    if b in {"apartments", "residential", "house", "detached", "semidetached_house"}:
        return "home"
    if b in {"office", "commercial", "industrial"}:
        return "work"
    if b in {"retail", "mall", "supermarket"}:
        return "retail"
    if tags.get("office"):
        return "work"
    return "other"
