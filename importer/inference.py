"""Attribute inference rules with GREEN/AMBER/RED quality flags."""
from __future__ import annotations

from enum import Enum


class QualityFlag(str, Enum):
    GREEN  = "green"   # directly from OSM tag
    AMBER  = "amber"   # inferred from highway class
    RED    = "red"     # fallback default applied
    MANUAL = "manual"  # set by user in the editor


# ---------------------------------------------------------------------- #
# Highway class defaults
# ---------------------------------------------------------------------- #

_LANE_DEFAULTS: dict[str, int] = {
    "motorway":       3,
    "motorway_link":  1,
    "trunk":          2,
    "trunk_link":     1,
    "primary":        2,
    "primary_link":   1,
    "secondary":      2,
    "secondary_link": 1,
    "tertiary":       1,
    "tertiary_link":  1,
    "residential":    1,
    "service":        1,
    "living_street":  1,
    "unclassified":   1,
    "road":           1,
}

# m/s equivalents for common speed classes
_SPEED_DEFAULTS_MS: dict[str, float] = {
    "motorway":       31.3,   # ~113 km/h
    "motorway_link":  22.2,
    "trunk":          22.2,   # ~80 km/h
    "trunk_link":     13.9,
    "primary":        13.9,   # ~50 km/h
    "primary_link":   13.9,
    "secondary":      13.9,
    "secondary_link": 11.1,
    "tertiary":       11.1,   # ~40 km/h
    "tertiary_link":  8.3,
    "residential":    8.3,    # ~30 km/h
    "service":        4.2,    # ~15 km/h
    "living_street":  2.8,    # ~10 km/h
    "unclassified":   8.3,
    "road":           8.3,
}

_FALLBACK_LANES: int = 1
_FALLBACK_SPEED_MS: float = 8.3   # 30 km/h


def _parse_maxspeed(tag: str) -> float | None:
    """Convert OSM maxspeed tag string to m/s, or return None on failure."""
    tag = tag.strip().lower()
    if tag in ("none", "signals", "variable", "walk"):
        return None
    # "50 mph" → mph conversion
    if tag.endswith(" mph") or tag.endswith("mph"):
        num = tag.replace("mph", "").strip()
        try:
            return float(num) * 0.44704
        except ValueError:
            return None
    # plain number → km/h
    try:
        return float(tag) / 3.6
    except ValueError:
        return None


def infer_edge_attributes(edge: dict) -> dict:
    """Apply inference rules to an edge dict and return enriched copy.

    Adds ``num_lanes``, ``speed_limit`` (m/s), and ``quality_flags`` keys.
    """
    tags = edge.get("tags", {})
    hw_class = tags.get("highway", "unclassified")
    result = dict(edge)

    # ---- Lane count -------------------------------------------------- #
    lanes_tag = tags.get("lanes")
    if lanes_tag is not None:
        try:
            num_lanes = max(1, int(lanes_tag))
            lanes_flag = QualityFlag.GREEN
        except ValueError:
            num_lanes = _LANE_DEFAULTS.get(hw_class, _FALLBACK_LANES)
            lanes_flag = QualityFlag.AMBER
    elif hw_class in _LANE_DEFAULTS:
        num_lanes = _LANE_DEFAULTS[hw_class]
        lanes_flag = QualityFlag.AMBER
    else:
        num_lanes = _FALLBACK_LANES
        lanes_flag = QualityFlag.RED

    # ---- Speed limit ------------------------------------------------- #
    maxspeed_tag = tags.get("maxspeed")
    if maxspeed_tag is not None:
        parsed = _parse_maxspeed(maxspeed_tag)
        if parsed is not None:
            speed_limit = parsed
            speed_flag = QualityFlag.GREEN
        else:
            speed_limit = _SPEED_DEFAULTS_MS.get(hw_class, _FALLBACK_SPEED_MS)
            speed_flag = QualityFlag.AMBER
    elif hw_class in _SPEED_DEFAULTS_MS:
        speed_limit = _SPEED_DEFAULTS_MS[hw_class]
        speed_flag = QualityFlag.AMBER
    else:
        speed_limit = _FALLBACK_SPEED_MS
        speed_flag = QualityFlag.RED

    result["num_lanes"] = num_lanes
    result["speed_limit"] = speed_limit
    result["quality_flags"] = {
        "lanes": lanes_flag.value,
        "speed_limit": speed_flag.value,
    }
    return result


def infer(parsed: dict) -> dict:
    """Apply inference to all edges in a parsed OSM structure.

    Returns the same structure with enriched ``edges`` list.
    """
    enriched_edges = [infer_edge_attributes(e) for e in parsed.get("edges", [])]
    return {**parsed, "edges": enriched_edges}
