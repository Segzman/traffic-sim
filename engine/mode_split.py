"""Distance-based mode split for Milestone 4.

Logistic functions of trip distance (metres):
    P_walk(d) = 1 / (1 + exp( 0.004  × (d − 2 000)))
    P_bike(d) = (1 − P_walk) / (1 + exp(0.0008 × (d − 7 000)))
    P_car(d)  = 1 − P_walk − P_bike
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    CAR  = "car"
    WALK = "walk"
    BIKE = "bike"


@dataclass
class ModeSplitConfig:
    """Optional overrides for mode split behaviour."""
    force_car: bool = False


def mode_split_probs(distance_m: float) -> dict[Mode, float]:
    """Return normalised mode probabilities for a trip of *distance_m* metres."""
    d      = max(0.0, float(distance_m))
    p_walk = 1.0 / (1.0 + math.exp( 0.004  * (d - 2_000.0)))
    p_bike = (1.0 - p_walk) / (1.0 + math.exp(0.0008 * (d - 7_000.0)))
    p_car  = max(0.0, 1.0 - p_walk - p_bike)
    total  = p_walk + p_bike + p_car or 1.0  # guard against fp underflow
    return {
        Mode.WALK: p_walk / total,
        Mode.BIKE: p_bike / total,
        Mode.CAR:  p_car  / total,
    }


def sample_mode(
    distance_m: float,
    seed: int | None = None,
    config: ModeSplitConfig | None = None,
) -> Mode:
    """Sample a travel mode for a trip of *distance_m* metres.

    Parameters
    ----------
    distance_m:
        Euclidean O–D distance in metres.
    seed:
        Optional RNG seed for reproducibility.
    config:
        Optional overrides, e.g. ``ModeSplitConfig(force_car=True)``.
    """
    if config is not None and config.force_car:
        return Mode.CAR

    probs = mode_split_probs(distance_m)
    rng   = random.Random(seed)
    r     = rng.random()
    cumulative = 0.0
    for mode in (Mode.WALK, Mode.BIKE, Mode.CAR):
        cumulative += probs[mode]
        if r < cumulative:
            return mode
    return Mode.CAR  # numerical fallback
