"""
Intelligent Driver Model (IDM) — pure acceleration function.

Pure function: no side effects, no state. Safe for NumPy vectorisation.
"""
import math


def idm_acceleration(
    v: float,           # current speed (m/s)
    v0: float,          # desired speed (m/s)
    s: float,           # gap to leader (m)
    delta_v: float,     # speed difference to leader (m/s), positive = closing
    s0: float,          # minimum jam gap (m)
    T: float,           # desired time headway (s)
    a: float,           # max acceleration (m/s²)
    b: float,           # comfortable braking deceleration (m/s²)
    delta: float = 4.0  # free-road acceleration exponent
) -> float:             # returns acceleration (m/s²)
    """
    Compute IDM longitudinal acceleration.

    Raises ValueError if s <= 0 (physically impossible gap).
    """
    if s <= 0:
        raise ValueError(f"Gap s must be positive, got {s}")

    # Desired dynamic gap
    s_star = s0 + max(0.0, v * T + v * delta_v / (2.0 * math.sqrt(a * b)))

    # IDM formula
    return a * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
