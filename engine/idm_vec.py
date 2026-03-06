"""Vectorised IDM — NumPy batch acceleration for N vehicles simultaneously.

Drop-in companion to ``engine.idm.idm_acceleration`` (scalar).
All array inputs must be 1-D float64-compatible arrays of the same length N.
Scalar parameters (s0, T, a, b, delta) are broadcast across all N vehicles,
OR they can themselves be 1-D arrays for per-vehicle heterogeneous params.

NumPy releases the GIL during heavy array ops, so calling this inside a
``concurrent.futures.ThreadPoolExecutor`` achieves genuine parallelism.
"""
from __future__ import annotations

import numpy as np


def idm_acceleration_vec(
    v:       "np.ndarray | list",   # current speed (m/s)         shape (N,)
    v0:      "np.ndarray | list",   # desired speed (m/s)         shape (N,) or scalar
    s:       "np.ndarray | list",   # gap to leader (m)           shape (N,)
    delta_v: "np.ndarray | list",   # speed diff to leader (m/s)  shape (N,)  +ve = closing
    s0:      "float | np.ndarray",  # minimum jam gap (m)         scalar or (N,)
    T:       "float | np.ndarray",  # desired time headway (s)    scalar or (N,)
    a:       "float | np.ndarray",  # max acceleration (m/s²)     scalar or (N,)
    b:       "float | np.ndarray",  # comfortable braking (m/s²)  scalar or (N,)
    delta:   float = 4.0,           # free-road acceleration exponent
) -> np.ndarray:
    """Compute IDM acceleration for N vehicles in one vectorised pass.

    Returns a float64 array of shape (N,) with accelerations in m/s².

    Numerical guards
    ----------------
    * ``v0`` floored at 1 × 10⁻⁶  m/s  — avoids division by zero.
    * ``s``  floored at 1 × 10⁻³  m    — avoids blowup when gap → 0
      (hard braking at ``-b`` should be applied separately for gaps ≤ 0).
    * Closing term clamped to ≥ 0 inside ``s_star`` so vehicles never get
      a phantom "push" when already pulling away from the leader.
    """
    v       = np.asarray(v,       dtype=np.float64)
    v0      = np.asarray(v0,      dtype=np.float64)
    s       = np.asarray(s,       dtype=np.float64)
    delta_v = np.asarray(delta_v, dtype=np.float64)
    s0      = np.asarray(s0,      dtype=np.float64)
    T       = np.asarray(T,       dtype=np.float64)
    a       = np.asarray(a,       dtype=np.float64)
    b       = np.asarray(b,       dtype=np.float64)

    # Desired dynamic gap s*(v, Δv)
    sqrt_ab = np.sqrt(a * b)                                      # (N,) or scalar
    s_star  = s0 + np.maximum(0.0, v * T + v * delta_v / (2.0 * sqrt_ab))

    # Safe denominators
    v0_safe = np.maximum(v0, 1e-6)
    s_safe  = np.maximum(s,  1e-3)

    # IDM formula: a·[1 − (v/v₀)^δ − (s*/s)²]
    return a * (1.0 - (v / v0_safe) ** delta - (s_star / s_safe) ** 2)
