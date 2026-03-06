"""Social Force Model pedestrian agent.

Reference: Helbing & Molnár (1995), "Social force model for pedestrian dynamics".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
# SFM constants
# ------------------------------------------------------------------ #
A_PED   = 2000.0   # repulsion magnitude (N)
B_PED   = 0.08     # repulsion decay length (m)
TAU     = 0.5      # relaxation time (s)
MASS    = 80.0     # pedestrian mass (kg)
V_MAX   = 3.0      # maximum walking speed (m/s)


@dataclass
class Pedestrian:
    id: int
    x: float            # position (m)
    y: float
    vx: float = 0.0     # velocity (m/s)
    vy: float = 0.0
    dest_x: float = 0.0 # destination
    dest_y: float = 0.0
    v_desired: float = 1.4  # desired walking speed (m/s)
    radius: float = 0.3     # body radius for repulsion (m)
    # Derived each step
    ax: float = 0.0     # acceleration (m/s²)
    ay: float = 0.0
    # Trip stats
    entry_time: float = 0.0
    exit_time:  float = -1.0   # -1 = not yet reached destination


# ------------------------------------------------------------------ #
# Force computation helpers
# ------------------------------------------------------------------ #

def _desired_force(ped: Pedestrian) -> tuple[float, float]:
    """Driving force toward destination: (1/τ) * (v0 * e_dest - v)."""
    dx = ped.dest_x - ped.x
    dy = ped.dest_y - ped.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return (0.0, 0.0)
    ex = dx / dist
    ey = dy / dist
    fx = (ped.v_desired * ex - ped.vx) / TAU
    fy = (ped.v_desired * ey - ped.vy) / TAU
    return (fx * MASS, fy * MASS)


def _ped_repulsion(ped: Pedestrian, other: Pedestrian) -> tuple[float, float]:
    """Exponential repulsion from another pedestrian."""
    dx = ped.x - other.x
    dy = ped.y - other.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        return (A_PED, 0.0)   # degenerate: push right
    # Surface-to-surface distance
    d_surf = dist - ped.radius - other.radius
    mag = A_PED * math.exp(-d_surf / B_PED)
    fx = mag * (dx / dist)
    fy = mag * (dy / dist)
    return (fx, fy)


def _obstacle_repulsion(
    ped: Pedestrian,
    obs_x: float,
    obs_y: float,
    obs_radius: float = 0.0,
) -> tuple[float, float]:
    """Exponential repulsion from a point obstacle."""
    dx = ped.x - obs_x
    dy = ped.y - obs_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        return (A_PED, 0.0)
    d_surf = max(0.0, dist - ped.radius - obs_radius)
    mag = A_PED * math.exp(-d_surf / B_PED)
    return (mag * dx / dist, mag * dy / dist)


def social_force(
    ped: Pedestrian,
    others: list[Pedestrian],
    obstacles: list[tuple[float, float, float]] | None = None,
) -> tuple[float, float]:
    """Net social force on *ped*.

    Parameters
    ----------
    ped:
        The pedestrian to evaluate.
    others:
        Other pedestrians in the scene (ped itself excluded automatically).
    obstacles:
        List of (x, y, radius) point obstacles.

    Returns
    -------
    (Fx, Fy) in Newtons.
    """
    fx, fy = _desired_force(ped)

    for other in others:
        if other.id == ped.id:
            continue
        rx, ry = _ped_repulsion(ped, other)
        fx += rx
        fy += ry

    if obstacles:
        for (ox, oy, orr) in obstacles:
            rx, ry = _obstacle_repulsion(ped, ox, oy, orr)
            fx += rx
            fy += ry

    return (fx, fy)


# ------------------------------------------------------------------ #
# Step function
# ------------------------------------------------------------------ #

def step_pedestrian(
    ped: Pedestrian,
    others: list[Pedestrian],
    dt: float,
    obstacles: list[tuple[float, float, float]] | None = None,
    sim_time: float = 0.0,
    arrival_radius: float = 1.0,
) -> None:
    """Advance *ped* by one timestep using Euler integration.

    Updates ped.x, ped.y, ped.vx, ped.vy, ped.ax, ped.ay in place.
    Sets ped.exit_time when the pedestrian reaches within *arrival_radius* of
    the destination.
    """
    if ped.exit_time >= 0.0:
        return  # already arrived — do nothing

    # Check arrival
    dx = ped.dest_x - ped.x
    dy = ped.dest_y - ped.y
    if math.sqrt(dx * dx + dy * dy) < arrival_radius:
        ped.exit_time = sim_time
        return

    fx, fy = social_force(ped, others, obstacles)
    ped.ax = fx / MASS
    ped.ay = fy / MASS

    ped.vx += ped.ax * dt
    ped.vy += ped.ay * dt

    # Clamp to V_MAX
    spd = math.sqrt(ped.vx ** 2 + ped.vy ** 2)
    if spd > V_MAX:
        ped.vx = ped.vx / spd * V_MAX
        ped.vy = ped.vy / spd * V_MAX

    ped.x += ped.vx * dt
    ped.y += ped.vy * dt
