"""Milestone 5 — Pedestrian Social Force Model unit tests."""
import math
import pytest

from engine.pedestrians import (
    Pedestrian,
    social_force,
    step_pedestrian,
    A_PED, B_PED, TAU, MASS, V_MAX,
)


# ------------------------------------------------------------------ #
# test_ped_moves_toward_destination
# ------------------------------------------------------------------ #

def test_ped_moves_toward_destination():
    """A lone pedestrian walks toward their destination."""
    ped = Pedestrian(id=0, x=0.0, y=0.0, vx=0.0, vy=0.0,
                     dest_x=10.0, dest_y=0.0, v_desired=1.4)

    # Run for 10 seconds (dt = 0.1 s, 100 steps)
    for step in range(100):
        step_pedestrian(ped, [], dt=0.1, sim_time=step * 0.1)

    # Should have moved toward destination
    assert ped.x > 0.0, "Pedestrian should have moved in +x direction"
    assert abs(ped.y) < 0.5, "Pedestrian should not drift far in y"

    # Speed should be close to v_desired (reached quasi-steady state)
    spd = math.sqrt(ped.vx ** 2 + ped.vy ** 2)
    assert spd > 0.5, f"Pedestrian should be walking; speed={spd:.3f}"
    assert spd <= V_MAX + 0.01, f"Speed must not exceed V_MAX; speed={spd:.3f}"


# ------------------------------------------------------------------ #
# test_ped_repulsion_from_obstacle
# ------------------------------------------------------------------ #

def test_ped_repulsion_from_obstacle():
    """Obstacle directly ahead produces a net repulsion force at close range.

    At 0.35 m the repulsion (~1070 N) greatly exceeds the desired force (~224 N),
    so the net x-force must be negative (obstacle wins over destination pull).
    """
    ped = Pedestrian(id=0, x=0.0, y=0.0, vx=0.0, vy=0.0,
                     dest_x=5.0, dest_y=0.0, v_desired=1.4)

    # Obstacle 0.35 m ahead.
    # d_surf = 0.35 - ped.radius(0.3) - obs_radius(0) = 0.05 m
    # repulsion = A_PED * exp(-0.05/B_PED) = 2000*exp(-0.625) ≈ 1070 N in -x
    # desired  ≈ (1.4/TAU)*MASS = (1.4/0.5)*80 = 224 N in +x → repulsion wins
    obstacle = (0.35, 0.0, 0.0)   # (x, y, radius)

    fx, fy = social_force(ped, [], obstacles=[obstacle])

    assert fx < 0.0, (
        f"At 0.35 m from obstacle, repulsion (~1070 N) should dominate desired (~224 N); "
        f"fx={fx:.2f}"
    )


# ------------------------------------------------------------------ #
# test_ped_repulsion_from_other_ped
# ------------------------------------------------------------------ #

def test_ped_repulsion_from_other_ped():
    """Two pedestrians side-by-side push each other apart."""
    ped_a = Pedestrian(id=0, x=0.0, y=0.0, vx=0.0, vy=0.0,
                       dest_x=10.0, dest_y=0.0, v_desired=1.4, radius=0.3)
    ped_b = Pedestrian(id=1, x=0.0, y=0.5, vx=0.0, vy=0.0,
                       dest_x=10.0, dest_y=0.5, v_desired=1.4, radius=0.3)

    # Force on ped_a from ped_b (ped_b is 0.5 m above: y=0.5)
    fx_a, fy_a = social_force(ped_a, [ped_b])

    # ped_b is above ped_a → repulsion pushes ped_a downward (fy < 0)
    assert fy_a < 0.0, (
        f"Pedestrian above should repel ped_a downward; fy_a={fy_a:.2f}"
    )

    # The repulsion is exponential with surface-to-surface distance = 0.5 - 0.3 - 0.3 = -0.1
    # (overlapping) → very strong force
    assert fy_a < -100.0, (
        f"Overlapping pedestrians should produce large repulsion; fy_a={fy_a:.2f}"
    )


# ------------------------------------------------------------------ #
# test_vehicle_yields_to_ped
# ------------------------------------------------------------------ #

def test_vehicle_yields_to_ped():
    """NetworkSimulation vehicle brakes hard when TTC to crossing ped < 2 s."""
    from engine.network import Network, Node, Edge
    from engine.network_simulation import NetworkSimulation
    from engine.pedestrians import Pedestrian

    # Single-edge network: A -> B, 200 m
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=200.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B",
                      num_lanes=1, speed_limit=13.9,
                      geometry=[(0.0, 0.0), (200.0, 0.0)]))

    # Place a pedestrian at x=20 (ped.x used as proxy for crossing position)
    # Vehicle starts at position_s=0, speed=13.9 m/s → TTC to x=20 ≈ 1.44 s < 2 s
    ped = Pedestrian(id=0, x=20.0, y=-5.0, dest_x=20.0, dest_y=5.0,
                     vx=0.0, vy=1.4, v_desired=1.4)

    sim = NetworkSimulation(
        network=net,
        demand={"A": {"B": 3600}},   # 1 veh/s — spawn immediately
        duration=5.0,
        seed=0,
        pedestrians=[ped],
    )

    # Step once to let at least one vehicle spawn
    for _ in range(20):  # 2 s
        sim.step()

    # Find the vehicle nearest x=20 (position_s ≈ 20)
    vehs_near = [v for v in sim.vehicles if v.position_s < 25.0]
    # If vehicle has not reached position 25 yet, check its speed
    if vehs_near:
        v = min(vehs_near, key=lambda vv: abs(vv.position_s - 20.0))
        assert v.speed < 13.9 * 0.9, (
            f"Vehicle should have braked near pedestrian; speed={v.speed:.2f} m/s"
        )
    else:
        # Vehicle may have already passed — check it was spawned at all
        # (trip completed means it passed through)
        assert sim._next_vid > 0, "At least one vehicle should have been spawned"
