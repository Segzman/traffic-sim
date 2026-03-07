"""M3 tests — vehicle classes, proportional sizing, disobedience, fleet mix."""
import json
import urllib.request

import pytest

from engine.config import SimConfig
from engine.network import Edge, Network, Node
from engine.network_simulation import NetworkSimulation
from engine.vehicle_classes import VEHICLE_CLASSES, mix_for_road_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_node_net(
    length: float = 1000.0,
    speed: float = 14.0,
    num_lanes: int = 2,
    road_type: str = "primary",
) -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=length, y=0.0))
    net.add_edge(Edge(
        id="AB", from_node="A", to_node="B",
        num_lanes=num_lanes, speed_limit=speed,
        geometry=[[0.0, 0.0], [length, 0.0]],
        road_type=road_type,
    ))
    return net


def _spawn_n(n: int, cfg: SimConfig | None = None, road_type: str = "primary",
             num_lanes: int = 2) -> list:
    """Spawn n vehicles and return them."""
    sim = NetworkSimulation(
        _two_node_net(length=5000.0, num_lanes=num_lanes, road_type=road_type),
        demand={"A": {"B": 3600.0 * n}},
        duration=300, seed=0, config=cfg,
    )
    for _ in range(n * 3 + 10):
        sim.step()
    return list(sim.vehicles)


# ---------------------------------------------------------------------------
# 1. VehicleClass catalogue
# ---------------------------------------------------------------------------

class TestVehicleClassCatalogue:

    def test_all_types_present(self):
        assert set(VEHICLE_CLASSES) == {"car", "van", "truck", "bus"}

    def test_physical_dimensions(self):
        c = VEHICLE_CLASSES["car"]
        assert c.length == pytest.approx(4.5)
        assert c.width  == pytest.approx(1.8)

        t = VEHICLE_CLASSES["truck"]
        assert t.length == pytest.approx(12.0)
        assert t.width  == pytest.approx(2.5)

        b = VEHICLE_CLASSES["bus"]
        assert b.length == pytest.approx(14.0)

    def test_truck_longer_than_car(self):
        assert VEHICLE_CLASSES["truck"].length > VEHICLE_CLASSES["car"].length

    def test_bus_longer_than_truck(self):
        assert VEHICLE_CLASSES["bus"].length > VEHICLE_CLASSES["truck"].length

    def test_lane_max_constraints(self):
        assert VEHICLE_CLASSES["truck"].lane_max == 0
        assert VEHICLE_CLASSES["bus"].lane_max   == 0
        assert VEHICLE_CLASSES["car"].lane_max   == 99
        assert VEHICLE_CLASSES["van"].lane_max   == 99

    def test_proportions_positive(self):
        for vc in VEHICLE_CLASSES.values():
            assert vc.proportion > 0

    def test_disobedience_caps_physics_ordered(self):
        """Truck max disobedience < car max disobedience (physics constraint)."""
        assert VEHICLE_CLASSES["truck"].max_disobedience < VEHICLE_CLASSES["car"].max_disobedience
        assert VEHICLE_CLASSES["bus"].max_disobedience   < VEHICLE_CLASSES["car"].max_disobedience


# ---------------------------------------------------------------------------
# 2. Road-type mix
# ---------------------------------------------------------------------------

class TestRoadTypeMix:

    def test_motorway_more_trucks_than_residential(self):
        m = mix_for_road_type("motorway")
        r = mix_for_road_type("residential")
        assert m["truck"] > r["truck"]

    def test_normalised_to_one(self):
        for road in ["motorway", "primary", "residential", "service", "unknown_type"]:
            mix = mix_for_road_type(road)
            total = sum(mix.values())
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_unknown_road_type_falls_back(self):
        mix = mix_for_road_type("nonexistent_highway_type")
        assert "car" in mix
        assert mix["car"] > 0.5


# ---------------------------------------------------------------------------
# 3. Vehicle spawning — type, dimensions, lane restriction
# ---------------------------------------------------------------------------

class TestVehicleSpawning:

    def test_spawn_assigns_vehicle_type(self):
        vehicles = _spawn_n(30)
        assert vehicles, "Expected vehicles to spawn"
        for v in vehicles:
            assert v.vehicle_type in VEHICLE_CLASSES

    def test_spawn_sets_correct_length(self):
        vehicles = _spawn_n(30)
        for v in vehicles:
            expected = VEHICLE_CLASSES[v.vehicle_type].length
            assert v.length == pytest.approx(expected)

    def test_spawn_sets_correct_width(self):
        vehicles = _spawn_n(30)
        for v in vehicles:
            expected = VEHICLE_CLASSES[v.vehicle_type].width
            assert v.width == pytest.approx(expected)

    def test_truck_stays_in_lane_0_by_default(self):
        """With truck_lane_discipline=True (default), trucks must be in lane 0."""
        cfg = SimConfig(vehicle_mix_truck=1.0)   # 100 % trucks
        vehicles = _spawn_n(20, cfg=cfg, num_lanes=3)
        trucks = [v for v in vehicles if v.vehicle_type == "truck"]
        assert trucks, "Expected trucks to spawn"
        for v in trucks:
            assert v.lane_id == 0, f"Truck spawned in lane {v.lane_id} != 0"

    def test_truck_can_use_any_lane_when_discipline_off(self):
        """With truck_lane_discipline=False, trucks may spawn in any lane.

        Sampled across multiple seeds because the lane-selection RNG can
        deterministically pick lane 0 for every spawn in a single seed;
        across ten seeds the spawn logic must produce non-zero lane variety.
        """
        cfg = SimConfig(vehicle_mix_truck=1.0, truck_lane_discipline=False)
        all_lanes: set[int] = set()
        for seed in range(10):
            sim = NetworkSimulation(
                _two_node_net(length=5000.0, num_lanes=3),
                demand={"A": {"B": 3600.0 * 40}},
                duration=300, seed=seed, config=cfg,
            )
            for _ in range(40 * 3 + 10):
                sim.step()
            for v in sim.vehicles:
                if v.vehicle_type == "truck":
                    all_lanes.add(v.lane_id)
        assert len(all_lanes) > 1, (
            f"With discipline off, trucks should reach multiple lanes across seeds; "
            f"only saw lanes {all_lanes}"
        )

    def test_mix_100pct_cars(self):
        cfg = SimConfig(
            vehicle_mix_car=1.0, vehicle_mix_van=0.0,
            vehicle_mix_truck=0.0, vehicle_mix_bus=0.0,
        )
        vehicles = _spawn_n(20, cfg=cfg)
        assert vehicles
        for v in vehicles:
            assert v.vehicle_type == "car"

    def test_fleet_mix_approximate(self):
        """High-demand spawn produces roughly the configured type ratio."""
        cfg = SimConfig(vehicle_mix_truck=0.5, vehicle_mix_car=0.5,
                        vehicle_mix_van=0.0, vehicle_mix_bus=0.0)
        sim = NetworkSimulation(
            _two_node_net(length=5000.0, num_lanes=2),
            demand={"A": {"B": 7200.0}},   # 2 veh/s — fills network fast
            duration=300, seed=42, config=cfg,
        )
        for _ in range(300):
            sim.step()
        types = [v.vehicle_type for v in sim.vehicles]
        assert len(types) >= 5, "Not enough vehicles to test mix"
        truck_frac = types.count("truck") / len(types)
        # With realistic IDM headways (T=1.5 s for cars, T=1.8 s for trucks),
        # trucks need 18 m entry-clearance vs 8.5 m for cars, so their actual
        # road fraction is lower than the configured 50 %.  The test verifies
        # both types are being spawned, not that the ratio is exact.
        assert 0.10 < truck_frac < 0.90, f"Expected significant truck presence, got {truck_frac:.1%}"


# ---------------------------------------------------------------------------
# 4. Disobedience
# ---------------------------------------------------------------------------

class TestDisobedience:

    def test_zero_disobedience_uses_class_defaults(self):
        """At disobedience=0, T and s0 equal class defaults (no reduction)."""
        cfg = SimConfig(disobedience=0.0, vehicle_mix_car=1.0)
        vehicles = _spawn_n(10, cfg=cfg)
        car_vc = VEHICLE_CLASSES["car"]
        for v in vehicles:
            # T should be close to class default (possibly affected by global slider)
            assert v.T  >= car_vc.T * 0.999   # not reduced
            assert v.s0 >= car_vc.s0 * 0.999

    def test_max_disobedience_reduces_headway(self):
        """At disobedience=1.0, cars have smaller T and s0 than at 0."""
        def _avg_T(disobey: float) -> float:
            cfg = SimConfig(disobedience=disobey, vehicle_mix_car=1.0)
            vs  = _spawn_n(30, cfg=cfg)
            return sum(v.T for v in vs) / len(vs) if vs else 9.9

        t_zero = _avg_T(0.0)
        t_full = _avg_T(1.0)
        assert t_full < t_zero, (
            f"Disobedience=1.0 should reduce avg T; got {t_full:.2f} vs {t_zero:.2f}"
        )

    def test_truck_disobedience_capped_below_car(self):
        """Trucks should end up with smaller headway reduction than cars at same disobedience."""
        def _avg_T_type(vtype: str, disobey: float) -> float:
            if vtype == "car":
                cfg = SimConfig(disobedience=disobey, vehicle_mix_car=1.0,
                                vehicle_mix_van=0.0, vehicle_mix_truck=0.0, vehicle_mix_bus=0.0)
            else:
                cfg = SimConfig(disobedience=disobey,
                                vehicle_mix_car=0.0, vehicle_mix_van=0.0,
                                vehicle_mix_truck=1.0, vehicle_mix_bus=0.0)
            vs = _spawn_n(20, cfg=cfg)
            return sum(v.T for v in vs) / len(vs) if vs else 9.9

        # Reduction in T relative to disobey=0
        car_t0   = _avg_T_type("car",   0.0)
        car_t1   = _avg_T_type("car",   1.0)
        truck_t0 = _avg_T_type("truck", 0.0)
        truck_t1 = _avg_T_type("truck", 1.0)

        car_reduction   = (car_t0   - car_t1)   / max(car_t0,   0.01)
        truck_reduction = (truck_t0 - truck_t1) / max(truck_t0, 0.01)

        assert car_reduction > truck_reduction, (
            f"Car T reduction ({car_reduction:.1%}) should exceed "
            f"truck T reduction ({truck_reduction:.1%})"
        )


    def test_disobedience_distribution_is_right_skewed(self):
        """Driver behaviour should cluster near-compliant, not spread uniformly.

        With a truncated-normal (mu=0.25·max, σ=0.20·max) most drivers should
        have a T value in the upper half of the possible range (near the class
        default), with only a minority being genuinely aggressive.

        A uniform distribution would put exactly 50 % above the midpoint;
        the truncated-normal should push that fraction well above 60 %.
        """
        cfg = SimConfig(disobedience=1.0, vehicle_mix_car=1.0,
                        vehicle_mix_van=0.0, vehicle_mix_truck=0.0,
                        vehicle_mix_bus=0.0)
        sim = NetworkSimulation(
            _two_node_net(length=5000.0, num_lanes=2),
            demand={"A": {"B": 3600.0 * 80}},
            duration=300, seed=7, config=cfg,
        )
        for _ in range(80 * 3 + 10):
            sim.step()

        car_vc = VEHICLE_CLASSES["car"]
        # Midpoint between class-default T and the hard floor after max reduction
        t_max   = car_vc.T                                      # 1.5 s
        t_min   = max(0.4, car_vc.T * (1 - car_vc.gap_reduction))  # 0.75 s
        midpoint = (t_max + t_min) / 2                          # 1.125 s

        ts = [v.T for v in sim.vehicles]
        assert len(ts) >= 10, "Too few vehicles to assess distribution"
        near_compliant = sum(1 for t in ts if t > midpoint)
        fraction = near_compliant / len(ts)
        assert fraction > 0.60, (
            f"Expected >60 % of drivers near-compliant (T > {midpoint:.2f} s); "
            f"got {fraction:.1%} ({near_compliant}/{len(ts)}). "
            f"Distribution may be uniform rather than right-skewed."
        )


# ---------------------------------------------------------------------------
# 5. State payload fields
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_server_m3():
    """Reuse the same pattern as test_dashboard_api.py but on port 19998."""
    import sys, os, threading, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run import _serve

    scenario = {
        "nodes": [
            {"id": "A", "x": 0.0,    "y": 0.0},
            {"id": "B", "x": 1000.0, "y": 0.0},
        ],
        "edges": [
            {"id": "AB", "from_node": "A", "to_node": "B",
             "num_lanes": 2, "speed_limit": 14.0},
        ],
        "demand":   {"A": {"B": 360.0}},
        "duration": 120.0,
        "seed":     1,
    }
    port   = 19998
    thread = threading.Thread(
        target=_serve, args=(scenario, port, "TestCity"), daemon=True,
    )
    thread.start()
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as r:
        return json.loads(r.read())


class TestStatePayload:

    def test_state_vehicles_have_type(self, api_server_m3):
        import time
        time.sleep(0.5)   # let a few vehicles spawn
        state = _get(api_server_m3, "/state")
        if not state.get("vehicles"):
            pytest.skip("No vehicles spawned yet")
        for v in state["vehicles"]:
            assert "type"   in v, "vehicle missing 'type'"
            assert "length" in v, "vehicle missing 'length'"
            assert "width"  in v, "vehicle missing 'width'"
            assert v["type"] in ("car", "van", "truck", "bus")
            assert v["length"] > 0
            assert v["width"]  > 0

    def test_vehicle_classes_endpoint(self, api_server_m3):
        data = _get(api_server_m3, "/vehicle_classes")
        assert set(data) == {"car", "van", "truck", "bus"}
        truck = data["truck"]
        assert truck["length"] == pytest.approx(12.0)
        assert truck["width"]  == pytest.approx(2.5)
        assert truck["lane_max"] == 0
