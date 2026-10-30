"""Vehicle agent dataclass defaults and per-instance state tests."""
from engine.agents import Vehicle


def test_vehicle_defaults():
    veh = Vehicle(id=7, lane_id=2, position_s=12.5, speed=8.0)

    assert veh.id == 7
    assert veh.lane_id == 2
    assert veh.position_s == 12.5
    assert veh.speed == 8.0

    assert veh.length == 4.5
    assert veh.width == 1.8
    assert veh.vehicle_type == "car"

    assert veh.v0 == 30.0
    assert veh.s0 == 2.0
    assert veh.T == 1.5
    assert veh.a_max == 1.4
    assert veh.b == 2.0
    assert veh.acceleration == 0.0

    assert veh.politeness == 0.3
    assert veh.b_safe == 4.0
    assert veh.delta_a_thr == 0.2
    assert veh.bias_right == 0.1
    assert veh.lane_change_cooldown == 0.0

    assert veh.route == []
    assert veh.route_index == 0
    assert veh.current_edge == ""
    assert veh.entry_time == 0.0
    assert veh.lane_changes == 0
    assert veh.stops == 0
    assert veh._was_stopped is False


def test_vehicle_route_lists_are_independent():
    first = Vehicle(id=1, lane_id=0, position_s=0.0, speed=0.0)
    second = Vehicle(id=2, lane_id=0, position_s=0.0, speed=0.0)

    first.route.append("E1")

    assert first.route == ["E1"]
    assert second.route == []


def test_vehicle_accepts_explicit_overrides():
    veh = Vehicle(
        id=3,
        lane_id=1,
        position_s=25.0,
        speed=14.0,
        length=12.0,
        width=2.5,
        vehicle_type="truck",
        v0=22.0,
        s0=3.0,
        T=2.0,
        a_max=1.0,
        b=1.5,
        acceleration=-0.4,
        politeness=0.1,
        b_safe=3.0,
        delta_a_thr=0.3,
        bias_right=0.0,
        lane_change_cooldown=1.5,
        route=["AB", "BC"],
        route_index=1,
        current_edge="BC",
        entry_time=5.0,
        lane_changes=2,
        stops=1,
        _was_stopped=True,
    )

    assert veh.length == 12.0
    assert veh.width == 2.5
    assert veh.vehicle_type == "truck"
    assert veh.v0 == 22.0
    assert veh.s0 == 3.0
    assert veh.T == 2.0
    assert veh.a_max == 1.0
    assert veh.b == 1.5
    assert veh.acceleration == -0.4
    assert veh.politeness == 0.1
    assert veh.b_safe == 3.0
    assert veh.delta_a_thr == 0.3
    assert veh.bias_right == 0.0
    assert veh.lane_change_cooldown == 1.5
    assert veh.route == ["AB", "BC"]
    assert veh.route_index == 1
    assert veh.current_edge == "BC"
    assert veh.entry_time == 5.0
    assert veh.lane_changes == 2
    assert veh.stops == 1
    assert veh._was_stopped is True
