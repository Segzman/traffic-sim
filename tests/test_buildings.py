"""Building use, floor, and capacity estimation tests."""
import pytest

from engine.buildings import (
    classify_building_purpose,
    estimate_building_capacity,
    estimate_floors,
    infer_building_use,
)


def test_infer_building_use():
    assert infer_building_use({"building": "apartments"}) == "residential"
    assert infer_building_use({"building": "house"}) == "residential"
    assert infer_building_use({"building": "semidetached_house"}) == "residential"
    assert infer_building_use({"building": "office"}) == "office"
    assert infer_building_use({"building": "commercial"}) == "office"
    assert infer_building_use({"building": "industrial"}) == "office"
    assert infer_building_use({"building": "retail"}) == "office"
    assert infer_building_use({"office": "yes"}) == "office"
    assert infer_building_use({"addr:housenumber": "12"}) == "residential"
    assert infer_building_use({"building": "yes"}) == "other"
    assert infer_building_use({}) == "other"


def test_estimate_floors_prefers_levels_over_height_and_footprint():
    assert estimate_floors({"building:levels": "4"}, footprint_area=0.0) == 4
    assert estimate_floors(
        {"building:levels": "2.6", "height": "30"},
        footprint_area=0.0,
    ) == 3
    assert estimate_floors(
        {"building:levels": "0"},
        footprint_area=0.0,
    ) == 1


def test_estimate_floors_uses_height_when_levels_missing():
    assert estimate_floors({"height": "9.9"}, footprint_area=0.0) == 3
    assert estimate_floors({"height": "10"}, footprint_area=0.0) == 3
    assert estimate_floors(
        {"height": "10"},
        footprint_area=0.0,
        meters_per_floor=5.0,
    ) == 2
    assert estimate_floors(
        {"building:levels": "bad", "height": "0"},
        footprint_area=1000.0,
    ) == 3


def test_estimate_floors_falls_back_to_footprint_area():
    assert estimate_floors({"building": "office"}, footprint_area=1000.0) == 2
    assert estimate_floors({"building": "house"}, footprint_area=1000.0) == 3
    assert estimate_floors({}, footprint_area=0.0) == 1
    assert estimate_floors({}, footprint_area=1.0) == 1


def test_estimate_building_capacity_uses_use_and_floors():
    office = {
        "footprint_area": 1000.0,
        "tags": {"building": "office", "building:levels": "2"},
    }
    assert estimate_building_capacity(office) == pytest.approx(2000.0 / 12.0)

    home = {
        "footprint_area": 700.0,
        "tags": {"building": "house"},
    }
    assert estimate_building_capacity(home) == pytest.approx(1400.0 / 25.0)


def test_estimate_building_capacity_supports_custom_density():
    building = {
        "footprint_area": 1000.0,
        "tags": {"building": "office", "building:levels": "2"},
    }
    assert estimate_building_capacity(
        building,
        office_sqm_per_person=20.0,
    ) == pytest.approx(100.0)


def test_estimate_building_capacity_clamps_negative_area():
    capacity = estimate_building_capacity(
        {"footprint_area": -500.0, "tags": {}},
    )
    assert capacity > 0.0
    assert capacity == pytest.approx(1.0 / 25.0)


def test_classify_building_purpose():
    assert classify_building_purpose({"building": "apartments"}) == "home"
    assert classify_building_purpose({"building": "detached"}) == "home"
    assert classify_building_purpose({"building": "office"}) == "work"
    assert classify_building_purpose({"building": "commercial"}) == "work"
    assert classify_building_purpose({"building": "industrial"}) == "work"
    assert classify_building_purpose({"building": "retail"}) == "retail"
    assert classify_building_purpose({"building": "mall"}) == "retail"
    assert classify_building_purpose({"building": "supermarket"}) == "retail"
    assert classify_building_purpose({"office": "yes"}) == "work"
    assert classify_building_purpose({"building": "yes"}) == "other"
    assert classify_building_purpose({}) == "other"
