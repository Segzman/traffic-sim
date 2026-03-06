"""Milestone 6 — Editor state: export schema, import/export roundtrip,
undo/redo, delete cascades, and OSM import."""
import json
import os
import pytest

from editor.state import EditorState, SCENARIO_SCHEMA


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _minimal_state() -> EditorState:
    """Return a small but complete EditorState with 2 nodes and 1 edge."""
    s = EditorState(source="freehand")
    s.add_node("N1", x=0.0, y=0.0, junction_type="uncontrolled")
    s.add_node("N2", x=100.0, y=0.0, junction_type="signal")
    s.add_edge(
        "E1",
        from_node="N1",
        to_node="N2",
        num_lanes=2,
        speed_limit=13.9,
        oneway=True,
        geometry=[[0.0, 0.0], [100.0, 0.0]],
        quality_flags={"lanes": "green", "speed_limit": "amber"},
    )
    return s


def _load_osm_fixture() -> dict:
    """Load the offline OSM fixture used in M4 tests."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "osm_small.json"
    )
    with open(fixture_path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ #
# test_export_schema_valid
# ------------------------------------------------------------------ #

def test_export_schema_valid():
    """A programmatically-constructed EditorState exports valid schema."""
    pytest.importorskip("jsonschema")
    import jsonschema

    s = _minimal_state()
    data = s.export()

    # Should not raise
    jsonschema.validate(data, SCENARIO_SCHEMA)

    # Spot-check key structure
    assert "meta" in data
    assert "nodes" in data
    assert "edges" in data
    assert data["meta"]["source"] == "freehand"
    assert "N1" in data["nodes"]
    assert "E1" in data["edges"]
    assert data["edges"]["E1"]["num_lanes"] == 2
    assert data["edges"]["E1"]["quality_flags"]["lanes"] == "green"


# ------------------------------------------------------------------ #
# test_import_export_roundtrip
# ------------------------------------------------------------------ #

def test_import_export_roundtrip():
    """Load exported state back in; re-export must match the original."""
    s1 = _minimal_state()
    exported = s1.export()

    s2 = EditorState()
    s2.load(exported)
    re_exported = s2.export()

    # Node and edge content must be identical
    assert re_exported["nodes"] == exported["nodes"], (
        "Nodes differ after roundtrip"
    )
    assert re_exported["edges"] == exported["edges"], (
        "Edges differ after roundtrip"
    )
    # meta.source is preserved
    assert re_exported["meta"]["source"] == exported["meta"]["source"]


# ------------------------------------------------------------------ #
# test_undo_redo_node
# ------------------------------------------------------------------ #

def test_undo_redo_node():
    """Add node → undo removes it → redo restores it with identical state."""
    s = EditorState()
    s.add_node("A", x=10.0, y=20.0, junction_type="stop")

    # Node should be present after add
    assert "A" in s.nodes, "Node 'A' should exist after add_node"
    assert s.nodes["A"]["x"] == pytest.approx(10.0)
    assert s.nodes["A"]["junction_type"] == "stop"

    # Undo
    result = s.undo()
    assert result is True, "undo() should return True when history is non-empty"
    assert "A" not in s.nodes, "Node 'A' should be absent after undo"

    # Redo
    result = s.redo()
    assert result is True, "redo() should return True when redo stack is non-empty"
    assert "A" in s.nodes, "Node 'A' should be restored after redo"
    assert s.nodes["A"]["x"] == pytest.approx(10.0)
    assert s.nodes["A"]["junction_type"] == "stop"

    # Double undo returns False when nothing more to undo
    s.undo()
    assert s.undo() is False


# ------------------------------------------------------------------ #
# test_undo_redo_edge
# ------------------------------------------------------------------ #

def test_undo_redo_edge():
    """Add edge → undo removes it → redo restores it."""
    s = EditorState()
    s.add_node("X", x=0.0, y=0.0)
    s.add_node("Y", x=50.0, y=0.0)
    s.add_edge("XY", from_node="X", to_node="Y", num_lanes=1, speed_limit=8.3)

    assert "XY" in s.edges, "Edge 'XY' should exist after add_edge"

    # Undo the edge (most recent action)
    s.undo()
    assert "XY" not in s.edges, "Edge 'XY' should be absent after undo"

    # Redo
    s.redo()
    assert "XY" in s.edges, "Edge 'XY' should be restored after redo"
    assert s.edges["XY"]["from_node"] == "X"
    assert s.edges["XY"]["speed_limit"] == pytest.approx(8.3)


# ------------------------------------------------------------------ #
# test_delete_node_removes_edges
# ------------------------------------------------------------------ #

def test_delete_node_removes_edges():
    """Deleting a node with 3 connected edges cascades to remove all 3 edges."""
    s = EditorState()
    # Hub node C connected to A, B, D
    for nid, x, y in [("A", -100, 0), ("B", 0, 100), ("C", 0, 0), ("D", 100, 0)]:
        s.add_node(nid, x=x, y=y)

    s.add_edge("CA", from_node="C", to_node="A", num_lanes=1, speed_limit=13.9)
    s.add_edge("CB", from_node="C", to_node="B", num_lanes=1, speed_limit=13.9)
    s.add_edge("DC", from_node="D", to_node="C", num_lanes=1, speed_limit=13.9)

    assert len(s.edges) == 3

    # Delete the hub node
    s.delete_node("C")

    assert "C" not in s.nodes, "Node C should be removed"
    assert "CA" not in s.edges, "Edge CA should be cascaded away"
    assert "CB" not in s.edges, "Edge CB should be cascaded away"
    assert "DC" not in s.edges, "Edge DC should be cascaded away"
    assert len(s.edges) == 0

    # Undo should restore node and all 3 edges
    s.undo()
    assert "C" in s.nodes
    assert len(s.edges) == 3
    assert "CA" in s.edges
    assert "CB" in s.edges
    assert "DC" in s.edges


# ------------------------------------------------------------------ #
# test_osm_import_loads_to_state
# ------------------------------------------------------------------ #

def test_osm_import_loads_to_state():
    """OSM fixture loaded via load_osm() produces edges for all highway ways."""
    osm_data = _load_osm_fixture()

    s = EditorState()
    s.load_osm(osm_data)

    # Fixture has 2 highway ways; way 201 gets split at shared node 102
    # → 3 edges total (101→102, 102→103, 102→104→105 or similar)
    assert len(s.edges) >= 2, (
        f"Expected at least 2 edges from OSM fixture; got {len(s.edges)}"
    )
    assert len(s.nodes) >= 3, (
        f"Expected at least 3 nodes from OSM fixture; got {len(s.nodes)}"
    )
    assert s.meta["source"] == "osm_import"

    # All edges have required keys
    for eid, e in s.edges.items():
        assert "from_node" in e, f"Edge {eid} missing from_node"
        assert "to_node" in e, f"Edge {eid} missing to_node"
        assert "num_lanes" in e, f"Edge {eid} missing num_lanes"
        assert "speed_limit" in e, f"Edge {eid} missing speed_limit"
        assert "quality_flags" in e, f"Edge {eid} missing quality_flags"
