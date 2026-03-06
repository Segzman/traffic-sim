"""Python implementation of the Map Editor state.

This module provides :class:`EditorState`, a pure-Python representation of
the editor's internal state.  It is used by the Python test-suite
(``tests/test_editor_export.py``) to exercise the state manipulation and
JSON export logic without needing a browser.

The exported schema is intentionally identical to the one consumed by the
simulation engine so that a scenario saved from the browser editor can be
directly passed to :class:`engine.network_simulation.NetworkSimulation`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ------------------------------------------------------------------ #
# JSON Schema (used by validate())
# ------------------------------------------------------------------ #

SCENARIO_SCHEMA: dict = {
    "type": "object",
    "required": ["meta", "nodes", "edges"],
    "properties": {
        "meta": {
            "type": "object",
            "required": ["source", "created_at"],
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["freehand", "osm_import", "hybrid"],
                },
                "osm_bbox": {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        {"type": "null"},
                    ]
                },
                "created_at": {"type": "string"},
            },
        },
        "nodes": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["x", "y", "junction_type"],
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "junction_type": {
                        "type": "string",
                        "enum": ["uncontrolled", "signal", "yield", "stop"],
                    },
                    "signal_plan": {},
                },
            },
        },
        "edges": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["from_node", "to_node", "num_lanes", "speed_limit"],
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "num_lanes": {"type": "integer", "minimum": 1},
                    "speed_limit": {"type": "number", "exclusiveMinimum": 0},
                    "oneway": {"type": "boolean"},
                    "geometry": {"type": "array"},
                    "quality_flags": {"type": "object"},
                },
            },
        },
    },
}


# ------------------------------------------------------------------ #
# EditorState
# ------------------------------------------------------------------ #

class EditorState:
    """In-memory editor state with undo/redo support.

    All mutation methods go through the command pattern so that every change
    can be undone with :meth:`undo` and re-applied with :meth:`redo`.

    Parameters
    ----------
    source:
        Initial ``meta.source`` tag — ``"freehand"``, ``"osm_import"``, or
        ``"hybrid"``.
    """

    def __init__(self, source: str = "freehand") -> None:
        self.meta: dict[str, Any] = {
            "source": source,
            "osm_bbox": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self._history: list[tuple] = []   # [(undo_fn, redo_fn), ...]
        self._redo_stack: list[tuple] = []

    # ------------------------------------------------------------------ #
    # Internal command execution
    # ------------------------------------------------------------------ #

    def _execute(self, undo_fn, redo_fn) -> None:
        """Execute *redo_fn*, record the pair, and clear the redo stack."""
        redo_fn()
        self._history.append((undo_fn, redo_fn))
        self._redo_stack.clear()

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        node_id: str,
        x: float,
        y: float,
        junction_type: str = "uncontrolled",
        signal_plan: Any = None,
    ) -> None:
        """Add a node; undoable."""
        snapshot = {
            "x": x,
            "y": y,
            "junction_type": junction_type,
            "signal_plan": signal_plan,
        }

        def _do():
            self.nodes[node_id] = dict(snapshot)

        def _undo():
            self.nodes.pop(node_id, None)

        self._execute(_undo, _do)

    def update_node(self, node_id: str, **kwargs) -> None:
        """Update node attributes; undoable."""
        before = dict(self.nodes.get(node_id, {}))
        after = {**before, **kwargs}

        def _do():
            self.nodes[node_id] = dict(after)

        def _undo():
            self.nodes[node_id] = dict(before)

        self._execute(_undo, _do)

    def delete_node(self, node_id: str) -> None:
        """Delete a node *and* all edges that reference it; undoable."""
        saved_node = dict(self.nodes.get(node_id, {}))
        connected = {
            eid: dict(e)
            for eid, e in self.edges.items()
            if e.get("from_node") == node_id or e.get("to_node") == node_id
        }

        def _do():
            self.nodes.pop(node_id, None)
            for eid in connected:
                self.edges.pop(eid, None)

        def _undo():
            self.nodes[node_id] = dict(saved_node)
            for eid, e in connected.items():
                self.edges[eid] = dict(e)

        self._execute(_undo, _do)

    # ------------------------------------------------------------------ #
    # Edge operations
    # ------------------------------------------------------------------ #

    def add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        num_lanes: int = 1,
        speed_limit: float = 13.9,
        oneway: bool = False,
        geometry: list | None = None,
        quality_flags: dict | None = None,
    ) -> None:
        """Add an edge; undoable."""
        snapshot = {
            "from_node": from_node,
            "to_node": to_node,
            "num_lanes": num_lanes,
            "speed_limit": speed_limit,
            "oneway": oneway,
            "geometry": list(geometry or []),
            "quality_flags": dict(quality_flags or {}),
        }

        def _do():
            self.edges[edge_id] = dict(snapshot)

        def _undo():
            self.edges.pop(edge_id, None)

        self._execute(_undo, _do)

    def update_edge(self, edge_id: str, **kwargs) -> None:
        """Update edge attributes; undoable."""
        before = dict(self.edges.get(edge_id, {}))
        after = {**before, **kwargs}

        def _do():
            self.edges[edge_id] = dict(after)

        def _undo():
            self.edges[edge_id] = dict(before)

        self._execute(_undo, _do)

    def delete_edge(self, edge_id: str) -> None:
        """Delete a single edge; undoable."""
        saved = dict(self.edges.get(edge_id, {}))

        def _do():
            self.edges.pop(edge_id, None)

        def _undo():
            self.edges[edge_id] = dict(saved)

        self._execute(_undo, _do)

    # ------------------------------------------------------------------ #
    # Undo / Redo
    # ------------------------------------------------------------------ #

    def undo(self) -> bool:
        """Undo the last command. Returns True if an action was undone."""
        if not self._history:
            return False
        undo_fn, redo_fn = self._history.pop()
        undo_fn()
        self._redo_stack.append((undo_fn, redo_fn))
        return True

    def redo(self) -> bool:
        """Redo the last undone command. Returns True if an action was redone."""
        if not self._redo_stack:
            return False
        undo_fn, redo_fn = self._redo_stack.pop()
        redo_fn()
        self._history.append((undo_fn, redo_fn))
        return True

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def export(self) -> dict:
        """Export the current state as a plain JSON-serialisable dict."""
        return {
            "meta": dict(self.meta),
            "nodes": {nid: dict(n) for nid, n in self.nodes.items()},
            "edges": {eid: dict(e) for eid, e in self.edges.items()},
        }

    def load(self, data: dict) -> None:
        """Replace the current state with *data*; clears undo/redo history.

        Parameters
        ----------
        data:
            A previously :meth:`export`-ed state dict.
        """
        self.meta = dict(data.get("meta", {}))
        self.nodes = {nid: dict(n) for nid, n in data.get("nodes", {}).items()}
        self.edges = {eid: dict(e) for eid, e in data.get("edges", {}).items()}
        self._history.clear()
        self._redo_stack.clear()

    def load_osm(self, osm_data: dict) -> None:
        """Populate state from a raw Overpass API JSON response.

        Uses the :mod:`importer` pipeline (parser → inference) to convert
        OSM ways into editor edges.  Clears the current state and undo history.

        Parameters
        ----------
        osm_data:
            Raw Overpass JSON (same format as ``tests/fixtures/osm_small.json``).
        """
        from importer import parser, inference  # lazy import avoids circular dep

        parsed = parser.parse_osm(osm_data)
        enriched = inference.infer(parsed)

        self.nodes.clear()
        self.edges.clear()
        self._history.clear()
        self._redo_stack.clear()
        self.meta["source"] = "osm_import"

        for nid, node in enriched["nodes"].items():
            self.nodes[str(nid)] = {
                "x": float(node["x"]),
                "y": float(node["y"]),
                "junction_type": "uncontrolled",
                "signal_plan": None,
            }

        for edge in enriched["edges"]:
            eid = str(edge["id"])
            self.edges[eid] = {
                "from_node": str(edge["from_node"]),
                "to_node": str(edge["to_node"]),
                "num_lanes": int(edge.get("num_lanes", 1)),
                "speed_limit": float(edge.get("speed_limit", 13.9)),
                "oneway": bool(edge.get("oneway", False)),
                "geometry": [
                    [float(p[0]), float(p[1])] for p in edge.get("geometry", [])
                ],
                "quality_flags": dict(edge.get("quality_flags", {})),
            }

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Validate exported state against :data:`SCENARIO_SCHEMA`.

        Returns True if valid, False otherwise.  Requires the ``jsonschema``
        package (added to requirements.txt in M6).
        """
        try:
            import jsonschema  # type: ignore
            jsonschema.validate(self.export(), SCENARIO_SCHEMA)
            return True
        except Exception:
            return False
