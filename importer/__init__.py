"""OSM import pipeline.

Public API
----------
import_bbox(south, west, north, east, osm_data=None) -> scenario dict
"""
from __future__ import annotations

from importer import overpass, parser, inference
from importer.projection import mercator_distance


def import_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    osm_data: dict | None = None,
) -> dict:
    """Import OSM road network for a bounding box and return a scenario dict.

    Parameters
    ----------
    south, west, north, east:
        Bounding box in WGS-84 decimal degrees.
    osm_data:
        Pre-loaded Overpass response dict.  When *None*, data is fetched
        from the Overpass API (with disk caching).

    Returns
    -------
    dict
        A scenario dict compatible with ``engine.simulation.Simulation``.
        Includes a ``"network"`` key with the full enriched graph for use
        in later milestones (routing, visualisation).
    """
    if osm_data is None:
        osm_data = overpass.fetch((south, west, north, east))

    parsed = parser.parse_osm(osm_data)
    enriched = inference.infer(parsed)

    # ------------------------------------------------------------------ #
    # Build a minimal simulation-compatible road section from the network.
    # We derive representative road parameters from all enriched edges.
    # ------------------------------------------------------------------ #
    edges = enriched.get("edges", [])
    nodes = enriched.get("nodes", {})

    if edges:
        # Compute length of each edge from its node geometry
        def _edge_length(edge: dict) -> float:
            all_nids = (
                [edge["from_node"]]
                + edge.get("via_nodes", [])
                + [edge["to_node"]]
            )
            total = 0.0
            coords = [
                (nodes[n]["x"], nodes[n]["y"])
                for n in all_nids if n in nodes
            ]
            for i in range(1, len(coords)):
                total += mercator_distance(coords[i - 1], coords[i])
            return total

        total_length = sum(_edge_length(e) for e in edges)
        avg_lanes = round(
            sum(e.get("num_lanes", 1) for e in edges) / len(edges)
        )
        avg_speed = sum(e.get("speed_limit", 13.9) for e in edges) / len(edges)
    else:
        total_length = 200.0
        avg_lanes = 1
        avg_speed = 13.9

    road_length = max(100.0, total_length)

    scenario: dict = {
        "road": {
            "length": round(road_length, 1),
            "num_lanes": max(1, avg_lanes),
            "speed_limit": round(avg_speed, 2),
        },
        "vehicles": {"count": 0},
        "duration": 60.0,
        "network": {
            "nodes": {
                str(nid): node
                for nid, node in nodes.items()
            },
            "edges": edges,
        },
    }
    return scenario
