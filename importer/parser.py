"""Parse raw OSM JSON into structured node/edge dicts; split ways at junctions."""
from __future__ import annotations

from importer.projection import latlng_to_mercator

# Highway types we want to include (excludes footways, cycleways, etc.)
_HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
    "service",
    "living_street",
    "unclassified",
    "road",
}


def parse_osm(data: dict) -> dict:
    """Parse raw Overpass/OSM JSON.

    Returns::

        {
          "nodes": {node_id: {"id", "lat", "lon", "x", "y"}, ...},
          "ways":  [{"id", "nodes": [node_ids], "tags": {...}}, ...],
        }

    Only ``highway`` ways matching ``_HIGHWAY_TYPES`` are included.
    Non-highway ways (buildings, waterways, etc.) are filtered out.
    """
    elements = data.get("elements", [])

    # Index all raw nodes first (we need coordinates for projection).
    # The Overpass response sometimes emits the same node twice: once with
    # full tags (from ``out body``) and once in skeletal form (from
    # ``out skel qt``).  We merge so that tags are never overwritten by an
    # empty dict.
    raw_nodes: dict[int, dict] = {}
    for el in elements:
        if el.get("type") == "node":
            nid = int(el["id"])
            lat = float(el["lat"])
            lon = float(el["lon"])
            x, y = latlng_to_mercator(lat, lon)
            new_tags = el.get("tags", {})
            if nid in raw_nodes:
                # Merge: only update tags when the new element has some
                if new_tags:
                    raw_nodes[nid]["tags"].update(new_tags)
            else:
                raw_nodes[nid] = {
                    "id": nid, "lat": lat, "lon": lon, "x": x, "y": y,
                    "tags": new_tags,
                }

    # Filter highway ways
    highway_ways: list[dict] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        hw = tags.get("highway", "")
        if hw not in _HIGHWAY_TYPES:
            continue
        highway_ways.append({
            "id": int(el["id"]),
            "nodes": [int(n) for n in el.get("nodes", [])],
            "tags": tags,
        })

    # ------------------------------------------------------------------ #
    # Find junction nodes: nodes that appear in > 1 way, OR appear in the
    # interior (non-endpoint) of a way.
    # ------------------------------------------------------------------ #
    node_way_count: dict[int, int] = {}
    for way in highway_ways:
        seen_in_way: set[int] = set()
        for nid in way["nodes"]:
            if nid not in seen_in_way:
                node_way_count[nid] = node_way_count.get(nid, 0) + 1
                seen_in_way.add(nid)

    junction_nodes: set[int] = set()
    for nid, cnt in node_way_count.items():
        if cnt > 1:
            junction_nodes.add(nid)
    # Also mark internal nodes (not start/end) within a way as junctions
    for way in highway_ways:
        nodes = way["nodes"]
        for nid in nodes[1:-1]:     # interior nodes
            junction_nodes.add(nid)

    # ------------------------------------------------------------------ #
    # Split ways at junction nodes → produce edge segments
    # ------------------------------------------------------------------ #
    edges: list[dict] = []
    for way in highway_ways:
        nodes = way["nodes"]
        if len(nodes) < 2:
            continue
        segment_start = 0
        for i in range(1, len(nodes)):
            nid = nodes[i]
            is_end = (i == len(nodes) - 1)
            is_junction = nid in junction_nodes
            if is_junction or is_end:
                seg_nodes = nodes[segment_start: i + 1]
                if len(seg_nodes) >= 2:
                    edge_id = f"way_{way['id']}_{segment_start}"
                    edges.append({
                        "id": edge_id,
                        "way_id": way["id"],
                        "from_node": seg_nodes[0],
                        "to_node": seg_nodes[-1],
                        "via_nodes": seg_nodes[1:-1],
                        "tags": way["tags"],
                    })
                segment_start = i

    # Collect only the nodes referenced by kept edges
    used_nids: set[int] = set()
    for edge in edges:
        used_nids.add(edge["from_node"])
        used_nids.add(edge["to_node"])
        for nid in edge["via_nodes"]:
            used_nids.add(nid)

    nodes_out = {nid: raw_nodes[nid] for nid in used_nids if nid in raw_nodes}

    return {
        "nodes": nodes_out,
        "ways": highway_ways,
        "edges": edges,
    }
