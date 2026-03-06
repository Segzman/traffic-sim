"""Generate commute demand for a Network.

Classifies nodes as *residential* or *commercial* based on the speed limits
of their adjacent edges, then builds an OD (origin–destination) demand matrix
simulating morning commute patterns (residential → commercial).

Public API
----------
classify_nodes(network)  → (residential_ids, commercial_ids)
demand_factor(hour)      → float multiplier for time-of-day scaling
generate_commute_demand(network, ...)  → {origin: {dest: veh_hr}}
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.network import Network

# --------------------------------------------------------------------------- #
# Speed threshold separating residential from commercial context
# 11.1 m/s ≈ 40 km/h.  Nodes only touching slow roads → residential.
# --------------------------------------------------------------------------- #
_RESIDENTIAL_MAX_MS: float = 11.1


def classify_nodes(
    network: "Network",
) -> tuple[list[str], list[str]]:
    """Classify network nodes as residential or commercial.

    A node is **residential** if the maximum speed of all adjacent edges
    (both outgoing and incoming) is at or below ``_RESIDENTIAL_MAX_MS``.
    All other reachable nodes are classified as **commercial**.

    Nodes with no adjacent edges are excluded from both lists.

    Returns
    -------
    (residential_ids, commercial_ids) : tuple[list[str], list[str]]
    """
    # Build incoming-edge map so we can check both in and out edges per node
    incoming: dict[str, list[str]] = {}
    for eid, edge in network.edges.items():
        incoming.setdefault(edge.to_node, []).append(eid)

    residential: list[str] = []
    commercial: list[str] = []

    for node_id in network.nodes:
        out_eids = network._adj.get(node_id, [])
        in_eids = incoming.get(node_id, [])
        all_eids = out_eids + in_eids
        if not all_eids:
            continue
        speeds = [
            network.edges[eid].speed_limit
            for eid in all_eids
            if eid in network.edges
        ]
        if not speeds:
            continue
        if max(speeds) <= _RESIDENTIAL_MAX_MS:
            residential.append(node_id)
        else:
            commercial.append(node_id)

    return residential, commercial


def demand_factor(hour: float) -> float:
    """Return a demand multiplier for the given time-of-day *hour* (0–24).

    Peaks:
    * 7–9 AM   →  3.0× (AM commute)
    * 16–19 PM →  2.5× (PM commute)

    Other periods scale proportionally down to 0.15× overnight.
    """
    h = hour % 24.0
    if   7.0  <= h < 9.0:   return 3.0
    elif 6.0  <= h < 7.0:   return 1.5
    elif 9.0  <= h < 10.0:  return 1.2
    elif 16.0 <= h < 19.0:  return 2.5
    elif 15.0 <= h < 16.0:  return 1.3
    elif 19.0 <= h < 20.0:  return 1.1
    elif 10.0 <= h < 15.0:  return 0.6
    else:                    return 0.15


def generate_commute_demand(
    network: "Network",
    seed: int = 42,
    peak_veh_hr: float = 400.0,
    max_pairs: int = 60,
) -> dict[str, dict[str, float]]:
    """Build an OD demand matrix for morning commute trips.

    Residential nodes are origins (home), commercial nodes are destinations
    (work / school).  Each origin gets 1–3 destinations chosen at random.
    The total expected throughput at peak hour is approximately *peak_veh_hr*.

    If classification yields an empty residential or commercial set the network
    is split 50/50 by node index as a fallback (useful for sparse imports).

    Parameters
    ----------
    network:
        Fully constructed Network (nodes + edges must be populated).
    seed:
        RNG seed for reproducibility.
    peak_veh_hr:
        Target total vehicle flow per hour across all OD pairs at peak demand.
    max_pairs:
        Cap on total OD pairs created (keeps spawn queue manageable).

    Returns
    -------
    ``{origin_node_id: {dest_node_id: flow_veh_hr}}``
    """
    rng = random.Random(seed)
    residential, commercial = classify_nodes(network)

    # Fallback: split network nodes 50/50
    if not residential or not commercial:
        all_nodes = list(network.nodes.keys())
        rng.shuffle(all_nodes)
        mid = max(1, len(all_nodes) // 2)
        residential = all_nodes[:mid]
        commercial = all_nodes[mid:]

    # Sample representative subsets to keep generation fast
    n_sample = min(max_pairs, max(len(residential), len(commercial)))
    sample_res = rng.sample(residential, min(len(residential), n_sample))
    sample_com = rng.sample(commercial,  min(len(commercial),  n_sample))

    # Flow per origin: share of peak_veh_hr spread across all origins
    flow_per_origin = peak_veh_hr / max(1, len(sample_res))

    demand: dict[str, dict[str, float]] = {}
    pairs_added = 0

    for origin in sample_res:
        if pairs_added >= max_pairs:
            break
        # Each origin drives to 1–3 destinations
        n_dests = rng.randint(1, min(3, len(sample_com)))
        dests = rng.sample(sample_com, n_dests)
        for dest in dests:
            if dest == origin or pairs_added >= max_pairs:
                continue
            # Distribute flow with ±40 % variation so not all pairs are equal
            flow = flow_per_origin / max(1, n_dests) * (0.6 + 0.8 * rng.random())
            demand.setdefault(origin, {})[dest] = round(flow, 2)
            pairs_added += 1

    return demand
