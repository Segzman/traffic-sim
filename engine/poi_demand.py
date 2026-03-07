"""Milestone 3 demand generation: WorldPop origins + POI purposes + building capacities."""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from engine.buildings import classify_building_purpose, estimate_building_capacity
from engine.worldpop import load_worldpop_weights, load_worldpop_surrounding_exposure

if TYPE_CHECKING:
    from engine.network import Network

try:
    from scipy.spatial import KDTree as _KDTree
except Exception:  # pragma: no cover - optional dependency
    _KDTree = None


class _NearestTree:
    def __init__(self, pts: np.ndarray):
        self._pts = pts
        self._tree = _KDTree(pts) if _KDTree is not None else None

    def query(self, point: tuple[float, float]) -> tuple[float, int]:
        if self._tree is not None:
            dist, idx = self._tree.query(point)
            return float(dist), int(idx)
        arr = self._pts - np.asarray(point, dtype=float)
        d2 = np.einsum("ij,ij->i", arr, arr)
        idx = int(np.argmin(d2))
        return float(np.sqrt(d2[idx])), idx


_PURPOSE_ORDER = ("home", "work", "school", "retail", "other")

# First-match-wins priority map.
PURPOSE_MAP = [
    (("building", "apartments"), "home"),
    (("building", "residential"), "home"),
    (("building", "house"), "home"),
    (("building", "office"), "work"),
    (("building", "commercial"), "work"),
    (("building", "industrial"), "work"),
    (("office", "*"), "work"),
    (("amenity", "school"), "school"),
    (("amenity", "university"), "school"),
    (("amenity", "college"), "school"),
    (("shop", "*"), "retail"),
    (("amenity", "restaurant"), "retail"),
    (("amenity", "cafe"), "retail"),
]

DEFAULT_PURPOSE_SPLIT = {
    "work": 0.40,
    "school": 0.15,
    "retail": 0.25,
    "other": 0.20,
}


def _classify_purpose(tags: dict) -> str:
    for (k, v), purpose in PURPOSE_MAP:
        if k not in tags:
            continue
        if v == "*" or str(tags.get(k, "")).lower() == v:
            return purpose

    if "building" in tags:
        return classify_building_purpose(tags)
    return "other"


def _safe_probs(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.ones_like(w) / max(1, len(w))
    return w / s


def _build_snap_tree(network: "Network") -> tuple[list[str], np.ndarray, _NearestTree]:
    node_ids = list(network.nodes.keys())
    xy = np.array([(network.nodes[n].x, network.nodes[n].y) for n in node_ids], dtype=float)
    return node_ids, xy, _NearestTree(xy)


def build_purpose_nodes(
    network: "Network",
    pois: list[dict] | None,
    buildings: list[dict] | None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Build purpose -> node list plus per-purpose node weights."""
    node_ids, _, tree = _build_snap_tree(network)

    buckets: dict[str, set[str]] = {p: set() for p in _PURPOSE_ORDER}
    weights: dict[str, dict[str, float]] = {p: {} for p in _PURPOSE_ORDER}

    for poi in pois or []:
        x = poi.get("x")
        y = poi.get("y")
        if x is None or y is None:
            continue
        purpose = _classify_purpose(poi.get("tags", {}))
        _, idx = tree.query((float(x), float(y)))
        nid = node_ids[int(idx)]
        buckets[purpose].add(nid)
        weights[purpose][nid] = weights[purpose].get(nid, 0.0) + 1.0

    for b in buildings or []:
        x = b.get("x")
        y = b.get("y")
        if x is None or y is None:
            continue
        purpose = _classify_purpose(b.get("tags", {}))
        _, idx = tree.query((float(x), float(y)))
        nid = node_ids[int(idx)]
        cap = max(1.0, estimate_building_capacity(b))
        buckets[purpose].add(nid)
        weights[purpose][nid] = weights[purpose].get(nid, 0.0) + cap

    # Fallbacks: keep model robust on sparse OSM areas.
    all_nodes = list(network.nodes.keys())
    if not buckets["home"]:
        buckets["home"] = set(all_nodes)
    for p in ("work", "school", "retail", "other"):
        if not buckets[p]:
            buckets[p] = set(all_nodes)

    for p in _PURPOSE_ORDER:
        for nid in buckets[p]:
            weights[p].setdefault(nid, 1.0)

    return ({k: sorted(v) for k, v in buckets.items()}, weights)


def _sample_destination(
    origin_id: str,
    purpose: str,
    candidates: list[str],
    node_weights: dict[str, float],
    purpose_weights: dict[str, float],
    network: "Network",
    rng: np.random.Generator,
    beta: float,
    gamma_km: float,
) -> str:
    ox = network.nodes[origin_id].x
    oy = network.nodes[origin_id].y
    w: list[float] = []

    for nid in candidates:
        if nid == origin_id:
            w.append(0.0)
            continue
        nx = network.nodes[nid].x
        ny = network.nodes[nid].y
        d_km = math.hypot(nx - ox, ny - oy) / 1000.0
        cap = max(1e-6, float(purpose_weights.get(nid, 1.0)))
        origin = max(1e-6, float(node_weights.get(origin_id, 1.0)))
        score = (cap ** beta) * math.exp(-gamma_km * d_km) * (origin ** 0.05)
        w.append(score)

    probs = _safe_probs(np.array(w, dtype=float))
    idx = int(rng.choice(len(candidates), p=probs))
    return candidates[idx]


def _weighted_sample_no_replace(
    items: list[str],
    probs: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> list[str]:
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    idx = rng.choice(len(items), size=k, replace=False, p=_safe_probs(probs))
    return [items[int(i)] for i in idx]


def _edge_capacity_veh_hr(edge, *, high_speed_ms: float) -> float:
    """Approximate directional edge capacity (veh/hr) from lanes + speed + class."""
    lanes = max(1.0, float(getattr(edge, "num_lanes", 1)))
    v = max(1.0, float(getattr(edge, "speed_limit", high_speed_ms)))
    tag = str(getattr(edge, "road_type", "")).lower()

    base_per_lane = 1800.0
    if tag in {"motorway", "motorway_link"}:
        cls = 1.0
    elif tag in {"trunk", "trunk_link"}:
        cls = 0.95
    elif tag in {"primary", "primary_link"}:
        cls = 0.80
    else:
        cls = 0.65

    speed_factor = min(1.15, max(0.60, v / max(1e-6, high_speed_ms)))
    return base_per_lane * lanes * cls * speed_factor


def _boundary_groups(network: "Network", eligible_nodes: list[str]) -> dict[str, list[str]]:
    if not eligible_nodes:
        return {"west": [], "east": [], "south": [], "north": []}
    xs = np.array([network.nodes[n].x for n in eligible_nodes], dtype=float)
    ys = np.array([network.nodes[n].y for n in eligible_nodes], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(0.0, x_max - x_min)
    y_span = max(0.0, y_max - y_min)
    x_pad = max(1.0, 0.10 * max(1.0, x_span))
    y_pad = max(1.0, 0.10 * max(1.0, y_span))
    use_x = x_span >= 10.0
    use_y = y_span >= 10.0
    return {
        "west": [n for n in eligible_nodes if use_x and network.nodes[n].x <= x_min + x_pad],
        "east": [n for n in eligible_nodes if use_x and network.nodes[n].x >= x_max - x_pad],
        "south": [n for n in eligible_nodes if use_y and network.nodes[n].y <= y_min + y_pad],
        "north": [n for n in eligible_nodes if use_y and network.nodes[n].y >= y_max - y_pad],
    }


def _directional_side_weights(
    surrounding_exposure: dict[str, float] | None,
) -> dict[str, float]:
    if not surrounding_exposure:
        return {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0}
    raw = {
        "west": max(0.0, float(surrounding_exposure.get("west", 0.0))),
        "east": max(0.0, float(surrounding_exposure.get("east", 0.0))),
        "south": max(0.0, float(surrounding_exposure.get("south", 0.0))),
        "north": max(0.0, float(surrounding_exposure.get("north", 0.0))),
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0}
    # Keep a floor so low-pop directions can still receive through traffic.
    return {k: 0.25 + 3.0 * (v / total) for k, v in raw.items()}


def estimate_intercity_through_veh_hr(
    network: "Network",
    *,
    base_peak_veh_hr: float,
    high_speed_ms: float = 19.0,
    surrounding_exposure: dict[str, float] | None = None,
) -> float:
    """Estimate through-demand from network structure (data-derived, no fixed share)."""
    if base_peak_veh_hr <= 0.0 or not network.edges:
        return 0.0

    fast_tags = {"motorway", "motorway_link", "trunk", "trunk_link"}
    total_lane_m = 0.0
    fast_lane_m = 0.0
    eligible_nodes: set[str] = set()

    for eid, edge in network.edges.items():
        length = max(1.0, float(network.edge_length(eid)))
        lane_m = length * max(1.0, float(edge.num_lanes))
        total_lane_m += lane_m
        is_fast = (str(edge.road_type) in fast_tags) or (float(edge.speed_limit) >= high_speed_ms)
        if is_fast:
            fast_lane_m += lane_m
            eligible_nodes.add(edge.from_node)
            eligible_nodes.add(edge.to_node)

    if total_lane_m <= 0.0 or not eligible_nodes:
        return 0.0

    fast_ratio = fast_lane_m / total_lane_m
    groups = _boundary_groups(network, sorted(eligible_nodes))
    boundary_nodes = set(groups["west"] + groups["east"] + groups["south"] + groups["north"])
    boundary_ratio = len(boundary_nodes) / max(1, len(eligible_nodes))

    # Gateway capacity from outbound fast edges at boundary nodes.
    gateway_cap = 0.0
    for nid in boundary_nodes:
        for eid in network._adj.get(nid, []):
            edge = network.edges[eid]
            if (str(edge.road_type) in fast_tags) or (float(edge.speed_limit) >= high_speed_ms):
                gateway_cap += _edge_capacity_veh_hr(edge, high_speed_ms=high_speed_ms)

    # Blend data-driven estimates:
    # 1) structural share from lane-km and boundary exposure
    # 2) physical gateway carrying potential
    structural = base_peak_veh_hr * min(0.85, max(0.0, fast_ratio * (0.6 + 0.9 * boundary_ratio)))
    gateway = gateway_cap * 0.03
    estimate = max(structural, gateway)

    # Optional WorldPop pull from surrounding cities/ring.
    if surrounding_exposure:
        ext_pop = max(0.0, float(surrounding_exposure.get("total", 0.0)))
        # Data-derived scaling: larger surrounding populations increase
        # boundary-through demand, bounded by gateway capacity.
        ext_pull = ext_pop / 4000.0
        estimate = max(estimate, ext_pull)

    return max(0.0, min(gateway_cap * 0.35, estimate))


def add_intercity_through_demand(
    network: "Network",
    demand: dict[str, dict[str, float]],
    *,
    seed: int = 42,
    through_veh_hr: float | None = None,
    base_peak_veh_hr: float = 0.0,
    max_pairs: int = 120,
    high_speed_ms: float = 19.0,
    surrounding_exposure: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Inject boundary-to-boundary through trips to populate highways.

    This approximates inter-city traffic when the imported network is only a
    city slice but includes motorway/trunk corridors.
    """
    if through_veh_hr is None:
        through_veh_hr = estimate_intercity_through_veh_hr(
            network,
            base_peak_veh_hr=max(0.0, float(base_peak_veh_hr)),
            high_speed_ms=high_speed_ms,
            surrounding_exposure=surrounding_exposure,
        )
    if through_veh_hr <= 0.0 or max_pairs <= 0 or len(network.nodes) < 2:
        return demand

    rng = np.random.default_rng(seed + 1337)

    incoming: dict[str, list[str]] = {}
    for eid, edge in network.edges.items():
        incoming.setdefault(edge.to_node, []).append(eid)

    fast_tags = {"motorway", "motorway_link", "trunk", "trunk_link"}
    eligible: list[str] = []
    gateway_weight: dict[str, float] = {}
    for nid in network.nodes:
        eids = list(network._adj.get(nid, [])) + incoming.get(nid, [])
        if not eids:
            continue
        max_spd = max(float(network.edges[e].speed_limit) for e in eids)
        has_fast_tag = any(str(network.edges[e].road_type) in fast_tags for e in eids)
        if max_spd >= high_speed_ms or has_fast_tag:
            eligible.append(nid)
            w = 0.0
            for eid in network._adj.get(nid, []):
                e = network.edges[eid]
                if (str(e.road_type) in fast_tags) or (float(e.speed_limit) >= high_speed_ms):
                    w += _edge_capacity_veh_hr(e, high_speed_ms=high_speed_ms)
            for eid in incoming.get(nid, []):
                e = network.edges[eid]
                if (str(e.road_type) in fast_tags) or (float(e.speed_limit) >= high_speed_ms):
                    w += 0.5 * _edge_capacity_veh_hr(e, high_speed_ms=high_speed_ms)
            gateway_weight[nid] = max(1.0, w)

    if len(eligible) < 2:
        return demand

    groups = _boundary_groups(network, eligible)
    west = groups["west"]
    east = groups["east"]
    south = groups["south"]
    north = groups["north"]

    directions = [
        ("west", west, "east", east),
        ("east", east, "west", west),
        ("south", south, "north", north),
        ("north", north, "south", south),
    ]
    side_w = _directional_side_weights(surrounding_exposure)
    dir_base_w: list[float] = []
    for side_o, origins, side_d, dests in directions:
        if not origins or not dests:
            dir_base_w.append(0.0)
            continue
        dir_base_w.append(math.sqrt(side_w.get(side_o, 1.0) * side_w.get(side_d, 1.0)))

    route_ok: dict[tuple[str, str], bool] = {}

    def _routable(o: str, d: str) -> bool:
        if o == d:
            return False
        key = (o, d)
        if key in route_ok:
            return route_ok[key]
        route_ok[key] = bool(network.shortest_path(o, d))
        return route_ok[key]

    pair_counts: dict[tuple[str, str], int] = {}
    dir_pairs: list[list[tuple[str, str]]] = [[] for _ in directions]

    w_sum = sum(dir_base_w)
    if w_sum <= 0.0:
        dir_pair_budget = [max_pairs // 4] * len(directions)
    else:
        raw = [(w / w_sum) * max_pairs for w in dir_base_w]
        dir_pair_budget = [int(math.floor(x)) for x in raw]
        rem = max(0, int(max_pairs) - sum(dir_pair_budget))
        frac_order = sorted(
            range(len(raw)),
            key=lambda i: (raw[i] - dir_pair_budget[i]),
            reverse=True,
        )
        for i in frac_order[:rem]:
            dir_pair_budget[i] += 1
    # Guarantee at least one pair for each valid direction when possible.
    valid_dirs = [i for i, w in enumerate(dir_base_w) if w > 0.0]
    for i in valid_dirs:
        if dir_pair_budget[i] == 0 and max_pairs >= len(valid_dirs):
            dir_pair_budget[i] = 1

    for di, (_, origins, _, dests) in enumerate(directions):
        if not origins or not dests:
            continue
        target_per_dir = max(0, int(dir_pair_budget[di]))
        if target_per_dir <= 0:
            continue
        # Gateway-balanced sampling: spread OD across multiple boundary nodes.
        o_probs = _safe_probs(np.array([gateway_weight.get(o, 1.0) for o in origins], dtype=float))
        d_probs = _safe_probs(np.array([gateway_weight.get(d, 1.0) for d in dests], dtype=float))

        k_o = min(len(origins), max(2, int(math.sqrt(target_per_dir * 2))))
        k_d = min(len(dests),   max(2, int(math.sqrt(target_per_dir * 2))))
        sel_o = _weighted_sample_no_replace(origins, o_probs, k_o, rng)
        sel_d = _weighted_sample_no_replace(dests, d_probs, k_d, rng)

        pool: list[tuple[str, str]] = []
        pool_w: list[float] = []
        for o in sel_o:
            for d in sel_d:
                if not _routable(o, d):
                    continue
                ox, oy = network.nodes[o].x, network.nodes[o].y
                dx, dy = network.nodes[d].x, network.nodes[d].y
                dist_km = max(0.1, math.hypot(dx - ox, dy - oy) / 1000.0)
                w = math.sqrt(gateway_weight.get(o, 1.0) * gateway_weight.get(d, 1.0)) * dist_km
                pool.append((o, d))
                pool_w.append(w)

        if not pool:
            continue

        take = min(target_per_dir, len(pool))
        idxs = rng.choice(len(pool), size=take, replace=False, p=_safe_probs(np.array(pool_w, dtype=float)))
        for i in idxs:
            o, d = pool[int(i)]
            pair_counts[(o, d)] = pair_counts.get((o, d), 0) + 1
            dir_pairs[di].append((o, d))

    if not pair_counts:
        return demand

    # Allocate directional through-flow with surrounding-population weighting.
    flow_w = [dir_base_w[i] if dir_pairs[i] else 0.0 for i in range(len(directions))]
    flow_sum = sum(flow_w)
    if flow_sum <= 0.0:
        flow_w = [1.0 if dir_pairs[i] else 0.0 for i in range(len(directions))]
        flow_sum = sum(flow_w)

    for i in range(len(directions)):
        pairs = dir_pairs[i]
        if not pairs:
            continue
        dir_flow = float(through_veh_hr) * (flow_w[i] / flow_sum)
        per_pair = dir_flow / max(1, len(pairs))
        for o, d in pairs:
            flow = round(per_pair, 2)
            if flow <= 0.0:
                continue
            demand.setdefault(o, {})
            demand[o][d] = round(demand[o].get(d, 0.0) + flow, 2)

    return demand


def add_intercity_exchange_demand(
    network: "Network",
    demand: dict[str, dict[str, float]],
    *,
    node_weights: dict[str, float],
    seed: int = 42,
    inflow_veh_hr: float | None = None,
    outflow_veh_hr: float | None = None,
    base_peak_veh_hr: float = 0.0,
    max_pairs: int = 120,
    high_speed_ms: float = 19.0,
    surrounding_exposure: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Inject boundary<->interior exchange trips (entering and leaving city)."""
    if max_pairs <= 0 or len(network.nodes) < 2:
        return demand

    est_through = estimate_intercity_through_veh_hr(
        network,
        base_peak_veh_hr=max(0.0, float(base_peak_veh_hr)),
        high_speed_ms=high_speed_ms,
        surrounding_exposure=surrounding_exposure,
    )
    if inflow_veh_hr is None:
        inflow_veh_hr = max(0.0, 0.35 * est_through + 0.06 * float(base_peak_veh_hr))
    if outflow_veh_hr is None:
        outflow_veh_hr = max(0.0, 0.35 * est_through + 0.06 * float(base_peak_veh_hr))
    if inflow_veh_hr <= 0.0 and outflow_veh_hr <= 0.0:
        return demand

    rng = np.random.default_rng(seed + 1911)
    fast_tags = {"motorway", "motorway_link", "trunk", "trunk_link"}

    incoming: dict[str, list[str]] = {}
    for eid, edge in network.edges.items():
        incoming.setdefault(edge.to_node, []).append(eid)

    eligible: list[str] = []
    gateway_weight: dict[str, float] = {}
    for nid in network.nodes:
        eids = list(network._adj.get(nid, [])) + incoming.get(nid, [])
        if not eids:
            continue
        max_spd = max(float(network.edges[e].speed_limit) for e in eids)
        has_fast_tag = any(str(network.edges[e].road_type) in fast_tags for e in eids)
        if max_spd >= high_speed_ms or has_fast_tag:
            eligible.append(nid)
            w = 0.0
            for eid in network._adj.get(nid, []):
                e = network.edges[eid]
                if (str(e.road_type) in fast_tags) or (float(e.speed_limit) >= high_speed_ms):
                    w += _edge_capacity_veh_hr(e, high_speed_ms=high_speed_ms)
            for eid in incoming.get(nid, []):
                e = network.edges[eid]
                if (str(e.road_type) in fast_tags) or (float(e.speed_limit) >= high_speed_ms):
                    w += 0.5 * _edge_capacity_veh_hr(e, high_speed_ms=high_speed_ms)
            gateway_weight[nid] = max(1.0, w)

    if len(eligible) < 2:
        return demand
    groups = _boundary_groups(network, eligible)
    boundary = list(dict.fromkeys(groups["west"] + groups["east"] + groups["south"] + groups["north"]))
    if not boundary:
        return demand

    side_w = _directional_side_weights(surrounding_exposure)
    node_side: dict[str, str] = {}
    for side in ("west", "east", "south", "north"):
        for nid in groups.get(side, []):
            node_side.setdefault(nid, side)

    boundary_set = set(boundary)
    interior = [n for n in network.nodes if n not in boundary_set and network._adj.get(n)]
    if not interior:
        return demand

    bnd_raw = np.array(
        [gateway_weight.get(n, 1.0) * side_w.get(node_side.get(n, "west"), 1.0) for n in boundary],
        dtype=float,
    )
    int_raw = np.array([max(1e-6, float(node_weights.get(n, 1.0))) for n in interior], dtype=float)
    bnd_probs = _safe_probs(bnd_raw)
    int_probs = _safe_probs(int_raw)

    route_ok: dict[tuple[str, str], bool] = {}

    def _routable(o: str, d: str) -> bool:
        if o == d:
            return False
        key = (o, d)
        if key in route_ok:
            return route_ok[key]
        route_ok[key] = bool(network.shortest_path(o, d))
        return route_ok[key]

    def _sample_counts(
        origins: list[str],
        o_probs: np.ndarray,
        dests: list[str],
        d_probs: np.ndarray,
        budget: int,
    ) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        if budget <= 0:
            return out
        attempts = 0
        max_attempts = budget * 20
        while sum(out.values()) < budget and attempts < max_attempts:
            attempts += 1
            o = origins[int(rng.choice(len(origins), p=o_probs))]
            d = dests[int(rng.choice(len(dests), p=d_probs))]
            if not _routable(o, d):
                continue
            out[(o, d)] = out.get((o, d), 0) + 1
        return out

    budget_in = max(0, int(max_pairs // 2))
    budget_out = max(0, int(max_pairs - budget_in))

    in_counts = _sample_counts(boundary, bnd_probs, interior, int_probs, budget_in)
    out_counts = _sample_counts(interior, int_probs, boundary, bnd_probs, budget_out)

    def _apply_flow(counts: dict[tuple[str, str], int], total_flow: float) -> None:
        if not counts or total_flow <= 0.0:
            return
        tot = sum(counts.values())
        per = float(total_flow) / max(1, tot)
        for (o, d), c in counts.items():
            f = round(c * per, 2)
            if f <= 0.0:
                continue
            demand.setdefault(o, {})
            demand[o][d] = round(demand[o].get(d, 0.0) + f, 2)

    _apply_flow(in_counts, float(inflow_veh_hr))
    _apply_flow(out_counts, float(outflow_veh_hr))
    return demand


def generate_spatial_demand(
    network: "Network",
    bbox: tuple[float, float, float, float],
    pois: list[dict] | None,
    buildings: list[dict] | None,
    *,
    seed: int = 42,
    peak_veh_hr: float = 400.0,
    max_pairs: int = 60,
    purpose_split: dict[str, float] | None = None,
    worldpop_raster_path: str | None = None,
    worldpop_cache_dir: str = ".overpass_cache",
    worldpop_city_slug: str | None = None,
    beta: float = 0.9,
    gamma_km: float = 0.12,
    min_unique_origins: int | None = None,
    intercity_share: float | None = None,
    intercity_max_pairs: int = 120,
    intercity_high_speed_ms: float = 19.0,
    intercity_ring_km: float = 60.0,
    intercity_exchange_pairs: int = 120,
    parallel_workers: int | None = None,
    max_candidate_eval: int = 2000,
) -> dict[str, dict[str, float]]:
    """Create OD matrix using WorldPop origin weights + purpose destinations."""
    rng = np.random.default_rng(seed)

    node_weights = load_worldpop_weights(
        bbox=bbox,
        network=network,
        cache_dir=worldpop_cache_dir,
        raster_path=worldpop_raster_path,
        city_slug=worldpop_city_slug,
    )
    surrounding_exposure = load_worldpop_surrounding_exposure(
        bbox=bbox,
        raster_path=worldpop_raster_path,
        cache_dir=worldpop_cache_dir,
        city_slug=worldpop_city_slug,
        ring_km=float(intercity_ring_km),
    )
    nodes = [n for n in network.nodes.keys() if network._adj.get(n)]
    if len(nodes) < 2:
        nodes = list(network.nodes.keys())
    if len(nodes) < 2:
        return {}
    all_dest_nodes = list(network.nodes.keys())

    origin_w = _safe_probs(np.array([node_weights.get(n, 0.0) for n in nodes], dtype=float))

    purpose_nodes, purpose_node_weights = build_purpose_nodes(network, pois, buildings)

    split = dict(DEFAULT_PURPOSE_SPLIT)
    split.update(purpose_split or {})
    split_keys = ["work", "school", "retail", "other"]
    split_probs = _safe_probs(np.array([split.get(k, 0.0) for k in split_keys], dtype=float))

    demand: dict[str, dict[str, float]] = {}
    pair_counts: dict[tuple[str, str], int] = {}

    # Fast vectorised destination scoring context.
    node_idx = {nid: i for i, nid in enumerate(nodes)}
    node_x = np.array([network.nodes[n].x for n in nodes], dtype=float)
    node_y = np.array([network.nodes[n].y for n in nodes], dtype=float)
    node_mass = np.array([max(1e-6, float(node_weights.get(n, 1.0))) for n in nodes], dtype=float)

    purpose_pool: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    max_eval = max(200, int(max_candidate_eval))
    for p in split_keys:
        raw = [n for n in (purpose_nodes.get(p, nodes) or nodes) if n in node_idx]
        if not raw:
            raw = list(nodes)
        cap = np.array(
            [max(1e-6, float(purpose_node_weights.get(p, {}).get(n, 1.0))) for n in raw],
            dtype=float,
        )
        if len(raw) > max_eval:
            pick = rng.choice(len(raw), size=max_eval, replace=False, p=_safe_probs(cap))
            raw = [raw[int(i)] for i in pick]
            cap = cap[pick]
        idx_arr = np.array([node_idx[n] for n in raw], dtype=np.int32)
        purpose_pool[p] = (raw, idx_arr, cap)

    route_ok: dict[tuple[str, str], bool] = {}
    route_lock = threading.Lock()
    _MISSING = object()

    def _routable(o: str, d: str) -> bool:
        if o == d:
            return False
        key = (o, d)
        with route_lock:
            cached = route_ok.get(key, _MISSING)
        if cached is not _MISSING:
            return bool(cached)
        ok = bool(network.shortest_path(o, d))
        with route_lock:
            route_ok[key] = ok
        return ok

    def _sample_destination_fast(
        origin_id: str,
        purpose: str,
        local_rng: np.random.Generator,
    ) -> str:
        candidates, cand_idx, cap = purpose_pool.get(purpose, purpose_pool[split_keys[0]])
        if not candidates:
            return origin_id
        oi = node_idx.get(origin_id, -1)
        if oi < 0:
            return candidates[int(local_rng.integers(0, len(candidates)))]

        d_km = np.hypot(node_x[cand_idx] - node_x[oi], node_y[cand_idx] - node_y[oi]) / 1000.0
        score = np.power(cap, beta) * np.exp(-gamma_km * d_km) * (node_mass[oi] ** 0.05)
        score = np.asarray(score, dtype=float)
        score[cand_idx == oi] = 0.0
        probs = _safe_probs(score)
        idx = int(local_rng.choice(len(candidates), p=probs))
        return candidates[idx]

    n_pairs = int(max(1, max_pairs))
    if min_unique_origins is None:
        min_unique_origins = max(20, n_pairs // 2)
    min_unique_origins = min(max(1, int(min_unique_origins)), n_pairs)

    # Coverage pass: guarantee a broad set of origins for visible spawn points.
    selected_origins = _weighted_sample_no_replace(
        nodes,
        origin_w,
        min_unique_origins,
        rng,
    )
    tasks: list[tuple[str, str, str, int]] = []
    for origin in selected_origins:
        purpose = split_keys[int(rng.choice(len(split_keys), p=split_probs))]
        tasks.append(("coverage", origin, purpose, int(rng.integers(0, 2**63 - 1))))

    for _ in range(max(0, n_pairs - len(tasks))):
        o_idx = int(rng.choice(len(nodes), p=origin_w))
        origin = nodes[o_idx]
        purpose = split_keys[int(rng.choice(len(split_keys), p=split_probs))]
        tasks.append(("regular", origin, purpose, int(rng.integers(0, 2**63 - 1))))

    def _propose(task: tuple[str, str, str, int]) -> tuple[str, str] | None:
        mode, origin, purpose, seed_i = task
        local_rng = np.random.default_rng(seed_i)
        tries = 8 if mode == "coverage" else 6
        for _ in range(tries):
            dest = _sample_destination_fast(origin, purpose, local_rng)
            if _routable(origin, dest):
                return (origin, dest)
        if mode == "coverage":
            for dest in all_dest_nodes:
                if _routable(origin, dest):
                    return (origin, dest)
        return None

    if parallel_workers is None:
        parallel_workers = max(1, min(8, os.cpu_count() or 2))
    if parallel_workers <= 1 or len(tasks) < 32:
        results = [_propose(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=int(parallel_workers), thread_name_prefix="m3-od") as pool:
            results = list(pool.map(_propose, tasks))

    for pair in results:
        if not pair:
            continue
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Deterministically enforce broad origin coverage even when random proposals
    # for some origins fail routability tests in directed/sparse graphs.
    covered_origins = {o for (o, _d) in pair_counts}
    for origin in selected_origins:
        if origin in covered_origins:
            continue
        for dest in all_dest_nodes:
            if _routable(origin, dest):
                pair_counts[(origin, dest)] = pair_counts.get((origin, dest), 0) + 1
                covered_origins.add(origin)
                break

    # Ensure a minimum breadth of unique OD pairs for calibration stability.
    target_unique_pairs = min(
        n_pairs,
        max(8, min_unique_origins * 2),
    )
    if len(pair_counts) < target_unique_pairs:
        fill_origins = list(dict.fromkeys(selected_origins + nodes))
        for o in fill_origins:
            if len(pair_counts) >= target_unique_pairs:
                break
            for d in all_dest_nodes:
                if len(pair_counts) >= target_unique_pairs:
                    break
                if o == d or (o, d) in pair_counts:
                    continue
                if _routable(o, d):
                    pair_counts[(o, d)] = 1

    if not pair_counts:
        # Deterministic fallback on sparse graphs.
        for i, origin in enumerate(nodes[:-1]):
            pair_counts[(origin, nodes[i + 1])] = 1

    total = sum(pair_counts.values())
    per_unit = float(peak_veh_hr) / max(1, total)

    for (origin, dest), cnt in pair_counts.items():
        flow = round(cnt * per_unit, 2)
        if flow <= 0.0:
            continue
        demand.setdefault(origin, {})[dest] = flow

    through_veh_hr = None
    if intercity_share is not None:
        through_veh_hr = max(0.0, float(peak_veh_hr) * float(intercity_share))
    demand = add_intercity_exchange_demand(
        network,
        demand,
        node_weights=node_weights,
        seed=seed,
        base_peak_veh_hr=float(peak_veh_hr),
        max_pairs=int(intercity_exchange_pairs),
        high_speed_ms=float(intercity_high_speed_ms),
        surrounding_exposure=surrounding_exposure,
    )
    demand = add_intercity_through_demand(
        network,
        demand,
        seed=seed,
        through_veh_hr=through_veh_hr,
        base_peak_veh_hr=float(peak_veh_hr),
        max_pairs=int(intercity_max_pairs),
        high_speed_ms=float(intercity_high_speed_ms),
        surrounding_exposure=surrounding_exposure,
    )

    return demand
