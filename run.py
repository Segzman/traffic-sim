"""CLI entry point — accepts scenario JSON path, duration, seed; prints metric summary.

Live-serving mode
-----------------
  python run.py --serve

Starts an HTTP server (default port 8765) that exposes:
  GET  /network        — road network JSON (re-fetched after each import)
  GET  /state          — live simulation snapshot (HTTP fallback)
  GET  /ws             — WebSocket endpoint; server pushes state after each batch
  GET  /import_status  — progress of an ongoing /import operation
  POST /control        — update paused / speed_mult / demand_mult / params{}
  POST /import         — geocode + fetch OSM + hot-swap simulation
"""
import argparse
import base64
import hashlib
import json
import math
import sys
import threading
import time as _time_module
import socketserver
from http.server import BaseHTTPRequestHandler

from engine.simulation import Simulation

# --------------------------------------------------------------------------- #
# Simulation duration / warmup used for freshly imported locations
# --------------------------------------------------------------------------- #
_IMPORT_DURATION_S = 7200.0   # 2-hour commute window
_IMPORT_WARMUP_S   =  120.0   # 2-minute warmup


def _blank_scenario() -> dict:
    """Return an empty network scenario for blank-map startup."""
    return {
        "_comment": "Blank map",
        "network": {
            "nodes": [],
            "edges": [],
            "signal_nodes": [],
            "signal_cycle_s": 90.0,
        },
        "demand": {},
        "duration": 24 * 3600.0,
        "warmup": 0.0,
        "seed": 42,
        "start_hour": 0.0,
        "temporal_demand": True,
        "continuous_demand": True,
        "auto_m3_demand": False,
    }


def _run_scenario(scenario: dict) -> dict:
    sim = Simulation(scenario)
    return sim.run()


def _print_metrics(metrics: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*40}")
        print(f"  {label}")
        print(f"{'='*40}")
    print(f"Throughput    : {metrics['throughput']:.4f} veh/s")
    print(f"Avg speed     : {metrics['avg_speed']:.2f} m/s")
    print(f"Density       : {metrics['density']:.6f} veh/m")
    if "queue_length" in metrics:
        print(f"Queue length  : {metrics['queue_length']:.2f} veh")
    if "avg_delay" in metrics:
        print(f"Avg delay     : {metrics['avg_delay']:.2f} s")
    lane_util = metrics.get("lane_utilisation", {})
    if lane_util:
        print("\nLane utilisation:")
        for lid, frac in sorted(lane_util.items()):
            print(f"  Lane {lid}: {frac * 100:.1f}%")


def _compare(scenario_path: str, runs: int) -> None:
    """Run two designs (roundabout vs signal) and print side-by-side table."""
    with open(scenario_path) as fh:
        cfg = json.load(fh)

    designs = {}
    for key in ("roundabout", "signal"):
        if key not in cfg:
            continue
        delays = []
        throughputs = []
        for run_idx in range(runs):
            sc = dict(cfg[key])
            sc["seed"] = run_idx
            m = _run_scenario(sc)
            delays.append(m.get("avg_delay", 0.0))
            throughputs.append(m["throughput"])
        designs[key] = {
            "mean_delay": sum(delays) / len(delays) if delays else 0.0,
            "mean_throughput": sum(throughputs) / len(throughputs),
            "runs": runs,
        }

    print(f"\n{'Design':<16} {'Mean delay (s)':>16} {'Mean throughput (veh/s)':>24} {'Runs':>6}")
    print("-" * 66)
    for name, d in designs.items():
        print(f"{name:<16} {d['mean_delay']:>16.2f} {d['mean_throughput']:>24.4f} {d['runs']:>6}")


def _import_bbox_cmd(bbox_str: str, output_path: str | None) -> None:
    """Fetch OSM data for a bounding box, run importer, save/print scenario."""
    from importer import import_bbox as _import_bbox
    try:
        parts = [float(v.strip()) for v in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("need 4 values")
        south, west, north, east = parts
    except ValueError as exc:
        print(f"Error: --import-bbox expects 'south,west,north,east' ({exc})", file=sys.stderr)
        sys.exit(1)

    print(f"Importing OSM data for bbox {south},{west},{north},{east} …", file=sys.stderr)
    scenario = _import_bbox(south, west, north, east)
    net = scenario.get("network", {})
    n_edges = len(net.get("edges", []))
    n_nodes = len(net.get("nodes", {}))
    print(f"  Imported {n_nodes} nodes, {n_edges} edges.", file=sys.stderr)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(scenario, fh, indent=2)
        print(f"  Saved to {output_path}", file=sys.stderr)
    else:
        print(json.dumps(scenario, indent=2))


def _scenario_from_import(scenario: dict) -> dict:
    """Convert an ``import_bbox()`` scenario to ``_build_network_sim()`` format.

    ``import_bbox`` returns nodes as a ``dict {str(nid): {id, lat, lon, x, y}}``
    and edges with ``int`` node references and a ``via_nodes`` list.

    ``_build_network_sim`` expects nodes as a ``list`` with string IDs and
    edges with geometry as ``[[x, y], ...]`` waypoint lists.
    """
    import copy
    scenario = copy.deepcopy(scenario)
    net_cfg = scenario.setdefault("network", {})

    nodes_raw = net_cfg.get("nodes", {})

    # ---- Convert nodes dict → list ----------------------------------------
    if isinstance(nodes_raw, dict):
        nodes_by_id: dict[str, dict] = {}
        nodes_list: list[dict] = []
        for nid, nd in nodes_raw.items():
            sid = str(nid)
            nodes_by_id[sid] = nd
            nodes_list.append({
                "id": sid,
                "x": float(nd.get("x", 0.0)),
                "y": float(nd.get("y", 0.0)),
                "junction_type": nd.get("junction_type", "uncontrolled"),
            })
        net_cfg["nodes"] = nodes_list
    else:
        # Already a list — build lookup only
        nodes_by_id = {n["id"]: n for n in nodes_raw}

    # ---- Fix edges: str IDs, geometry from via_nodes ----------------------
    new_edges: list[dict] = []
    for e in net_cfg.get("edges", []):
        fn = str(e.get("from_node", ""))
        tn = str(e.get("to_node",   ""))
        ne: dict = {
            "id":          str(e.get("id", "")),
            "from_node":   fn,
            "to_node":     tn,
            "num_lanes":   int(e.get("num_lanes",  1)),
            "speed_limit": float(e.get("speed_limit", 8.3)),
            "road_type":   str(e.get("road_type", "primary")),
        }

        # Build geometry list [[x,y], ...] from via_nodes if not present
        if e.get("geometry"):
            ne["geometry"] = e["geometry"]
        else:
            via_nodes = e.get("via_nodes", [])
            all_nids = [fn] + [str(v) for v in via_nodes] + [tn]
            pts: list[list[float]] = []
            for nid in all_nids:
                nd = nodes_by_id.get(nid)
                if nd:
                    pts.append([float(nd.get("x", 0.0)), float(nd.get("y", 0.0))])
            ne["geometry"] = pts

        new_edges.append(ne)

    net_cfg["edges"] = new_edges

    # ---- Compute node degree + max adjacent speed -------------------------
    in_deg: dict[str, int]    = {}
    out_deg: dict[str, int]   = {}
    max_spd: dict[str, float] = {}
    for e in new_edges:
        fn, tn = e["from_node"], e["to_node"]
        out_deg[fn] = out_deg.get(fn, 0) + 1
        in_deg[tn]  = in_deg.get(tn,  0) + 1
        sp = float(e.get("speed_limit", 0.0))
        max_spd[fn] = max(max_spd.get(fn, 0.0), sp)
        max_spd[tn] = max(max_spd.get(tn, 0.0), sp)

    # ---- Detect signal nodes: prefer OSM highway=traffic_signals tags ----
    # Sort OSM-tagged signal nodes by total degree (busiest first) so the
    # most important intersections are always included in the 300-node cap.
    osm_signals = sorted(
        [
            (in_deg.get(nid, 0) + out_deg.get(nid, 0), nid)
            for nid, nd in nodes_by_id.items()
            if nd.get("tags", {}).get("highway") == "traffic_signals"
            and nid in in_deg   # must be reachable in our graph
        ],
        reverse=True,
    )

    if osm_signals:
        # Real OSM data: use ALL signalised nodes (no cap)
        signal_nodes = [nid for _, nid in osm_signals]
        print(
            f"[import] {len(osm_signals)} OSM signal nodes → using all "
            f"{len(signal_nodes)}",
            file=sys.stderr,
        )
    else:
        # Fallback heuristic: in-degree ≥ 2, out-degree ≥ 2, speed ≥ 9.5 m/s
        candidates = sorted(
            [
                (in_deg.get(nid, 0) + out_deg.get(nid, 0), nid)
                for nid in nodes_by_id
                if in_deg.get(nid, 0) >= 2
                and out_deg.get(nid, 0) >= 2
                and max_spd.get(nid, 0.0) >= 9.5
            ],
            reverse=True,
        )
        signal_nodes = [nid for _, nid in candidates[:300]]
        print(
            f"[import] No OSM signal tags found — heuristic: {len(signal_nodes)} nodes",
            file=sys.stderr,
        )

    net_cfg["signal_nodes"] = signal_nodes

    return scenario


def _build_network_sim(scenario: dict):
    """Build a NetworkSimulation from a scenario dict.

    Handles both network-style scenarios (``scenario["network"]``) and
    simple road-style scenarios (``scenario["road"]``), wrapping the latter
    in a minimal two-node network.
    """
    from engine.network import Network, Node, Edge
    from engine.network_simulation import NetworkSimulation
    from engine.signals import SignalPlan, Phase
    from importer.projection import mercator_to_latlng
    from engine.worldpop import slugify_city
    import os as _os

    net = Network()

    if "network" in scenario:
        net_cfg = scenario["network"]

        for n in net_cfg.get("nodes", []):
            net.add_node(Node(
                id=n["id"],
                x=float(n["x"]),
                y=float(n["y"]),
                junction_type=n.get("junction_type", "uncontrolled"),
            ))

        for e in net_cfg.get("edges", []):
            from_node = e.get("from_node") or e.get("from", "")
            to_node   = e.get("to_node")   or e.get("to",   "")
            num_lanes  = int(e.get("num_lanes") or e.get("lanes", 1))
            speed_lim  = float(e.get("speed_limit") or e.get("speed_limit_ms", 13.9))
            geom = [list(pt) for pt in e.get("geometry", [])]
            net.add_edge(Edge(
                id=e["id"],
                from_node=from_node,
                to_node=to_node,
                num_lanes=num_lanes,
                speed_limit=speed_lim,
                geometry=geom,
                road_type=str(e.get("road_type", "primary")),
            ))

        # Build signal plans for signal nodes.
        signal_nodes = net_cfg.get("signal_nodes", [])
        cycle_s   = float(net_cfg.get("signal_cycle_s", 90.0))
        yellow_s  = 4.0
        all_red_s = 2.0
        overhead  = yellow_s + all_red_s
        g_dur     = max(5.0, cycle_s / 2.0 - overhead)
        signal_plans: dict = {}
        for i, nid in enumerate(signal_nodes):
            offset = (i * cycle_s / max(1, len(signal_nodes))) % cycle_s
            plan = SignalPlan(
                node_id=nid,
                phases=[
                    Phase(green_movements=[], green_duration=g_dur,
                          yellow_duration=yellow_s, all_red_duration=all_red_s),
                    Phase(green_movements=[], green_duration=g_dur,
                          yellow_duration=yellow_s, all_red_duration=all_red_s),
                ],
                offset=offset,
            )
            signal_plans[nid] = plan

    else:
        road = scenario.get("road", {})
        length    = float(road.get("length", 500.0))
        speed_lim = float(road.get("speed_limit", 13.9))
        num_lanes = int(road.get("num_lanes", 1))
        net.add_node(Node(id="A", x=0.0,    y=0.0))
        net.add_node(Node(id="B", x=length, y=0.0))
        net.add_edge(Edge(
            id="AB",
            from_node="A",
            to_node="B",
            num_lanes=num_lanes,
            speed_limit=speed_lim,
            geometry=[[0.0, 0.0], [length, 0.0]],
            road_type=str(road.get("road_type", "primary")),
        ))
        signal_plans = {}

    demand  = scenario.get("demand", {})
    dur     = float(scenario.get("duration", 300.0))
    seed    = int(scenario.get("seed", 42))
    warmup  = float(scenario.get("warmup", 0.0))
    continuous = bool(scenario.get("continuous_demand", False))
    start_hour = float(scenario.get("start_hour", 0.0))
    temporal_demand = bool(scenario.get("temporal_demand", False))
    day_type = scenario.get("day_type")  # "weekday" | "weekend" | None
    auto_m3 = bool(scenario.get("auto_m3_demand", False))
    worldpop_city_slug = scenario.get("worldpop_city_slug")
    if not worldpop_city_slug:
        worldpop_city_slug = slugify_city(str(scenario.get("_comment", "unknown-city")))

    if auto_m3 and "network" in scenario:
        try:
            # Promote sparse legacy OD configs to broad M3 spatial demand.
            n_orig = len(demand)
            n_pairs = sum(len(v) for v in demand.values()) if isinstance(demand, dict) else 0
            if n_orig < 25 or n_pairs < 150:
                from engine.poi_demand import generate_spatial_demand
                from importer import overpass, parser, inference

                net_cfg = scenario["network"]
                bbox_cfg = net_cfg.get("bbox", {})
                if all(k in bbox_cfg for k in ("south", "west", "north", "east")):
                    bbox = (
                        float(bbox_cfg["south"]),
                        float(bbox_cfg["west"]),
                        float(bbox_cfg["north"]),
                        float(bbox_cfg["east"]),
                    )
                else:
                    lats: list[float] = []
                    lons: list[float] = []
                    for n in net_cfg.get("nodes", []):
                        if "lat" in n and "lon" in n:
                            lats.append(float(n["lat"]))
                            lons.append(float(n["lon"]))
                        else:
                            lat, lon = mercator_to_latlng(float(n["x"]), float(n["y"]))
                            lats.append(lat)
                            lons.append(lon)
                    if lats and lons:
                        pad = 0.002
                        bbox = (
                            min(lats) - pad,
                            min(lons) - pad,
                            max(lats) + pad,
                            max(lons) + pad,
                        )
                    else:
                        bbox = None

                pois = net_cfg.get("pois", [])
                buildings = net_cfg.get("buildings", [])
                if bbox is not None and (not pois or not buildings):
                    try:
                        osm = overpass.fetch(bbox)
                        parsed = parser.parse_osm(osm)
                        enriched = inference.infer(parsed)
                        pois = pois or enriched.get("pois", [])
                        buildings = buildings or enriched.get("buildings", [])
                    except Exception as exc:
                        print(f"[m3] context fetch failed: {exc}", file=sys.stderr)

                if bbox is not None:
                    existing_total = 0.0
                    if isinstance(demand, dict):
                        existing_total = sum(sum(v.values()) for v in demand.values())
                    peak_veh_hr = float(scenario.get("peak_veh_hr", max(1200.0, existing_total)))
                    max_pairs = int(scenario.get("max_pairs", min(2500, max(400, len(net.nodes) // 4))))
                    min_unique = int(scenario.get("min_unique_origins", max(40, max_pairs // 4)))
                    intercity_share = scenario.get("intercity_share")
                    worldpop_raster = net_cfg.get("worldpop_raster_path")

                    m3_demand = generate_spatial_demand(
                        network=net,
                        bbox=bbox,
                        pois=pois,
                        buildings=buildings,
                        seed=seed,
                        peak_veh_hr=peak_veh_hr,
                        max_pairs=max_pairs,
                        worldpop_raster_path=worldpop_raster,
                        worldpop_city_slug=worldpop_city_slug,
                        min_unique_origins=min_unique,
                        intercity_share=intercity_share,
                        intercity_ring_km=float(scenario.get("intercity_ring_km", 80.0)),
                        parallel_workers=(
                            int(scenario.get("m3_parallel_workers", 0))
                            or int(_os.getenv("M3_PARALLEL_WORKERS", "0"))
                            or None
                        ),
                    )
                    if m3_demand:
                        m3_orig = len(m3_demand)
                        m3_pairs = sum(len(v) for v in m3_demand.values())
                        print(
                            f"[m3] demand upgrade: origins {n_orig}->{m3_orig}, pairs {n_pairs}->{m3_pairs}",
                            file=sys.stderr,
                        )
                        demand = m3_demand
        except Exception as exc:
            print(f"[m3] auto demand disabled by error: {exc}", file=sys.stderr)

    return NetworkSimulation(
        network=net,
        demand=demand,
        duration=dur,
        seed=seed,
        warmup=warmup,
        signal_plans=signal_plans,
        continuous_demand=continuous,
        start_hour=start_hour,
        temporal_demand=temporal_demand,
        day_type=day_type,
    )


# --------------------------------------------------------------------------- #
# Threaded TCP server (needed so /state keeps polling during long imports)
# --------------------------------------------------------------------------- #
class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True   # don't block shutdown on active handler threads


def _serve(scenario: dict, port: int = 8765, location: str = "Traffic Sim") -> None:
    """Run the simulation continuously and serve state over HTTP.

    Endpoints
    ---------
    GET  /network        Road network JSON (nodes + edges).
    GET  /state          Live simulation snapshot (polled at ~10 Hz).
    GET  /import_status  Progress of an ongoing /import operation.
    POST /control        Update paused / speed_mult / demand_mult.
    POST /import         Geocode + fetch OSM + hot-swap simulation.
    """
    import os as _os
    from engine.worldpop import slugify_city

    # ---- Initial simulation build ----------------------------------------
    scenario_live = dict(scenario)
    scenario_live["continuous_demand"] = True
    scenario_live.setdefault("temporal_demand", True)
    scenario_live.setdefault("auto_m3_demand", True)
    scenario_live.setdefault("worldpop_city_slug", slugify_city(location))
    sim = _build_network_sim(scenario_live)
    net = sim.network

    def _mk_net_payload(n):
        nodes_out = {
            nid: {"x": nd.x, "y": nd.y, "junction_type": nd.junction_type}
            for nid, nd in n.nodes.items()
        }
        edges_out = {
            eid: {
                "from_node": e.from_node,
                "to_node":   e.to_node,
                "num_lanes": e.num_lanes,
                "speed_limit": e.speed_limit,
                "geometry":  e.geometry,
            }
            for eid, e in n.edges.items()
        }
        return json.dumps({"nodes": nodes_out, "edges": edges_out}).encode()

    # ---- Mutable world state (hot-swappable) ------------------------------
    _world: dict = {
        "sim":            sim,
        "net":            net,
        "net_payload":    _mk_net_payload(net),
        "signal_node_ids": list(sim._signal_plans.keys()),
        "location":       location,
    }

    # ---- Thread-safety locks ---------------------------------------------
    _lock     = threading.Lock()   # guards _world["sim"] / _world["net"] etc.
    _ctrl_lock = threading.Lock()  # guards _ctrl
    _imp_lock  = threading.Lock()  # guards _import_status

    # ---- Runtime controls -----------------------------------------------
    _ctrl: dict = {"paused": False, "speed_mult": 1.0, "demand_mult": 1.0}

    # ---- Import status --------------------------------------------------
    _import_status: dict = {"stage": "idle", "progress": 0.0, "message": ""}
    _import_job: dict = {
        "id": 0,
        "thread": None,
        "started_mono": 0.0,
    }
    _IMPORT_STALL_S = 120.0

    # Sim clock start-of-day (default 00:00 unless scenario overrides start_hour).
    _start_hour = int(float(scenario.get("start_hour", 0.0))) % 24
    _START_S = _start_hour * 3600

    # ---- WebSocket client registry --------------------------------------
    _ws_clients: set = set()
    _ws_lock_ws       = threading.Lock()
    _WS_MAGIC         = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def _ws_encode(payload: bytes) -> bytes:
        """Wrap payload in a WebSocket text frame (server→client, unmasked)."""
        n = len(payload)
        if n <= 125:
            return bytes([0x81, n]) + payload
        elif n <= 65535:
            return bytes([0x81, 0x7E]) + n.to_bytes(2, "big") + payload
        else:
            return bytes([0x81, 0x7F]) + n.to_bytes(8, "big") + payload

    def _ws_broadcast(payload: bytes) -> None:
        if not _ws_clients:
            return
        frame = _ws_encode(payload)
        dead: set = set()
        with _ws_lock_ws:
            for sock in list(_ws_clients):
                try:
                    sock.sendall(frame)
                except OSError:
                    dead.add(sock)
            _ws_clients.difference_update(dead)

    # ---- Simulation loop (daemon thread) ---------------------------------
    _TARGET_BATCH_S = 0.05   # target ~20 batches/s → smooth UI at any speed

    def _sim_loop() -> None:
        while True:
            with _ctrl_lock:
                paused     = _ctrl["paused"]
                speed_mult = _ctrl["speed_mult"]
            if paused:
                _time_module.sleep(0.05)
                continue
            t0 = _time_module.monotonic()
            with _lock:
                s  = _world["sim"]
                dt = s.dt
                # Run enough steps to cover ~50 ms of real time at current speed.
                # At 1× (dt=0.1): batch=1; at 288× (dt=0.5): batch=29.
                batch = max(1, round(speed_mult * _TARGET_BATCH_S / dt))
                for _ in range(batch):
                    s.step()
            elapsed = _time_module.monotonic() - t0
            _time_module.sleep(max(0.0, _TARGET_BATCH_S - elapsed))
            # Push fresh state to all connected WebSocket clients
            _ws_broadcast(_state_payload())

    sim_thread = threading.Thread(target=_sim_loop, daemon=True, name="sim-loop")
    sim_thread.start()

    # ---- Background import function -------------------------------------
    def _do_import(query: str, job_id: int) -> None:
        def _active() -> bool:
            with _imp_lock:
                return int(_import_job.get("id", -1)) == int(job_id)

        def _upd(**kw):
            with _imp_lock:
                if int(_import_job.get("id", -1)) != int(job_id):
                    return
                _import_status.update(**kw)

        try:
            if not _active():
                return
            # 1. Geocode
            _upd(stage="geocoding", progress=0.05, message=f'Geocoding "{query}"…')
            from importer.geocode import geocode
            loc = geocode(query)
            if not _active():
                return
            short_name = loc["display_name"].split(",")[0].strip()
            from engine.worldpop import slugify_city
            city_slug = slugify_city(short_name)

            # 2. Fetch OSM
            _upd(stage="fetching_osm", progress=0.15,
                 message=f"Fetching OSM data for {short_name}…")
            from importer import import_bbox
            raw_scenario = import_bbox(*loc["bbox"])
            if not _active():
                return

            # 3. Convert importer format
            _upd(stage="building_network", progress=0.50,
                 message="Building road network…")
            converted = _scenario_from_import(raw_scenario)
            if not _active():
                return
            n_nodes = len(converted["network"]["nodes"])
            n_edges = len(converted["network"]["edges"])
            print(
                f"[import] {short_name}: {n_nodes} nodes, {n_edges} edges",
                file=sys.stderr,
            )

            # 4. Build temporary network for M3 OD generation.
            _upd(stage="generating_demand", progress=0.65,
                 message="Generating M3 spatial demand (WorldPop + POI + buildings)…")
            from engine.network import Network, Node, Edge
            from engine.commute import generate_commute_demand
            from engine.poi_demand import generate_spatial_demand
            _temp_net = Network()
            for nd in converted["network"]["nodes"]:
                _temp_net.add_node(Node(
                    id=nd["id"], x=float(nd["x"]), y=float(nd["y"])
                ))
            for ed in converted["network"]["edges"]:
                _temp_net.add_edge(Edge(
                    id=ed["id"],
                    from_node=ed["from_node"],
                    to_node=ed["to_node"],
                    num_lanes=int(ed.get("num_lanes", 1)),
                    speed_limit=float(ed.get("speed_limit", 8.3)),
                    geometry=ed.get("geometry", []),
                    road_type=str(ed.get("road_type", "primary")),
                ))
            # Milestone 3 demand: WorldPop-weighted origins + POI/building
            # purpose destinations.  Falls back to legacy commute demand.
            m3_bbox = converted["network"].get("bbox", {})
            m3_pois = converted["network"].get("pois", [])
            m3_buildings = converted["network"].get("buildings", [])
            m3_worldpop_raster = (
                converted["network"].get("worldpop_raster_path")
                or _os.getenv("WORLDPOP_RASTER_PATH")
            )
            bbox_tuple = (
                float(m3_bbox.get("south", loc["bbox"][0])),
                float(m3_bbox.get("west", loc["bbox"][1])),
                float(m3_bbox.get("north", loc["bbox"][2])),
                float(m3_bbox.get("east", loc["bbox"][3])),
            )
            try:
                demand_peak_veh_hr = min(4500.0, max(1200.0, 1200.0 * math.sqrt(max(1.0, n_edges / 120.0))))
                demand_max_pairs = int(min(900, max(120, n_nodes // 5)))
                demand_min_unique = int(min(300, max(30, demand_max_pairs // 4)))
                demand = generate_spatial_demand(
                    network=_temp_net,
                    bbox=bbox_tuple,
                    pois=m3_pois,
                    buildings=m3_buildings,
                    seed=42,
                    peak_veh_hr=demand_peak_veh_hr,
                    max_pairs=demand_max_pairs,
                    worldpop_raster_path=m3_worldpop_raster,
                    worldpop_city_slug=city_slug,
                    min_unique_origins=demand_min_unique,
                    intercity_share=None,
                    intercity_max_pairs=max(80, demand_max_pairs // 4),
                    intercity_ring_km=80.0,
                    parallel_workers=(
                        int(_os.getenv("M3_PARALLEL_WORKERS", "0")) or None
                    ),
                )
            except Exception as exc:
                print(f"[import] M3 demand fallback: {exc}", file=sys.stderr)
                demand = generate_commute_demand(
                    _temp_net,
                    seed=42,
                    peak_veh_hr=400.0,
                    max_pairs=60,
                )
            converted["demand"]   = demand
            converted["duration"] = _IMPORT_DURATION_S
            converted["warmup"]   = _IMPORT_WARMUP_S
            converted["seed"]     = 42
            converted["continuous_demand"] = True
            # Keep imported runs on the same midnight temporal profile as blank startup
            # so we do not inject peak-hour flow at t=00:00.
            converted["temporal_demand"] = True
            converted["start_hour"] = 0.0
            converted["auto_m3_demand"] = False
            converted["worldpop_city_slug"] = city_slug
            print(
                f"[import] demand: {sum(len(v) for v in demand.values())} OD pairs",
                file=sys.stderr,
            )

            # 5. Build full simulation
            _upd(stage="spawning", progress=0.82,
                 message="Initialising simulation…")
            new_sim = _build_network_sim(converted)
            new_net = new_sim.network
            new_payload     = _mk_net_payload(new_net)
            new_signal_ids  = list(new_sim._signal_plans.keys())
            if not _active():
                return

            # 6. Atomic hot-swap
            _upd(stage="swapping", progress=0.95,
                 message="Hot-swapping simulation…")
            if not _active():
                return
            with _lock:
                _world["sim"]            = new_sim
                _world["net"]            = new_net
                _world["net_payload"]    = new_payload
                _world["signal_node_ids"] = new_signal_ids
                _world["location"]       = short_name

            _upd(stage="done", progress=1.0,
                 message=f"Ready — {short_name}", location=short_name)
            print(f"[import] Done: {short_name}", file=sys.stderr)

        except Exception as exc:
            _upd(stage="error", progress=0.0, message=str(exc))
            print(f"[import] Error: {exc}", file=sys.stderr)

    # ---- State serialiser -----------------------------------------------
    def _state_payload() -> bytes:
        with _lock:
            s   = _world["sim"]
            net = _world["net"]
            sig = _world["signal_node_ids"]
            loc = _world["location"]
            vehicles = [
                {
                    "id":         v.id,
                    "edge_id":    v.current_edge,
                    "position_s": round(v.position_s, 3),
                    "speed":      round(v.speed, 3),
                    "lane_id":    v.lane_id,
                    "speed_limit": (
                        net.edges[v.current_edge].speed_limit
                        if v.current_edge in net.edges else 13.9
                    ),
                    "length":     round(v.length, 1),
                    "width":      round(v.width, 2),
                    "type":       v.vehicle_type,
                }
                for v in s.vehicles
            ]
            pedestrians = [
                {"id": p.id, "x": round(p.x, 3), "y": round(p.y, 3)}
                for p in s.pedestrians
            ]
            signals = [
                {"node_id": nid, "state": s._signal_color(nid)}
                for nid in sig
            ]
            sim_time = s.time
        with _ctrl_lock:
            ctrl_snap = dict(_ctrl)

        # Wall-clock from scenario start hour (default midnight).
        wall_s  = _START_S + sim_time
        wall_h  = int(wall_s // 3600) % 24
        wall_m  = int(wall_s % 3600) // 60

        snapshot = {
            "time":        round(sim_time, 2),
            "clock":       f"{wall_h:02d}:{wall_m:02d}",
            "location":    loc,
            "vehicles":    vehicles,
            "pedestrians": pedestrians,
            "signals":     signals,
            "ctrl":        ctrl_snap,
        }
        return json.dumps(snapshot).encode()

    # ---- Static file map ------------------------------------------------
    _viz_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "viz")
    _static = {
        "/":            (_os.path.join(_viz_dir, "index.html"),  "text/html; charset=utf-8"),
        "/index.html":  (_os.path.join(_viz_dir, "index.html"),  "text/html; charset=utf-8"),
        "/renderer.js": (_os.path.join(_viz_dir, "renderer.js"), "application/javascript"),
    }

    # ---- HTTP handler ---------------------------------------------------
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[http] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]

            if path == "/network":
                with _lock:
                    body = _world["net_payload"]
                self._json(200, body)

            elif path == "/state":
                self._json(200, _state_payload())

            elif path == "/vehicle_classes":
                from engine.vehicle_classes import VEHICLE_CLASSES
                import dataclasses as _dc
                body = {k: _dc.asdict(vc) for k, vc in VEHICLE_CLASSES.items()}
                self._json(200, json.dumps(body).encode())

            elif path == "/ws":
                ws_key = self.headers.get("Sec-WebSocket-Key", "")
                if self.headers.get("Upgrade", "").lower() != "websocket" or not ws_key:
                    self._json(400, b'{"error":"websocket upgrade required"}')
                    return
                accept = base64.b64encode(
                    hashlib.sha1((ws_key + _WS_MAGIC).encode()).digest()
                ).decode()
                self.send_response(101)
                self.send_header("Upgrade",              "websocket")
                self.send_header("Connection",           "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                self.wfile.flush()
                sock = self.connection
                with _ws_lock_ws:
                    _ws_clients.add(sock)
                try:
                    while True:
                        hdr = sock.recv(2)
                        if len(hdr) < 2:
                            break
                        opcode  = hdr[0] & 0x0F
                        if opcode == 0x8:       # close frame
                            break
                        masked  = bool(hdr[1] & 0x80)
                        plen    = hdr[1] & 0x7F
                        if plen == 126:
                            plen = int.from_bytes(sock.recv(2), "big")
                        elif plen == 127:
                            plen = int.from_bytes(sock.recv(8), "big")
                        to_skip = (4 if masked else 0) + plen
                        if to_skip:
                            sock.recv(to_skip)
                except OSError:
                    pass
                finally:
                    with _ws_lock_ws:
                        _ws_clients.discard(sock)

            elif path == "/config":
                with _ctrl_lock:
                    self._json(200, json.dumps(_ctrl).encode())

            elif path == "/import_status":
                with _imp_lock:
                    body = json.dumps(_import_status).encode()
                self._json(200, body)

            elif path in _static:
                fpath, mime = _static[path]
                try:
                    with open(fpath, "rb") as fh:
                        body = fh.read()
                    self._raw(200, body, mime)
                except OSError:
                    self._json(404, b'{"error":"file not found"}')

            else:
                self._json(404, b'{"error":"not found"}')

        def do_POST(self):
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
            except Exception:
                self._json(400, b'{"error":"bad json"}')
                return

            if path == "/control":
                with _ctrl_lock:
                    if "paused" in body:
                        _ctrl["paused"] = bool(body["paused"])
                    if "speed_mult" in body:
                        mult = max(0.25, min(288.0, float(body["speed_mult"])))
                        _ctrl["speed_mult"] = mult
                        with _lock:
                            _world["sim"].set_speed_mult(mult)
                    if "demand_mult" in body:
                        mult = max(0.0, min(5.0, float(body["demand_mult"])))
                        _ctrl["demand_mult"] = mult
                        with _lock:
                            _world["sim"].set_demand_mult(mult)
                    # M2: forward SimConfig params dict
                    if "params" in body and isinstance(body["params"], dict):
                        with _lock:
                            cfg = getattr(_world["sim"], "config", None)
                            if cfg is not None:
                                cfg.update(**body["params"])
                            # speed_mult in params also updates ctrl + sim
                            if "speed_mult" in body["params"]:
                                sm = max(0.25, min(288.0, float(body["params"]["speed_mult"])))
                                _ctrl["speed_mult"] = sm
                                _world["sim"].set_speed_mult(sm)
                    resp = json.dumps(_ctrl).encode()
                self._json(200, resp)

            elif path == "/import":
                query = str(body.get("query", "")).strip()
                if not query:
                    self._json(400, b'{"error":"missing query"}')
                    return
                force = bool(body.get("force", False))
                # Reject if another import is running
                with _imp_lock:
                    stage = _import_status.get("stage", "idle")
                    active_stages = {
                        "geocoding", "fetching_osm", "building_network",
                        "generating_demand", "spawning", "swapping",
                    }
                    thr = _import_job.get("thread")
                    started = float(_import_job.get("started_mono", 0.0))
                    age = max(0.0, _time_module.monotonic() - started) if started else 0.0
                    running = bool(getattr(thr, "is_alive", lambda: False)())
                    if stage in active_stages and running and age < _IMPORT_STALL_S and not force:
                        msg = (
                            f'import already in progress '
                            f'({stage}, {int(age)}s)'
                        ).encode()
                        self._json(409, b'{"error":"' + msg + b'"}')
                        return
                    # Stale/dead import guard: supersede previous job.
                    if stage in active_stages and (force or not running or age >= _IMPORT_STALL_S):
                        _import_status.clear()
                        _import_status.update(
                            stage="idle",
                            progress=0.0,
                            message=("Previous import cancelled." if force
                                     else "Previous import cleared (stale)."),
                        )
                        _import_job["id"] = int(_import_job.get("id", 0)) + 1

                    new_job_id = int(_import_job.get("id", 0)) + 1
                    _import_job["id"] = new_job_id
                    _import_status.clear()
                    _import_status.update(
                        stage="geocoding", progress=0.0,
                        message=f'Starting import for "{query}"…',
                    )
                t = threading.Thread(
                    target=_do_import, args=(query, new_job_id),
                    daemon=True, name=f"import-{query[:20]}",
                )
                with _imp_lock:
                    _import_job["thread"] = t
                    _import_job["started_mono"] = _time_module.monotonic()
                t.start()
                resp = json.dumps({"status": "importing", "query": query}).encode()
                self._json(200, resp)

            else:
                self._json(404, b'{"error":"not found"}')

        def _json(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _raw(self, code: int, body: bytes, mime: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type",   mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control",  "no-cache")
            self.end_headers()
            self.wfile.write(body)

    # ---- Start server (blocks until Ctrl-C) -----------------------------
    print(
        f"[serve] HTTP server → http://localhost:{port}  "
        f"(scenario: {location}, duration={sim.duration:.0f} s)",
        file=sys.stderr,
    )
    print(f"[serve]   http://localhost:{port}/          — live canvas viewer", file=sys.stderr)
    print(f"[serve]   http://localhost:{port}/network   — road network JSON",  file=sys.stderr)
    print(f"[serve]   http://localhost:{port}/state     — live simulation",     file=sys.stderr)
    print(f"[serve]   http://localhost:{port}/import    — POST {{\"query\":\"...\"}}", file=sys.stderr)
    print("[serve]   Press Ctrl-C to stop.", file=sys.stderr)

    with _ThreadedServer(("localhost", port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] Stopped.", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Traffic Simulation Platform")
    parser.add_argument("scenario", nargs="?", help="Path to scenario JSON file")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--compare",  action="store_true",
                        help="Compare roundabout vs signal designs")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--import-bbox", metavar="BBOX",
                        help="Fetch OSM for 'south,west,north,east' and save scenario")
    parser.add_argument("--output", metavar="FILE",
                        help="Output path for --import-bbox")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server streaming live state (default port 8765)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--worldpop-cache-list", action="store_true",
                        help="List city-specific WorldPop caches")
    parser.add_argument("--worldpop-cache-delete", metavar="CITY",
                        help="Delete city-specific WorldPop cache by city name/slug")
    parser.add_argument("--worldpop-cache-delete-all", action="store_true",
                        help="Delete all city-specific WorldPop caches")
    parser.add_argument("--worldpop-cache-dir", default=".overpass_cache",
                        help="Cache directory for WorldPop city caches")
    args = parser.parse_args(argv)

    if args.worldpop_cache_list or args.worldpop_cache_delete or args.worldpop_cache_delete_all:
        import os as _os
        from engine.worldpop import list_worldpop_caches, delete_worldpop_city_cache

        cache_dir = args.worldpop_cache_dir
        if args.worldpop_cache_delete_all:
            entries = list_worldpop_caches(cache_dir=cache_dir)
            for e in entries:
                try:
                    _os.remove(e["path"])
                except OSError:
                    pass
            print(f"Deleted {len(entries)} WorldPop city cache(s) from {cache_dir}")
            return

        if args.worldpop_cache_delete:
            ok = delete_worldpop_city_cache(args.worldpop_cache_delete, cache_dir=cache_dir)
            if ok:
                print(f"Deleted WorldPop cache for '{args.worldpop_cache_delete}'")
            else:
                print(f"No WorldPop cache found for '{args.worldpop_cache_delete}'")
            return

        entries = list_worldpop_caches(cache_dir=cache_dir)
        if not entries:
            print(f"No WorldPop city caches in {cache_dir}")
            return
        print(f"WorldPop city caches in {cache_dir}:")
        for e in entries:
            print(f"  - {e['city_slug']}: {e['size_bytes']} bytes ({e['path']})")
        return

    if args.import_bbox:
        _import_bbox_cmd(args.import_bbox, args.output)
        return

    if args.scenario is None and not args.serve:
        parser.print_help()
        sys.exit(1)

    if args.compare:
        if args.scenario is None:
            print("Error: --compare requires a scenario file", file=sys.stderr)
            sys.exit(1)
        _compare(args.scenario, args.runs)
        return

    if args.scenario is None:
        scenario = _blank_scenario()
        location = "Blank map"
    else:
        with open(args.scenario) as fh:
            scenario = json.load(fh)

        # Derive a short location name from the scenario file or its _comment field
        import os as _os
        _raw_name = _os.path.basename(args.scenario).replace(".json", "").replace("_", " ")
        location  = scenario.get("_comment", "").split("—")[0].strip() or _raw_name.title()

    if args.serve:
        if args.duration is not None:
            scenario["duration"] = args.duration
        if args.seed is not None:
            scenario["seed"] = args.seed
        _serve(scenario, port=args.port, location=location)
        return

    if args.duration is not None:
        scenario["duration"] = args.duration
    if args.seed is not None:
        scenario["seed"] = args.seed

    sim = Simulation(scenario)
    metrics = sim.run()
    _print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
