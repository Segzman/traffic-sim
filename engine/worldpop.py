"""WorldPop node-weight loading for Milestone 3 demand generation."""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from typing import TYPE_CHECKING

import numpy as np

from importer.projection import mercator_to_latlng

if TYPE_CHECKING:
    from engine.network import Network

try:
    from scipy.spatial import KDTree as _KDTree
except Exception:  # pragma: no cover - optional dependency
    _KDTree = None


class _NearestTree:
    """Small fallback nearest-neighbour index when scipy is unavailable."""

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


def _cache_file(
    bbox: tuple[float, float, float, float],
    network: "Network",
    cache_dir: str,
    city_slug: str | None = None,
) -> str:
    if city_slug:
        return os.path.join(cache_dir, f"worldpop_city_{city_slug}.pkl")
    south, west, north, east = bbox
    key = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}|{len(network.nodes)}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"worldpop_nodes_{digest}.pkl")


def _surrounding_cache_file(
    bbox: tuple[float, float, float, float],
    cache_dir: str,
    city_slug: str | None = None,
    ring_km: float = 60.0,
) -> str:
    if city_slug:
        return os.path.join(
            cache_dir,
            f"worldpop_surrounding_{city_slug}_{int(round(ring_km))}km.pkl",
        )
    south, west, north, east = bbox
    key = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}|{ring_km:.1f}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"worldpop_surrounding_{digest}.pkl")


def slugify_city(name: str) -> str:
    """Return filesystem-safe city slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower()).strip("-")
    return s or "unknown-city"


def _extract_cached_weights(payload) -> dict | None:
    if isinstance(payload, dict) and "weights" in payload and isinstance(payload["weights"], dict):
        return payload["weights"]
    if isinstance(payload, dict):
        return payload
    return None


def _validate_and_fill(weights: dict, network: "Network") -> tuple[dict[str, float], float]:
    """Return cleaned weights and coverage fraction on current network nodes."""
    cleaned: dict[str, float] = {}
    hit = 0
    eps = 1e-6
    for nid in network.nodes:
        v = weights.get(nid)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if np.isfinite(fv) and fv > 0:
            hit += 1
            cleaned[nid] = fv
        else:
            cleaned[nid] = eps
    coverage = hit / max(1, len(network.nodes))
    return cleaned, coverage


def list_worldpop_caches(cache_dir: str = ".overpass_cache") -> list[dict]:
    """List available city-specific WorldPop cache files."""
    out: list[dict] = []
    if not os.path.isdir(cache_dir):
        return out
    for fn in sorted(os.listdir(cache_dir)):
        if not (fn.startswith("worldpop_city_") and fn.endswith(".pkl")):
            continue
        slug = fn[len("worldpop_city_"):-4]
        path = os.path.join(cache_dir, fn)
        out.append({
            "city_slug": slug,
            "path": path,
            "size_bytes": os.path.getsize(path),
        })
    return out


def delete_worldpop_city_cache(city: str, cache_dir: str = ".overpass_cache") -> bool:
    """Delete cached WorldPop weights for one city. Returns True when removed."""
    slug = slugify_city(city)
    path = os.path.join(cache_dir, f"worldpop_city_{slug}.pkl")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _uniform_weights(network: "Network") -> dict[str, float]:
    return {nid: 1.0 for nid in network.nodes}


def _raster_surrounding_exposure(
    bbox: tuple[float, float, float, float],
    raster_path: str,
    ring_km: float,
) -> dict[str, float]:
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import xy
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("rasterio is required for raster-based WorldPop loading") from exc

    south, west, north, east = bbox
    # Rough km -> degrees expansion; good enough for directional weighting.
    lat_mid = 0.5 * (south + north)
    d_lat = max(0.02, float(ring_km) / 111.0)
    cos_lat = np.clip(np.cos(np.deg2rad(lat_mid)), 0.2, 1.0)
    d_lon = max(0.02, float(ring_km) / (111.0 * float(cos_lat)))
    outer_s, outer_w = south - d_lat, west - d_lon
    outer_n, outer_e = north + d_lat, east + d_lon

    out = {"west": 0.0, "east": 0.0, "south": 0.0, "north": 0.0}
    with rasterio.open(raster_path) as ds:
        win = from_bounds(outer_w, outer_s, outer_e, outer_n, ds.transform)
        arr = ds.read(1, window=win, masked=True)
        win_t = ds.window_transform(win)

        rows, cols = arr.shape
        for r in range(rows):
            for c in range(cols):
                v = float(arr[r, c])
                if not np.isfinite(v) or v <= 0.0:
                    continue
                lon, lat = xy(win_t, r, c, offset="center")
                # Skip inner city box; we only want external catchment exposure.
                if west <= lon <= east and south <= lat <= north:
                    continue

                dx = 0.0
                if lon < west:
                    dx = west - lon
                elif lon > east:
                    dx = lon - east

                dy = 0.0
                if lat < south:
                    dy = south - lat
                elif lat > north:
                    dy = lat - north

                if dx <= 0.0 and dy <= 0.0:
                    continue
                if dx >= dy:
                    out["west" if lon < west else "east"] += v
                else:
                    out["south" if lat < south else "north"] += v
    return out


def _raster_weights(
    bbox: tuple[float, float, float, float],
    network: "Network",
    raster_path: str,
) -> dict[str, float]:
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import xy
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("rasterio is required for raster-based WorldPop loading") from exc

    south, west, north, east = bbox

    node_ids = list(network.nodes.keys())
    node_latlon = np.array([mercator_to_latlng(network.nodes[n].x, network.nodes[n].y) for n in node_ids])
    tree = _NearestTree(node_latlon)

    out = {nid: 0.0 for nid in node_ids}
    with rasterio.open(raster_path) as ds:
        win = from_bounds(west, south, east, north, ds.transform)
        arr = ds.read(1, window=win, masked=True)
        win_t = ds.window_transform(win)

        rows, cols = arr.shape
        for r in range(rows):
            for c in range(cols):
                v = float(arr[r, c])
                if not np.isfinite(v) or v <= 0:
                    continue
                lon, lat = xy(win_t, r, c, offset="center")
                _, idx = tree.query((lat, lon))
                out[node_ids[int(idx)]] += v

    # ensure every node has at least a tiny positive mass
    eps = 1e-6
    for nid in out:
        out[nid] = max(eps, out[nid])
    return out


def load_worldpop_weights(
    bbox: tuple[float, float, float, float],
    network: "Network",
    cache_dir: str = ".overpass_cache",
    raster_path: str | None = None,
    city_slug: str | None = None,
) -> dict[str, float]:
    """Return {node_id: weight} using WorldPop raster if available.

    If *raster_path* is omitted, a deterministic uniform prior is used so M3
    demand logic remains functional without optional GIS dependencies.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_file(bbox, network, cache_dir, city_slug=city_slug)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fh:
                data = _extract_cached_weights(pickle.load(fh))
            if data:
                cleaned, coverage = _validate_and_fill(data, network)
                if coverage >= 0.80:
                    return cleaned
        except Exception:
            pass

    if raster_path:
        try:
            weights = _raster_weights(bbox, network, raster_path)
        except Exception:
            weights = _uniform_weights(network)
    else:
        weights = _uniform_weights(network)

    with open(cache_file, "wb") as fh:
        payload = {
            "version": 1,
            "city_slug": city_slug,
            "bbox": tuple(float(v) for v in bbox),
            "node_count": len(network.nodes),
            "weights": weights,
        }
        pickle.dump(payload, fh)
    return weights


def load_worldpop_surrounding_exposure(
    bbox: tuple[float, float, float, float],
    *,
    raster_path: str | None = None,
    cache_dir: str = ".overpass_cache",
    city_slug: str | None = None,
    ring_km: float = 60.0,
) -> dict[str, float] | None:
    """Directional population exposure around bbox from WorldPop raster.

    Returns ``{"west","east","south","north","total"}`` where each directional
    field is the external population in a ring around the city bbox.
    """
    if not raster_path:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _surrounding_cache_file(
        bbox=bbox,
        cache_dir=cache_dir,
        city_slug=city_slug,
        ring_km=ring_km,
    )
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fh:
                payload = pickle.load(fh)
            if isinstance(payload, dict):
                d = payload.get("directional", payload)
                if isinstance(d, dict):
                    west = max(0.0, float(d.get("west", 0.0)))
                    east = max(0.0, float(d.get("east", 0.0)))
                    south = max(0.0, float(d.get("south", 0.0)))
                    north = max(0.0, float(d.get("north", 0.0)))
                    return {
                        "west": west,
                        "east": east,
                        "south": south,
                        "north": north,
                        "total": west + east + south + north,
                    }
        except Exception:
            pass

    try:
        directional = _raster_surrounding_exposure(
            bbox=bbox,
            raster_path=raster_path,
            ring_km=ring_km,
        )
    except Exception:
        return None

    west = max(0.0, float(directional.get("west", 0.0)))
    east = max(0.0, float(directional.get("east", 0.0)))
    south = max(0.0, float(directional.get("south", 0.0)))
    north = max(0.0, float(directional.get("north", 0.0)))
    payload = {
        "version": 1,
        "city_slug": city_slug,
        "bbox": tuple(float(v) for v in bbox),
        "ring_km": float(ring_km),
        "directional": {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        },
    }
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(payload, fh)
    except Exception:
        pass

    return {
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "total": west + east + south + north,
    }
