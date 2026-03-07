"""WorldPop city-cache management tests."""
from __future__ import annotations

from engine.network import Network, Node, Edge
from engine.worldpop import (
    slugify_city,
    load_worldpop_weights,
    list_worldpop_caches,
    delete_worldpop_city_cache,
)


def _tiny_net() -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=100.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1, speed_limit=10.0,
                      geometry=[[0.0, 0.0], [100.0, 0.0]]))
    return net


def test_city_cache_create_list_delete(tmp_path):
    net = _tiny_net()
    city = "Oakville ON"
    slug = slugify_city(city)

    weights = load_worldpop_weights(
        bbox=(43.0, -79.8, 43.2, -79.6),
        network=net,
        cache_dir=str(tmp_path),
        raster_path=None,
        city_slug=slug,
    )
    assert set(weights.keys()) == {"A", "B"}

    entries = list_worldpop_caches(cache_dir=str(tmp_path))
    slugs = {e["city_slug"] for e in entries}
    assert slug in slugs

    assert delete_worldpop_city_cache(city, cache_dir=str(tmp_path)) is True
    entries2 = list_worldpop_caches(cache_dir=str(tmp_path))
    slugs2 = {e["city_slug"] for e in entries2}
    assert slug not in slugs2


def test_delete_missing_city_cache_returns_false(tmp_path):
    assert delete_worldpop_city_cache("does-not-exist", cache_dir=str(tmp_path)) is False
