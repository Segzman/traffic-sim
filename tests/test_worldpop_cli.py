"""CLI hooks for city WorldPop cache management."""
from __future__ import annotations

import subprocess
import sys

from engine.network import Network, Node, Edge
from engine.worldpop import load_worldpop_weights, slugify_city


def _tiny_net() -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0, y=0.0))
    net.add_node(Node(id="B", x=100.0, y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1, speed_limit=10.0,
                      geometry=[[0.0, 0.0], [100.0, 0.0]]))
    return net


def test_worldpop_cli_list_and_delete(tmp_path):
    net = _tiny_net()
    slug = slugify_city("Test City")
    load_worldpop_weights(
        bbox=(43.0, -79.8, 43.2, -79.6),
        network=net,
        cache_dir=str(tmp_path),
        city_slug=slug,
    )

    cmd_list = [
        sys.executable,
        "run.py",
        "--worldpop-cache-list",
        "--worldpop-cache-dir",
        str(tmp_path),
    ]
    out = subprocess.run(cmd_list, cwd="/Users/sekun/Traffic sim", capture_output=True, text=True, check=True)
    assert slug in out.stdout

    cmd_del = [
        sys.executable,
        "run.py",
        "--worldpop-cache-delete",
        "Test City",
        "--worldpop-cache-dir",
        str(tmp_path),
    ]
    out2 = subprocess.run(cmd_del, cwd="/Users/sekun/Traffic sim", capture_output=True, text=True, check=True)
    assert "Deleted" in out2.stdout
