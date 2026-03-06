"""Network graph — Node, Edge, Lane dataclasses with directed graph structure."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    x: float                        # Web Mercator metres
    y: float
    junction_type: str = "uncontrolled"  # 'uncontrolled'|'signal'|'yield'|'stop'


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    num_lanes: int
    speed_limit: float              # m/s
    geometry: list = field(default_factory=list)  # list of (x, y) Mercator waypoints


@dataclass
class Lane:
    id: str
    edge_id: str
    lane_index: int                 # 0 = rightmost
    width: float = 3.5             # metres


class Network:
    """Directed road graph: nodes, edges, and their lanes."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.lanes: dict[str, Lane] = {}
        self._adj: dict[str, list[str]] = {}   # node_id -> list of outgoing edge IDs

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        self._adj.setdefault(edge.from_node, [])
        self._adj[edge.from_node].append(edge.id)
        self._adj.setdefault(edge.to_node, [])
        # Create Lane objects for each lane on this edge
        for idx in range(edge.num_lanes):
            lane = Lane(
                id=f"{edge.id}_lane{idx}",
                edge_id=edge.id,
                lane_index=idx,
            )
            self.lanes[lane.id] = lane

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def adjacency(self) -> dict[str, list[str]]:
        """Return adjacency list: node_id -> [outgoing edge IDs]."""
        return dict(self._adj)

    def get_lanes_for_edge(self, edge_id: str) -> list[Lane]:
        """All Lane objects belonging to edge_id, sorted by lane_index."""
        lanes = [la for la in self.lanes.values() if la.edge_id == edge_id]
        return sorted(lanes, key=lambda la: la.lane_index)

    # ------------------------------------------------------------------
    # Lane geometry helpers
    # ------------------------------------------------------------------

    def lane_right_edge_offset(self, lane: Lane) -> float:
        """Lateral offset (m) of the lane RIGHT edge from the edge centreline.

        Negative means right of centreline; lane 0 right edge is at -(num_lanes * w / 2).
        """
        edge = self.edges[lane.edge_id]
        half = edge.num_lanes * lane.width / 2.0
        return -half + lane.lane_index * lane.width

    def lane_left_edge_offset(self, lane: Lane) -> float:
        """Lateral offset (m) of the lane LEFT edge from the edge centreline.

        Positive means left of centreline; lane (N-1) left edge is at +(num_lanes * w / 2).
        """
        edge = self.edges[lane.edge_id]
        half = edge.num_lanes * lane.width / 2.0
        return -half + (lane.lane_index + 1) * lane.width

    def lane_centre_offset(self, lane: Lane) -> float:
        """Lateral offset of the lane centreline."""
        return (self.lane_right_edge_offset(lane) + self.lane_left_edge_offset(lane)) / 2.0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def edge_length(self, edge_id: str) -> float:
        """Euclidean length of an edge (m), using geometry or node positions."""
        edge = self.edges[edge_id]
        pts = edge.geometry
        if not pts and edge.from_node in self.nodes and edge.to_node in self.nodes:
            fn = self.nodes[edge.from_node]
            tn = self.nodes[edge.to_node]
            return math.sqrt((tn.x - fn.x) ** 2 + (tn.y - fn.y) ** 2)
        total = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total if total > 0.0 else 1.0  # degenerate guard

    def edge_travel_time(self, edge_id: str) -> float:
        """Expected travel time (s) = length / speed_limit."""
        edge = self.edges[edge_id]
        length = self.edge_length(edge_id)
        return length / max(0.1, edge.speed_limit)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        from_node: str,
        to_node: str,
        weight: str = "travel_time",
    ) -> list[str]:
        """Dijkstra's algorithm — returns ordered list of edge IDs.

        Parameters
        ----------
        from_node, to_node:
            Node IDs (strings) in this network.
        weight:
            ``"travel_time"`` (default) or ``"length"``.

        Returns
        -------
        list[str]
            Ordered list of edge IDs forming the shortest path.
            Returns an empty list if no path exists.

        Raises
        ------
        KeyError
            If either node ID is not in the network.
        """
        if from_node not in self.nodes:
            raise KeyError(f"from_node '{from_node}' not in network")
        if to_node not in self.nodes:
            raise KeyError(f"to_node '{to_node}' not in network")

        if from_node == to_node:
            return []

        # Cost function per edge
        def _cost(edge_id: str) -> float:
            if weight == "length":
                return self.edge_length(edge_id)
            return self.edge_travel_time(edge_id)

        # dist[node] = best known cost from from_node
        dist: dict[str, float] = {from_node: 0.0}
        prev_edge: dict[str, str] = {}   # node -> edge used to reach it
        heap: list[tuple[float, str]] = [(0.0, from_node)]

        while heap:
            cost_u, u = heapq.heappop(heap)
            if u == to_node:
                break
            if cost_u > dist.get(u, math.inf):
                continue  # stale entry
            for edge_id in self._adj.get(u, []):
                edge = self.edges[edge_id]
                v = edge.to_node
                new_cost = cost_u + _cost(edge_id)
                if new_cost < dist.get(v, math.inf):
                    dist[v] = new_cost
                    prev_edge[v] = edge_id
                    heapq.heappush(heap, (new_cost, v))

        if to_node not in prev_edge:
            return []  # unreachable

        # Reconstruct path
        path: list[str] = []
        node = to_node
        while node in prev_edge:
            eid = prev_edge[node]
            path.append(eid)
            node = self.edges[eid].from_node
        path.reverse()
        return path
