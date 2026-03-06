"""Find well-connected OD pairs in the Oakville network and rewrite the scenario."""
import json, sys, random
sys.path.insert(0, '.')
from run import _build_network_sim
from collections import deque

sc = json.load(open('scenarios/oakville_on.json'))
sim = _build_network_sim(sc)
net = sim.network

def reachable_from(start):
    visited = {start}
    q = deque([start])
    while q:
        n = q.popleft()
        for eid in net._adj.get(n, []):
            nb = net.edges[eid].to_node
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited

# Find the hub with widest reach
candidates = [nid for nid, edges in net._adj.items() if len(edges) >= 2]
random.seed(42)
sample = random.sample(candidates, min(300, len(candidates)))

best_node, best_count = None, 0
for nid in sample:
    r = reachable_from(nid)
    if len(r) > best_count:
        best_count = len(r)
        best_node = nid

print(f"Hub: {best_node} reaches {best_count} nodes")
hub_reach = list(reachable_from(best_node))
print(f"Hub reach list: {len(hub_reach)} nodes")

# Sample random pairs from within the reachable set and find valid paths
random.seed(123)
reach_sample = random.sample(hub_reach, min(100, len(hub_reach)))

good_pairs = []
for i, a in enumerate(reach_sample):
    for b in reach_sample:
        if a != b:
            p = net.shortest_path(a, b)
            if p and len(p) > 5:   # must be non-trivial path
                good_pairs.append((a, b, len(p)))
    if len(good_pairs) >= 20:
        break

print(f"Found {len(good_pairs)} good pairs")
good_pairs.sort(key=lambda x: -x[2])  # longest paths first
for a, b, n in good_pairs[:6]:
    na = net.nodes[a]; nb_node = net.nodes[b]
    print(f"  {a}({na.x:.0f},{na.y:.0f}) -> {b}({nb_node.x:.0f},{nb_node.y:.0f}): {n} edges")

# Pick 4 diverse pairs for demand
demand = {}
used_origins = set()
used_dests = set()
chosen = []
for a, b, n in good_pairs:
    if a not in used_origins and b not in used_dests:
        chosen.append((a, b))
        used_origins.add(a)
        used_dests.add(b)
        demand.setdefault(a, {})[b] = 25
    if len(chosen) >= 4:
        break

print(f"Chosen OD pairs: {chosen}")
print(f"Demand: {demand}")

sc['demand'] = demand
with open('scenarios/oakville_on.json', 'w') as f:
    json.dump(sc, f)
print("Saved.")
