"""GPU / CPU compute backend detection and correctness (Milestone 6)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from engine.compute_backend import detect_backend, get_idm_backend


def test_detect_backend_returns_valid_string():
    backend = detect_backend()
    assert backend in ("cuda", "metal", "numpy")


def test_numpy_backend_always_available():
    fn = get_idm_backend(prefer="numpy")
    assert fn is not None


def test_idm_backend_output_matches_numpy():
    """Any backend must produce the same result as the NumPy reference."""
    rng = np.random.default_rng(77)
    N = 500
    v   = rng.uniform(0, 25, N).astype(np.float32)
    v0  = np.full(N, 30.0, np.float32)
    s   = rng.uniform(3, 80, N).astype(np.float32)
    dv  = rng.uniform(-3, 3, N).astype(np.float32)

    ref_fn  = get_idm_backend(prefer="numpy")
    test_fn = get_idm_backend(prefer="auto")

    ref    = np.array(ref_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0))
    result = np.array(test_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0))
    np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-4)


def test_idm_backend_benchmark_numpy():
    """NumPy backend: 100 k vehicles in < 50 ms (average over 10 runs)."""
    N = 100_000
    rng = np.random.default_rng(0)
    v  = rng.uniform(0, 25, N)
    v0 = np.full(N, 30.0)
    s  = rng.uniform(3, 80, N)
    dv = np.zeros(N)

    fn = get_idm_backend(prefer="numpy")
    t0 = time.perf_counter()
    for _ in range(10):
        fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    avg = (time.perf_counter() - t0) / 10
    assert avg < 0.05, f"NumPy IDM 100k: {avg * 1000:.1f} ms — too slow"


@pytest.mark.skipif(
    detect_backend() not in ("cuda", "metal"),
    reason="GPU not available",
)
def test_gpu_backend_benchmark():
    """GPU backend must be ≥5× faster than NumPy on 100 k vehicles."""
    N = 100_000
    rng = np.random.default_rng(0)
    v  = rng.uniform(0, 25, N)
    v0 = np.full(N, 30.0)
    s  = rng.uniform(3, 80, N)
    dv = np.zeros(N)

    numpy_fn = get_idm_backend(prefer="numpy")
    gpu_fn   = get_idm_backend(prefer="auto")

    # Warmup JIT / device transfer
    gpu_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)

    t0 = time.perf_counter()
    for _ in range(50):
        numpy_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    numpy_t = (time.perf_counter() - t0) / 50

    t0 = time.perf_counter()
    for _ in range(50):
        gpu_fn(v, v0, s, dv, 2.0, 1.5, 1.4, 2.0)
    gpu_t = (time.perf_counter() - t0) / 50

    speedup = numpy_t / gpu_t
    assert speedup >= 5, f"GPU speedup only {speedup:.1f}×"


def test_graceful_fallback_to_numpy(monkeypatch):
    """If CuPy / JAX fail to import, backend must silently use NumPy."""
    import importlib
    import sys

    # Simulate missing GPU libraries
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "jax",  None)

    import engine.compute_backend as cb
    importlib.reload(cb)

    assert cb.detect_backend() == "numpy"


def test_simulation_deterministic_across_backends():
    """numpy and auto backends must yield the same vehicle count after 100 steps."""
    from engine.network import Network, Node, Edge
    from engine.network_simulation import NetworkSimulation

    def _run(backend_name: str) -> int:
        net = Network()
        net.add_node(Node(id="A", x=0.0,    y=0.0))
        net.add_node(Node(id="B", x=1000.0, y=0.0))
        net.add_edge(Edge(id="AB", from_node="A", to_node="B", num_lanes=1,
                          speed_limit=14.0, geometry=[[0.0, 0.0], [1000.0, 0.0]]))
        sim = NetworkSimulation(
            net,
            demand={"A": {"B": 300.0}},
            duration=60.0,
            seed=0,
            compute_backend=backend_name,
        )
        for _ in range(100):
            sim.step()
        return len(sim.vehicles)

    cpu_count  = _run("numpy")
    auto_count = _run("auto")
    assert cpu_count == auto_count
