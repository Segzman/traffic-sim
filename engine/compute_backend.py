"""Compute backend detection and IDM dispatch (Milestone 6).

Priority order: CUDA (CuPy) → Apple Metal (JAX) → NumPy (always available).

All backends expose the same signature::

    fn(v, v0, s, dv, s0, T, a, b) -> np.ndarray

where scalar s0, T, a, b are broadcast constants and v, v0, s, dv are 1-D
arrays of shape (N,).
"""
from __future__ import annotations

import numpy as np

from engine.idm_vec import idm_acceleration_vec


# ---------------------------------------------------------------------------
# NumPy reference backend (always available)
# ---------------------------------------------------------------------------

def _idm_numpy(
    v, v0, s, dv,
    s0: float, T: float, a: float, b: float,
) -> np.ndarray:
    return idm_acceleration_vec(
        np.asarray(v,  dtype=np.float64),
        np.asarray(v0, dtype=np.float64),
        np.asarray(s,  dtype=np.float64),
        np.asarray(dv, dtype=np.float64),
        s0, T, a, b,
    )


# ---------------------------------------------------------------------------
# CuPy (NVIDIA CUDA) backend — optional
# ---------------------------------------------------------------------------

def _try_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
        # When monkeypatched to None in sys.modules the import raises; the
        # except below catches it.  Extra guard for other edge cases:
        if cp is None:
            return None, False

        def idm_cuda(v, v0, s, dv, s0, T, a, b):  # noqa: ANN001
            vc  = cp.asarray(v,  dtype=cp.float64)
            v0c = cp.asarray(v0, dtype=cp.float64)
            sc  = cp.asarray(s,  dtype=cp.float64)
            dvc = cp.asarray(dv, dtype=cp.float64)
            s_star = s0 + cp.maximum(0.0, vc * T + vc * dvc / (2.0 * cp.sqrt(a * b)))
            result = a * (
                1.0
                - (vc  / cp.maximum(v0c, 1e-6)) ** 4
                - (s_star / cp.maximum(sc,  1e-3)) ** 2
            )
            return cp.asnumpy(result)

        return idm_cuda, True
    except Exception:  # noqa: BLE001
        return None, False


# ---------------------------------------------------------------------------
# JAX (Apple Metal / XLA) backend — optional
# ---------------------------------------------------------------------------

def _try_jax():
    try:
        import jax            # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        if jax is None:
            return None, False

        @jax.jit
        def _inner(vc, v0c, sc, dvc, s0, T, a, b):
            s_star = s0 + jnp.maximum(0.0, vc * T + vc * dvc / (2.0 * jnp.sqrt(a * b)))
            return a * (
                1.0
                - (vc   / jnp.maximum(v0c, 1e-6)) ** 4
                - (s_star / jnp.maximum(sc,  1e-3)) ** 2
            )

        def idm_metal(v, v0, s, dv, s0, T, a, b):  # noqa: ANN001
            result = _inner(
                jnp.asarray(v),  jnp.asarray(v0),
                jnp.asarray(s),  jnp.asarray(dv),
                float(s0), float(T), float(a), float(b),
            )
            return np.array(result)

        return idm_metal, True
    except Exception:  # noqa: BLE001
        return None, False


# ---------------------------------------------------------------------------
# Module-level initialisation — runs once at import time
# ---------------------------------------------------------------------------

_cuda_fn, _has_cuda  = _try_cupy()
_metal_fn, _has_metal = _try_jax()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Return the best available compute backend identifier.

    Returns
    -------
    str
        ``"cuda"`` if CuPy is available, ``"metal"`` if JAX is available,
        ``"numpy"`` otherwise.
    """
    if _has_cuda:
        return "cuda"
    if _has_metal:
        return "metal"
    return "numpy"


def get_idm_backend(prefer: str = "auto"):
    """Return the IDM batch function for the requested backend.

    Parameters
    ----------
    prefer:
        ``"auto"`` selects the best available backend.  ``"numpy"``,
        ``"cuda"``, or ``"metal"`` request a specific backend; falls back
        to NumPy if the requested one is unavailable.

    Returns
    -------
    callable
        A function with signature
        ``(v, v0, s, dv, s0, T, a, b) -> np.ndarray``.
    """
    if prefer == "numpy":
        return _idm_numpy
    if prefer == "cuda":
        return _cuda_fn if _has_cuda else _idm_numpy
    if prefer == "metal":
        return _metal_fn if _has_metal else _idm_numpy
    # "auto": best available
    if _has_cuda:
        return _cuda_fn
    if _has_metal:
        return _metal_fn
    return _idm_numpy
