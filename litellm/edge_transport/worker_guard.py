"""Phase 3.5 §2/§21: single-worker enforcement for the edge profile.

Edge state and per-context asyncio locks are PROCESS-LOCAL. With more than
one uvicorn/gunicorn worker, two processes would serve the same context_id
with independent acknowledged generations and independent pending
transitions — silent corruption.

Rule: when the edge profile is enabled (``EDGE_SSEPROXY_ENABLED``),
``num_workers MUST equal 1`` unless a future cross-process shared
``EdgeStateBackend`` is explicitly configured (``EDGE_STATE_SHARED_BACKEND``).

Startup fails with a clear configuration error otherwise. One async worker
is enough for many concurrent HTTP/2/SSE contexts.
"""

from __future__ import annotations

from .config import get_config

_EDGE_WORKER_ERROR = (
    "EDGE PROFILE ERROR: differential_sseproxy (edge transport) is enabled "
    "(EDGE_SSEPROXY_ENABLED) with --num_workers > 1, but no cross-process "
    "shared EdgeStateBackend is configured (EDGE_STATE_SHARED_BACKEND is "
    "unset). Edge state and per-context locks are process-local; a second "
    "worker would corrupt acknowledged generations. Run with --num_workers 1 "
    "or configure EDGE_STATE_SHARED_BACKEND explicitly."
)


def assert_edge_worker_safety(num_workers: int) -> None:
    """Raise SystemExit when the edge profile is enabled with more than one
    worker and no shared backend. No-op when the edge profile is disabled
    or num_workers <= 1."""
    config = get_config()
    if not config.enabled:
        return
    if num_workers <= 1:
        return
    if config.shared_state_backend:
        return
    raise SystemExit(_EDGE_WORKER_ERROR)


__all__ = ["assert_edge_worker_safety", "_EDGE_WORKER_ERROR"]