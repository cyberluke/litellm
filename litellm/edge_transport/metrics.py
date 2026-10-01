"""Edge transport metrics (Phase 3 §46 "Metrics required before live
benchmark").

Prometheus-safe names map to the spec's dotted logical names:

    edge.request.full_logical_bytes  -> edge_request_full_logical_bytes_total
    edge.request.delta_bytes         -> edge_request_delta_bytes_total
    edge.request.wire_bytes          -> edge_request_wire_bytes_total
    edge.request.compression_ratio   -> edge_request_compression_ratio
    edge.request.compression_ms      -> edge_request_compression_ms
    edge.http.negotiated_version     -> edge_http_negotiated_version_total
    edge.http.active_streams         -> edge_http_active_streams
    edge.dc.operation                -> edge_dc_operation_total
    edge.dc.generation               -> edge_dc_generation
    edge.dc.conflicts                -> edge_dc_conflicts_total
    edge.dc.disabled_reason          -> edge_dc_disabled_reason_total
    edge.dc.lost_ack_recovered       -> edge_dc_lost_ack_recovered_total
    edge.dc.resync_full              -> edge_dc_resync_full_total
    edge.dc.unrecoverable_conflict   -> edge_dc_unrecoverable_conflict_total
"""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram

    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PROMETHEUS = False

    class _Counter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_Counter":
            return self

        def inc(self, n: int = 1) -> None:
            pass

    class _Gauge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_Gauge":
            return self

        def set(self, value: float | int) -> None:
            pass

        def inc(self, n: int = 1) -> None:
            pass

        def dec(self, n: int = 1) -> None:
            pass

    class _Histogram:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_Histogram":
            return self

        def observe(self, value: float) -> None:
            pass

    Counter = _Counter  # type: ignore[assignment,misc]
    Gauge = _Gauge  # type: ignore[assignment,misc]
    Histogram = _Histogram  # type: ignore[assignment,misc]


_RATIO_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)

M_FULL = Counter("edge_request_full_logical_bytes_total", "full logical request bytes")
M_DELTA = Counter("edge_request_delta_bytes_total", "edge delta envelope bytes")
M_WIRE = Counter("edge_request_wire_bytes_total", "compressed wire bytes sent")
M_RATIO = Histogram("edge_request_compression_ratio", "logical/delta byte ratio", buckets=_RATIO_BUCKETS)
M_COMPRESS_MS = Histogram("edge_request_compression_ms", "envelope compression duration ms")
M_NEGOTIATED = Counter(
    "edge_http_negotiated_version_total", "negotiated HTTP version on the edge route", ["version"]
)
M_ACTIVE_STREAMS = Gauge("edge_http_active_streams", "active edge SSE streams")
M_OP = Counter("edge_dc_operation_total", "edge differential operations", ["op"])
M_GENERATION = Gauge("edge_dc_generation", "acknowledged edge generation per context", ["context_id"])
M_CONFLICTS = Counter("edge_dc_conflicts_total", "edge baseline conflicts (409)")
M_DISABLED = Counter(
    "edge_dc_disabled_reason_total", "differential disabled reasons", ["reason"]
)
# Phase 3.5 §8/§9/§10 recovery metrics.
M_LOST_ACK = Counter("edge_dc_lost_ack_recovered_total", "lost-ACK recoveries (pending promoted)")
M_RESYNC = Counter("edge_dc_resync_full_total", "explicit RESYNC_FULL sends", ["reason"])
M_UNRECOVERABLE = Counter(
    "edge_dc_unrecoverable_conflict_total",
    "unmatched authoritative conflicts raised (no invisible retry)",
    ["reason"],
)


__all__ = [
    "M_FULL",
    "M_DELTA",
    "M_WIRE",
    "M_RATIO",
    "M_COMPRESS_MS",
    "M_NEGOTIATED",
    "M_ACTIVE_STREAMS",
    "M_OP",
    "M_GENERATION",
    "M_CONFLICTS",
    "M_DISABLED",
    "M_LOST_ACK",
    "M_RESYNC",
    "M_UNRECOVERABLE",
]