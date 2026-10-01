"""Edge transport configuration (Phase 3 §46 steps 10-13).

Environment-driven; the module exposes a lazy singleton (``get_config``) so
tests can override fields per-case. All values mirror the SSEProxy-side
limits; the WAN base URL is the Caddy TLS frontend, never the aiohttp port.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

DEFAULT_ROUTE_MODEL = "differential_sseproxy"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(token.strip().lower() for token in value.split(",") if token.strip())


def default_state_db_path() -> str:
    """Phase 3.5 §3: %LOCALAPPDATA%/VIVERRA/LiteLLM/edge-state.sqlite3 (or
    the platform equivalent)."""
    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_DATA_HOME")
        or os.path.expanduser("~")
    )
    return os.path.join(base, "VIVERRA", "LiteLLM", "edge-state.sqlite3")


@dataclass(slots=True)
class EdgeTransportConfig:
    # Master switch; everything below is inert while disabled.
    enabled: bool = False
    # WAN base URL (Caddy TLS + ALPN h2 frontend). Never the aiohttp port.
    base_url: str = "https://localhost:8443"
    api_key: Optional[str] = None
    # Client-side identity: Authorization header sent to SSEProxy.
    # Strict HTTP/2: any negotiated HTTP/1.1 on the edge route raises
    # transport_error — no silent downgrade (step 13).
    strict_http2: bool = True
    # Request compression producer order preference: "zstd" | "gzip" |
    # "identity" (step 12: compress the EDGE DELTA envelope, never the
    # full logical request).
    request_compression: str = "zstd"
    compression_threshold: int = 1024
    max_request_bytes: int = 8 * 1024 * 1024
    # Session identity (step 11): explicit configured header/body metadata
    # first, then known coding-agent adapter headers. No prompt-similarity
    # guessing ever.
    session_headers: tuple[str, ...] = ("x-dc-session-id",)
    session_body_keys: tuple[str, ...] = ("session_id", "metadata.session_id")
    adapter_session_headers: tuple[str, ...] = (
        "x-kilo-session-id",
        "x-cline-session-id",
        "x-kelvin-session-id",
        "x-session-id",
    )
    # Route: requests whose model equals this (or starts with "<this>/")
    # take the Differential Context edge path.
    route_model: str = DEFAULT_ROUTE_MODEL
    # The model name the WAN chain/engine knows (e.g. the engine's served
    # model path). When set, the OUTBOUND payload model is rewritten to this
    # value on BOTH paths (edge envelope and ordinary fallback) while the
    # client-facing model name stays the route model. Empty = send the
    # client model name unchanged.
    upstream_model: str = ""
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 10.0
    # Capabilities cache TTL (step 5 contract validation). Phase 3.5 §18:
    # additionally refreshed on startup, on server-epoch mismatch and on
    # capability/protocol errors — never per inference.
    capabilities_refresh_seconds: float = 300.0
    # TLS verification for the WAN frontend. Local loopback integration
    # tests use Caddy's internal CA and may set this to 0 (the HTTP/2
    # proof is the negotiated version, not the certificate).
    verify_tls: bool = True
    # HTTPX pool bounds: persistent client, wide keepalive pool.
    max_connections: int = 50
    max_keepalive_connections: int = 20
    # Phase 3.5 §3/§5: durable local edge state. "sqlite" (default) is the
    # workstation profile; "memory" is for tests/throwaway runs. The
    # persistence lives HERE, never in the shared differential-context
    # package.
    state_persistence: str = "sqlite"
    state_db_path: str = ""
    # Phase 3.5 §2: when set, the edge profile may run with num_workers > 1
    # (a future cross-process shared EdgeStateBackend). The worker guard
    # fails startup otherwise.
    shared_state_backend: Optional[str] = None
    # Phase 3.5 §16: local idle TTL + cleanup cadence.
    idle_ttl_seconds: float = 24 * 3600.0
    idle_cleanup_interval_seconds: float = 300.0


def load_config() -> EdgeTransportConfig:
    return EdgeTransportConfig(
        enabled=_env_bool("EDGE_SSEPROXY_ENABLED", False),
        base_url=os.environ.get("EDGE_SSEPROXY_BASE_URL", "https://localhost:8443").rstrip("/"),
        api_key=os.environ.get("EDGE_SSEPROXY_API_KEY") or None,
        strict_http2=_env_bool("EDGE_SSEPROXY_STRICT_HTTP2", True),
        request_compression=os.environ.get("EDGE_SSEPROXY_REQUEST_COMPRESSION", "zstd").strip().lower(),
        compression_threshold=_env_int("EDGE_SSEPROXY_COMPRESSION_THRESHOLD", 1024),
        max_request_bytes=_env_int("EDGE_SSEPROXY_MAX_REQUEST_BYTES", 8 * 1024 * 1024),
        session_headers=_env_list("EDGE_SSEPROXY_SESSION_HEADERS", ("x-dc-session-id",)),
        session_body_keys=_env_list(
            "EDGE_SSEPROXY_SESSION_BODY_KEYS", ("session_id", "metadata.session_id")
        ),
        adapter_session_headers=_env_list(
            "EDGE_SSEPROXY_ADAPTER_SESSION_HEADERS",
            ("x-kilo-session-id", "x-cline-session-id", "x-kelvin-session-id", "x-session-id"),
        ),
        route_model=os.environ.get("EDGE_SSEPROXY_ROUTE_MODEL", DEFAULT_ROUTE_MODEL),
        upstream_model=os.environ.get("EDGE_SSEPROXY_UPSTREAM_MODEL", "").strip(),
        timeout_seconds=float(os.environ.get("EDGE_SSEPROXY_TIMEOUT_SECONDS", "600")),
        connect_timeout_seconds=float(os.environ.get("EDGE_SSEPROXY_CONNECT_TIMEOUT_SECONDS", "10")),
        capabilities_refresh_seconds=float(
            os.environ.get("EDGE_SSEPROXY_CAPABILITIES_REFRESH_SECONDS", "300")
        ),
        verify_tls=_env_bool("EDGE_SSEPROXY_VERIFY_TLS", True),
        max_connections=_env_int("EDGE_SSEPROXY_MAX_CONNECTIONS", 50),
        max_keepalive_connections=_env_int("EDGE_SSEPROXY_MAX_KEEPALIVE_CONNECTIONS", 20),
        state_persistence=os.environ.get("EDGE_STATE_PERSISTENCE", "sqlite").strip().lower(),
        state_db_path=os.environ.get("EDGE_STATE_DB_PATH") or default_state_db_path(),
        shared_state_backend=os.environ.get("EDGE_STATE_SHARED_BACKEND") or None,
        idle_ttl_seconds=float(os.environ.get("EDGE_CONTEXT_IDLE_TTL", str(24 * 3600))),
        idle_cleanup_interval_seconds=float(
            os.environ.get("EDGE_IDLE_CLEANUP_INTERVAL_SECONDS", "300")
        ),
    )


_CONFIG: Optional[EdgeTransportConfig] = None


def get_config() -> EdgeTransportConfig:
    """Lazy singleton; tests may call ``set_config`` to override."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def set_config(config: EdgeTransportConfig) -> None:
    global _CONFIG
    _CONFIG = config


def reset_config() -> None:
    global _CONFIG
    _CONFIG = None


__all__ = [
    "EdgeTransportConfig",
    "DEFAULT_ROUTE_MODEL",
    "default_state_db_path",
    "load_config",
    "get_config",
    "set_config",
    "reset_config",
]