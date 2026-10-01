"""litellm.edge_transport — Differential Context edge transport (Phase 3
§46 steps 10-13).

Isolated module inside the LiteLLM fork. All Differential Context protocol
logic comes from the shared ``differential-context`` package; this package
owns only transport concerns (config, session resolution, compression,
capabilities validation, HTTPX HTTP/2 client, metrics) and never duplicates
the edge state machine.

Importing this package must stay cheap: heavy imports (httpx, transport)
are deferred to first use so litellm's ordinary paths are unaffected when
the edge route is disabled.
"""

from .config import (
    DEFAULT_ROUTE_MODEL,
    EdgeTransportConfig,
    get_config,
    load_config,
    reset_config,
    set_config,
)
from .worker_guard import assert_edge_worker_safety

__all__ = [
    "DEFAULT_ROUTE_MODEL",
    "EdgeTransportConfig",
    "get_config",
    "load_config",
    "set_config",
    "reset_config",
    "assert_edge_worker_safety",
]