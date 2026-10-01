"""Edge capabilities contract validation (Phase 3 §46 step 5/13).

The local LiteLLM edge fetches ``GET /v1/transport/capabilities`` from the
SSEProxy WAN frontend and validates the DEPLOYED contract before using the
edge endpoint: edge protocol version, representation, and (when strict
HTTP/2 is configured) the WAN HTTP/2 claim. Cached with a TTL.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from differential_context.edge.models import EDGE_PROTOCOL_VERSION, EDGE_REPRESENTATION

from .config import EdgeTransportConfig


class CapabilitiesError(Exception):
    pass


class CapabilitiesCache:
    def __init__(self, config: EdgeTransportConfig) -> None:
        self._config = config
        self._cached: Optional[dict[str, Any]] = None
        self._fetched_at = 0.0

    async def get(self, client: Any) -> dict[str, Any]:
        now = time.monotonic()
        if self._cached is not None and now - self._fetched_at < self._config.capabilities_refresh_seconds:
            return self._cached
        caps = await fetch_capabilities(client, self._config)
        validate_capabilities(caps, self._config)
        self._cached = caps
        self._fetched_at = now
        return caps

    def clear(self) -> None:
        self._cached = None
        self._fetched_at = 0.0


async def fetch_capabilities(client: Any, config: EdgeTransportConfig) -> dict[str, Any]:
    try:
        response = await client.get(f"{config.base_url}/v1/transport/capabilities")
    except Exception as exc:
        raise CapabilitiesError(f"capabilities fetch failed: {exc}") from exc
    if response.status_code != 200:
        raise CapabilitiesError(f"capabilities endpoint returned {response.status_code}")
    try:
        doc = response.json()
    except Exception as exc:
        raise CapabilitiesError(f"capabilities endpoint returned non-JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CapabilitiesError("capabilities endpoint returned a non-object")
    return doc


def validate_capabilities(caps: dict[str, Any], config: EdgeTransportConfig) -> None:
    """Validate the DEPLOYED capabilities against what the edge transport
    needs. Raises CapabilitiesError when the deployment cannot serve the
    edge contract."""
    # Phase 3.5 §6: the edge server process epoch is part of the contract —
    # without it the producer cannot detect remote state loss.
    epoch = caps.get("edge_server_epoch")
    if not isinstance(epoch, str) or not epoch:
        raise CapabilitiesError("capabilities missing edge_server_epoch")
    dc = caps.get("differential_context")
    if not isinstance(dc, dict):
        raise CapabilitiesError("capabilities missing differential_context")
    versions = dc.get("edge_versions")
    if not isinstance(versions, list) or EDGE_PROTOCOL_VERSION not in versions:
        raise CapabilitiesError(
            f"capabilities do not advertise edge protocol version {EDGE_PROTOCOL_VERSION}"
        )
    representations = dc.get("edge_representations")
    if not isinstance(representations, list) or EDGE_REPRESENTATION not in representations:
        raise CapabilitiesError(
            f"capabilities do not advertise representation {EDGE_REPRESENTATION!r}"
        )
    if config.strict_http2:
        http = caps.get("http")
        if not isinstance(http, dict):
            raise CapabilitiesError("capabilities missing http section")
        wan_versions = http.get("wan_versions")
        if not isinstance(wan_versions, list) or "2" not in wan_versions:
            raise CapabilitiesError(
                "strict_http2 configured but the deployment does not advertise WAN HTTP/2"
            )
        if http.get("strict_http2_supported") is not True:
            raise CapabilitiesError(
                "strict_http2 configured but capabilities report strict_http2_supported=false"
            )


__all__ = ["CapabilitiesError", "CapabilitiesCache", "fetch_capabilities", "validate_capabilities"]