"""Deterministic session resolution (Phase 3 §46 step 11).

Priority:
1. explicit configured session header/body metadata
   (``EDGE_SSEPROXY_SESSION_HEADERS`` / ``EDGE_SSEPROXY_SESSION_BODY_KEYS``);
2. known LiteLLM request metadata (``metadata.session_id`` body key);
3. known coding-agent adapter headers (Kilo/Cline/Kelvin).

Subagent semantics stay as corrected: a NEW child resolves to its own
independent context; RESUME of the same child reuses the same context id;
parent ids are lineage metadata only, never a base, never a FORK trigger
(the edge producer never auto-forks).

If no stable session identity exists, Differential Context is DISABLED for
that request (``dc_disabled_reason=no_stable_session_identity``) while
transport (HTTP/2, compression, full ordinary OpenAI request) still runs.
Session identity is NEVER guessed from prompt similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import EdgeTransportConfig

DISABLED_NO_STABLE_IDENTITY = "no_stable_session_identity"


@dataclass(slots=True)
class SessionResolution:
    context_id: Optional[str]
    disabled_reason: Optional[str]
    source: str  # header/body key that produced the id, or "none"


def _valid_context_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 128:
        return None
    if any(ord(ch) < 0x20 for ch in value):
        return None
    return value


def _header(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    value = headers.get(name)
    if isinstance(value, str) and value:
        return value
    return None


def _body_value(body: Any, dotted_key: str) -> Optional[Any]:
    if not isinstance(body, dict):
        return None
    current: Any = body
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def resolve_session(
    headers: Any,
    body: Optional[dict[str, Any]],
    config: EdgeTransportConfig,
) -> SessionResolution:
    # 1. Explicit configured session header metadata.
    for name in config.session_headers:
        value = _valid_context_id(_header(headers, name))
        if value is not None:
            return SessionResolution(value, None, f"header:{name}")
    # 2. Explicit configured session body metadata (incl. LiteLLM's
    #    metadata.session_id).
    if isinstance(body, dict):
        for key in config.session_body_keys:
            value = _valid_context_id(_body_value(body, key))
            if value is not None:
                return SessionResolution(value, None, f"body:{key}")
    # 3. Known coding-agent adapters (identity fields only — adapters never
    #    own protocol semantics).
    for name in config.adapter_session_headers:
        value = _valid_context_id(_header(headers, name))
        if value is not None:
            return SessionResolution(value, None, f"adapter:{name}")
    # No stable identity: Differential Context disabled for this request.
    return SessionResolution(None, DISABLED_NO_STABLE_IDENTITY, "none")


__all__ = ["SessionResolution", "resolve_session", "DISABLED_NO_STABLE_IDENTITY"]