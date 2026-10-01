"""litellm.acompletion integration hook (Phase 3 §46 steps 12/13).

A request whose ``model`` matches the configured edge route model
(``differential_sseproxy`` by default) is handled by the edge transport:

    ordinary OpenAI client request
            |
            v
    litellm.acompletion (this hook)
            |
            v
    resolve context id (session resolver)
            |
            v
    canonical prompt state -> REGISTER_FULL / APPEND / REPLACE_TAIL
            |
            v
    serialize edge envelope -> zstd/gzip -> HTTP/2
            |
            v
    SSEProxy edge endpoint

Unrelated providers never share the edge transport (the hook returns None
for every other model). For stream=True the hook returns an async
generator of ordinary OpenAI chunk dicts (wrapped as ModelResponse); the
Differential Context ACK control event is consumed and stripped by the
transport — never exposed to the coding client.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from .config import get_config

logger = logging.getLogger("litellm.edge_transport")

# OpenAI generation parameters forwarded beside the edge envelope (the
# delta carries message content; these are transport/generation-only).
_GENERATION_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "max_completion_tokens",
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "seed",
        "n",
        "user",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "metadata",
        "reasoning_effort",
        "stream_options",
        "timeout",
    }
)

_EDGE_TRANSPORT: Optional[Any] = None


def _transport() -> Any:
    """Lazy singleton (created on first use inside the running loop)."""
    global _EDGE_TRANSPORT
    if _EDGE_TRANSPORT is None:
        from .transport import EdgeTransport

        _EDGE_TRANSPORT = EdgeTransport(get_config())
    return _EDGE_TRANSPORT


def route_matches(model: str) -> bool:
    route = get_config().route_model
    return model == route or model.startswith(route + "/")


async def maybe_route_edge(
    *,
    model: str,
    messages: list[Any],
    stream: Optional[bool],
    kwargs: dict[str, Any],
    extra_headers: Optional[dict[str, Any]] = None,
) -> Optional[Any]:
    """Return the edge route result, or None when the route is not active
    (disabled config or non-matching model). Errors on an ACTIVE route
    propagate — no silent fallback for the edge route."""
    config = get_config()
    if not config.enabled:
        return None
    if not route_matches(model):
        return None
    if kwargs.get("custom_llm_provider"):
        return None

    from litellm.types.utils import ModelResponse

    headers = extra_headers or kwargs.get("extra_headers") or {}
    if not isinstance(headers, dict):
        headers = dict(headers) if headers else {}

    generation_params = {
        key: value for key, value in kwargs.items() if key in _GENERATION_PARAM_KEYS and value is not None
    }

    transport = _transport()
    response, meta = await transport.acompletion(
        model=model,
        messages=list(messages),
        stream=stream is True,
        headers=headers,
        generation_params=generation_params,
    )

    if stream is True:
        return _chunk_wrapper(response, meta)

    if isinstance(response, dict):
        return ModelResponse.model_validate(response)
    return response


async def _chunk_wrapper(
    chunks: AsyncIterator[dict[str, Any]], meta: Any
) -> AsyncIterator[Any]:
    """Wrap ordinary OpenAI SSE chunk dicts as ModelResponse objects. The
    transport already stripped every Differential Context control frame."""
    from litellm.types.utils import ModelResponse

    async for chunk in chunks:
        yield ModelResponse.model_validate(chunk)


__all__ = ["maybe_route_edge", "route_matches", "_transport"]