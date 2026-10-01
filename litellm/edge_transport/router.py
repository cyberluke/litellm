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
from .transport import EdgeTransportError

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
        from .transport import EdgeTransport, EdgeTransportError

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
    # NOTE: there is deliberately NO custom_llm_provider bail-out here. The
    # edge route model name is reserved for the edge transport; an explicit
    # provider qualifier on that exact name must not fall through to normal
    # dispatch (which would hit an unrelated provider with edge semantics).

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


async def route_edge_proxy(
    *, model: str, data: dict[str, Any], headers: Optional[dict[str, str]] = None
) -> Optional[Any]:
    """Proxy-layer edge route handler.

    The LiteLLM proxy (proxy_server.chat_completion) dispatches through its
    own ``llm_router.schedule_acompletion`` and NEVER reaches
    ``litellm.acompletion``, so the acompletion-level edge hook cannot fire
    on the HTTP path. proxy_server intercepts the edge route BEFORE the
    router and calls this: edge semantics for non-stream requests return a
    ``ModelResponse`` (jsonified by the handler); stream requests return an
    SSE response whose frames are ordinary OpenAI chunk dicts followed by
    ``[DONE]`` (the Differential Context ACK was already stripped by the
    transport). Nothing here ever calls the router or a provider.
    """
    config = get_config()
    if not config.enabled:
        return None
    if not route_matches(model):
        return None

    messages = data.get("messages")
    if not messages:
        raise ValueError("edge route: request has no messages")

    stream = data.get("stream") is True
    generation_params = {
        key: value
        for key, value in data.items()
        if key in _GENERATION_PARAM_KEYS and value is not None
    }

    transport = _transport()
    try:
        response, meta = await transport.acompletion(
            model=model,
            messages=list(messages),
            stream=stream,
            headers=headers or {},
            generation_params=generation_params,
        )
    except EdgeTransportError as exc:
        # Surface the WAN chain's rejection to the client as a CLEAN error
        # (e.g. "edge endpoint returned 502: image input not supported")
        # instead of a generic 500 or a silently broken stream. Kilo Code
        # renders this JSON error directly.
        from fastapi.responses import JSONResponse

        status = exc.status or 502
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": str(exc),
                    "type": "edge_transport_error",
                    "code": status,
                }
            },
        )

    if not stream:
        from litellm.types.utils import ModelResponse

        if isinstance(response, dict):
            return ModelResponse.model_validate(response)
        return response

    import json as _json
    import time as _time
    from uuid import uuid4 as _uuid4

    from sse_starlette.sse import EventSourceResponse

    async def _frames() -> AsyncIterator[dict[str, str]]:
        # The WAN stream yields ENGINE-NATIVE frames
        # ({"text", "output_ids", "meta_info", ...}) — convert each into an
        # ordinary OpenAI chat.completion.chunk so standard OpenAI clients
        # (Kilo, etc.) parse it. The Differential Context ACK was already
        # stripped by the transport.
        async for chunk in response:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            meta = chunk.get("meta_info")
            meta = meta if isinstance(meta, dict) else {}
            finish = None
            fr = meta.get("finish_reason")
            if isinstance(fr, dict):
                finish = fr.get("type") or None
            elif isinstance(fr, str):
                finish = fr or None
            oai_chunk = {
                "id": meta.get("id") or f"dc_{_uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(_time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text} if isinstance(text, str) else {},
                        "finish_reason": finish,
                    }
                ],
            }
            yield {"data": _json.dumps(oai_chunk, default=str)}
        yield {"data": "[DONE]"}

    return EventSourceResponse(
        _frames(),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["maybe_route_edge", "route_edge_proxy", "route_matches", "_transport"]