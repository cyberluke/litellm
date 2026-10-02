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
            # The engine's Differential Context route returns RAW model
            # output: DeepSeek tool calls arrive as <|DSML|...|> markup in
            # the content. Extract them into OpenAI tool_calls and strip the
            # markup so clients execute the calls instead of rendering the
            # raw tokens. The client's tools schema makes the coercion
            # schema-aware (string params like write.content stay strings).
            from .deepseek_tools import _tool_param_types, parse_deepseek_tool_calls

            param_types = _tool_param_types(data.get("tools"))

            choices = response.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    clean, calls = parse_deepseek_tool_calls(msg["content"], param_types)
                    if calls:
                        msg["content"] = clean
                        msg["tool_calls"] = calls
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
        #
        # DeepSeek tool calls arrive as <|DSML|...|> markup INSIDE the text
        # (the engine's Differential Context route does not post-process the
        # output). The full text is buffered, parsed at the end, and the
        # markup is re-emitted as OpenAI tool_calls deltas (the engine emits
        # the whole response in one frame, so buffering adds no latency).
        from .deepseek_tools import _tool_param_types, parse_deepseek_tool_calls

        # The client's tools schema drives the parameter coercion (string
        # params such as write.content stay strings even when the content
        # is itself JSON text).
        param_types = _tool_param_types(data.get("tools"))

        text_parts: list[str] = []
        meta_info: dict[str, Any] = {}
        finish: Optional[str] = None

        def _oai(delta: dict[str, Any], fin: Optional[str] = None) -> dict[str, str]:
            nonlocal meta_info
            oai_chunk = {
                "id": meta_info.get("id") or f"dc_{_uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(_time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": fin,
                    }
                ],
            }
            return {"data": _json.dumps(oai_chunk, default=str)}

        async for chunk in response:
            if not isinstance(chunk, dict):
                continue
            err = chunk.get("error")
            if err is not None:
                # Upstream/engine rejection frame (e.g. "Model only supports
                # text input"): surface it as an OpenAI stream error instead
                # of an empty chunk + [DONE], which clients render as
                # "Response ended unexpectedly and may be incomplete".
                msg = err.get("message") if isinstance(err, dict) else str(err)
                code = err.get("code") if isinstance(err, dict) else None
                yield {
                    "data": _json.dumps(
                        {
                            "error": {
                                "message": msg,
                                "type": "upstream_error",
                                "code": code or 502,
                            }
                        },
                        default=str,
                    )
                }
                return
            text = chunk.get("text")
            meta = chunk.get("meta_info")
            meta = meta if isinstance(meta, dict) else {}
            if isinstance(text, str):
                text_parts.append(text)
            if meta:
                meta_info.update({k: v for k, v in meta.items() if v is not None})
            fr = meta.get("finish_reason")
            if isinstance(fr, dict):
                fr = fr.get("type") or None
            elif not isinstance(fr, str):
                fr = None
            if isinstance(fr, str) and fr:
                finish = fr

        full_text = "".join(text_parts)
        clean_text, tool_calls = (
            parse_deepseek_tool_calls(full_text, param_types)
            if full_text
            else (full_text, [])
        )
        if tool_calls:
            if clean_text:
                yield _oai({"content": clean_text})
            for i, call in enumerate(tool_calls):
                fn = call.get("function") or {}
                yield _oai(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": call.get("id") or f"call_{i + 1}",
                                "type": "function",
                                "function": {
                                    "name": fn.get("name") or "",
                                    "arguments": fn.get("arguments") or "",
                                },
                            }
                        ]
                    }
                )
            yield _oai({}, "tool_calls")
        else:
            if full_text:
                yield _oai({"content": full_text})
            yield _oai({}, finish or "stop")
        yield {"data": "[DONE]"}

    return EventSourceResponse(
        _frames(),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["maybe_route_edge", "route_edge_proxy", "route_matches", "_transport"]