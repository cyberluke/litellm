"""DeepSeek tool-call markup parser for the Differential Context edge.

The engine's Differential Context route returns the model's RAW output: tool
calls arrive as DeepSeek markup inside the assistant text (``<|DSML|tool_calls|>``
blocks with ``<|DSML|invoke name="..."|>`` / ``<|DSML|parameter name="..."|>``),
which the text-only DeepSeek engine does not post-process on this route (the
regular OpenAI route applies ``--tool-call-parser deepseekv4``).

This module extracts the markup into OpenAI ``tool_calls`` and strips it from
the message content so standard clients (Kilo Code, ...) can execute the calls
instead of rendering raw tokens as chat text.

The bars in the markup may be ASCII ``|`` or the tokenizer's fullwidth variant
(U+FF5C), so matching is done on the tag body ("DSML", "tool_calls", "invoke",
"parameter") rather than on the exact delimiter characters.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# A tag like <|DSML|tool_calls|> with any delimiter characters between < >.
_TAG = re.compile(r"<([^>]*)>")
_BLOCK_START = re.compile(r"<\s*([^>]*)DSML[^>]*tool_calls[^>]*>", re.IGNORECASE)
_BLOCK_END = re.compile(r"<\s*/\s*([^>]*)DSML[^>]*tool_calls[^>]*>", re.IGNORECASE)
_INVOKE_START = re.compile(
    r"<\s*([^>]*)DSML[^>]*(?:tool\s+)?invoke(?:\s+(?:name|invoke))?\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
_INVOKE_END = re.compile(r"<\s*/\s*([^>]*)DSML[^>]*invoke[^>]*>", re.IGNORECASE)
_PARAM_START = re.compile(
    r"<\s*([^>]*)DSML[^>]*parameter\s+(?:name|parameter)\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
_PARAM_END = re.compile(r"<\s*/\s*([^>]*)DSML[^>]*parameter[^>]*>", re.IGNORECASE)

# The model sometimes emits the tool call as a bare JSON array instead of the
# DSML markup, e.g.:
#   [{"name": "bash", "parameters": {"command": "pwd"}}]
# (the engine's own deepseekv4 parser leaves this unparsed too). The array may
# be wrapped in prose. Extract the first JSON array whose items carry a
# "name" and "parameters" object.
_JSON_ARRAY = re.compile(r"\[[\s\S]*?\]")


def _find_tag(text: str, pattern: re.Pattern[str], start: int = 0) -> Optional[re.Match[str]]:
    return pattern.search(text, start)


def _coerce_param_value(raw: str) -> Any:
    """Parameter values from the DSML markup are raw text; the model encodes
    structured values (arrays/objects) either directly (``[{...}]``) or as a
    JSON-encoded string (``"[{\"content\": ...}]"``). Coerce to the real JSON
    value so the client's schema validation (e.g. todowrite's ``todos`` array)
    accepts the call. Plain strings stay strings (``pwd`` / ``"pwd"``)."""
    stripped = raw.strip()
    try:
        value = json.loads(stripped)
    except (ValueError, TypeError):
        return raw
    if isinstance(value, str):
        inner = value.strip()
        try:
            inner_value = json.loads(inner)
        except (ValueError, TypeError):
            return value
        if isinstance(inner_value, (dict, list)):
            return inner_value
        return value
    return value


def parse_deepseek_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract DeepSeek tool-call markup from ``text``.

    Returns ``(clean_text, tool_calls)``:
    - ``clean_text``: the text with the tool-call block removed (leading
      prose before the block is preserved).
    - ``tool_calls``: OpenAI chat.completion ``tool_calls`` list
      (``id`` / ``type`` / ``function.{name, arguments}``); empty when no
      markup is present.
    """
    if not text:
        return text, []

    # 1) DSML markup block.
    if "DSML" in text:
        block = _BLOCK_START.search(text)
        if block is not None:
            block_end = _BLOCK_END.search(text, block.end())
            if block_end is None:
                body = text[block.end():]
                clean_text = text[:block.start()].strip()
            else:
                body = text[block.end():block_end.start()]
                clean_text = (text[:block.start()] + text[block_end.end():]).strip()

            calls: list[dict[str, Any]] = []
            pos = 0
            while True:
                invoke = _INVOKE_START.search(body, pos)
                if invoke is None:
                    break
                name = invoke.group(2).strip()
                invoke_end_m = _INVOKE_END.search(body, invoke.end())
                invoke_end = invoke_end_m.start() if invoke_end_m is not None else len(body)
                inner = body[invoke.end():invoke_end]

                params: dict[str, Any] = {}
                ppos = 0
                while True:
                    pm = _PARAM_START.search(inner, ppos)
                    if pm is None:
                        break
                    pname = pm.group(2).strip()
                    pend_m = _PARAM_END.search(inner, pm.end())
                    pend = pend_m.start() if pend_m is not None else len(inner)
                    value = _coerce_param_value(inner[pm.end():pend])
                    params[pname] = value
                    ppos = pm.end() if pend_m is None else pend_m.end()

                # The model's native bash-tool parameter is sometimes "cmd"
                # instead of the schema's "command"; map it so the client's
                # tool validation accepts the call (guarded: only when the
                # schema name is absent).
                if "cmd" in params and "command" not in params:
                    params["command"] = params.pop("cmd")

                arguments = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
                calls.append(
                    {
                        "id": f"call_{len(calls) + 1}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                pos = invoke.end() if invoke_end_m is None else invoke_end_m.end()

            if calls:
                return clean_text, calls

    # 2) Bare JSON array fallback (model-native output, no DSML wrapper).
    for m in _JSON_ARRAY.finditer(text):
        raw = m.group(0)
        try:
            items = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(items, list) or not items:
            continue
        parsed: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                break
            name = item.get("name")
            parameters = item.get("parameters")
            if not isinstance(name, str) or not isinstance(parameters, dict):
                break
            parsed.append(
                {
                    "id": f"call_{len(parsed) + 1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            parameters, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )
        if parsed:
            clean_text = (text[: m.start()] + text[m.end() :]).strip()
            return clean_text, parsed

    return text, []