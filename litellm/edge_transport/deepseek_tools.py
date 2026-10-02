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

import html
import json
import re
from typing import Any, Optional

# A tag like <|DSML|tool_calls|> with any delimiter characters between < >.
_TAG = re.compile(r"<([^>]*)>")
_BLOCK_START = re.compile(r"<\s*([^>]*)DSML[^>]*tool_calls[^>]*>", re.IGNORECASE)
_INVOKE_START = re.compile(
    r"<\s*([^>]*)DSML[^>]*(?:tool\s+)?invoke(?:\s+(?:name|invoke))?\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
_PARAM_START = re.compile(
    r"<\s*([^>]*)DSML[^>]*parameter\s+(?:name|parameter)\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)

# The model sometimes emits the tool call as a bare JSON array instead of the
# DSML markup, e.g.:
#   [{"name": "bash", "parameters": {"command": "pwd"}}]
# (the engine's own deepseekv4 parser leaves this unparsed too). The array may
# be wrapped in prose. Extract the first JSON array whose items carry a
# "name" and "parameters" object.
_JSON_ARRAY = re.compile(r"\[[\s\S]*?\]")

# The model also emits a plain XML tool-call block (observed live, 2026-10-02):
#   <tool_calls>
#     <invoke name="read">
#       <parameter name="filePath" string="true">C:/temp/foo.py</parameter>
#       <parameter name="offset" string="false">1</parameter>
#     </invoke>
#   </tool_calls>
# The optional ``string="true|false"`` attribute is the model's type hint; the
# value is additionally JSON-coerced so numbers stay numbers.
_XML_BLOCK_START = re.compile(r"<\s*([^>]*)tool_calls[^>]*>", re.IGNORECASE)
_XML_BLOCK_END = re.compile(r"<\s*/\s*([^>]*)tool_calls[^>]*>", re.IGNORECASE)
_XML_INVOKE = re.compile(
    r"<\s*([^>]*)invoke\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE
)
_XML_PARAM = re.compile(
    r"<\s*([^>]*)parameter\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>([\s\S]*?)"
    r"<\s*/\s*([^>]*)parameter[^>]*>",
    re.IGNORECASE,
)


def _find_json_array(text: str, start: int = 0) -> Optional[tuple[int, int]]:
    """Return (start, end) of the next balanced JSON array in ``text``,
    honoring string quotes and escapes (the naive non-greedy regex breaks on
    nested arrays such as a JSON-encoded ``todos`` string)."""
    i = text.find("[", start)
    while i != -1:
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < len(text):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return i, j + 1
            j += 1
        i = text.find("[", i + 1)
    return None


def _find_tag(text: str, pattern: re.Pattern[str], start: int = 0) -> Optional[re.Match[str]]:
    return pattern.search(text, start)


def _find_end_tag(text: str, start: int, name: str) -> Optional[re.Match[str]]:
    """Next DSML closing tag for ``name`` (parameter / invoke / tool_calls).
    The model emits both ``</DSML|parameter|>`` and ``<|DSML|/parameter|>``;
    accept any tag containing DSML + the name + a slash."""
    for m in _TAG.finditer(text, start):
        body = m.group(1)
        if "DSML" in body and name in body and "/" in body:
            return m
    return None


def _coerce_param_value(raw: Any, ptype: Optional[str] = None) -> Any:
    """Coerce one tool-call parameter value to its JSON schema type.

    ``raw`` is either the RAW text from the markup (DSML/XML) or an
    already-parsed JSON value (bare-array path). The model encodes structured
    values either directly (``[{...}]``) or as a JSON-encoded string
    (``"[{\"content\": ...}]"``), sometimes with HTML-escaped quotes
    (``&quot;``). ``ptype`` is the parameter's declared JSON schema type from
    the client's ``tools`` definition:

    - ``"string"``: ALWAYS a string. A JSON-looking value (file content that
      is itself JSON, e.g. the ``write`` tool's ``content``) must NOT be
      decoded into an object/array — the client schema rejects it. A
      structured value emitted for a string param is serialized back to its
      compact JSON string form.
    - ``"array"`` / ``"object"``: decode JSON-encoded strings to the real
      list/dict (e.g. todowrite's ``todos`` array).
    - ``"integer"`` / ``"number"`` / ``"boolean"``: coerce ``"1"`` -> 1,
      ``"true"`` -> True.
    - ``None`` (no schema known): legacy behavior — decode JSON-encoded
      strings to list/dict, plain strings stay strings.

    Order matters: parse the RAW text as JSON FIRST so backslash escapes
    inside a JSON-encoded string survive (unescaping first would turn
    ``\"`` into ``"`` and corrupt the inner JSON); unescape and retry only
    for HTML-escaped output; then parse string values as inner JSON.
    """
    if not isinstance(raw, str):
        # Already a parsed JSON value (bare-array path).
        if ptype == "string":
            return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        return raw
    text = raw.strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        unescaped = html.unescape(text)
        if unescaped == text:
            value = text
        else:
            try:
                value = json.loads(unescaped)
            except (ValueError, TypeError):
                return unescaped
    if ptype == "string":
        if isinstance(value, str):
            return html.unescape(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        inner = html.unescape(value).strip()
        try:
            inner_value = json.loads(inner)
        except (ValueError, TypeError):
            return inner
        if ptype in ("array", "object"):
            if isinstance(inner_value, (dict, list)):
                return inner_value
            return inner
        if ptype in ("integer", "number", "boolean"):
            if isinstance(inner_value, (int, float, bool)) or inner_value is None:
                return inner_value
            return inner
        # No schema: legacy behavior.
        if isinstance(inner_value, (dict, list)):
            return inner_value
        return inner
    return value


def _tool_param_types(tools: Any) -> dict[str, dict[str, str]]:
    """Extract ``{tool_name: {param_name: json_type}}`` from the client's
    ``tools`` definitions (OpenAI function schema). Empty when the request
    carries no usable schema — coercion then falls back to legacy behavior."""
    result: dict[str, dict[str, str]] = {}
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        parameters = fn.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        props = parameters.get("properties")
        if not isinstance(props, dict):
            continue
        types = {
            pname: prop.get("type")
            for pname, prop in props.items()
            if isinstance(prop, dict) and isinstance(prop.get("type"), str)
        }
        if types:
            result[name] = types
    return result


def _param_type(
    param_types: Optional[dict[str, dict[str, str]]],
    tool_name: str,
    param_name: str,
) -> Optional[str]:
    if param_types:
        types = param_types.get(tool_name)
        if types:
            return types.get(param_name)
    return None


def parse_deepseek_tool_calls(
    text: str, param_types: Optional[dict[str, dict[str, str]]] = None
) -> tuple[str, list[dict[str, Any]]]:
    """Extract DeepSeek tool-call markup from ``text``.

    ``param_types`` (``{tool_name: {param_name: json_type}}``) makes the
    parameter coercion schema-aware: string-typed params (e.g. the ``write``
    tool's ``content``, which is itself JSON text) stay strings instead of
    being decoded into objects.

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
            block_end = _find_end_tag(text, block.end(), "tool_calls")
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
                invoke_end_m = _find_end_tag(body, invoke.end(), "invoke")
                invoke_end = invoke_end_m.start() if invoke_end_m is not None else len(body)
                inner = body[invoke.end():invoke_end]

                params: dict[str, Any] = {}
                ppos = 0
                while True:
                    pm = _PARAM_START.search(inner, ppos)
                    if pm is None:
                        break
                    pname = pm.group(2).strip()
                    pend_m = _find_end_tag(inner, pm.end(), "parameter")
                    pend = pend_m.start() if pend_m is not None else len(inner)
                    value = _coerce_param_value(
                        inner[pm.end():pend], _param_type(param_types, name, pname)
                    )
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

    # 2) Plain XML tool-call block (no DSML wrapper).
    xml_block = _XML_BLOCK_START.search(text)
    if xml_block is not None:
        block_end = _XML_BLOCK_END.search(text, xml_block.end())
        body = (
            text[xml_block.end():block_end.start()]
            if block_end is not None
            else text[xml_block.end():]
        )
        xml_calls: list[dict[str, Any]] = []
        for invoke in _XML_INVOKE.finditer(body):
            name = invoke.group(2).strip()
            inner = body[invoke.end():]
            close = re.compile(
                r"<\s*/\s*([^>]*)invoke[^>]*>", re.IGNORECASE
            ).search(inner)
            if close is not None:
                inner = inner[: close.start()]
            params: dict[str, Any] = {}
            for pm in _XML_PARAM.finditer(inner):
                pname = pm.group(2).strip()
                pvalue = pm.group(3)
                params[pname] = _coerce_param_value(
                    pvalue, _param_type(param_types, name, pname)
                )
            if "cmd" in params and "command" not in params:
                params["command"] = params.pop("cmd")
            xml_calls.append(
                {
                    "id": f"call_{len(xml_calls) + 1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(params, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )
        if xml_calls:
            clean_text = (
                (text[: xml_block.start()] + text[xml_block.end() :]).strip()
                if block_end is not None
                else text[: xml_block.start()].strip()
            )
            return clean_text, xml_calls

    # 3) Bare JSON array fallback (model-native output, no DSML wrapper).
    search_from = 0
    while True:
        span = _find_json_array(text, search_from)
        if span is None:
            break
        m_start, m_end = span
        raw = text[m_start:m_end]
        try:
            items = json.loads(raw)
        except (ValueError, TypeError):
            search_from = m_start + 1
            continue
        if not isinstance(items, list) or not items:
            search_from = m_start + 1
            continue
        parsed: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                break
            name = item.get("name")
            parameters = item.get("parameters")
            if not isinstance(name, str) or not isinstance(parameters, dict):
                break
            # Coerce stringified values ("1" -> 1, "[{...}]" -> list) so the
            # client schema validation accepts the call.
            coerced = {
                k: _coerce_param_value(v, _param_type(param_types, name, k))
                for k, v in parameters.items()
            }
            if "cmd" in coerced and "command" not in coerced:
                coerced["command"] = coerced.pop("cmd")
            parsed.append(
                {
                    "id": f"call_{len(parsed) + 1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            coerced, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )
        if parsed:
            clean_text = (text[:m_start] + text[m_end:]).strip()
            return clean_text, parsed
        # Not a valid tool-call array (e.g. stringified ``parameters`` or
        # non-call objects): never re-loop on the same array — keep scanning.
        search_from = m_end + 1

    return text, []