"""Unit tests for litellm.edge_transport (Phase 3 §46 steps 10-13) — no
network, no engine. Protocol behavior (diff rules, envelope shape, ACK
validation) is exercised through the shared ``differential-context``
package; the HTTPX client is mocked so the strict HTTP/2 proof stays for
the local Caddy integration test.

Run:  python -m pytest tests/edge_transport -q
(with PYTHONPATH = <litellm repo root>;<differential-context src>)
"""

import asyncio
import json
import unittest
from unittest import mock

import httpx
import zstandard

from differential_context.common.errors import DifferentialContextError
from differential_context.edge.ack import build_edge_ack_sse, ACK_EVENT_NAME
from differential_context.edge.canonical_request import CanonicalPromptState

from litellm.edge_transport import EdgeTransportConfig
from litellm.edge_transport.capabilities import CapabilitiesError, validate_capabilities
from litellm.edge_transport.compression import decompress_body, maybe_compress
from litellm.edge_transport.differential import (
    EdgeStateStore,
    LocalEdgeState,
    OP_APPEND,
    OP_REGISTER,
    apply_edge_ack,
    build_edge_plan,
    resync_register_full,
)
from litellm.edge_transport.session_resolver import (
    DISABLED_NO_STABLE_IDENTITY,
    resolve_session,
)
from litellm.edge_transport.transport import EdgeTransport, EdgeTransportError

SYS = {"role": "system", "content": "sys"}
U1 = {"role": "user", "content": "hello"}
A1 = {"role": "assistant", "content": "hi"}
U2 = {"role": "user", "content": "more"}


def _config(**overrides) -> EdgeTransportConfig:
    base = dict(
        enabled=True,
        base_url="https://edge.test",
        api_key="k",
        strict_http2=False,
        request_compression="zstd",
        compression_threshold=0,
        # Phase 3.5: never touch a real SQLite file in unit tests.
        state_persistence="memory",
    )
    base.update(overrides)
    return EdgeTransportConfig(**base)


def _caps(epoch="epoch-1") -> dict:
    return {
        "edge_server_epoch": epoch,
        "http": {"wan_versions": ["2"], "strict_http2_supported": True, "termination": "caddy"},
        "differential_context": {
            "edge_versions": [1],
            "edge_representations": ["openai_prompt_state"],
        },
    }


def _with_caps(handler):
    """Route GET /v1/transport/capabilities before the edge handler."""

    def routed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/transport/capabilities":
            return httpx.Response(200, json=_caps(), request=request)
        return handler(request)

    return routed


def _payload(messages, **extra):
    payload = {"model": "differential_sseproxy", "messages": messages}
    payload.update(extra)
    return payload


class TestSessionResolver(unittest.TestCase):
    def test_configured_header_wins(self):
        cfg = _config(session_headers=("x-dc-session-id",))
        res = resolve_session({"x-dc-session-id": "ctx-1"}, None, cfg)
        self.assertEqual(res.context_id, "ctx-1")
        self.assertIsNone(res.disabled_reason)
        self.assertEqual(res.source, "header:x-dc-session-id")

    def test_body_keys_priority(self):
        cfg = _config(session_body_keys=("session_id", "metadata.session_id"))
        res = resolve_session(None, {"session_id": "ctx-b"}, cfg)
        self.assertEqual(res.context_id, "ctx-b")

    def test_metadata_session_id(self):
        cfg = _config(session_body_keys=("session_id", "metadata.session_id"))
        res = resolve_session(None, {"metadata": {"session_id": "ctx-m"}}, cfg)
        self.assertEqual(res.context_id, "ctx-m")

    def test_adapter_headers(self):
        cfg = _config(adapter_session_headers=("x-kilo-session-id", "x-cline-session-id"))
        res = resolve_session({"x-cline-session-id": "ctx-c"}, None, cfg)
        self.assertEqual(res.context_id, "ctx-c")
        self.assertEqual(res.source, "adapter:x-cline-session-id")

    def test_no_stable_identity_disables(self):
        cfg = _config()
        res = resolve_session({}, {}, cfg)
        self.assertIsNone(res.context_id)
        self.assertEqual(res.disabled_reason, DISABLED_NO_STABLE_IDENTITY)

    def test_empty_or_oversized_rejected(self):
        cfg = _config()
        res = resolve_session({"x-dc-session-id": "   "}, None, cfg)
        self.assertIsNone(res.context_id)
        res = resolve_session({"x-dc-session-id": "x" * 200}, None, cfg)
        self.assertIsNone(res.context_id)


class TestBuildEdgePlan(unittest.TestCase):
    def setUp(self):
        self.store = EdgeStateStore()

    def test_first_request_register_full(self):
        plan = build_edge_plan("ctx-a", _payload([SYS, U1]), self.store)
        self.assertEqual(plan.operation, OP_REGISTER)
        self.assertTrue(plan.requires_full)
        self.assertEqual(plan.envelope["context_mode"], "independent")
        self.assertIsNone(plan.base_generation)

    def test_resume_appends_only_delta(self):
        first = build_edge_plan("ctx-a", _payload([SYS, U1]), self.store)
        apply_edge_ack(
            self.store, "ctx-a", _ack("ctx-a", 1, first.envelope["prompt_state"]["messages"]),
            [SYS, U1], model="m",
        )
        plan = build_edge_plan("ctx-a", _payload([SYS, U1, A1, U2]), self.store)
        self.assertEqual(plan.operation, OP_APPEND)
        self.assertEqual(plan.delta_units, [A1, U2])
        self.assertEqual(plan.base_generation, 1)
        body = json.loads(plan.envelope_bytes)
        self.assertEqual(body["operations"][0]["units"], [A1, U2])

    def test_replace_tail_on_edit(self):
        first = build_edge_plan("ctx-a", _payload([SYS, U1]), self.store)
        apply_edge_ack(
            self.store, "ctx-a", _ack("ctx-a", 1, first.envelope["prompt_state"]["messages"]),
            [SYS, U1], model="m",
        )
        edited = [SYS, {"role": "user", "content": "changed"}]
        plan = build_edge_plan("ctx-a", _payload(edited), self.store)
        self.assertEqual(plan.operation, "replace_tail")
        self.assertEqual(plan.envelope["operations"][0]["keep"], 1)

    def test_noop_resumes_with_empty_append(self):
        first = build_edge_plan("ctx-a", _payload([SYS, U1]), self.store)
        apply_edge_ack(
            self.store, "ctx-a", _ack("ctx-a", 1, first.envelope["prompt_state"]["messages"]),
            [SYS, U1], model="m",
        )
        plan = build_edge_plan("ctx-a", _payload([SYS, U1]), self.store)
        self.assertEqual(plan.operation, OP_APPEND)
        self.assertTrue(plan.is_noop)
        self.assertEqual(plan.envelope["operations"][0]["units"], [])
        self.assertEqual(plan.envelope["context_mode"], "resume")

    def test_explicit_fork(self):
        first = build_edge_plan("parent-P", _payload([SYS, U1]), self.store)
        apply_edge_ack(
            self.store, "parent-P", _ack("parent-P", 1, first.envelope["prompt_state"]["messages"]),
            [SYS, U1], model="m",
        )
        parent = self.store.get("parent-P")
        plan = build_edge_plan(
            "branch-B", _payload([SYS, U1, U2]), self.store,
            fork_source=("parent-P", parent.generation, parent.state_hash),
        )
        self.assertEqual(plan.operation, "fork")
        self.assertEqual(plan.envelope["fork"]["source_context_id"], "parent-P")

    def test_resync_register_full(self):
        plan = resync_register_full("ctx-a", _payload([SYS, U1, U2]))
        self.assertEqual(plan.operation, OP_REGISTER)
        self.assertTrue(plan.requires_full)


def _ack(context_id, generation, units, state_hash=None, edge_server_epoch="epoch-1"):
    from differential_context.edge.canonical_request import CanonicalPromptState

    prompt = CanonicalPromptState(model="m", messages=units)
    return {
        "representation": "openai_prompt_state",
        "context_id": context_id,
        "generation": generation,
        "state_hash": state_hash or prompt.state_hash(),
        "unit_count": len(units),
        "protocol_version": 1,
        "edge_server_epoch": edge_server_epoch,
    }


class TestApplyEdgeAck(unittest.TestCase):
    def test_commit_advances_local_state(self):
        store = EdgeStateStore()
        state = apply_edge_ack(store, "ctx-a", _ack("ctx-a", 2, [SYS, U1, A1]), [SYS, U1, A1], model="m")
        self.assertEqual(store.get("ctx-a").generation, 2)
        self.assertTrue(store.get("ctx-a").state_hash.startswith("edge-v1:"))

    def test_wrong_context_rejected(self):
        store = EdgeStateStore()
        with self.assertRaises(DifferentialContextError):
            apply_edge_ack(store, "ctx-a", _ack("ctx-b", 2, [SYS, U1]), [SYS, U1], model="m")


class TestCompression(unittest.TestCase):
    BIG = json.dumps({"messages": [{"role": "user", "content": "x" * 5000}]}).encode()

    def test_zstd_roundtrip_and_threshold(self):
        body, enc, _ = maybe_compress(self.BIG, "zstd", 0)
        self.assertEqual(enc, "zstd")
        self.assertEqual(decompress_body(body, "zstd"), self.BIG)

    def test_gzip_roundtrip(self):
        body, enc, _ = maybe_compress(self.BIG, "gzip", 0)
        self.assertEqual(enc, "gzip")
        self.assertEqual(decompress_body(body, "gzip"), self.BIG)

    def test_below_threshold_identity(self):
        body, enc, _ = maybe_compress(self.BIG, "zstd", 10**9)
        self.assertEqual(enc, "identity")
        self.assertIs(body, self.BIG)

    def test_unsupported_response_encoding(self):
        from litellm.edge_transport.compression import CompressionError

        with self.assertRaises(CompressionError):
            decompress_body(b"x", "br")


class TestCapabilitiesValidation(unittest.TestCase):
    def test_valid(self):
        caps = {
            "edge_server_epoch": "epoch-1",
            "http": {"wan_versions": ["2"], "strict_http2_supported": True},
            "differential_context": {"edge_versions": [1], "edge_representations": ["openai_prompt_state"]},
        }
        validate_capabilities(caps, _config(strict_http2=True))

    def test_missing_epoch_rejected(self):
        caps = {
            "http": {"wan_versions": ["1.1"], "strict_http2_supported": False},
            "differential_context": {"edge_versions": [1], "edge_representations": ["openai_prompt_state"]},
        }
        with self.assertRaises(CapabilitiesError):
            validate_capabilities(caps, _config(strict_http2=False))

    def test_missing_edge_version(self):
        caps = {"edge_server_epoch": "e", "differential_context": {"edge_versions": [2], "edge_representations": ["openai_prompt_state"]}}
        with self.assertRaises(CapabilitiesError):
            validate_capabilities(caps, _config(strict_http2=False))

    def test_strict_http2_requires_wan_claim(self):
        caps = {
            "edge_server_epoch": "e",
            "http": {"wan_versions": ["1.1"], "strict_http2_supported": False},
            "differential_context": {"edge_versions": [1], "edge_representations": ["openai_prompt_state"]},
        }
        with self.assertRaises(CapabilitiesError):
            validate_capabilities(caps, _config(strict_http2=True))


def _sse_chunks(*events):
    # SSE events are newline-terminated; the parser needs the boundaries.
    return b"".join(
        (e if isinstance(e, bytes) else e.encode()) + b"\n" for e in events
    )


class TestTransportBehavior(unittest.TestCase):
    """Mocked HTTPX transport: ACK stripping, 409 resync, disabled path."""

    def _transport(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return EdgeTransport(_config(strict_http2=False), client=client)

    def test_stream_acks_stripped_state_committed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/differential-context/chat/completions")
            self.assertIn("Content-Encoding", request.headers)
            body = decompress_body(request.content, request.headers.get("content-encoding"))
            envelope = json.loads(body)
            self.assertEqual(envelope["context_id"], "ctx-s")
            chunks = [
                'data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"a"}}]}',
                build_edge_ack_sse(_ack("ctx-s", 1, [SYS, U1])).decode(),
                'data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"b"}}]}',
                "data: [DONE]",
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_chunks(*chunks),
                request=request,
            )

        transport = self._transport(_with_caps(handler))
        meta_holder = {}

        async def run():
            stream, meta = await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1], stream=True,
                headers={"x-dc-session-id": "ctx-s"},
            )
            meta_holder["meta"] = meta
            chunks = [chunk async for chunk in stream]
            return chunks

        chunks = asyncio.run(run())
        meta = meta_holder["meta"]
        contents = [c["choices"][0]["delta"]["content"] for c in chunks]
        self.assertEqual(contents, ["a", "b"])
        # ACK consumed: local state advanced, no control frame visible.
        self.assertEqual(transport.store.get("ctx-s").generation, 1)
        self.assertEqual(transport.store.get("ctx-s").server_epoch, "epoch-1")
        self.assertEqual(meta.edge_generation, 1)
        self.assertEqual(meta.operation, OP_REGISTER)

    def test_409_matching_pending_lost_ack_recovery(self):
        """§8: authoritative identity == durable pending -> promote locally
        (no replay of the prior inference) and re-plan the CURRENT request
        against the promoted baseline."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            body = decompress_body(request.content, request.headers.get("content-encoding"))
            envelope = json.loads(body)
            if envelope.get("base_generation") == 1:
                # the stale append -> authoritative gen 2 == pending prediction
                predicted = CanonicalPromptState(model="m", messages=[SYS, U1, A1])
                return httpx.Response(
                    409,
                    json={
                        "error": {"message": "stale", "type": "edge_conflict", "code": 409},
                        "authoritative_generation": 2,
                        "authoritative_state_hash": predicted.state_hash(),
                        "edge_server_epoch": "epoch-1",
                    },
                    request=request,
                )
            return httpx.Response(
                200, json={"id": "r", "object": "chat.completion", "created": 1,
                           "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
                request=request,
            )

        transport = self._transport(_with_caps(handler))
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        transport.store.set(LocalEdgeState("ctx-r", 1, prompt.state_hash(), prompt, server_epoch="epoch-1"))
        transport.store.set_server_epoch("epoch-1")

        async def run():
            doc, meta = await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1, A1], stream=False,
                headers={"x-dc-session-id": "ctx-r"},
            )
            return doc, meta

        doc, meta = asyncio.run(run())
        self.assertEqual(doc["choices"][0]["message"]["content"], "ok")
        # two wire calls: the stale append (409) + the replanned noop append (200)
        self.assertEqual(len(calls), 2)
        second = json.loads(decompress_body(calls[1].content, calls[1].headers.get("content-encoding")))
        self.assertEqual(second["context_mode"], "resume")
        self.assertEqual(second["base_generation"], 2)
        self.assertEqual(second["operations"][0]["units"], [])
        self.assertTrue(meta.lost_ack_recovered)
        # the promoted baseline is authoritative now
        self.assertEqual(transport.store.get("ctx-r").generation, 2)

    def test_409_not_matching_pending_unrecoverable(self):
        """§9: authoritative state matches neither acknowledged nor pending
        -> raise UnrecoverableEdgeConflict, never an invisible retry."""
        from litellm.edge_transport.transport import UnrecoverableEdgeConflict

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                409,
                json={
                    "error": {"message": "stale", "type": "edge_conflict", "code": 409},
                    "authoritative_generation": 2,
                    "authoritative_state_hash": "edge-v1:totally-different",
                    "edge_server_epoch": "epoch-1",
                },
                request=request,
            )

        transport = self._transport(_with_caps(handler))
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        transport.store.set(LocalEdgeState("ctx-u", 1, prompt.state_hash(), prompt, server_epoch="epoch-1"))
        transport.store.set_server_epoch("epoch-1")

        async def run():
            await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1, A1], stream=False,
                headers={"x-dc-session-id": "ctx-u"},
            )

        with self.assertRaises(UnrecoverableEdgeConflict):
            asyncio.run(run())
        # exactly one wire call: no hidden resync, no retry.
        self.assertEqual(len(calls), 1)

    def test_409_epoch_change_resyncs_explicitly(self):
        """§7: a 409 carrying a NEW epoch (stale capabilities cache) marks
        the binding invalid and the request is re-sent as an explicit
        RESYNC_FULL against freshly fetched capabilities."""
        calls = []
        state = {"epoch": "epoch-1"}

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if request.url.path == "/v1/transport/capabilities":
                return httpx.Response(200, json=_caps(epoch=state["epoch"]), request=request)
            body = decompress_body(request.content, request.headers.get("content-encoding"))
            envelope = json.loads(body)
            if envelope.get("context_mode") == "resume":
                # the remote restarted between the caps fetch and this send
                state["epoch"] = "epoch-2"
                return httpx.Response(
                    409,
                    json={
                        "error": {"message": "stale", "type": "edge_conflict", "code": 409},
                        "authoritative_generation": 2,
                        "authoritative_state_hash": "edge-v1:x",
                        "edge_server_epoch": "epoch-2",
                    },
                    request=request,
                )
            return httpx.Response(
                200, json={"id": "r", "object": "chat.completion", "created": 1,
                           "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
                request=request,
            )

        transport = self._transport(handler)
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        transport.store.set(LocalEdgeState("ctx-e", 1, prompt.state_hash(), prompt, server_epoch="epoch-1"))

        async def run():
            doc, meta = await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1, A1], stream=False,
                headers={"x-dc-session-id": "ctx-e"},
            )
            return doc, meta

        doc, meta = asyncio.run(run())
        self.assertEqual(doc["choices"][0]["message"]["content"], "ok")
        # caps + stale resume (409) + refreshed caps + explicit resync
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0].url.path, "/v1/transport/capabilities")
        self.assertEqual(calls[2].url.path, "/v1/transport/capabilities")
        resync = json.loads(decompress_body(calls[3].content, calls[3].headers.get("content-encoding")))
        self.assertEqual(resync["context_mode"], "resync")
        self.assertEqual(resync["reason"], "server_epoch_changed")
        self.assertEqual(resync["server_epoch"], "epoch-2")
        self.assertEqual(meta.resync_reason, "server_epoch_changed")

    def test_no_stable_identity_disabled_path(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(
                200,
                json={"id": "r", "object": "chat.completion", "created": 1, "model": "m",
                      "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
                request=request,
            )

        transport = self._transport(handler)
        from litellm.edge_transport import metrics as em

        async def run():
            doc, meta = await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1], stream=False,
                headers={},
            )
            return doc, meta

        doc, meta = asyncio.run(run())
        self.assertEqual(meta.disabled_reason, DISABLED_NO_STABLE_IDENTITY)
        self.assertEqual(doc["choices"][0]["message"]["content"], "ok")

    def test_non_stream_ack_removed_from_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = decompress_body(request.content, request.headers.get("content-encoding"))
            envelope = json.loads(body)
            resp = {
                "id": "r", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "differential_context": _ack(envelope["context_id"], 1, [SYS, U1]),
            }
            return httpx.Response(200, json=resp, request=request)

        transport = self._transport(_with_caps(handler))

        async def run():
            doc, meta = await transport.acompletion(
                model="differential_sseproxy", messages=[SYS, U1], stream=False,
                headers={"x-dc-session-id": "ctx-n"},
            )
            return doc, meta

        doc, meta = asyncio.run(run())
        self.assertNotIn("differential_context", doc)
        self.assertEqual(transport.store.get("ctx-n").generation, 1)
        self.assertEqual(transport.store.get("ctx-n").server_epoch, "epoch-1")


if __name__ == "__main__":
    unittest.main()