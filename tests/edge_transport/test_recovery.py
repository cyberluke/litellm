"""Phase 3.5 recovery tests (LiteLLM edge transport): worker guard,
startup epoch reconciliation, pending/lost-ACK recovery via the store,
explicit RESYNC_FULL, idle cleanup + RELEASE + tombstone, reconcile helper.
No network (HTTPX mocked where the transport is exercised)."""

import asyncio
import json
import os
import tempfile
import time
import unittest
import uuid
from unittest import mock

import httpx

from differential_context.edge.canonical_request import CanonicalPromptState

from litellm.edge_transport import EdgeTransportConfig, assert_edge_worker_safety
from litellm.edge_transport.differential import (
    EdgeStateStore,
    LocalEdgeState,
    OP_RESYNC,
    build_edge_plan,
    build_release_plan,
    build_resync_plan,
    resync_register_full,
)
from litellm.edge_transport.state_backend import (
    MemoryEdgeStateBackend,
    PersistedEdgeContext,
)
from litellm.edge_transport.worker_guard import _EDGE_WORKER_ERROR

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
        state_persistence="memory",
        idle_ttl_seconds=3600,
        idle_cleanup_interval_seconds=3600,
    )
    base.update(overrides)
    return EdgeTransportConfig(**base)


def _caps(epoch="epoch-1"):
    return {
        "edge_server_epoch": epoch,
        "http": {"wan_versions": ["2"], "strict_http2_supported": True},
        "differential_context": {
            "edge_versions": [1],
            "edge_representations": ["openai_prompt_state"],
        },
    }


def _ack(context_id, generation, units, epoch="epoch-1"):
    prompt = CanonicalPromptState(model="m", messages=units)
    return {
        "representation": "openai_prompt_state",
        "context_id": context_id,
        "generation": generation,
        "state_hash": prompt.state_hash(),
        "protocol_version": 1,
        "edge_server_epoch": epoch,
    }


class TestWorkerGuard(unittest.TestCase):
    def setUp(self):
        from litellm.edge_transport import config as edge_config

        edge_config.reset_config()

    def test_disabled_profile_allows_workers(self):
        edge_config = _config(enabled=False)
        from litellm.edge_transport import config as cfg

        cfg.set_config(edge_config)
        assert_edge_worker_safety(8)  # must not raise

    def test_single_worker_allowed(self):
        edge_config = _config(enabled=True)
        from litellm.edge_transport import config as cfg

        cfg.set_config(edge_config)
        assert_edge_worker_safety(1)  # must not raise

    def test_multi_worker_fails_startup(self):
        edge_config = _config(enabled=True)
        from litellm.edge_transport import config as cfg

        cfg.set_config(edge_config)
        with self.assertRaises(SystemExit) as caught:
            assert_edge_worker_safety(2)
        self.assertIn("num_workers", str(caught.exception))
        self.assertIn("EDGE_STATE_SHARED_BACKEND", str(caught.exception))

    def test_multi_worker_with_shared_backend_allowed(self):
        edge_config = _config(enabled=True, shared_state_backend="redis://x")
        from litellm.edge_transport import config as cfg

        cfg.set_config(edge_config)
        assert_edge_worker_safety(4)  # must not raise

    def test_error_message_exposed(self):
        self.assertIn("--num_workers 1", _EDGE_WORKER_ERROR)


class TestResyncPlan(unittest.TestCase):
    def test_explicit_resync_envelope(self):
        plan = build_resync_plan(
            "ctx-r",
            {"model": "m", "messages": [SYS, U1, U2]},
            reason="server_epoch_changed",
            server_epoch="epoch-9",
            known_generation=3,
            known_state_hash="edge-v1:old",
        )
        self.assertEqual(plan.operation, OP_RESYNC)
        self.assertEqual(plan.envelope["context_mode"], "resync")
        self.assertEqual(plan.envelope["reason"], "server_epoch_changed")
        self.assertEqual(plan.envelope["server_epoch"], "epoch-9")
        self.assertEqual(plan.envelope["known_generation"], 3)
        self.assertTrue(plan.envelope["resync_nonce"])
        self.assertTrue(plan.edge_request_id)
        self.assertEqual(plan.predicted_next_generation, 4)
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1, U2])
        self.assertEqual(plan.predicted_state_hash, prompt.state_hash())

    def test_request_id_stable_and_not_context_derived(self):
        plan_a = build_resync_plan(
            "ctx-a", {"model": "m", "messages": [SYS]}, reason="operator_requested", server_epoch="e"
        )
        plan_b = build_resync_plan(
            "ctx-b", {"model": "m", "messages": [SYS]}, reason="operator_requested", server_epoch="e"
        )
        self.assertNotEqual(plan_a.edge_request_id, plan_b.edge_request_id)
        self.assertEqual(str(uuid.UUID(plan_a.edge_request_id)), plan_a.edge_request_id)

    def test_release_plan(self):
        plan = build_release_plan("ctx-c", 5)
        self.assertEqual(plan.operation, "release")
        self.assertEqual(plan.envelope["operations"], [{"op": "release"}])
        self.assertEqual(plan.envelope["base_generation"], 5)


class TestStoreRecovery(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.backend = MemoryEdgeStateBackend()
        self.store = EdgeStateStore(self.backend)

    def tearDown(self):
        self.loop.run_until_complete(self.store.close())
        self.loop.close()

    def _seed(self, context_id, epoch="epoch-1", generation=1, units=None):
        units = units or [SYS, U1]
        prompt = CanonicalPromptState(model="m", messages=units)
        row = PersistedEdgeContext(
            context_id=context_id,
            server_epoch=epoch,
            acknowledged_generation=generation,
            acknowledged_state_hash=prompt.state_hash(),
            canonical_prompt_state=prompt.canonical_dict(),
            canonical_prompt_hash=prompt.state_hash(),
            updated_at=time.time(),
            last_seen_at=time.time(),
        )
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))

    def test_startup_matching_epoch_resumable(self):
        self._seed("ctx-ok", epoch="epoch-1")
        self.store.set_server_epoch("epoch-1")
        self.loop.run_until_complete(self.store.open())
        self.assertIsNotNone(self.store.get("ctx-ok"))
        self.assertFalse(self.store.is_invalid("ctx-ok"))

    def test_startup_epoch_mismatch_requires_reregistration(self):
        self._seed("ctx-stale", epoch="epoch-1")
        self.store.set_server_epoch("epoch-2")
        self.loop.run_until_complete(self.store.open())
        self.assertIsNotNone(self.store.get("ctx-stale"))
        self.assertTrue(self.store.is_invalid("ctx-stale"))

    def test_startup_preserves_pending_for_lost_ack(self):
        self._seed("ctx-p", epoch="epoch-1", generation=1)
        self.store.set_server_epoch("epoch-1")
        self.loop.run_until_complete(self.store.open())
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1, U2])
        plan = build_edge_plan("ctx-p", {"model": "m", "messages": [SYS, U1, U2]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-p", plan))

        # simulate restart: fresh store over the same backend
        store2 = EdgeStateStore(self.backend)
        store2.set_server_epoch("epoch-1")
        self.loop.run_until_complete(store2.open())
        pending = store2.pending_for("ctx-p")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.next_generation, 2)
        self.assertTrue(
            store2.pending_matches("ctx-p", 2, prompt.state_hash(), "epoch-1")
        )
        self.loop.run_until_complete(store2.close())

    def test_pending_matches_requires_epoch(self):
        self.store.set_server_epoch("epoch-1")
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        plan = build_edge_plan("ctx-m", {"model": "m", "messages": [SYS, U1]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-m", plan))
        self.assertTrue(self.store.pending_matches("ctx-m", 1, prompt.state_hash(), "epoch-1"))
        self.assertFalse(self.store.pending_matches("ctx-m", 1, prompt.state_hash(), "epoch-2"))
        self.assertFalse(self.store.pending_matches("ctx-m", 2, prompt.state_hash(), "epoch-1"))

    def test_promote_pending_durable(self):
        self.store.set_server_epoch("epoch-1")
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        plan = build_edge_plan("ctx-d", {"model": "m", "messages": [SYS, U1]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-d", plan))
        state = self.loop.run_until_complete(
            self.store.promote_pending(
                "ctx-d", ack_generation=1, ack_state_hash=prompt.state_hash(), epoch="epoch-1"
            )
        )
        self.assertEqual(state.generation, 1)
        rows = self.loop.run_until_complete(self.store.rows())
        self.assertEqual(rows[0].acknowledged_generation, 1)
        self.assertFalse(rows[0].has_pending)

    def test_commit_ack_stores_epoch_and_marks_invalid_on_change(self):
        self.store.set_server_epoch("epoch-1")
        plan = build_edge_plan("ctx-e", {"model": "m", "messages": [SYS, U1]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-e", plan))
        ack = _ack("ctx-e", 1, [SYS, U1], epoch="epoch-1")
        state = self.loop.run_until_complete(
            self.store.commit_ack("ctx-e", ack, [SYS, U1], model="m")
        )
        self.assertEqual(state.server_epoch, "epoch-1")
        self.assertEqual(self.store.epoch_for("ctx-e"), "epoch-1")
        self.assertFalse(self.store.is_invalid("ctx-e"))

    def test_commit_ack_from_new_epoch_marks_invalid(self):
        self.store.set_server_epoch("epoch-1")
        plan = build_edge_plan("ctx-f", {"model": "m", "messages": [SYS, U1]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-f", plan))
        ack = _ack("ctx-f", 1, [SYS, U1], epoch="epoch-2")
        state = self.loop.run_until_complete(
            self.store.commit_ack("ctx-f", ack, [SYS, U1], model="m")
        )
        # the ACK is authoritative for its own commit, but the binding is
        # invalid: the remote process restarted mid-flight
        self.assertEqual(state.server_epoch, "epoch-2")
        self.assertTrue(self.store.is_invalid("ctx-f"))

    def test_delete_context_removes_durable_row(self):
        self.store.set_server_epoch("epoch-1")
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        plan = build_edge_plan("ctx-g", {"model": "m", "messages": [SYS, U1]}, self.store)
        self.loop.run_until_complete(self.store.persist_pending("ctx-g", plan))
        self.loop.run_until_complete(self.store.delete_context("ctx-g"))
        rows = self.loop.run_until_complete(self.store.rows())
        self.assertEqual(len(rows), 0)


class TestIdleCleanup(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_list_idle_by_last_seen(self):
        backend = MemoryEdgeStateBackend()
        now = time.time()
        backend.save_pending(
            PersistedEdgeContext(
                context_id="idle",
                server_epoch="epoch-1",
                acknowledged_generation=1,
                acknowledged_state_hash="edge-v1:x",
                updated_at=now - 10_000,
                last_seen_at=now - 10_000,
            )
        )
        backend.save_pending(
            PersistedEdgeContext(
                context_id="fresh",
                server_epoch="epoch-1",
                acknowledged_generation=1,
                acknowledged_state_hash="edge-v1:y",
                updated_at=now,
                last_seen_at=now,
            )
        )
        store = EdgeStateStore(backend)
        idle = self.loop.run_until_complete(store.list_idle(3600))
        self.assertEqual(idle, ["idle"])
        self.loop.run_until_complete(store.close())

    def test_tombstone_and_retry_flow(self):
        backend = MemoryEdgeStateBackend()
        now = time.time()
        backend.save_pending(
            PersistedEdgeContext(
                context_id="t",
                server_epoch="epoch-1",
                acknowledged_generation=1,
                acknowledged_state_hash="edge-v1:x",
                updated_at=now,
                last_seen_at=now - 10_000,
            )
        )
        store = EdgeStateStore(backend)
        self.loop.run_until_complete(store.mark_tombstoned("t"))
        rows = self.loop.run_until_complete(store.rows())
        self.assertTrue(rows[0].tombstoned)
        # tombstoned contexts are never re-listed for release
        idle = self.loop.run_until_complete(store.list_idle(3600))
        self.assertEqual(idle, [])
        self.loop.run_until_complete(store.close())


class TestTransportReleaseAndReconcile(unittest.TestCase):
    """Release + reconcile via the transport with a mocked HTTPX client."""

    def _transport(self, handler):
        from litellm.edge_transport.transport import EdgeTransport

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return EdgeTransport(_config(), client=client)

    def test_release_confirmed_deletes_state(self):
        from litellm.edge_transport.compression import decompress_body

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if request.url.path == "/v1/transport/capabilities":
                return httpx.Response(200, json=_caps(), request=request)
            body = json.loads(
                decompress_body(request.content, request.headers.get("content-encoding"))
            )
            self.assertEqual(body["operations"], [{"op": "release"}])
            return httpx.Response(
                200,
                json={
                    "differential_context": {
                        "representation": "openai_prompt_state",
                        "context_id": "ctx-rel",
                        "generation": 2,
                        "state_hash": "edge-v1:x",
                        "released": True,
                        "edge_server_epoch": "epoch-1",
                        "edge_request_id": "req-r",
                    }
                },
                request=request,
            )

        transport = self._transport(handler)
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        transport.store.set(
            LocalEdgeState("ctx-rel", 2, prompt.state_hash(), prompt, server_epoch="epoch-1")
        )
        transport.store.set_server_epoch("epoch-1")

        async def run():
            return await transport.release_context("ctx-rel", 2)

        confirmed = asyncio.run(run())
        self.assertTrue(confirmed)
        self.assertIsNone(transport.store.get("ctx-rel"))

    def test_release_conflict_tombstones_via_caller(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/transport/capabilities":
                return httpx.Response(200, json=_caps(), request=request)
            return httpx.Response(
                409,
                json={
                    "error": {"message": "gen", "type": "edge_conflict", "code": 409},
                    "authoritative_generation": 3,
                    "edge_server_epoch": "epoch-1",
                },
                request=request,
            )

        transport = self._transport(handler)
        prompt = CanonicalPromptState(model="m", messages=[SYS, U1])
        transport.store.set(
            LocalEdgeState("ctx-c", 2, prompt.state_hash(), prompt, server_epoch="epoch-1")
        )
        transport.store.set_server_epoch("epoch-1")

        async def run():
            try:
                await transport.release_context("ctx-c", 2)
                return False
            except Exception:
                return True

        self.assertTrue(asyncio.run(run()))

    def test_reconcile_roundtrip(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/transport/capabilities":
                return httpx.Response(200, json=_caps(), request=request)
            body = json.loads(request.content)
            self.assertEqual(body["context_id"], "ctx-q")
            return httpx.Response(
                200,
                json={
                    "server_epoch": "epoch-1",
                    "context_exists": True,
                    "authoritative_generation": 3,
                    "authoritative_state_hash": "edge-v1:h",
                    "request_commit_status": "committed",
                },
                request=request,
            )

        transport = self._transport(handler)

        async def run():
            return await transport.reconcile(
                "ctx-q",
                edge_request_id="req-1",
                known_generation=3,
                known_state_hash="edge-v1:h",
            )

        doc = asyncio.run(run())
        self.assertEqual(doc["request_commit_status"], "committed")
        self.assertEqual(doc["authoritative_generation"], 3)


if __name__ == "__main__":
    unittest.main()