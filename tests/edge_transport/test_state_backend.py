"""Phase 3.5 §3/§4/§5: durable EdgeStateBackend tests (SQLite WAL + memory):
versioned schema, one row per context, pending-before-send persistence,
atomic promote transaction, tombstone, delete. No network."""

import asyncio
import os
import tempfile
import time
import unittest

from differential_context.edge.canonical_request import CanonicalPromptState

from litellm.edge_transport.state_backend import (
    MemoryEdgeStateBackend,
    PersistedEdgeContext,
    SqliteEdgeStateBackend,
    open_backend,
)
from litellm.edge_transport.config import EdgeTransportConfig


def _context(context_id="ctx-a", epoch="epoch-1", generation=2):
    prompt = CanonicalPromptState(model="m", messages=[{"role": "user", "content": "hi"}])
    return PersistedEdgeContext(
        context_id=context_id,
        server_epoch=epoch,
        acknowledged_generation=generation,
        acknowledged_state_hash=prompt.state_hash(),
        canonical_prompt_state=prompt.canonical_dict(),
        canonical_prompt_hash=prompt.state_hash(),
        updated_at=time.time(),
        last_seen_at=time.time(),
    )


def _with_pending(context, request_id="req-1"):
    prompt = CanonicalPromptState(
        model="m", messages=[{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]
    )
    context.pending_request_id = request_id
    context.pending_base_generation = context.acknowledged_generation
    context.pending_base_state_hash = context.acknowledged_state_hash
    context.pending_next_generation = (context.acknowledged_generation or 0) + 1
    context.pending_predicted_state_hash = prompt.state_hash()
    context.pending_canonical_prompt_state = prompt.canonical_dict()
    context.pending_operation_type = "append"
    context.pending_created_at = time.time()
    return context


class _BackendCase:
    """Shared behavior for both backends (plain mixin — the concrete
    subclasses are the unittest.TestCase classes pytest collects)."""

    def make_backend(self):
        raise NotImplementedError

    def setUp(self):
        self.backend = self.make_backend()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.run_until_complete(asyncio.to_thread(self.backend.close))
        self.loop.close()

    def test_save_and_load_pending(self):
        row = _with_pending(_context())
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].has_pending)
        self.assertEqual(loaded[0].pending_request_id, "req-1")
        self.assertEqual(loaded[0].pending_next_generation, 3)
        self.assertEqual(loaded[0].acknowledged_generation, 2)

    def test_promote_pending_atomic(self):
        row = _with_pending(_context())
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        self.loop.run_until_complete(
            asyncio.to_thread(
                self.backend.promote_pending,
                "ctx-a",
                epoch="epoch-1",
                updated_at=time.time(),
            )
        )
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(loaded[0].acknowledged_generation, 3)  # pending promoted
        self.assertFalse(loaded[0].has_pending)  # pending cleared
        self.assertIsNotNone(loaded[0].canonical_prompt_state)

    def test_new_pending_retains_previous_as_last(self):
        # §8: a second unconfirmed send must not destroy the first one —
        # the previous pending is retained durably for lost-ACK matching.
        row = _with_pending(_context(), request_id="req-1")
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        newer = _with_pending(_context(), request_id="req-2")
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, newer))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(loaded[0].pending_request_id, "req-2")
        self.assertEqual(loaded[0].last_pending_request_id, "req-1")
        self.assertEqual(loaded[0].last_pending_next_generation, 3)
        # promote clears BOTH slots
        self.loop.run_until_complete(
            asyncio.to_thread(
                self.backend.promote_pending,
                "ctx-a",
                ack_generation=4,
                ack_state_hash="edge-v1:ack",
                epoch="epoch-1",
                updated_at=time.time(),
            )
        )
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertFalse(loaded[0].has_pending)
        self.assertFalse(loaded[0].has_last_pending)
        self.assertEqual(loaded[0].acknowledged_generation, 4)

    def test_promote_with_ack_overrides_pending(self):
        row = _with_pending(_context())
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        self.loop.run_until_complete(
            asyncio.to_thread(
                self.backend.promote_pending,
                "ctx-a",
                ack_generation=7,
                ack_state_hash="edge-v1:ack",
                epoch="epoch-1",
                updated_at=time.time(),
            )
        )
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(loaded[0].acknowledged_generation, 7)
        self.assertEqual(loaded[0].acknowledged_state_hash, "edge-v1:ack")
        self.assertFalse(loaded[0].has_pending)

    def test_promote_without_row_creates_acknowledged(self):
        self.loop.run_until_complete(
            asyncio.to_thread(
                self.backend.promote_pending,
                "ctx-new",
                ack_generation=1,
                ack_state_hash="edge-v1:first",
                epoch="epoch-9",
                updated_at=time.time(),
            )
        )
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(loaded[0].acknowledged_generation, 1)
        self.assertEqual(loaded[0].server_epoch, "epoch-9")

    def test_one_row_per_context(self):
        a = _context("ctx-a")
        b = _context("ctx-b")
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, a))
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, b))
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, a))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(len(loaded), 2)

    def test_tombstone_and_delete(self):
        row = _context()
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        self.loop.run_until_complete(asyncio.to_thread(self.backend.mark_tombstoned, "ctx-a"))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertTrue(loaded[0].tombstoned)
        self.loop.run_until_complete(asyncio.to_thread(self.backend.delete, "ctx-a"))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(len(loaded), 0)

    def test_touch_refreshes_last_seen(self):
        row = _context()
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        later = time.time() + 100
        self.loop.run_until_complete(asyncio.to_thread(self.backend.touch, "ctx-a", later))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        self.assertEqual(loaded[0].last_seen_at, later)

    def test_never_persists_secrets(self):
        # No column exists for keys/headers; a persisted row carries only the
        # documented fields.
        row = _with_pending(_context())
        self.loop.run_until_complete(asyncio.to_thread(self.backend.save_pending, row))
        loaded = self.loop.run_until_complete(asyncio.to_thread(self.backend.load_all))
        doc = {f: getattr(loaded[0], f) for f in loaded[0].__dataclass_fields__}
        for secret in ("api_key", "authorization", "Authorization", "token"):
            self.assertNotIn(secret, doc)


class TestMemoryBackend(_BackendCase, unittest.TestCase):
    def make_backend(self):
        return MemoryEdgeStateBackend()


class TestSqliteBackend(_BackendCase, unittest.TestCase):
    def make_backend(self):
        tmp = tempfile.mkdtemp(prefix="edge-state-")
        self._db = os.path.join(tmp, "edge-state.sqlite3")
        return SqliteEdgeStateBackend(self._db)

    def test_wal_mode_enabled(self):
        import sqlite3

        conn = sqlite3.connect(self._db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode.lower(), "wal")

    def test_versioned_schema(self):
        import sqlite3

        conn = sqlite3.connect(self._db)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        self.assertEqual(version, 2)  # schema v2: last_pending columns (§8)

    def test_busy_timeout_set(self):
        import sqlite3

        conn = sqlite3.connect(self._db)
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        self.assertEqual(timeout, 5000)


class TestOpenBackend(unittest.TestCase):
    def test_memory_config(self):
        config = EdgeTransportConfig(state_persistence="memory")
        backend = open_backend(config)
        self.assertIsInstance(backend, MemoryEdgeStateBackend)
        backend.close()

    def test_sqlite_config_uses_path(self):
        tmp = tempfile.mkdtemp(prefix="edge-state-open-")
        db = os.path.join(tmp, "edge.sqlite3")
        config = EdgeTransportConfig(state_persistence="sqlite", state_db_path=db)
        backend = open_backend(config)
        self.assertIsInstance(backend, SqliteEdgeStateBackend)
        self.assertTrue(os.path.exists(db))
        backend.close()

    def test_unknown_persistence_rejected(self):
        config = EdgeTransportConfig(state_persistence="redis")
        with self.assertRaises(ValueError):
            open_backend(config)


if __name__ == "__main__":
    unittest.main()