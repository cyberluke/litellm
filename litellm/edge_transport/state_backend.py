"""Phase 3.5 §3/§4/§5: durable local edge state backend (LiteLLM-owned).

The acknowledged edge state and the uncertain in-flight transition are
persisted per context. SQLite (WAL) is the first implementation; a memory
backend exists for tests and throwaway runs. The shared
``differential-context`` package is deliberately untouched by persistence.

Persistence invariants (§4/§5):

- persist the pending transition BEFORE every WAN send (the transport calls
  ``save_pending`` before posting the request);
- on ACK the transport calls ``promote_pending``: a single atomic UPDATE
  moves pending -> acknowledged and clears pending in one statement;
- acknowledged generation NEVER advances before an ACK;
- one row per context; short transactions; WAL mode; busy_timeout;
  versioned schema (PRAGMA user_version);
- NEVER store API keys or Authorization headers.

Per-context async locks stay in ``EdgeStateStore`` (process-local, one lock
per context, never a global lock). The SQLite connection itself is guarded
by a threading lock because backend calls run in executor threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from differential_context.edge.models import (
    EDGE_PROTOCOL_VERSION,
    EDGE_REPRESENTATION,
)

from .config import default_state_db_path

_SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS edge_state (
    context_id                  TEXT PRIMARY KEY,
    server_epoch                TEXT NOT NULL,
    representation              TEXT NOT NULL,
    protocol_version            INTEGER NOT NULL,
    acknowledged_generation     INTEGER,
    acknowledged_state_hash     TEXT,
    canonical_prompt_state      TEXT,
    canonical_prompt_hash       TEXT,
    parent_context_id           TEXT,
    updated_at                  REAL NOT NULL,
    last_seen_at                REAL NOT NULL,
    pending_request_id          TEXT,
    pending_base_generation     INTEGER,
    pending_base_state_hash     TEXT,
    pending_next_generation     INTEGER,
    pending_predicted_state_hash TEXT,
    pending_canonical_prompt_state TEXT,
    pending_operation_type      TEXT,
    pending_created_at          REAL,
    -- Phase 3.5 schema v2 (§8): the PREVIOUS unconfirmed send is retained
    -- so a 409 on the CURRENT request can still match the pending of the
    -- send whose ACK was lost (the current request's pending is only
    -- useful for ITS own later recovery).
    last_pending_request_id          TEXT,
    last_pending_base_generation     INTEGER,
    last_pending_base_state_hash     TEXT,
    last_pending_next_generation     INTEGER,
    last_pending_predicted_state_hash TEXT,
    last_pending_canonical_prompt_state TEXT,
    last_pending_operation_type      TEXT,
    last_pending_created_at          REAL,
    tombstoned                  INTEGER NOT NULL DEFAULT 0
)
"""

_PENDING_COLUMNS = (
    "pending_request_id",
    "pending_base_generation",
    "pending_base_state_hash",
    "pending_next_generation",
    "pending_predicted_state_hash",
    "pending_canonical_prompt_state",
    "pending_operation_type",
    "pending_created_at",
)

_LAST_PENDING_COLUMNS = (
    "last_pending_request_id",
    "last_pending_base_generation",
    "last_pending_base_state_hash",
    "last_pending_next_generation",
    "last_pending_predicted_state_hash",
    "last_pending_canonical_prompt_state",
    "last_pending_operation_type",
    "last_pending_created_at",
)


@dataclass(slots=True)
class PersistedEdgeContext:
    """One durable row: acknowledged state + (optional) pending transition."""

    context_id: str
    server_epoch: str
    representation: str = EDGE_REPRESENTATION
    protocol_version: int = EDGE_PROTOCOL_VERSION
    acknowledged_generation: Optional[int] = None
    acknowledged_state_hash: Optional[str] = None
    canonical_prompt_state: Optional[dict[str, Any]] = None
    canonical_prompt_hash: Optional[str] = None
    parent_context_id: Optional[str] = None
    updated_at: float = 0.0
    last_seen_at: float = 0.0
    pending_request_id: Optional[str] = None
    pending_base_generation: Optional[int] = None
    pending_base_state_hash: Optional[str] = None
    pending_next_generation: Optional[int] = None
    pending_predicted_state_hash: Optional[str] = None
    pending_canonical_prompt_state: Optional[dict[str, Any]] = None
    pending_operation_type: Optional[str] = None
    pending_created_at: Optional[float] = None
    # Schema v2: the previous unconfirmed send (see module docstring).
    last_pending_request_id: Optional[str] = None
    last_pending_base_generation: Optional[int] = None
    last_pending_base_state_hash: Optional[str] = None
    last_pending_next_generation: Optional[int] = None
    last_pending_predicted_state_hash: Optional[str] = None
    last_pending_canonical_prompt_state: Optional[dict[str, Any]] = None
    last_pending_operation_type: Optional[str] = None
    last_pending_created_at: Optional[float] = None
    tombstoned: bool = False

    @property
    def has_pending(self) -> bool:
        return self.pending_request_id is not None

    @property
    def has_last_pending(self) -> bool:
        return self.last_pending_request_id is not None


class EdgeStateBackend:
    """Synchronous backend contract (SQLite/memory). All methods are short
    and run in executor threads; callers hold the per-context asyncio lock."""

    def load_all(self) -> list[PersistedEdgeContext]:
        raise NotImplementedError

    def save_pending(self, context: PersistedEdgeContext) -> None:
        """Persist/refresh the pending transition (BEFORE WAN send)."""
        raise NotImplementedError

    def promote_pending(
        self,
        context_id: str,
        *,
        ack_generation: Optional[int] = None,
        ack_state_hash: Optional[str] = None,
        epoch: str,
        canonical_prompt_state: Optional[dict[str, Any]] = None,
        parent_context_id: Optional[str] = None,
        updated_at: float,
    ) -> None:
        """Atomic transaction: pending -> acknowledged, pending cleared."""
        raise NotImplementedError

    def upsert_acknowledged(self, context: PersistedEdgeContext) -> None:
        raise NotImplementedError

    def touch(self, context_id: str, last_seen_at: float) -> None:
        raise NotImplementedError

    def mark_tombstoned(self, context_id: str, tombstoned: bool = True) -> None:
        raise NotImplementedError

    def delete(self, context_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MemoryEdgeStateBackend(EdgeStateBackend):
    """Non-durable backend (tests / throwaway runs). Same contract."""

    def __init__(self) -> None:
        self._rows: dict[str, PersistedEdgeContext] = {}
        self._lock = threading.Lock()

    def load_all(self) -> list[PersistedEdgeContext]:
        with self._lock:
            return [self._copy(row) for row in self._rows.values()]

    def save_pending(self, context: PersistedEdgeContext) -> None:
        with self._lock:
            existing = self._rows.get(context.context_id)
            if (
                existing is not None
                and existing.has_pending
                and existing.pending_request_id != context.pending_request_id
            ):
                # §8: retain the previous unconfirmed send as last_pending so
                # a later 409 can still match the lost-ACK transition.
                for target, source in zip(_LAST_PENDING_COLUMNS, _PENDING_COLUMNS):
                    setattr(context, target, getattr(existing, source))
            self._rows[context.context_id] = self._copy(context)

    def promote_pending(
        self,
        context_id: str,
        *,
        ack_generation: Optional[int] = None,
        ack_state_hash: Optional[str] = None,
        epoch: str,
        canonical_prompt_state: Optional[dict[str, Any]] = None,
        parent_context_id: Optional[str] = None,
        updated_at: float,
    ) -> None:
        with self._lock:
            row = self._rows.get(context_id)
            if row is None:
                row = PersistedEdgeContext(context_id=context_id, server_epoch=epoch)
            pending_state = row.pending_canonical_prompt_state
            row.acknowledged_generation = (
                ack_generation
                if ack_generation is not None
                else row.pending_next_generation
            )
            row.acknowledged_state_hash = (
                ack_state_hash if ack_state_hash is not None else row.pending_predicted_state_hash
            )
            row.canonical_prompt_state = (
                canonical_prompt_state if canonical_prompt_state is not None else pending_state
            )
            row.canonical_prompt_hash = row.acknowledged_state_hash
            row.server_epoch = epoch
            if parent_context_id is not None:
                row.parent_context_id = parent_context_id
            row.updated_at = updated_at
            row.last_seen_at = max(row.last_seen_at, updated_at)
            for column in _PENDING_COLUMNS + _LAST_PENDING_COLUMNS:
                setattr(row, column, None)
            self._rows[context_id] = row

    def upsert_acknowledged(self, context: PersistedEdgeContext) -> None:
        with self._lock:
            self._rows[context.context_id] = self._copy(context)

    def touch(self, context_id: str, last_seen_at: float) -> None:
        with self._lock:
            row = self._rows.get(context_id)
            if row is not None:
                row.last_seen_at = last_seen_at

    def mark_tombstoned(self, context_id: str, tombstoned: bool = True) -> None:
        with self._lock:
            row = self._rows.get(context_id)
            if row is not None:
                row.tombstoned = tombstoned

    def delete(self, context_id: str) -> None:
        with self._lock:
            self._rows.pop(context_id, None)

    def close(self) -> None:
        with self._lock:
            self._rows.clear()

    @staticmethod
    def _copy(row: PersistedEdgeContext) -> PersistedEdgeContext:
        return PersistedEdgeContext(**{f: getattr(row, f) for f in row.__dataclass_fields__})


class SqliteEdgeStateBackend(EdgeStateBackend):
    """Durable backend: SQLite WAL, busy_timeout, versioned schema, one row
    per context, short transactions (single-statement writes)."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={5 * 1000}")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.execute(_DDL)
            self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def load_all(self) -> list[PersistedEdgeContext]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM edge_state ORDER BY context_id"
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
        return [_row_to_context(row, columns) for row in rows]

    def save_pending(self, context: PersistedEdgeContext) -> None:
        # §8: retain the previous unconfirmed send as last_pending so a 409
        # on the CURRENT request can still match the lost-ACK transition.
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE edge_state SET
                        last_pending_request_id = pending_request_id,
                        last_pending_base_generation = pending_base_generation,
                        last_pending_base_state_hash = pending_base_state_hash,
                        last_pending_next_generation = pending_next_generation,
                        last_pending_predicted_state_hash = pending_predicted_state_hash,
                        last_pending_canonical_prompt_state = pending_canonical_prompt_state,
                        last_pending_operation_type = pending_operation_type,
                        last_pending_created_at = pending_created_at
                    WHERE context_id = ?
                      AND pending_request_id IS NOT NULL
                      AND pending_request_id != ?
                    """,
                    (context.context_id, context.pending_request_id),
                )
                self._conn.execute(
                    """
                    INSERT INTO edge_state (
                        context_id, server_epoch, representation, protocol_version,
                        acknowledged_generation, acknowledged_state_hash,
                        canonical_prompt_state, canonical_prompt_hash, parent_context_id,
                        updated_at, last_seen_at,
                        pending_request_id, pending_base_generation, pending_base_state_hash,
                        pending_next_generation, pending_predicted_state_hash,
                        pending_canonical_prompt_state, pending_operation_type,
                        pending_created_at, tombstoned
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(context_id) DO UPDATE SET
                        server_epoch=excluded.server_epoch,
                        representation=excluded.representation,
                        protocol_version=excluded.protocol_version,
                        acknowledged_generation=excluded.acknowledged_generation,
                        acknowledged_state_hash=excluded.acknowledged_state_hash,
                        canonical_prompt_state=excluded.canonical_prompt_state,
                        canonical_prompt_hash=excluded.canonical_prompt_hash,
                        parent_context_id=excluded.parent_context_id,
                        updated_at=excluded.updated_at,
                        last_seen_at=excluded.last_seen_at,
                        pending_request_id=excluded.pending_request_id,
                        pending_base_generation=excluded.pending_base_generation,
                        pending_base_state_hash=excluded.pending_base_state_hash,
                        pending_next_generation=excluded.pending_next_generation,
                        pending_predicted_state_hash=excluded.pending_predicted_state_hash,
                        pending_canonical_prompt_state=excluded.pending_canonical_prompt_state,
                        pending_operation_type=excluded.pending_operation_type,
                        pending_created_at=excluded.pending_created_at,
                        tombstoned=excluded.tombstoned
                    """,
                    (
                        context.context_id,
                        context.server_epoch,
                        context.representation,
                        context.protocol_version,
                        context.acknowledged_generation,
                        context.acknowledged_state_hash,
                        _dumps(context.canonical_prompt_state),
                        context.canonical_prompt_hash,
                        context.parent_context_id,
                        context.updated_at,
                        context.last_seen_at,
                        context.pending_request_id,
                        context.pending_base_generation,
                        context.pending_base_state_hash,
                        context.pending_next_generation,
                        context.pending_predicted_state_hash,
                        _dumps(context.pending_canonical_prompt_state),
                        context.pending_operation_type,
                        context.pending_created_at,
                        1 if context.tombstoned else 0,
                    ),
                )

    def promote_pending(
        self,
        context_id: str,
        *,
        ack_generation: Optional[int] = None,
        ack_state_hash: Optional[str] = None,
        epoch: str,
        canonical_prompt_state: Optional[dict[str, Any]] = None,
        parent_context_id: Optional[str] = None,
        updated_at: float,
    ) -> None:
        """§4: ONE atomic statement — pending becomes acknowledged, pending
        cleared. Never advances acknowledged state before an ACK."""
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """
                    UPDATE edge_state SET
                        acknowledged_generation = COALESCE(?, pending_next_generation),
                        acknowledged_state_hash = COALESCE(?, pending_predicted_state_hash),
                        canonical_prompt_state = COALESCE(?, pending_canonical_prompt_state),
                        canonical_prompt_hash = COALESCE(?, pending_predicted_state_hash),
                        server_epoch = ?,
                        parent_context_id = COALESCE(?, parent_context_id),
                        updated_at = ?,
                        last_seen_at = MAX(last_seen_at, ?),
                        pending_request_id = NULL,
                        pending_base_generation = NULL,
                        pending_base_state_hash = NULL,
                        pending_next_generation = NULL,
                        pending_predicted_state_hash = NULL,
                        pending_canonical_prompt_state = NULL,
                        pending_operation_type = NULL,
                        pending_created_at = NULL,
                        last_pending_request_id = NULL,
                        last_pending_base_generation = NULL,
                        last_pending_base_state_hash = NULL,
                        last_pending_next_generation = NULL,
                        last_pending_predicted_state_hash = NULL,
                        last_pending_canonical_prompt_state = NULL,
                        last_pending_operation_type = NULL,
                        last_pending_created_at = NULL,
                        tombstoned = 0
                    WHERE context_id = ?
                    """,
                    (
                        ack_generation,
                        ack_state_hash,
                        _dumps(canonical_prompt_state),
                        ack_state_hash,
                        epoch,
                        parent_context_id,
                        updated_at,
                        updated_at,
                        context_id,
                    ),
                )
                if cursor.rowcount == 0:
                    # No row yet (e.g. ACK raced a deleted row): create the
                    # acknowledged row directly.
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO edge_state (
                            context_id, server_epoch, representation, protocol_version,
                            acknowledged_generation, acknowledged_state_hash,
                            canonical_prompt_state, canonical_prompt_hash,
                            updated_at, last_seen_at, tombstoned
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,0)
                        """,
                        (
                            context_id,
                            epoch,
                            EDGE_REPRESENTATION,
                            EDGE_PROTOCOL_VERSION,
                            ack_generation,
                            ack_state_hash,
                            _dumps(canonical_prompt_state),
                            ack_state_hash,
                            updated_at,
                            updated_at,
                        ),
                    )

    def upsert_acknowledged(self, context: PersistedEdgeContext) -> None:
        row = PersistedEdgeContext(
            context_id=context.context_id,
            server_epoch=context.server_epoch,
            representation=context.representation,
            protocol_version=context.protocol_version,
            acknowledged_generation=context.acknowledged_generation,
            acknowledged_state_hash=context.acknowledged_state_hash,
            canonical_prompt_state=context.canonical_prompt_state,
            canonical_prompt_hash=context.canonical_prompt_hash,
            parent_context_id=context.parent_context_id,
            updated_at=context.updated_at,
            last_seen_at=context.last_seen_at,
            pending_request_id=context.pending_request_id,
            pending_base_generation=context.pending_base_generation,
            pending_base_state_hash=context.pending_base_state_hash,
            pending_next_generation=context.pending_next_generation,
            pending_predicted_state_hash=context.pending_predicted_state_hash,
            pending_canonical_prompt_state=context.pending_canonical_prompt_state,
            pending_operation_type=context.pending_operation_type,
            pending_created_at=context.pending_created_at,
            tombstoned=context.tombstoned,
        )
        self.save_pending(row)

    def touch(self, context_id: str, last_seen_at: float) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE edge_state SET last_seen_at = ? WHERE context_id = ?",
                    (last_seen_at, context_id),
                )

    def mark_tombstoned(self, context_id: str, tombstoned: bool = True) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE edge_state SET tombstoned = ? WHERE context_id = ?",
                    (1 if tombstoned else 0, context_id),
                )

    def delete(self, context_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM edge_state WHERE context_id = ?", (context_id,))

    def close(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _loads(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _row_to_context(row: tuple, columns: list[str]) -> PersistedEdgeContext:
    data = dict(zip(columns, row))
    return PersistedEdgeContext(
        context_id=str(data["context_id"]),
        server_epoch=str(data["server_epoch"] or ""),
        representation=str(data["representation"] or EDGE_REPRESENTATION),
        protocol_version=int(data["protocol_version"] or EDGE_PROTOCOL_VERSION),
        acknowledged_generation=data["acknowledged_generation"],
        acknowledged_state_hash=data["acknowledged_state_hash"],
        canonical_prompt_state=_loads(data["canonical_prompt_state"]),
        canonical_prompt_hash=data["canonical_prompt_hash"],
        parent_context_id=data["parent_context_id"],
        updated_at=float(data["updated_at"] or 0.0),
        last_seen_at=float(data["last_seen_at"] or 0.0),
        pending_request_id=data["pending_request_id"],
        pending_base_generation=data["pending_base_generation"],
        pending_base_state_hash=data["pending_base_state_hash"],
        pending_next_generation=data["pending_next_generation"],
        pending_predicted_state_hash=data["pending_predicted_state_hash"],
        pending_canonical_prompt_state=_loads(data["pending_canonical_prompt_state"]),
        pending_operation_type=data["pending_operation_type"],
        pending_created_at=data["pending_created_at"],
        last_pending_request_id=data.get("last_pending_request_id"),
        last_pending_base_generation=data.get("last_pending_base_generation"),
        last_pending_base_state_hash=data.get("last_pending_base_state_hash"),
        last_pending_next_generation=data.get("last_pending_next_generation"),
        last_pending_predicted_state_hash=data.get("last_pending_predicted_state_hash"),
        last_pending_canonical_prompt_state=_loads(data.get("last_pending_canonical_prompt_state")),
        last_pending_operation_type=data.get("last_pending_operation_type"),
        last_pending_created_at=data.get("last_pending_created_at"),
        tombstoned=bool(data["tombstoned"]),
    )


def open_backend(config: Any) -> EdgeStateBackend:
    """Build the configured backend (sqlite default per §5)."""
    persistence = getattr(config, "state_persistence", "sqlite").strip().lower()
    if persistence == "memory":
        return MemoryEdgeStateBackend()
    if persistence != "sqlite":
        raise ValueError(
            f"EDGE_STATE_PERSISTENCE must be 'sqlite' or 'memory', got {persistence!r}"
        )
    return SqliteEdgeStateBackend(getattr(config, "state_db_path", "") or default_state_db_path())


__all__ = [
    "PersistedEdgeContext",
    "EdgeStateBackend",
    "MemoryEdgeStateBackend",
    "SqliteEdgeStateBackend",
    "open_backend",
    "default_state_db_path",
]