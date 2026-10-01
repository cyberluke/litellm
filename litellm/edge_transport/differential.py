"""Local edge differential producer/consumer (Phase 3 §46 step 10/12,
Phase 3.5 §3/§4/§8/§10/§12).

Uses the shared ``differential-context`` package for ALL protocol logic:
canonical state, edge diff rules, envelope construction, ACK validation,
RESYNC_FULL construction. This module adds the LOCAL acknowledged-state
mirror (the LiteLLM side of the two-phase commit), the durable
``EdgeStateBackend`` (SQLite WAL by default), per-context locks and the
durable pending transition that makes lost-ACK recovery deterministic.

Invariants:
- normal new child   -> REGISTER_FULL (independent), lineage metadata only;
- resume same child  -> APPEND / REPLACE_TAIL against the SAME context;
- FORK is explicit-only (build_edge_fork), never derived from lineage;
- every request carries a per-request ``edge_request_id`` (UUID, stable for
  the request lifetime, never derived from context_id) — §12;
- the pending transition is persisted BEFORE every WAN send — §4;
- acknowledged generation NEVER advances before an ACK — §4;
- a 409 whose authoritative identity matches the durable pending is a
  lost-ACK recovery (promote, no replay) — §8;
- a 409 matching nothing is an explicit unrecoverable conflict — §9;
- epoch mismatch marks the binding invalid; the next request performs an
  explicit RESYNC_FULL (server_epoch_changed) — §7/§10;
- the acknowledged baseline after a request is the request's OWN canonical
  units (client is the semantic truth — mirrors SSEProxy apply_ack_to_session).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from differential_context.common.errors import DifferentialContextError
from differential_context.common.generations import next_generation
from differential_context.common.lineage import Lineage
from differential_context.edge.ack import validate_edge_ack
from differential_context.edge.canonical_request import CanonicalPromptState
from differential_context.edge.models import (
    OP_REPLACE_TAIL as _OP_REPLACE_TAIL,
    OP_RESYNC_FULL as _OP_RESYNC_FULL,
)
from differential_context.edge.operations import EdgeDiff, diff_edge_units
from differential_context.edge.producer import (
    build_edge_append,
    build_edge_fork,
    build_edge_register_full,
    build_edge_release,
    build_edge_replace_tail,
    build_edge_resync_full,
)

from .state_backend import (
    EdgeStateBackend,
    MemoryEdgeStateBackend,
    PersistedEdgeContext,
)

# Human-readable diff reasons for observability (edge.dc.operation labels).
OP_REGISTER = "register_full"
OP_APPEND = "append"
OP_REPLACE = "replace_tail"
OP_FORK = "fork"
OP_NOOP = "noop"
OP_RESYNC = "resync_full"
OP_RELEASE = "release"


@dataclass(slots=True)
class LocalEdgeState:
    """Acknowledged edge state of ONE logical context (client-side mirror).
    ``server_epoch`` is the SSEProxy process epoch stored with the ACK (§6)."""

    context_id: str
    generation: int
    state_hash: str
    prompt_state: CanonicalPromptState
    server_epoch: Optional[str] = None

    @property
    def units(self) -> list[dict[str, Any]]:
        return self.prompt_state.canonical_units()


@dataclass(slots=True)
class PendingTransition:
    """The uncertain in-flight transition (§4/§8): persisted BEFORE the WAN
    send, promoted to acknowledged on ACK, matched against authoritative
    409s for lost-ACK recovery."""

    request_id: str
    base_generation: Optional[int]
    base_state_hash: Optional[str]
    next_generation: int
    predicted_state_hash: str
    canonical_prompt_state: dict[str, Any]
    operation: str
    epoch: str
    created_at: float


@dataclass(slots=True)
class EdgeRequestPlan:
    """The prepared edge mutation for one request (Phase 3.5: carries the
    per-request id, the predicted next identity and the full canonical
    state so the pending transition is durable before the send)."""

    context_id: str
    envelope: dict[str, Any]
    operation: str
    base_generation: Optional[int]
    base_state_hash: Optional[str]
    full_units: list[dict[str, Any]]
    delta_units: list[dict[str, Any]]
    is_noop: bool = False
    requires_full: bool = False
    # Envelope serialized WITHOUT compression (step 12 order: diff first,
    # then compress the delta envelope).
    envelope_bytes: Optional[bytes] = None
    # Phase 3.5 §12/§4: request identity + deterministic prediction.
    edge_request_id: str = ""
    predicted_next_generation: Optional[int] = None
    predicted_state_hash: Optional[str] = None
    canonical_state: Optional[dict[str, Any]] = None
    server_epoch: Optional[str] = None


class EdgeStateStore:
    """Per-context state + per-context locks + durable pending transitions.
    Ten independent contexts run fully concurrently; only one context ever
    contends with itself. One backend; never a global Differential Context
    lock."""

    def __init__(self, backend: Optional[EdgeStateBackend] = None) -> None:
        self._backend = backend or MemoryEdgeStateBackend()
        self._states: dict[str, LocalEdgeState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, PendingTransition] = {}
        # §8: the PREVIOUS unconfirmed send. A 409 on the CURRENT request
        # matches against this when the lost ACK belonged to the previous
        # send (the current request's pending only covers ITS own ACK).
        self._previous_pending: dict[str, PendingTransition] = {}
        self._invalid: set[str] = set()
        self._tombstoned: set[str] = set()
        self._server_epoch: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Epoch / binding (§6/§7)
    # ------------------------------------------------------------------ #

    def set_server_epoch(self, epoch: str) -> None:
        """Record the SSEProxy epoch reported by capabilities. Every context
        whose stored epoch differs becomes invalid (§7)."""
        if self._server_epoch == epoch:
            return
        self._server_epoch = epoch
        for context_id, state in self._states.items():
            if state.server_epoch is not None and state.server_epoch != epoch:
                self._invalid.add(context_id)

    def server_epoch(self) -> Optional[str]:
        return self._server_epoch

    def epoch_for(self, context_id: str) -> Optional[str]:
        state = self._states.get(context_id)
        return state.server_epoch if state is not None else None

    def mark_invalid(self, context_id: str) -> None:
        self._invalid.add(context_id)

    def is_invalid(self, context_id: str) -> bool:
        return context_id in self._invalid

    # ------------------------------------------------------------------ #
    # Startup recovery (§15)
    # ------------------------------------------------------------------ #

    async def open(self) -> None:
        """Load the durable state ONCE: acknowledged contexts become the
        in-memory mirror; epoch-mismatched ones are marked as requiring an
        explicit re-registration; pending transitions are preserved for
        lost-ACK reconciliation. Recovery is lazy — nothing is uploaded."""
        rows = await asyncio.to_thread(self._backend.load_all)
        for row in rows:
            if row.tombstoned:
                self._tombstoned.add(row.context_id)
            if (
                row.acknowledged_generation is not None
                and row.acknowledged_state_hash
                and row.canonical_prompt_state
            ):
                prompt_state = CanonicalPromptState.from_dict(row.canonical_prompt_state)
                state = LocalEdgeState(
                    context_id=row.context_id,
                    generation=row.acknowledged_generation,
                    state_hash=row.acknowledged_state_hash,
                    prompt_state=prompt_state,
                    server_epoch=row.server_epoch or None,
                )
                self._states[row.context_id] = state
                if self._server_epoch is not None and row.server_epoch != self._server_epoch:
                    self._invalid.add(row.context_id)
            if row.has_pending and row.pending_next_generation is not None:
                self._pending[row.context_id] = PendingTransition(
                    request_id=row.pending_request_id or "",
                    base_generation=row.pending_base_generation,
                    base_state_hash=row.pending_base_state_hash,
                    next_generation=row.pending_next_generation,
                    predicted_state_hash=row.pending_predicted_state_hash or "",
                    canonical_prompt_state=row.pending_canonical_prompt_state or {},
                    operation=row.pending_operation_type or "",
                    epoch=row.server_epoch or "",
                    created_at=row.pending_created_at or 0.0,
                )
            if row.has_last_pending and row.last_pending_next_generation is not None:
                self._previous_pending[row.context_id] = PendingTransition(
                    request_id=row.last_pending_request_id or "",
                    base_generation=row.last_pending_base_generation,
                    base_state_hash=row.last_pending_base_state_hash,
                    next_generation=row.last_pending_next_generation,
                    predicted_state_hash=row.last_pending_predicted_state_hash or "",
                    canonical_prompt_state=row.last_pending_canonical_prompt_state or {},
                    operation=row.last_pending_operation_type or "",
                    epoch=row.server_epoch or "",
                    created_at=row.last_pending_created_at or 0.0,
                )

    async def close(self) -> None:
        await asyncio.to_thread(self._backend.close)

    # ------------------------------------------------------------------ #
    # Pending transition (§4/§8)
    # ------------------------------------------------------------------ #

    async def persist_pending(self, context_id: str, plan: EdgeRequestPlan) -> None:
        """§4 atomic rule: persist the pending transition BEFORE the WAN
        send. Also refreshes last_seen_at for idle cleanup (§16).

        §8: when a previous pending is still unconfirmed it is retained as
        the LAST pending (durably), so a 409 on the current request can
        still match the previous send's lost ACK."""
        now = time.time()
        state = self._states.get(context_id)
        previous = self._pending.get(context_id)
        if previous is not None and previous.request_id != plan.edge_request_id:
            self._previous_pending[context_id] = previous
        pending = PendingTransition(
            request_id=plan.edge_request_id,
            base_generation=plan.base_generation,
            base_state_hash=plan.base_state_hash,
            next_generation=(
                plan.predicted_next_generation
                if plan.predicted_next_generation is not None
                else (state.generation + 1 if state is not None else 1)
            ),
            predicted_state_hash=plan.predicted_state_hash or "",
            canonical_prompt_state=plan.canonical_state or {},
            operation=plan.operation,
            epoch=self._server_epoch or "",
            created_at=now,
        )
        self._pending[context_id] = pending
        context = PersistedEdgeContext(
            context_id=context_id,
            server_epoch=self._server_epoch or "",
            acknowledged_generation=state.generation if state is not None else None,
            acknowledged_state_hash=state.state_hash if state is not None else None,
            canonical_prompt_state=(
                state.prompt_state.canonical_dict() if state is not None else None
            ),
            canonical_prompt_hash=state.state_hash if state is not None else None,
            updated_at=now,
            last_seen_at=now,
            pending_request_id=pending.request_id,
            pending_base_generation=pending.base_generation,
            pending_base_state_hash=pending.base_state_hash,
            pending_next_generation=pending.next_generation,
            pending_predicted_state_hash=pending.predicted_state_hash,
            pending_canonical_prompt_state=pending.canonical_prompt_state,
            pending_operation_type=pending.operation,
            pending_created_at=pending.created_at,
            last_pending_request_id=(
                previous.request_id if previous is not None else None
            ),
            last_pending_base_generation=(
                previous.base_generation if previous is not None else None
            ),
            last_pending_base_state_hash=(
                previous.base_state_hash if previous is not None else None
            ),
            last_pending_next_generation=(
                previous.next_generation if previous is not None else None
            ),
            last_pending_predicted_state_hash=(
                previous.predicted_state_hash if previous is not None else None
            ),
            last_pending_canonical_prompt_state=(
                previous.canonical_prompt_state if previous is not None else None
            ),
            last_pending_operation_type=(
                previous.operation if previous is not None else None
            ),
            last_pending_created_at=(
                previous.created_at if previous is not None else None
            ),
            tombstoned=context_id in self._tombstoned,
        )
        await asyncio.to_thread(self._backend.save_pending, context)

    def pending_for(self, context_id: str) -> Optional[PendingTransition]:
        return self._pending.get(context_id)

    def previous_pending_for(self, context_id: str) -> Optional[PendingTransition]:
        return self._previous_pending.get(context_id)

    def pending_matches(
        self, context_id: str, generation: Any, state_hash: Any, epoch: Any
    ) -> Optional[PendingTransition]:
        """§8: authoritative 409 identity matches the durable pending
        transition (generation + predicted hash + server epoch). Checks the
        CURRENT pending first, then the previous (lost-ACK) one. Returns
        the matching transition or None."""
        if not isinstance(generation, int) or not isinstance(state_hash, str):
            return None
        for pending in (
            self._pending.get(context_id),
            self._previous_pending.get(context_id),
        ):
            if pending is None:
                continue
            if (
                pending.next_generation == generation
                and pending.predicted_state_hash == state_hash
                and (epoch is None or pending.epoch == epoch)
            ):
                return pending
        return None

    async def promote_pending(
        self,
        context_id: str,
        *,
        ack_generation: Optional[int] = None,
        ack_state_hash: Optional[str] = None,
        epoch: Optional[str] = None,
        model: str = "",
        pending: Optional[PendingTransition] = None,
    ) -> Optional[LocalEdgeState]:
        """§4: transaction — pending becomes acknowledged, pending cleared.
        ACK values are authoritative when provided; otherwise the pending
        prediction is used. Never advances acknowledged state before an
        ACK (this is only ever called on ACK or on an authoritative 409
        matching the pending). ``pending`` selects the transition to
        promote (default: the current one)."""
        pending = pending or self._pending.get(context_id)
        state = self._states.get(context_id)
        generation = (
            ack_generation if ack_generation is not None else (pending.next_generation if pending else None)
        )
        state_hash = (
            ack_state_hash if ack_state_hash is not None else (pending.predicted_state_hash if pending else None)
        )
        if generation is None or not state_hash:
            raise DifferentialContextError(
                "invalid_ack",
                "promote requires an acknowledged generation + state hash (ACK or pending)",
                context_id=context_id,
            )
        epoch = epoch or (pending.epoch if pending else None) or self._server_epoch or ""
        canonical = (
            pending.canonical_prompt_state
            if pending is not None and pending.canonical_prompt_state
            else (state.prompt_state.canonical_dict() if state is not None else None)
        )
        if canonical:
            prompt_state = CanonicalPromptState.from_dict(canonical)
        else:
            prompt_state = CanonicalPromptState(model=model, messages=[])
        new_state = LocalEdgeState(
            context_id=context_id,
            generation=int(generation),
            state_hash=str(state_hash),
            prompt_state=prompt_state,
            server_epoch=epoch or None,
        )
        self._states[context_id] = new_state
        self._pending.pop(context_id, None)
        self._previous_pending.pop(context_id, None)
        self._invalid.discard(context_id)
        await asyncio.to_thread(
            self._backend.promote_pending,
            context_id,
            ack_generation=int(generation),
            ack_state_hash=str(state_hash),
            epoch=epoch,
            canonical_prompt_state=canonical,
            updated_at=time.time(),
        )
        return new_state

    async def commit_ack(
        self,
        context_id: str,
        ack: dict[str, Any],
        full_units: list[dict[str, Any]],
        model: str = "",
    ) -> Optional[LocalEdgeState]:
        """Validate the edge ACK for THIS context and promote the pending
        transition (§4 transaction). The ACK's epoch is stored with the
        acknowledged context (§6); when it differs from the cached epoch the
        binding is marked invalid (the remote process restarted mid-flight;
        the ACK itself is authoritative for its own commit)."""
        validate_edge_ack(ack, context_id)
        epoch = ack.get("edge_server_epoch")
        epoch_str = epoch if isinstance(epoch, str) and epoch else ""
        state = await self.promote_pending(
            context_id,
            ack_generation=int(ack["generation"]),
            ack_state_hash=str(ack["state_hash"]),
            epoch=epoch_str or None,
            model=model,
        )
        if (
            epoch_str
            and self._server_epoch
            and epoch_str != self._server_epoch
        ):
            self.mark_invalid(context_id)
        return state

    # ------------------------------------------------------------------ #
    # Idle cleanup / release (§16)
    # ------------------------------------------------------------------ #

    async def list_idle(self, ttl_seconds: float) -> list[str]:
        """Contexts whose durable last_seen_at is older than the TTL and
        which are not already tombstoned."""
        rows = await asyncio.to_thread(self._backend.load_all)
        now = time.time()
        return [
            row.context_id
            for row in rows
            if not row.tombstoned and (now - row.last_seen_at) > ttl_seconds
        ]

    def is_tombstoned(self, context_id: str) -> bool:
        return context_id in self._tombstoned

    async def mark_tombstoned(self, context_id: str) -> None:
        self._tombstoned.add(context_id)
        await asyncio.to_thread(self._backend.mark_tombstoned, context_id)

    async def delete_context(self, context_id: str) -> None:
        self._states.pop(context_id, None)
        self._pending.pop(context_id, None)
        self._previous_pending.pop(context_id, None)
        self._invalid.discard(context_id)
        self._tombstoned.discard(context_id)
        await asyncio.to_thread(self._backend.delete, context_id)

    async def rows(self) -> list[PersistedEdgeContext]:
        return await asyncio.to_thread(self._backend.load_all)

    # ------------------------------------------------------------------ #
    # Per-context access (unchanged API)
    # ------------------------------------------------------------------ #

    def lock_for(self, context_id: str) -> asyncio.Lock:
        lock = self._locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context_id] = lock
        return lock

    def get(self, context_id: str) -> Optional[LocalEdgeState]:
        return self._states.get(context_id)

    def set(self, state: LocalEdgeState) -> None:
        self._states[state.context_id] = state

    def clear(self) -> None:
        self._states.clear()
        self._locks.clear()
        self._pending.clear()
        self._previous_pending.clear()
        self._invalid.clear()
        self._tombstoned.clear()

    def __len__(self) -> int:
        return len(self._states)


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #


def _predict_resume_state(
    baseline: LocalEdgeState, full_units: list[dict[str, Any]]
) -> CanonicalPromptState:
    """Mirror of the shared consumer's ``_prepare_resume`` state
    reconstruction, so the client's predicted hash equals the server's
    committed hash deterministically."""
    prompt = baseline.prompt_state
    return CanonicalPromptState(
        model=prompt.model,
        messages=full_units,
        system=prompt.system,
        tools=prompt.tools,
        tool_choice=prompt.tool_choice,
        parallel_tool_calls=prompt.parallel_tool_calls,
        chat_template_kwargs=prompt.chat_template_kwargs,
        reasoning_mode=prompt.reasoning_mode,
    )


def build_edge_plan(
    context_id: str,
    payload: dict[str, Any],
    store: EdgeStateStore,
    *,
    fork_source: Optional[tuple[str, int, str]] = None,
    lineage_parent: Optional[str] = None,
    edge_request_id: Optional[str] = None,
) -> EdgeRequestPlan:
    """Diff the FULL logical request against the acknowledged baseline and
    build the edge envelope (steps 11/12). The diff happens on logical
    units BEFORE any compression. A noop diff degrades to an APPEND with
    empty units (correct resume; the completion still runs)."""
    request_id = edge_request_id or str(uuid.uuid4())
    prompt_state = CanonicalPromptState.from_openai_payload(payload)
    new_units = prompt_state.canonical_units()

    # EXPLICIT fork (only when the caller demands it — never from lineage).
    if fork_source is not None:
        source_context_id, source_generation, source_hash = fork_source
        child_units = new_units
        envelope = build_edge_fork(
            context_id,
            source_context_id,
            source_generation,
            source_hash,
            child_append_units=child_units,
        )
        source_state = store.get(source_context_id)
        # The child state = source units + child append units (mirror of the
        # shared consumer's _prepare_fork reconstruction).
        predicted = (
            _predict_resume_state(source_state, source_state.units + child_units)
            if source_state is not None
            else prompt_state
        )
        plan = EdgeRequestPlan(
            context_id=context_id,
            envelope=envelope,
            operation=OP_FORK,
            base_generation=source_generation,
            base_state_hash=source_hash,
            full_units=new_units,
            delta_units=child_units,
            edge_request_id=request_id,
            predicted_next_generation=1,
            predicted_state_hash=predicted.state_hash(),
            canonical_state=predicted.canonical_dict(),
        )
        return _serialize(plan)

    baseline = store.get(context_id)
    if baseline is None:
        envelope = build_edge_register_full(context_id, prompt_state)
        plan = EdgeRequestPlan(
            context_id=context_id,
            envelope=envelope,
            operation=OP_REGISTER,
            base_generation=None,
            base_state_hash=None,
            full_units=new_units,
            delta_units=new_units,
            requires_full=True,
            edge_request_id=request_id,
            predicted_next_generation=1,
            predicted_state_hash=prompt_state.state_hash(),
            canonical_state=prompt_state.canonical_dict(),
        )
        return _serialize(plan)

    diff = diff_edge_units(baseline.units, new_units)
    if diff.op == "noop":
        # Identical logical conversation (e.g. re-run with different
        # generation params): an APPEND with EMPTY units is the correct
        # resume — the edge consumer accepts it and the completion still
        # runs against the same state.
        envelope = build_edge_append(context_id, baseline.generation, baseline.state_hash, [])
        predicted = _predict_resume_state(baseline, new_units)
        plan = EdgeRequestPlan(
            context_id=context_id,
            envelope=envelope,
            operation=OP_APPEND,
            base_generation=baseline.generation,
            base_state_hash=baseline.state_hash,
            full_units=new_units,
            delta_units=[],
            is_noop=True,
            edge_request_id=request_id,
            predicted_next_generation=next_generation(baseline.generation),
            predicted_state_hash=predicted.state_hash(),
            canonical_state=predicted.canonical_dict(),
        )
        return _serialize(plan)

    if diff.op == OP_APPEND:
        envelope = build_edge_append(context_id, baseline.generation, baseline.state_hash, diff.units)
        operation = OP_APPEND
    elif diff.op == _OP_REPLACE_TAIL:
        envelope = build_edge_replace_tail(
            context_id, baseline.generation, baseline.state_hash, diff.keep or 0, diff.units
        )
        operation = OP_REPLACE
    else:  # register_full
        envelope = build_edge_register_full(context_id, prompt_state)
        operation = OP_REGISTER

    predicted = _predict_resume_state(baseline, new_units)
    plan = EdgeRequestPlan(
        context_id=context_id,
        envelope=envelope,
        operation=operation,
        base_generation=baseline.generation,
        base_state_hash=baseline.state_hash,
        full_units=new_units,
        delta_units=diff.units,
        requires_full=diff.requires_full,
        edge_request_id=request_id,
        predicted_next_generation=next_generation(baseline.generation),
        predicted_state_hash=predicted.state_hash(),
        canonical_state=predicted.canonical_dict(),
    )
    return _serialize(plan)


def build_resync_plan(
    context_id: str,
    payload: dict[str, Any],
    *,
    reason: str,
    server_epoch: str,
    known_generation: Optional[int] = None,
    known_state_hash: Optional[str] = None,
    lineage_parent: Optional[str] = None,
    edge_request_id: Optional[str] = None,
) -> EdgeRequestPlan:
    """Phase 3.5 §10: EXPLICIT RESYNC_FULL — full canonical prompt state +
    the producer's belief (epoch / known generation/hash) + reason + nonce.
    Only ever sent deliberately: epoch change (§7), local state loss,
    operator request, unrecoverable conflict. Never an automatic answer to
    an arbitrary 409."""
    prompt_state = CanonicalPromptState.from_openai_payload(payload)
    new_units = prompt_state.canonical_units()
    request_id = edge_request_id or str(uuid.uuid4())
    lineage = Lineage(parent_context_id=lineage_parent) if lineage_parent else None
    envelope = build_edge_resync_full(
        context_id,
        prompt_state,
        server_epoch,
        reason,
        resync_nonce=str(uuid.uuid4()),
        known_generation=known_generation,
        known_state_hash=known_state_hash,
        lineage=lineage,
        edge_request_id=request_id,
    )
    plan = EdgeRequestPlan(
        context_id=context_id,
        envelope=envelope,
        operation=OP_RESYNC,
        base_generation=known_generation,
        base_state_hash=known_state_hash,
        full_units=new_units,
        delta_units=new_units,
        requires_full=True,
        edge_request_id=request_id,
        predicted_next_generation=(
            next_generation(known_generation)
            if isinstance(known_generation, int) and known_generation >= 1
            else None
        ),
        predicted_state_hash=prompt_state.state_hash(),
        canonical_state=prompt_state.canonical_dict(),
        server_epoch=server_epoch,
    )
    return _serialize(plan)


def build_release_plan(
    context_id: str, generation: int, *, edge_request_id: Optional[str] = None
) -> EdgeRequestPlan:
    """Phase 3.5 §16: RELEASE envelope for idle cleanup. Control-only on the
    server (no model invocation); the release ACK confirms and the local
    durable state is deleted."""
    envelope = build_edge_release(context_id, generation)
    return EdgeRequestPlan(
        context_id=context_id,
        envelope=envelope,
        operation=OP_RELEASE,
        base_generation=generation,
        base_state_hash=None,
        full_units=[],
        delta_units=[],
        edge_request_id=edge_request_id or str(uuid.uuid4()),
    )


def _serialize(plan: EdgeRequestPlan) -> EdgeRequestPlan:
    plan.envelope_bytes = json.dumps(
        plan.envelope, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return plan


def resync_register_full(context_id: str, payload: dict[str, Any]) -> EdgeRequestPlan:
    """Explicit REGISTER_FULL resync after a 409 (edge §31) — never a
    silent retry of the stale diff. Kept for compatibility; the Phase 3.5
    recovery paths use ``build_resync_plan`` / lost-ACK promotion."""
    prompt_state = CanonicalPromptState.from_openai_payload(payload)
    envelope = build_edge_register_full(context_id, prompt_state)
    new_units = prompt_state.canonical_units()
    plan = EdgeRequestPlan(
        context_id=context_id,
        envelope=envelope,
        operation=OP_REGISTER,
        base_generation=None,
        base_state_hash=None,
        full_units=new_units,
        delta_units=new_units,
        requires_full=True,
        edge_request_id=str(uuid.uuid4()),
        predicted_next_generation=1,
        predicted_state_hash=prompt_state.state_hash(),
        canonical_state=prompt_state.canonical_dict(),
    )
    return _serialize(plan)


def apply_edge_ack(
    store: EdgeStateStore,
    context_id: str,
    ack: dict[str, Any],
    full_units: list[dict[str, Any]],
    model: str = "",
) -> LocalEdgeState:
    """Synchronous in-memory ACK adoption (used by tests and fast paths).
    The production path is ``EdgeStateStore.commit_ack`` which additionally
    persists + promotes the pending transition."""
    validate_edge_ack(ack, context_id)
    prompt_state = CanonicalPromptState(model=model, messages=full_units)
    state = LocalEdgeState(
        context_id=context_id,
        generation=int(ack["generation"]),
        state_hash=str(ack["state_hash"]),
        prompt_state=prompt_state,
        server_epoch=ack.get("edge_server_epoch"),
    )
    store.set(state)
    return state


__all__ = [
    "LocalEdgeState",
    "PendingTransition",
    "EdgeRequestPlan",
    "EdgeStateStore",
    "build_edge_plan",
    "build_resync_plan",
    "build_release_plan",
    "resync_register_full",
    "apply_edge_ack",
    "OP_REGISTER",
    "OP_APPEND",
    "OP_REPLACE",
    "OP_FORK",
    "OP_NOOP",
    "OP_RESYNC",
    "OP_RELEASE",
]