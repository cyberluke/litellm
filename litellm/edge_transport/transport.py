"""Edge transport — the HTTPX HTTP/2 client for the Differential Context
edge route (Phase 3 §46 steps 12/13, Phase 3.5 §4/§6/§7/§8/§9/§12/§15/§16/§18).

- persistent HTTPX pool with HTTP/2 enabled (``http2=True``) for the edge
  route only; unrelated providers keep their own transports;
- strict negotiated-version verification: any response on the edge route
  that is NOT HTTP/2 raises transport_error (no silent downgrade);
- streaming: HTTP/2 verified BEFORE the stream is exposed to the caller;
  the ``differential_context.ack`` SSE control event is consumed, validated
  and committed (pending -> acknowledged, §4 transaction), then stripped;
- non-stream: the ``differential_context`` ACK metadata is consumed and
  removed from the response body;
- Phase 3.5 recovery discipline:
    * every request carries a per-request ``edge_request_id`` (UUID) — §12;
    * the pending transition is persisted BEFORE every WAN send — §4;
    * a 409 whose authoritative identity matches the durable pending is a
      lost-ACK recovery: promote locally, re-plan the CURRENT request
      against the promoted baseline, resend ONCE — §8, no replay of the
      prior inference;
    * a 409 whose epoch differs marks the binding invalid and the request
      is re-sent as an explicit RESYNC_FULL (server_epoch_changed) against
      freshly fetched capabilities — §7/§10;
    * a 409 matching neither pending nor epoch is an UNRECOVERABLE conflict
      — raised, never hidden behind a resync — §9;
    * startup loads the durable store and reconciles epochs lazily (§15);
    * idle contexts are RELEASEd (control-only) and their durable state
      deleted; unconfirmed releases are tombstoned and retried — §16;
    * capabilities are cached and refreshed on startup / epoch mismatch /
      protocol error / TTL — §18;
- one async lock per context; ten independent contexts run concurrently;
- compression order: full logical request -> differential diff -> serialize
  delta envelope -> zstd/gzip -> HTTP/2 (the full request is never
  compressed before diffing);
- no stable session identity -> Differential Context disabled for the
  request, but the transport still runs: HTTP/2 + compression + full
  ordinary OpenAI request to the standard endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from differential_context.edge.ack import ACK_EVENT_NAME, parse_edge_ack
from differential_context.edge.models import EDGE_REPRESENTATION
from differential_context.common.errors import DifferentialContextError

from . import metrics as m
from .capabilities import CapabilitiesCache, CapabilitiesError
from .compression import decompress_body, maybe_compress
from .config import EdgeTransportConfig
from .differential import (
    EdgeRequestPlan,
    EdgeStateStore,
    build_edge_plan,
    build_release_plan,
    build_resync_plan,
)
from .session_resolver import DISABLED_NO_STABLE_IDENTITY, SessionResolution, resolve_session
from .state_backend import open_backend

logger = logging.getLogger("litellm.edge_transport")

# Envelope keys consumed by the SSEProxy edge endpoint (mirror of
# sseproxy.edge_dc._ENVELOPE_KEYS); every other top-level key is a
# generation parameter. ``messages`` is excluded explicitly — the edge
# delta IS the message content; forwarding the full messages would
# silently defeat the delta.
_ENVELOPE_KEYS = frozenset(
    {
        "protocol_version",
        "representation",
        "context_id",
        "context_mode",
        "base_generation",
        "base_state_hash",
        "operations",
        "fork",
        "lineage",
        "prompt_state",
        "edge_request_id",
        "server_epoch",
        "known_generation",
        "known_state_hash",
        "reason",
        "resync_nonce",
    }
)


class EdgeTransportError(Exception):
    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class UnrecoverableEdgeConflict(EdgeTransportError):
    """§9: the authoritative state matches neither the acknowledged state
    nor the one durable pending transition. No guess, no invisible retry —
    the caller must decide on an explicit RESYNC_FULL."""


def _error_detail(response: httpx.Response) -> str:
    """Bounded upstream error excerpt for EdgeTransportError messages, so
    clients see WHY the WAN chain rejected the request (e.g. the engine
    refusing multimodal input) instead of a bare status code."""
    try:
        text = response.text.strip()
    except Exception:
        return ""
    if not text:
        return ""
    try:
        doc = response.json()
        if isinstance(doc, dict) and "error" in doc:
            msg = doc["error"]
            if isinstance(msg, dict):
                msg = msg.get("message") or ""
            if isinstance(msg, str) and msg:
                return f": {msg[:300]}"
    except Exception:
        pass
    return f": {text[:300]}"


@dataclass(slots=True)
class EdgeMeta:
    """Observability metadata returned beside the OpenAI response."""

    context_id: Optional[str] = None
    operation: Optional[str] = None
    disabled_reason: Optional[str] = None
    negotiated_http_version: Optional[str] = None
    full_logical_bytes: int = 0
    delta_bytes: int = 0
    wire_bytes: int = 0
    compression_encoding: str = "identity"
    edge_generation: Optional[int] = None
    edge_state_hash: Optional[str] = None
    edge_request_id: Optional[str] = None
    server_epoch: Optional[str] = None
    resync_reason: Optional[str] = None
    lost_ack_recovered: bool = False


class EdgeTransport:
    """One persistent HTTPX client for ALL edge contexts (pool reuse); the
    per-context locks live in the store."""

    def __init__(self, config: EdgeTransportConfig, client: Optional[httpx.AsyncClient] = None) -> None:
        self.config = config
        self.store = EdgeStateStore(open_backend(config))
        self._client = client or httpx.AsyncClient(
            http2=True,
            verify=config.verify_tls,
            # 0 disables the bound (slow-network profile: a generation is
            # never killed; the TCP/TLS connect gets a generous 120s default).
            timeout=httpx.Timeout(
                config.timeout_seconds if config.timeout_seconds > 0 else None,
                connect=(
                    config.connect_timeout_seconds
                    if config.connect_timeout_seconds > 0
                    else None
                ),
            ),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
            ),
            headers={
                "Authorization": f"Bearer {config.api_key}" if config.api_key else "",
                "Accept-Encoding": "zstd, gzip, identity",
            },
        )
        self._owns_client = client is None
        self._capabilities = CapabilitiesCache(config)
        self._started = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def aclose(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError, Exception):
                await self._cleanup_task
            self._cleanup_task = None
        if self._owns_client:
            await self._client.aclose()
        await self.store.close()

    async def capabilities(self) -> dict[str, Any]:
        return await self._capabilities.get(self._client)

    # ------------------------------------------------------------------ #
    # Startup (§15) / epoch reconciliation (§6/§7)
    # ------------------------------------------------------------------ #

    async def ensure_started(self) -> None:
        """Fetch capabilities once, record the server epoch, load the
        durable store (lazy recovery — nothing is uploaded) and start the
        idle-cleanup loop."""
        if self._started:
            return
        caps = await self._capabilities.get(self._client)
        epoch = caps.get("edge_server_epoch") or ""
        self.store.set_server_epoch(epoch)
        await self.store.open()
        self._started = True
        if self.config.idle_ttl_seconds > 0:
            self._cleanup_task = asyncio.create_task(
                self._idle_cleanup_loop(), name="edge:idle-cleanup"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def acompletion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        headers: Optional[Dict[str, str]] = None,
        generation_params: Optional[dict[str, Any]] = None,
        fork_source: Optional[Tuple[str, int, str]] = None,
    ) -> Tuple[Any, EdgeMeta]:
        """Edge route completion. Returns (response, meta):

        - stream=False -> the ordinary OpenAI JSON dict (edge ACK removed);
        - stream=True  -> an async iterator of ordinary OpenAI chunk dicts
          (the ACK control event is consumed and never yielded).
        """
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if generation_params:
            payload.update(generation_params)
        payload["stream"] = stream
        # Header keys are matched case-insensitively by the session resolver
        # (coding-agent adapters send e.g. X-Kilo-Session-ID); normalize so
        # direct SDK callers with mixed-case extra_headers work identically
        # to the proxy path (whose Starlette headers are already lowercase).
        if headers:
            headers = {str(k).lower(): str(v) for k, v in headers.items()}
        # The WAN chain/engine validates the model name against ITS model
        # (e.g. the engine's served model path). When upstream_model is
        # configured, rewrite the OUTBOUND model on both paths (edge
        # envelope and ordinary fallback); the client-facing route model
        # name is unaffected.
        if self.config.upstream_model:
            payload["model"] = self.config.upstream_model

        resolution = resolve_session(headers, payload, self.config)
        if resolution.context_id is None:
            return await self._completion_disabled(payload, resolution)

        await self.ensure_started()
        caps = await self.capabilities()
        epoch = caps.get("edge_server_epoch") or ""
        self.store.set_server_epoch(epoch)

        context_id = resolution.context_id
        # §12: per-request edge identity, stable for the request lifetime
        # (including its 409-recovery resend), never derived from context_id.
        edge_request_id = str(uuid.uuid4())

        lock = self.store.lock_for(context_id)
        async with lock:
            plan, resync_reason = self._plan_for(
                context_id, payload, epoch, fork_source, edge_request_id
            )
            m.M_OP.labels(plan.operation).inc()
            meta = EdgeMeta(
                context_id=context_id,
                operation=plan.operation,
                edge_request_id=edge_request_id,
                server_epoch=epoch,
                resync_reason=resync_reason,
                full_logical_bytes=len(
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ),
                delta_bytes=len(plan.envelope_bytes or b""),
            )
            # §4: persist the pending transition BEFORE the WAN send.
            await self.store.persist_pending(context_id, plan)
            response, negotiated = await self._post_edge(
                payload, plan, meta, context_id, recovery=True, fork_source=fork_source
            )
            if stream:
                return self._stream_response(response, resolution, payload, meta), meta
            return await self._plain_response(response, resolution, payload, meta), meta

    async def resync_context(
        self,
        context_id: str,
        payload: dict[str, Any],
        *,
        reason: str,
        known_generation: Optional[int] = None,
        known_state_hash: Optional[str] = None,
    ) -> Tuple[Any, EdgeMeta]:
        """Phase 3.5 §10: EXPLICIT RESYNC_FULL (operator/local-loss/
        unrecoverable-conflict recovery). Only ever called deliberately;
        the transport never hides an arbitrary 409 behind this."""
        await self.ensure_started()
        caps = await self.capabilities()
        epoch = caps.get("edge_server_epoch") or ""
        self.store.set_server_epoch(epoch)
        edge_request_id = str(uuid.uuid4())
        plan = build_resync_plan(
            context_id,
            payload,
            reason=reason,
            server_epoch=epoch,
            known_generation=known_generation,
            known_state_hash=known_state_hash,
            edge_request_id=edge_request_id,
        )
        m.M_RESYNC.labels(reason).inc()
        lock = self.store.lock_for(context_id)
        async with lock:
            meta = EdgeMeta(
                context_id=context_id,
                operation=plan.operation,
                edge_request_id=edge_request_id,
                server_epoch=epoch,
                resync_reason=reason,
                full_logical_bytes=len(
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ),
                delta_bytes=len(plan.envelope_bytes or b""),
            )
            await self.store.persist_pending(context_id, plan)
            stream = bool(payload.get("stream", False))
            response, negotiated = await self._post_edge(
                payload, plan, meta, context_id, recovery=False
            )
            if stream:
                return self._stream_response(response, None, payload, meta), meta
            return await self._plain_response(response, None, payload, meta), meta

    async def release_context(self, context_id: str, generation: int) -> bool:
        """Phase 3.5 §16: RELEASE one context (control-only — no model
        invocation). Confirmed release deletes the durable local state.
        Raises on failure so the caller can tombstone and retry later."""
        await self.ensure_started()
        plan = build_release_plan(context_id, generation)
        wire: dict[str, Any] = dict(plan.envelope)
        envelope_bytes = json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        body, encoding, compress_ms = maybe_compress(
            envelope_bytes, self.config.request_compression, self.config.compression_threshold
        )
        m.M_COMPRESS_MS.observe(compress_ms)
        headers = {}
        if encoding != "identity":
            headers["Content-Encoding"] = encoding
        url = f"{self.config.base_url}/v1/differential-context/chat/completions"
        response = await self._client.post(url, content=body, headers=headers)
        negotiated = response.http_version
        self._verify_http2(negotiated)
        if response.status_code == 200:
            try:
                doc = json.loads(
                    decompress_body(response.content, response.headers.get("content-encoding"))
                )
            except (ValueError, UnicodeDecodeError) as exc:
                raise EdgeTransportError(f"release response is not JSON: {exc}") from exc
            dc = doc.get("differential_context") if isinstance(doc, dict) else None
            if (
                isinstance(dc, dict)
                and dc.get("released") is True
                and dc.get("context_id") == context_id
            ):
                await self.store.delete_context(context_id)
                return True
            raise EdgeTransportError("release response missing release ack", status=200)
        if response.status_code == 409:
            # §16: an epoch mismatch means the remote binding is invalid —
            # nothing to release there, drop the local state. Any other
            # conflict is retried later (caller tombstones).
            try:
                doc = response.json()
                conflict_epoch = doc.get("edge_server_epoch")
            except Exception:
                conflict_epoch = None
            if (
                conflict_epoch is not None
                and conflict_epoch != self.store.server_epoch()
            ):
                self.store.mark_invalid(context_id)
                await self.store.delete_context(context_id)
                return True
            raise EdgeTransportError(
                f"release conflict for context {context_id}", status=409
            )
        raise EdgeTransportError(
            f"release endpoint returned {response.status_code}", status=response.status_code
        )

    async def reconcile(
        self,
        context_id: str,
        *,
        edge_request_id: Optional[str] = None,
        known_generation: Optional[int] = None,
        known_state_hash: Optional[str] = None,
        pending_generation: Optional[int] = None,
        pending_state_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """Phase 3.5 §14: metadata-only commit-status check. Never prompt
        content, never token ids, never a model invocation."""
        body: dict[str, Any] = {
            "context_id": context_id,
            "server_epoch": self.store.server_epoch() or "",
        }
        if edge_request_id:
            body["edge_request_id"] = edge_request_id
        if known_generation is not None:
            body["known_generation"] = known_generation
        if known_state_hash is not None:
            body["known_state_hash"] = known_state_hash
        if pending_generation is not None:
            body["pending_generation"] = pending_generation
        if pending_state_hash is not None:
            body["pending_state_hash"] = pending_state_hash
        response = await self._client.post(
            f"{self.config.base_url}/v1/differential-context/reconcile", json=body
        )
        negotiated = response.http_version
        self._verify_http2(negotiated)
        if response.status_code >= 400:
            raise EdgeTransportError(
                f"reconcile endpoint returned {response.status_code}",
                status=response.status_code,
            )
        return response.json()

    # ------------------------------------------------------------------ #
    # Request path
    # ------------------------------------------------------------------ #

    def _plan_for(
        self,
        context_id: str,
        payload: dict[str, Any],
        epoch: str,
        fork_source: Optional[Tuple[str, int, str]],
        edge_request_id: str,
    ) -> Tuple[EdgeRequestPlan, Optional[str]]:
        """§7: when the stored per-context epoch differs from the reported
        one (or the binding is already marked invalid), the NEXT request is
        an explicit full re-registration (RESYNC_FULL,
        dc_resync_reason=server_epoch_changed) from the full canonical
        request already present locally. One large compressed WAN request
        per active context after a restart — acceptable and explicit."""
        state = self.store.get(context_id)
        if self.store.is_invalid(context_id) or (
            state is not None
            and state.server_epoch is not None
            and state.server_epoch != epoch
        ):
            known_generation = state.generation if state is not None else None
            known_state_hash = state.state_hash if state is not None else None
            logger.info(
                "dc_resync_reason=server_epoch_changed context_id=%s known_generation=%s",
                context_id,
                known_generation,
            )
            m.M_RESYNC.labels("server_epoch_changed").inc()
            return (
                build_resync_plan(
                    context_id,
                    payload,
                    reason="server_epoch_changed",
                    server_epoch=epoch,
                    known_generation=known_generation,
                    known_state_hash=known_state_hash,
                    edge_request_id=edge_request_id,
                ),
                "server_epoch_changed",
            )
        return build_edge_plan(
            context_id, payload, self.store, fork_source=fork_source, edge_request_id=edge_request_id
        ), None

    async def _post_edge(
        self,
        payload: dict[str, Any],
        plan: EdgeRequestPlan,
        meta: EdgeMeta,
        context_id: str,
        *,
        recovery: bool,
        fork_source: Optional[Tuple[str, int, str]] = None,
    ) -> Tuple[httpx.Response, str]:
        """Compress the DELTA envelope and POST it over HTTP/2. On 409 the
        deterministic recovery runs ONCE per request:

        1. epoch changed  -> mark invalid, refresh capabilities, resend as
           an explicit RESYNC_FULL (server_epoch_changed) — §7;
        2. authoritative matches the durable pending -> lost-ACK recovery:
           promote locally, re-plan the CURRENT request against the
           promoted baseline, resend — §8 (never replays the prior
           inference);
        3. anything else -> UnrecoverableEdgeConflict, raised — §9.

        The wire body = the edge envelope PLUS the generation parameters
        (stream, temperature, ...). ``messages`` never ride along — the
        delta IS the message content.
        """
        wire: dict[str, Any] = dict(plan.envelope)
        wire.update(
            {k: v for k, v in payload.items() if k not in _ENVELOPE_KEYS and k != "messages"}
        )
        envelope_bytes = json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        body, encoding, compress_ms = maybe_compress(
            envelope_bytes, self.config.request_compression, self.config.compression_threshold
        )
        m.M_COMPRESS_MS.observe(compress_ms)
        meta.delta_bytes = len(envelope_bytes)
        meta.wire_bytes = len(body)
        meta.compression_encoding = encoding
        if encoding != "identity" and envelope_bytes:
            m.M_RATIO.observe(len(envelope_bytes) / max(1, len(body)))
        m.M_FULL.inc(meta.full_logical_bytes)
        m.M_DELTA.inc(meta.delta_bytes)
        m.M_WIRE.inc(meta.wire_bytes)

        headers = {}
        if encoding != "identity":
            headers["Content-Encoding"] = encoding

        url = f"{self.config.base_url}/v1/differential-context/chat/completions"
        response = await self._client.post(url, content=body, headers=headers)
        negotiated = response.http_version
        self._verify_http2(negotiated)
        meta.negotiated_http_version = negotiated or "unknown"
        m.M_NEGOTIATED.labels(negotiated or "unknown").inc()

        if response.status_code == 409:
            m.M_CONFLICTS.inc()
            if not recovery:
                raise EdgeTransportError(
                    f"edge endpoint returned 409 after recovery for context {context_id}",
                    status=409,
                )
            doc: dict[str, Any] = {}
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    doc = parsed
            except Exception:
                pass
            conflict_epoch = doc.get("edge_server_epoch")
            authoritative_generation = doc.get("authoritative_generation")
            authoritative_state_hash = doc.get("authoritative_state_hash")

            if (
                conflict_epoch is not None
                and conflict_epoch != self.store.server_epoch()
            ):
                # §7: capabilities were stale — the remote restarted. The
                # current request becomes the explicit RESYNC_FULL against
                # freshly fetched capabilities (one request per context).
                logger.info(
                    "dc_resync_reason=server_epoch_changed context_id=%s old_epoch=%s new_epoch=%s",
                    context_id,
                    self.store.server_epoch(),
                    conflict_epoch,
                )
                self.store.mark_invalid(context_id)
                self._capabilities.clear()
                fresh_caps = await self._capabilities.get(self._client)
                fresh_epoch = fresh_caps.get("edge_server_epoch") or ""
                self.store.set_server_epoch(fresh_epoch)
                state = self.store.get(context_id)
                plan2 = build_resync_plan(
                    context_id,
                    payload,
                    reason="server_epoch_changed",
                    server_epoch=fresh_epoch,
                    known_generation=state.generation if state is not None else None,
                    known_state_hash=state.state_hash if state is not None else None,
                    edge_request_id=plan.edge_request_id,
                )
                m.M_RESYNC.labels("server_epoch_changed").inc()
                meta.operation = plan2.operation
                meta.resync_reason = "server_epoch_changed"
                meta.server_epoch = fresh_epoch
                await self.store.persist_pending(context_id, plan2)
                return await self._post_edge(
                    payload, plan2, meta, context_id, recovery=False
                )

            matched_pending = (
                self.store.pending_matches(
                    context_id, authoritative_generation, authoritative_state_hash, conflict_epoch
                )
                if authoritative_generation is not None
                and authoritative_state_hash is not None
                else None
            )
            if matched_pending is not None:
                # §8: lost-ACK recovery — the remote committed OUR pending
                # transition (current or previous send) but the ACK never
                # arrived. Promote locally (no replay of the prior
                # inference) and re-plan the CURRENT request against the
                # promoted baseline.
                m.M_LOST_ACK.inc()
                meta.lost_ack_recovered = True
                logger.info(
                    "dc_lost_ack_recovered context_id=%s generation=%s",
                    context_id,
                    authoritative_generation,
                )
                await self.store.promote_pending(
                    context_id,
                    ack_generation=authoritative_generation,
                    ack_state_hash=authoritative_state_hash,
                    epoch=conflict_epoch,
                    model=str(payload.get("model") or ""),
                    pending=matched_pending,
                )
                plan2 = build_edge_plan(
                    context_id,
                    payload,
                    self.store,
                    fork_source=None,
                    edge_request_id=plan.edge_request_id,
                )
                m.M_OP.labels(plan2.operation).inc()
                meta.operation = plan2.operation
                meta.delta_bytes = len(plan2.envelope_bytes or b"")
                await self.store.persist_pending(context_id, plan2)
                return await self._post_edge(
                    payload, plan2, meta, context_id, recovery=False
                )

            # §9: authoritative state matches neither the acknowledged nor
            # the one durable pending transition. Do not guess, do not
            # retry invisibly.
            m.M_UNRECOVERABLE.labels("unmatched_conflict").inc()
            raise UnrecoverableEdgeConflict(
                f"unrecoverable edge conflict for context {context_id}: "
                f"authoritative_generation={authoritative_generation} "
                f"authoritative_state_hash={authoritative_state_hash} "
                f"edge_server_epoch={conflict_epoch}",
                status=409,
            )

        if response.status_code >= 400:
            raise EdgeTransportError(
                f"edge endpoint returned {response.status_code}{_error_detail(response)}",
                status=response.status_code,
            )
        return response, negotiated

    def _verify_http2(self, negotiated: Optional[str]) -> None:
        meta_version = negotiated or "unknown"
        if self.config.strict_http2 and meta_version != "HTTP/2":
            raise EdgeTransportError(
                f"strict HTTP/2 violated: negotiated {meta_version} on the edge route "
                "(no silent downgrade)",
            )

    # ------------------------------------------------------------------ #
    # Disabled path (no stable session identity)
    # ------------------------------------------------------------------ #

    async def _completion_disabled(
        self, payload: dict[str, Any], resolution: SessionResolution
    ) -> Tuple[Any, EdgeMeta]:
        m.M_DISABLED.labels(resolution.disabled_reason or "unknown").inc()
        meta = EdgeMeta(
            context_id=None,
            disabled_reason=resolution.disabled_reason,
            full_logical_bytes=len(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ),
        )
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        compressed, encoding, compress_ms = maybe_compress(
            body, self.config.request_compression, self.config.compression_threshold
        )
        m.M_COMPRESS_MS.observe(compress_ms)
        meta.wire_bytes = len(compressed)
        meta.compression_encoding = encoding
        headers = {}
        if encoding != "identity":
            headers["Content-Encoding"] = encoding
        url = f"{self.config.base_url}/v1/chat/completions"
        response = await self._client.post(url, content=compressed, headers=headers)
        negotiated = response.http_version
        self._verify_http2(negotiated)
        meta.negotiated_http_version = negotiated or "unknown"
        m.M_NEGOTIATED.labels(negotiated or "unknown").inc()
        if response.status_code >= 400:
            raise EdgeTransportError(
                f"ordinary endpoint returned {response.status_code}{_error_detail(response)}",
                status=response.status_code,
            )
        if payload.get("stream"):
            return self._stream_response(response, resolution, payload, meta, disabled=True), meta
        return await self._plain_response(response, resolution, payload, meta, disabled=True), meta

    # ------------------------------------------------------------------ #
    # Response path
    # ------------------------------------------------------------------ #

    async def _plain_response(
        self,
        response: httpx.Response,
        resolution: Optional[SessionResolution],
        payload: dict[str, Any],
        meta: EdgeMeta,
        *,
        disabled: bool = False,
    ) -> dict[str, Any]:
        body = decompress_body(response.content, response.headers.get("content-encoding"))
        try:
            doc = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise EdgeTransportError(f"edge response is not JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise EdgeTransportError("edge response is not a JSON object")

        ack = doc.pop("differential_context", None)
        if isinstance(ack, dict) and not disabled:
            await self._commit_ack(ack, payload, meta)
        return doc

    def _stream_response(
        self,
        response: httpx.Response,
        resolution: Optional[SessionResolution],
        payload: dict[str, Any],
        meta: EdgeMeta,
        *,
        disabled: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        # HTTP/2 was already verified before this generator was returned.
        m.M_ACTIVE_STREAMS.inc()
        return self._iter_stream(response, resolution, payload, meta, disabled)

    async def _iter_stream(
        self,
        response: httpx.Response,
        resolution: Optional[SessionResolution],
        payload: dict[str, Any],
        meta: EdgeMeta,
        disabled: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            pending_ack_event = False
            async for line in response.aiter_lines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("event: "):
                    if stripped[7:].strip() == ACK_EVENT_NAME:
                        pending_ack_event = True
                    continue
                if pending_ack_event:
                    pending_ack_event = False
                    if stripped.startswith("data:"):
                        ack = parse_edge_ack(stripped[5:].strip().encode("utf-8"))
                        if ack is not None and not disabled:
                            await self._commit_ack(ack, payload, meta)
                    continue
                if not stripped.startswith("data:"):
                    continue
                payload_line = stripped[5:].strip()
                if payload_line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload_line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(chunk, dict):
                    yield chunk
        finally:
            m.M_ACTIVE_STREAMS.dec()
            await response.aclose()

    async def _commit_ack(
        self, ack: dict[str, Any], payload: dict[str, Any], meta: EdgeMeta
    ) -> None:
        context_id = meta.context_id
        if not context_id:
            return
        try:
            state = await self.store.commit_ack(
                context_id,
                ack,
                payload.get("messages") or [],
                model=str(payload.get("model") or ""),
            )
        except DifferentialContextError as exc:
            raise EdgeTransportError(f"invalid edge ACK: {exc.reason}") from exc
        if state is None:
            return
        meta.edge_generation = state.generation
        meta.edge_state_hash = state.state_hash
        m.M_GENERATION.labels(context_id).set(state.generation)

    # ------------------------------------------------------------------ #
    # Idle cleanup (§16)
    # ------------------------------------------------------------------ #

    async def _idle_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.idle_cleanup_interval_seconds)
            try:
                await self._cleanup_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("edge idle cleanup error: %s", exc)

    async def _cleanup_idle_once(self) -> None:
        """§16: on idle TTL expiry send RELEASE when the remote epoch still
        matches; delete the durable local state after a confirmed release;
        tombstone unconfirmed releases and retry on the next pass."""
        idle = await self.store.list_idle(self.config.idle_ttl_seconds)
        for context_id in idle:
            async with self.store.lock_for(context_id):
                state = self.store.get(context_id)
                epoch_ok = (
                    state is not None
                    and state.server_epoch is not None
                    and state.server_epoch == self.store.server_epoch()
                )
                if not epoch_ok:
                    # The remote binding is invalid (restart/TTL drop) — the
                    # release goal is already achieved; drop local state.
                    await self.store.delete_context(context_id)
                    continue
                try:
                    await self.release_context(
                        context_id, generation=state.generation
                    )
                    # confirmed release -> durable delete (inside
                    # release_context); also clears the tombstone.
                except Exception as exc:
                    await self.store.mark_tombstoned(context_id)
                    logger.warning(
                        "edge release unconfirmed, tombstoned context_id=%s: %s",
                        context_id,
                        exc,
                    )


__all__ = [
    "EdgeTransport",
    "EdgeTransportError",
    "UnrecoverableEdgeConflict",
    "EdgeMeta",
    "EDGE_REPRESENTATION",
]