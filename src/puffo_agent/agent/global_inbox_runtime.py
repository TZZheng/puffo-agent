"""Serial orchestration for the durable global agent Inbox.

Storage, prefix selection, provider context control, and sending deliberately
remain in their leaf modules.  This module joins those contracts and owns only
the mutable state of the one active turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .context_controller import (
    AdmissionCandidate,
    ContextController,
    DecisionOutcome,
    ProviderAdmissionEvent,
    ToolResultAdmission,
)
from .inbox_scheduler import (
    MAX_ESTIMATED_TOKENS,
    MAX_FORMATTED_BYTES,
    InboxCoalescer,
    InboxNoticeDelivery,
    InboxPlanner,
    NoticeDeliveryCapability,
    PlannedBatch,
)
from .message_store import (
    PRIOR_CONTEXT_MAX_BYTES,
    PRIOR_CONTEXT_MAX_ITEMS,
    MessageStore,
    ProcessingState,
    StoredMessage,
)
from ._logging import log_runtime_event
from .message_projection import format_message_group

logger = logging.getLogger(__name__)

OUTPUT_TOOL_RESERVE_TOKENS = 4_096
CURRENT_TURN_VERSION = 2


def conservative_token_estimate(text: str) -> int:
    """Conservative leaf estimate used for Inbox prompt text."""
    return len(text.encode("utf-8"))


async def await_listener_with_runtime(
    listener: Awaitable[Any],
    runtime_task: asyncio.Task[Any],
    *,
    label: str,
) -> Any:
    """Stop a transport listener as soon as its Inbox consumer exits."""
    listener_task = asyncio.ensure_future(listener)
    try:
        done, _pending = await asyncio.wait(
            {listener_task, runtime_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runtime_task in done:
            listener_failure: BaseException | None = None
            if listener_task in done:
                try:
                    await listener_task
                except (asyncio.CancelledError, Exception) as exc:
                    listener_failure = exc
            listener_diagnostic = (
                f"; listener also failed: {listener_failure}"
                if listener_failure is not None
                else ""
            )
            if runtime_task.cancelled():
                raise RuntimeError(
                    f"{label} was cancelled unexpectedly{listener_diagnostic}"
                )
            failure = runtime_task.exception()
            if failure is not None:
                raise RuntimeError(
                    f"{label} crashed: {failure}{listener_diagnostic}"
                ) from failure
            raise RuntimeError(f"{label} exited unexpectedly{listener_diagnostic}")
        return await listener_task
    finally:
        if not listener_task.done():
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass


@dataclass(frozen=True)
class MessageRoute:
    envelope_id: str
    kind: str
    space_id: str = ""
    channel_id: str = ""
    thread_root_id: str = ""
    dm_peer: str = ""

    @property
    def target(self) -> tuple[str, ...]:
        if self.kind == "dm":
            return ("dm", self.dm_peer)
        if self.kind == "thread":
            return ("thread", self.space_id, self.channel_id, self.thread_root_id)
        return ("channel", self.space_id, self.channel_id)


@dataclass(frozen=True)
class PlannedTurn:
    turn_id: str
    planning_cycle_key: str
    message_ids: tuple[str, ...]
    items: tuple[StoredMessage, ...]
    routes: tuple[MessageRoute, ...]
    targets: tuple[tuple[str, ...], ...]
    pending_targets: tuple[tuple[str, ...], ...]
    target_summary: str
    formatted_blocks: tuple[str, ...]
    provider_input: str
    formatted_tokens: int
    wrapper_overhead_tokens: int
    formatted_bytes: int
    wrapper_overhead_bytes: int
    more_available: bool = False
    notice_generation: int = 0
    requires_encryption: bool = False

    @property
    def candidate(self) -> AdmissionCandidate:
        first = (
            conservative_token_estimate(self.formatted_blocks[0])
            if self.formatted_blocks
            else 0
        )
        return AdmissionCandidate(
            planning_cycle_key=self.planning_cycle_key,
            formatted_batch_tokens=self.formatted_tokens,
            wrapper_overhead_tokens=self.wrapper_overhead_tokens,
            output_tool_reserve_tokens=OUTPUT_TOOL_RESERVE_TOKENS,
            payload=self,
            minimum_formatted_batch_tokens=first,
        )


@dataclass
class ActiveExactUnion:
    turn_id: str = ""
    message_ids: list[str] = field(default_factory=list)
    # In-memory only: rows whose plaintext has crossed a provider boundary.
    visible_message_ids: list[str] = field(default_factory=list)
    provider_session_id: str | None = None
    provider_turn_id: str | None = None
    routes: list[MessageRoute] = field(default_factory=list)
    through_by_channel: dict[tuple[str, str], int] = field(default_factory=dict)

    def clear(self) -> None:
        self.turn_id = ""
        self.message_ids.clear()
        self.visible_message_ids.clear()
        self.provider_session_id = None
        self.provider_turn_id = None
        self.routes.clear()
        self.through_by_channel.clear()


@dataclass(frozen=True)
class RuntimeHealth:
    state: str = "idle"
    diagnostic: str = ""


@dataclass
class HeldStaging:
    # Compatibility/status snapshot only. Admission never reads this mutable
    # field; each callback uses HeldAdmissionEvidence instead.
    message_ids: tuple[str, ...] = ()
    latest_seq: int | None = None
    latest_envelope_id: str | None = None
    recovered_through_seq: int | None = None
    more_pending: bool = False
    synchronized: bool = False
    diagnostic: str = ""
    correlation_key: str = ""


@dataclass(frozen=True)
class HeldAdmissionEvidence:
    """Local-only recovery evidence frozen for one held send result."""

    active_turn_id: str
    provider_session_id: str
    provider_turn_id: str
    space_id: str
    channel_id: str
    latest_seq: int
    latest_envelope_id: str
    displayed_ids: tuple[str, ...]


@dataclass
class SendAttemptState:
    attempts: int = 0
    destinations: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)

    def record(self, destination: str, state: str) -> None:
        self.attempts += 1
        self.destinations.append(destination)
        self.states.append(state)

    def reset(self) -> None:
        self.attempts = 0
        self.destinations.clear()
        self.states.clear()


def route_for(item: StoredMessage) -> MessageRoute:
    if item.envelope_kind == "dm":
        peer = item.sender_slug or item.recipient_slug or ""
        return MessageRoute(item.envelope_id, "dm", dm_peer=peer)
    # Intro prompts self-reference only to remain non-replyable in local
    # history. They still must authorize the top-level send they request.
    is_intro_prompt = (
        item.sender_slug == "system"
        and item.envelope_id.startswith("intro-prompt-")
        and item.thread_root_id == item.envelope_id
    )
    if item.thread_root_id and not is_intro_prompt:
        return MessageRoute(
            item.envelope_id,
            "thread",
            item.space_id or "",
            item.channel_id or "",
            item.thread_root_id,
        )
    return MessageRoute(
        item.envelope_id,
        "channel",
        item.space_id or "",
        item.channel_id or "",
    )


def format_stored_message(
    item: StoredMessage,
    *,
    current_agent_aliases: Sequence[str] = (),
) -> str:
    """Readable per-message model view (legacy entry point for callers)."""
    return format_message_group((item,), current_agent_aliases=current_agent_aliases)


class BaselineAdapter:
    def __init__(self, store: MessageStore):
        self.store = store

    async def get_context_baseline_seq(
        self, space_id: str, channel_id: str
    ) -> int | None:
        return await self.store.get_context_baseline(space_id, channel_id)


class ActiveBoundaryAdapter:
    """Inject the active turn id into the authoritative store query."""

    def __init__(self, store: MessageStore, active: ActiveExactUnion):
        self.store = store
        self.active = active

    async def get_active_turn_through_seq(
        self, space_id: str, channel_id: str
    ) -> int | None:
        if not self.active.turn_id:
            return None
        persisted = await self.store.get_model_visible_through_seq(
            self.active.turn_id, space_id, channel_id
        )
        # ``through_by_channel`` contains only Store-proven values written by
        # ``advance`` below (including a history-tool candidate), never raw
        # selected rows.
        advanced = self.active.through_by_channel.get((space_id, channel_id))
        values = [value for value in (persisted, advanced) if value is not None]
        return max(values) if values else None

    async def advance_active_turn_through_seq(
        self, space_id: str, channel_id: str, seq: int
    ) -> None:
        if not self.active.turn_id:
            return
        key = (space_id, channel_id)
        proven = await self.store.get_model_visible_through_seq(
            self.active.turn_id, space_id, channel_id, candidate_seq=seq,
        )
        # With no locally-known rows there is no local blocker to prove; this
        # preserves the stateless boundary adapter contract used before the
        # receipt path has populated the Store.
        if proven is not None:
            self.active.through_by_channel[key] = proven
        elif not await self.store.has_known_channel_rows_above_baseline(
            space_id, channel_id,
        ):
            self.active.through_by_channel[key] = seq


class TrackingSendDelegate:
    """Mark semantic attempts before awaiting the worker-owned coordinator."""

    def __init__(
        self,
        coordinator: Any,
        attempts: SendAttemptState,
        runtime: "GlobalInboxRuntime | None" = None,
    ):
        self.coordinator = coordinator
        self.attempts = attempts
        self.runtime = runtime

    async def send(self, request: Any = None, **kwargs: Any) -> dict[str, Any]:
        destination = ""
        if isinstance(request, Mapping):
            destination = str(
                request.get("destination", request.get("channel", ""))
            )
        elif request is not None:
            destination = str(getattr(request, "destination", ""))
        else:
            destination = str(kwargs.get("destination", kwargs.get("channel", "")))
        send_anyway = bool(
            request.get("send_anyway", False)
            if isinstance(request, Mapping)
            else getattr(request, "send_anyway", False)
            if request is not None
            else kwargs.get("send_anyway", False)
        )
        prior_held = any(
            prior_destination == destination and prior_state == "held"
            for prior_destination, prior_state in zip(
                self.attempts.destinations, self.attempts.states
            )
        )
        is_dm = destination.startswith("@")
        transport = (
            "keyless"
            if bool(getattr(getattr(self.coordinator, "http_client", None), "keyless", False))
            else "dm"
            if is_dm
            else "channel"
        )
        mode = (
            "send_anyway" if send_anyway else "require_current"
        ) if not is_dm else None
        attempt_phase = "reconsider" if prior_held else "initial"
        self.attempts.record(destination, "attempting")
        attempt = self.attempts.attempts
        runtime = self.runtime
        active = runtime.active if runtime is not None else None
        route = (
            runtime.resolve_active_send_route(destination, request, kwargs)
            if runtime is not None
            else None
        )
        started = time.monotonic()
        send_attempt_id = (
            f"{active.turn_id}:{attempt}"
            if active is not None and active.turn_id
            else f"unbound:{attempt}"
        )
        log_runtime_event(
            logger,
            "send.attempted",
            level=logging.INFO,
            agent_id=runtime.agent_id if runtime is not None else None,
            turn_id=active.turn_id if active is not None else None,
            provider_session_id=(
                active.provider_session_id if active is not None else None
            ),
            provider_turn_id=(
                active.provider_turn_id if active is not None else None
            ),
            space_id=route.space_id if route is not None else None,
            channel_id=route.channel_id if route is not None else None,
            thread_root_id=route.thread_root_id if route is not None else None,
            dm_peer=route.dm_peer if route is not None else None,
            mode=mode,
            attempt_phase=attempt_phase,
            transport=transport,
            attempt=attempt,
            send_attempt_id=send_attempt_id,
            state="attempting",
        )
        try:
            result = await self.coordinator.send(request, **kwargs)
        except BaseException:
            self.attempts.states[-1] = "failed"
            log_runtime_event(
                logger,
                "send.failed",
                level=logging.INFO,
                agent_id=runtime.agent_id if runtime is not None else None,
                turn_id=active.turn_id if active is not None else None,
                provider_session_id=(
                    active.provider_session_id if active is not None else None
                ),
                provider_turn_id=(
                    active.provider_turn_id if active is not None else None
                ),
                space_id=route.space_id if route is not None else None,
                channel_id=route.channel_id if route is not None else None,
                thread_root_id=route.thread_root_id if route is not None else None,
                dm_peer=route.dm_peer if route is not None else None,
                mode=mode,
                attempt_phase=attempt_phase,
                transport=transport,
                attempt=attempt,
                send_attempt_id=send_attempt_id,
                state="failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                error_category="delegate_exception",
            )
            raise
        reconsideration_audit = result.pop("_reconsideration_audit", {})
        if not isinstance(reconsideration_audit, Mapping):
            reconsideration_audit = {}
        state = str(result.get("state", "failed"))
        self.attempts.states[-1] = state
        if send_anyway:
            eligible = bool(reconsideration_audit.get("eligible"))
            if not reconsideration_audit:
                eligible = (
                    result.get("error_kind") != "reconsideration_ineligible"
                )
            log_runtime_event(
                logger,
                (
                    "reconsideration.blocked"
                    if not eligible
                    else "reconsideration.eligible"
                ),
                agent_id=runtime.agent_id if runtime is not None else None,
                turn_id=(
                    reconsideration_audit.get("turn_id")
                    or (active.turn_id if active is not None else None)
                ),
                provider_session_id=(
                    reconsideration_audit.get("provider_session_id")
                    or (
                        active.provider_session_id
                        if active is not None else None
                    )
                ),
                provider_turn_id=(
                    active.provider_turn_id if active is not None else None
                ),
                space_id=route.space_id if route is not None else None,
                channel_id=route.channel_id if route is not None else None,
                envelope_id=reconsideration_audit.get(
                    "latest_envelope_id"
                ),
                latest_seq=reconsideration_audit.get("latest_seq"),
                seen_seq=reconsideration_audit.get("admitted_seq"),
                transport=transport,
                mode=mode,
                send_attempt_id=send_attempt_id,
                outcome="accepted" if eligible else "rejected",
                decision_reason=(
                    reconsideration_audit.get("decision_reason")
                    or (
                        "synchronized_and_admitted"
                        if eligible else "reconsideration_ineligible"
                    )
                ),
            )
        if state == "held":
            source = getattr(self.coordinator, "held_recovery_source", None)
            runtime = getattr(source, "runtime", None)
            staging = getattr(runtime, "held", None)
            if staging is not None:
                result["synchronized"] = bool(staging.synchronized)
                if staging.recovered_through_seq is not None:
                    result["recovered_through_seq"] = staging.recovered_through_seq
                result["recovery_more_pending"] = bool(staging.more_pending)
                if staging.diagnostic:
                    result["synchronization_diagnostic"] = staging.diagnostic
                if staging.correlation_key:
                    result["continuation_correlation_key"] = (
                        staging.correlation_key
                    )
            log_runtime_event(
                logger,
                "held.synchronized",
                agent_id=runtime.agent_id if runtime is not None else None,
                turn_id=active.turn_id if active is not None else None,
                provider_session_id=(
                    active.provider_session_id if active is not None else None
                ),
                send_attempt_id=send_attempt_id,
                latest_seq=result.get("latest_seq"),
                outcome=(
                    "available" if result.get("synchronized") else "unavailable"
                ),
            )
        event = {
            "held": "send.held",
            "sent": "send.committed",
        }.get(state, "send.failed")
        log_runtime_event(
            logger,
            event,
            level=logging.INFO,
            agent_id=runtime.agent_id if runtime is not None else None,
            turn_id=active.turn_id if active is not None else None,
            provider_session_id=(
                active.provider_session_id if active is not None else None
            ),
            provider_turn_id=(
                active.provider_turn_id if active is not None else None
            ),
            correlation_key=result.get("continuation_correlation_key"),
            envelope_id=result.get("envelope_id"),
            space_id=route.space_id if route is not None else None,
            channel_id=route.channel_id if route is not None else None,
            thread_root_id=route.thread_root_id if route is not None else None,
            dm_peer=route.dm_peer if route is not None else None,
            context_baseline_seq=result.get("context_baseline_seq"),
            seen_seq=result.get("seen_seq"),
            latest_seq=result.get("latest_seq") if state == "held" else None,
            latest_seq_before_send=(
                result.get("latest_seq_before_send")
                if state == "sent"
                else None
            ),
            seq=result.get("seq"),
            mode=mode,
            attempt_phase=attempt_phase,
            transport=transport,
            attempt=attempt,
            send_attempt_id=send_attempt_id,
            state=state,
            duration_ms=int((time.monotonic() - started) * 1000),
            error_category=result.get("error_kind"),
        )
        return result

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        return await self.send(kwargs)


class HeldRecoverySource:
    """Live durable held catch-up for the one active exact turn.

    This deliberately does not invent a remote catch-up API.  It waits for
    receipt commits, then proves the exact terminal watermark exists as a
    pending durable row for the active provider session. The proof is
    synchronization metadata only; it is independent of content pagination
    and never claims that the provider saw the recovered content.
    """

    def __init__(
        self,
        runtime: "GlobalInboxRuntime",
        *,
        wait_timeout_s: float = 2.0,
        catchup_pending: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self.runtime = runtime
        self.wait_timeout_s = wait_timeout_s
        self.catchup_pending = catchup_pending
        self._changed = asyncio.Event()

    def notify_delivery(self) -> None:
        self._changed.set()

    async def wait_for_held_delivery(
        self,
        space_id: str,
        channel_id: str,
        latest_seq: int,
        latest_envelope_id: str,
    ) -> bool:
        async def proven() -> bool:
            row = await self.runtime.store.get_message_by_envelope(
                latest_envelope_id
            )
            return bool(
                row is not None
                and row.server_seq == latest_seq
                and row.space_id == space_id
                and row.channel_id == channel_id
            )

        deadline = time.monotonic() + self.wait_timeout_s
        while not await proven():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._changed.clear()
            # Re-check after clearing so a commit between the first query and
            # clear cannot be lost.
            if await proven():
                return True
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
        if await proven():
            return True
        if self.catchup_pending is not None:
            try:
                await self.catchup_pending(latest_envelope_id)
            except Exception:
                pass
            if await proven():
                return True
        self.runtime.held = HeldStaging(
            latest_seq=latest_seq,
            latest_envelope_id=latest_envelope_id,
            diagnostic=(
                "exact held watermark unavailable after WebSocket wait "
                "and signed pending catch-up"
            ),
        )
        return False

    async def query_held_messages(
        self,
        space_id: str,
        channel_id: str,
        latest_seq: int,
        latest_envelope_id: str,
        provider_session_id: str | None,
    ) -> Sequence[Mapping[str, Any]]:
        active = self.runtime.active
        if (
            not active.turn_id
            or not provider_session_id
            or active.provider_session_id != provider_session_id
        ):
            self.runtime.held = HeldStaging(
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                diagnostic="stateful active provider session unavailable",
            )
            return ()
        watermark = await self.runtime.store.get_message_by_envelope(
            latest_envelope_id
        )
        if (
            watermark is None
            or watermark.server_seq != latest_seq
            or watermark.space_id != space_id
            or watermark.channel_id != channel_id
            or watermark.processing_state is not ProcessingState.PENDING
        ):
            self.runtime.held = HeldStaging(
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                diagnostic="exact held watermark mismatch",
            )
            return ()
        self.runtime.held = HeldStaging(
            latest_seq=latest_seq,
            latest_envelope_id=latest_envelope_id,
            recovered_through_seq=latest_seq,
            synchronized=True,
        )
        through = await self.runtime.store.get_model_visible_through_seq(
            active.turn_id, space_id, channel_id,
        )
        rows = await self.runtime.store.get_held_reconsideration_rows(
            space_id=space_id, channel_id=channel_id,
            after_seq=through if through is not None else -1,
            through_seq=latest_seq, limit=51,
        )
        if len(rows) > 50:
            self.runtime.held = HeldStaging(
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                diagnostic="held context exceeds the bounded recovery limit",
            )
            return ()
        # The terminal pair has already been checked above; retain it in the
        # returned local row set so the coordinator can prove exact readiness.
        projected: list[Mapping[str, Any]] = []
        for row in rows:
            projected.append({
                "space_id": row.space_id, "channel_id": row.channel_id,
                "thread_root_id": row.thread_root_id or "",
                "envelope_id": row.envelope_id, "server_seq": row.server_seq,
                "latest_seq": latest_seq, "latest_envelope_id": latest_envelope_id,
                "provider_session_id": provider_session_id, "sender_slug": row.sender_slug,
                "envelope_kind": row.envelope_kind, "sent_at": row.sent_at,
                "is_encrypted": row.is_encrypted, "content": row.content,
            })
        evidence = HeldAdmissionEvidence(
            active_turn_id=active.turn_id,
            provider_session_id=provider_session_id,
            provider_turn_id=active.provider_turn_id or "",
            space_id=space_id, channel_id=channel_id,
            latest_seq=latest_seq, latest_envelope_id=latest_envelope_id,
            displayed_ids=tuple(row.envelope_id for row in rows),
        )
        self.runtime._held_admission_evidence[
            (active.turn_id, provider_session_id, space_id, channel_id,
             latest_seq, latest_envelope_id)
        ] = evidence
        self.runtime.held.message_ids = evidence.displayed_ids
        return tuple(projected)


TurnRunner = Callable[[PlannedTurn], Awaitable[Any]]
UnfitPolicy = Callable[..., bool | Awaitable[bool]]


class GlobalInboxRuntime:
    """One serial provider boundary over the durable global Inbox."""

    def __init__(
        self,
        *,
        store: MessageStore,
        adapter: Any,
        run_turn: TurnRunner,
        workspace: str | Path,
        context_controller: ContextController | None = None,
        planner: InboxPlanner | None = None,
        coalescer: InboxCoalescer | None = None,
        formatter: Callable[[StoredMessage], str] = format_stored_message,
        estimator: Callable[[str], int] = conservative_token_estimate,
        unfit_policy: UnfitPolicy | None = None,
        coordinator: Any | None = None,
        max_context_decisions: int = 12,
        max_api_retries: int = 2,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        held_catchup: Callable[[str], Awaitable[bool]] | None = None,
        send_mode_keys: Sequence[str] = (),
        agent_id: str = "",
        notice_delivery: InboxNoticeDelivery | None = None,
        runtime_event_outbox: Any | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.run_turn = run_turn
        self.workspace = Path(workspace)
        self.context_controller = context_controller or ContextController(adapter)
        self.planner = planner or InboxPlanner()
        self.coalescer = coalescer or InboxCoalescer()
        self._configured_formatter = formatter
        self.estimator = estimator
        self.unfit_policy = unfit_policy or (lambda *_args, **_kwargs: True)
        self.coordinator = coordinator
        self.active = ActiveExactUnion()
        self.attempts = SendAttemptState()
        self.health = RuntimeHealth()
        self.held = HeldStaging()
        self._held_admission_evidence: dict[
            tuple[str, str, str, str, int, str], HeldAdmissionEvidence
        ] = {}
        self._boundary = asyncio.Lock()
        self._stopping = False
        self._degraded = False
        self._defer_requeued_resume = False
        self.max_context_decisions = max_context_decisions
        self.max_api_retries = max_api_retries
        self.retry_sleep = retry_sleep
        self.send_mode_keys = tuple(
            dict.fromkeys(key for key in send_mode_keys if key)
        )
        self.agent_id = agent_id
        # ``send_mode_keys`` is the existing runtime identity-alias set used
        # by the send-mode guard (normally the configured agent id and the
        # wire slug).  Inbox attribution is derived from the same identities;
        # no durable or provider state is introduced.
        self.formatter = self._format_for_provider
        self.notice_delivery = notice_delivery or InboxNoticeDelivery(
            NoticeDeliveryCapability.NEXT_TURN
        )
        self.runtime_event_outbox = runtime_event_outbox
        self.send_delegate: TrackingSendDelegate | None = None
        self.held_recovery_source = HeldRecoverySource(
            self,
            catchup_pending=held_catchup,
        )

    def _current_agent_identity_aliases(self) -> tuple[str, ...]:
        """Return only runtime-owned identities usable for self attribution."""
        values: list[str] = [self.agent_id, *self.send_mode_keys]
        for owner in (self.adapter, self.coordinator):
            for name in ("slug", "agent_id", "agent_slug", "self_slug"):
                value = getattr(owner, name, "")
                if value:
                    values.append(str(value))
        return tuple(dict.fromkeys(value for value in values if value))

    def _format_for_provider(self, item: StoredMessage) -> str:
        """Apply current-Agent attribution only to the production formatter."""
        if self._configured_formatter is format_stored_message:
            return format_stored_message(
                item,
                current_agent_aliases=self._current_agent_identity_aliases(),
            )
        return self._configured_formatter(item)

    @property
    def current_turn_path(self) -> Path:
        return self.workspace / ".puffo-agent" / "current_turn.json"

    def notify(self) -> None:
        self._degraded = False
        self.coalescer.notify()

    def notify_delivery(self) -> None:
        self.held_recovery_source.notify_delivery()

    @property
    def _admits_tool_results_on_return(self) -> bool:
        return (
            getattr(
                self.adapter,
                "tool_result_admission_boundary",
                "provider_completion",
            )
            == "tool_return"
        )

    async def _admit_returned_tool_result(
        self,
        callback: Callable[[ProviderAdmissionEvent], Awaitable[None]],
        *,
        planning_cycle_key: str,
        provider_session_id: str,
        provider_turn_id: str | None = None,
    ) -> None:
        await callback(ProviderAdmissionEvent(
            planning_cycle_key=planning_cycle_key,
            provider_session_id=provider_session_id,
            provider_turn_id=(
                provider_turn_id
                if provider_turn_id is not None else self.active.provider_turn_id
            ),
            admitted_at=datetime.now(timezone.utc),
        ))

    async def stage_held_send_result(
        self,
        result: dict[str, Any],
        *,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Stage exactly the semantic held rows at the original tool boundary."""
        payload = result.get("reconsideration")
        if not isinstance(payload, Mapping) or not payload.get("context_ready"):
            return result
        active_turn_id = self.active.turn_id
        provider_session_id = self.active.provider_session_id
        if not active_turn_id or not provider_session_id:
            payload = dict(payload)
            payload["context_ready"] = False
            payload["diagnostic"] = "active held admission identity unavailable"
            result["reconsideration"] = payload
            return result
        target = payload.get("target") if isinstance(payload.get("target"), Mapping) else {}
        space_id = str(target.get("space_id") or "")
        channel_id = str(target.get("channel_id") or "")
        latest_seq = payload.get("latest_seq")
        latest_envelope_id = payload.get("latest_envelope_id")
        if (
            not space_id or not channel_id or not isinstance(latest_seq, int)
            or not isinstance(latest_envelope_id, str)
        ):
            return result
        evidence_key = (active_turn_id, provider_session_id, space_id, channel_id,
                        latest_seq, latest_envelope_id)
        evidence = self._held_admission_evidence.get(evidence_key)
        if evidence is None or not evidence.displayed_ids:
            payload = dict(payload)
            payload["context_ready"] = False
            payload["diagnostic"] = "exact held recovery evidence unavailable"
            result["reconsideration"] = payload
            return result
        displayed_ids = evidence.displayed_ids
        correlation_key = f"held_send_{uuid.uuid4().hex}"
        receipt = uuid.uuid4().hex
        marker = ToolResultAdmission.build(
            lambda _event: None, correlation_key,
            evidence.provider_turn_id,
            correlation_receipt=receipt,
        ).receipt_marker
        fired = False

        async def admit_held(event: ProviderAdmissionEvent) -> None:
            nonlocal fired
            if fired or event.planning_cycle_key != correlation_key:
                return
            if (
                self.active.turn_id != active_turn_id
                or self.active.provider_session_id != provider_session_id
                or self.active.provider_turn_id != evidence.provider_turn_id
                or event.provider_session_id != provider_session_id
                or event.provider_turn_id != evidence.provider_turn_id
            ):
                return
            rows = [await self.store.get_message_by_envelope(item) for item in displayed_ids]
            if any(
                row is None or row.space_id != space_id or row.channel_id != channel_id
                or row.server_seq is None or row.server_seq > latest_seq
                for row in rows
            ):
                return
            if any(
                row is not None and row.processing_state is not ProcessingState.PENDING
                and not (row.processing_state is ProcessingState.IN_TURN
                         and row.processing_turn_id == active_turn_id)
                for row in rows
            ):
                return
            pending_ids = [row.envelope_id for row in rows if row is not None and row.processing_state is ProcessingState.PENDING]
            try:
                if pending_ids:
                    await self.store.admit_messages(
                        pending_ids, turn_id=active_turn_id,
                        provider_session_id=provider_session_id,
                    )
                await self._add_visible_message_ids(
                    displayed_ids, space_id=space_id, channel_id=channel_id,
                    through_seq=latest_seq,
                )
                # The durable turn-membership table is authoritative.  In
                # particular, do not construct this union from the previous
                # in-memory snapshot: another same-turn admission may already
                # have changed it, and callers retain this list object.
                durable_rows = await self.store.get_in_turn_messages(
                    active_turn_id, provider_session_id,
                )
                durable_ids = [row.envelope_id for row in durable_rows]
                self.active.message_ids[:] = durable_ids
                durable_routes = [route_for(row) for row in durable_rows]
                self.active.routes[:] = list(dict.fromkeys(durable_routes))
                self._write_active_current_turn()
                await ActiveBoundaryAdapter(self.store, self.active).advance_active_turn_through_seq(
                    space_id, channel_id, latest_seq,
                )
            except Exception:
                return
            fired = True
            self._held_admission_evidence.pop(evidence_key, None)

        if self._admits_tool_results_on_return:
            await self._admit_returned_tool_result(
                admit_held, planning_cycle_key=correlation_key,
                provider_session_id=provider_session_id,
                provider_turn_id=evidence.provider_turn_id,
            )
        else:
            register = getattr(self.adapter, "register_continuation_callback", None)
            if not callable(register):
                payload = dict(payload)
                payload["context_ready"] = False
                payload["diagnostic"] = "provider cannot correlate held tool results"
                result["reconsideration"] = payload
                return result
            register(
                admit_held, correlation_key, tool_names=(tool_name,),
                tool_arguments=dict(tool_arguments), correlation_receipt=receipt,
            )
        result["tool_result_admission"] = marker
        self.held.correlation_key = correlation_key
        return result

    def _write_active_current_turn(self) -> None:
        """Atomically persist the exact active union after held admission."""
        targets = tuple(dict.fromkeys(route.target for route in self.active.routes))
        self._write_current_turn(PlannedTurn(
            turn_id=self.active.turn_id,
            planning_cycle_key="held_admission",
            message_ids=tuple(self.active.message_ids),
            items=(), routes=tuple(self.active.routes), targets=targets,
            pending_targets=(), target_summary="", formatted_blocks=(),
            provider_input="", formatted_tokens=0, wrapper_overhead_tokens=0,
            formatted_bytes=0, wrapper_overhead_bytes=0,
        ))

    def note_input_ready(self, turn_id: str) -> None:
        self.notice_delivery.note_input_ready(turn_id)

    async def offer_busy_notice(self, *, turn_id: str) -> bool:
        """Offer metadata-only work to the named active Turn when safe."""
        if turn_id != self.active.turn_id:
            return False
        planned = await self.plan_pending()
        if planned is None:
            return False
        decision = await self.context_controller.decide(
            planned.candidate, self._replacement_candidate
        )
        if decision.outcome is not DecisionOutcome.ADMIT:
            return False
        planned = decision.candidate.payload
        if not isinstance(planned, PlannedTurn):
            return False
        offer = getattr(self.adapter, "offer_inbox_notice", None)
        if not callable(offer):
            return False

        async def deliver() -> bool:
            return bool(await offer(turn_id, planned.provider_input))

        accepted = await self.notice_delivery.offer(
            named_turn_id=turn_id,
            active_turn_id=self.active.turn_id,
            deliver=deliver,
        )
        if not accepted:
            return False
        return await self.store.mark_notice_delivered(planned.notice_generation)

    def resolve_active_send_route(
        self,
        destination: str,
        request: Any,
        kwargs: Mapping[str, Any],
    ) -> MessageRoute | None:
        """Return only a route already resolved for the active durable turn."""
        root_id = (
            request.get("root_id", "")
            if isinstance(request, Mapping)
            else getattr(request, "root_id", "")
            if request is not None
            else kwargs.get("root_id", "")
        )
        dm_peer = destination[1:] if destination.startswith("@") else ""
        for route in self.active.routes:
            if route.kind == "dm":
                if dm_peer and route.dm_peer == dm_peer:
                    return route
                continue
            if route.channel_id != destination:
                continue
            if root_id:
                if route.thread_root_id == str(root_id):
                    return route
                continue
            if route.kind == "channel":
                return route
        return None

    def resolve_plain_fallback_route(self) -> MessageRoute | None:
        """Return an implicit route only when admitted context is unambiguous."""
        unique: dict[tuple[str, ...], MessageRoute] = {}
        for route in self.active.routes:
            unique.setdefault(route.target, route)
        if len(unique) != 1:
            return None
        return next(iter(unique.values()))

    @staticmethod
    def _batch_route_projection(planned: PlannedTurn) -> list[dict[str, Any]]:
        projected: dict[tuple[str, ...], dict[str, Any]] = {}
        for item, route in zip(planned.items, planned.routes):
            entry = projected.setdefault(
                route.target,
                {
                    "space_id": route.space_id,
                    "channel_id": route.channel_id,
                    "thread_root_id": route.thread_root_id,
                    "dm_peer": route.dm_peer,
                    "count": 0,
                    "min_seq": None,
                    "max_seq": None,
                },
            )
            entry["count"] += 1
            if item.server_seq is not None:
                current_min = entry["min_seq"]
                current_max = entry["max_seq"]
                entry["min_seq"] = (
                    item.server_seq
                    if current_min is None
                    else min(current_min, item.server_seq)
                )
                entry["max_seq"] = (
                    item.server_seq
                    if current_max is None
                    else max(current_max, item.server_seq)
                )
        return [
            {
                key: value
                for key, value in route.items()
                if value not in (None, "")
            }
            for route in projected.values()
        ]

    async def stage_model_visible_read(
        self,
        *,
        space_id: str,
        channel_id: str,
        through_seq: int,
        through_envelope_id: str,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        visible_message_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Advance freshness at the adapter's proven model-visible boundary."""
        if through_seq < 0:
            raise RuntimeError("model-visible read sequence must be non-negative")
        row = await self.store.get_message_by_envelope(through_envelope_id)
        if (
            row is None
            or row.server_seq != through_seq
            or row.space_id != space_id
            or row.channel_id != channel_id
            or row.envelope_kind == "dm"
        ):
            log_runtime_event(
                logger,
                "history.read_staged",
                level=logging.DEBUG,
                agent_id=self.agent_id,
                turn_id=self.active.turn_id,
                provider_session_id=self.active.provider_session_id,
                envelope_id=through_envelope_id,
                space_id=space_id,
                channel_id=channel_id,
                latest_seq=through_seq,
                state=(
                    "dm_unsupported"
                    if row is not None and row.envelope_kind == "dm"
                    else "invalid_watermark"
                ),
            )
            raise RuntimeError("model-visible read watermark does not match local storage")
        active_turn_id = self.active.turn_id
        provider_session_id = self.active.provider_session_id
        if not active_turn_id or not provider_session_id:
            log_runtime_event(
                logger,
                "history.read_staged",
                level=logging.DEBUG,
                agent_id=self.agent_id,
                envelope_id=through_envelope_id,
                space_id=space_id,
                channel_id=channel_id,
                latest_seq=through_seq,
                state="no_active_turn",
            )
            raise RuntimeError("no active provider turn for model-visible read")
        visible_ids = list(visible_message_ids or ())
        await self._add_visible_message_ids(
            visible_ids,
            space_id=space_id,
            channel_id=channel_id,
            through_seq=through_seq,
            mutate=False,
        )
        correlation_key = f"visible_read_{uuid.uuid4().hex}"
        correlation_receipt = uuid.uuid4().hex
        fired = False

        async def admit_read(event: ProviderAdmissionEvent) -> None:
            nonlocal fired
            if fired or event.planning_cycle_key != correlation_key:
                return
            if (
                self.active.turn_id != active_turn_id
                or self.active.provider_session_id != provider_session_id
                or event.provider_session_id != provider_session_id
            ):
                raise RuntimeError(
                    "model-visible read admission crossed the active provider turn"
                )
            fired = True
            try:
                await self._add_visible_message_ids(
                    visible_ids,
                    space_id=space_id,
                    channel_id=channel_id,
                    through_seq=through_seq,
                )
                await ActiveBoundaryAdapter(
                    self.store, self.active
                ).advance_active_turn_through_seq(
                    space_id,
                    channel_id,
                    through_seq,
                )
            except Exception:
                log_runtime_event(
                    logger,
                    "history.read_staged",
                    level=logging.DEBUG,
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    provider_turn_id=event.provider_turn_id,
                    tool_call_id=event.tool_call_id,
                    correlation_key=correlation_key,
                    envelope_id=through_envelope_id,
                    space_id=space_id,
                    channel_id=channel_id,
                    latest_seq=through_seq,
                    state="admission_failed",
                )
                raise
            log_runtime_event(
                logger,
                "history.read_admitted",
                level=logging.DEBUG,
                agent_id=self.agent_id,
                turn_id=active_turn_id,
                provider_session_id=provider_session_id,
                provider_turn_id=event.provider_turn_id,
                tool_call_id=event.tool_call_id,
                correlation_key=correlation_key,
                envelope_id=through_envelope_id,
                space_id=space_id,
                channel_id=channel_id,
                latest_seq=through_seq,
                state="admitted",
            )

        state = "staged"
        if self._admits_tool_results_on_return:
            log_runtime_event(
                logger,
                "history.read_staged",
                level=logging.DEBUG,
                agent_id=self.agent_id,
                turn_id=active_turn_id,
                provider_session_id=provider_session_id,
                correlation_key=correlation_key,
                envelope_id=through_envelope_id,
                space_id=space_id,
                channel_id=channel_id,
                latest_seq=through_seq,
                state="tool_return",
            )
            await self._admit_returned_tool_result(
                admit_read,
                planning_cycle_key=correlation_key,
                provider_session_id=provider_session_id,
            )
            state = "admitted"
        else:
            register_continuation = getattr(
                self.adapter, "register_continuation_callback", None
            )
            if not callable(register_continuation):
                log_runtime_event(
                    logger,
                    "history.read_staged",
                    level=logging.DEBUG,
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    correlation_key=correlation_key,
                    envelope_id=through_envelope_id,
                    space_id=space_id,
                    channel_id=channel_id,
                    latest_seq=through_seq,
                    state="unsupported_adapter",
                )
                raise RuntimeError(
                    "provider cannot correlate model-visible history results"
                )
            try:
                register_continuation(
                    admit_read,
                    correlation_key,
                    tool_names=(tool_name,),
                    tool_arguments=dict(tool_arguments),
                    correlation_receipt=correlation_receipt,
                )
            except Exception:
                log_runtime_event(
                    logger,
                    "history.read_staged",
                    level=logging.DEBUG,
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    correlation_key=correlation_key,
                    envelope_id=through_envelope_id,
                    space_id=space_id,
                    channel_id=channel_id,
                    latest_seq=through_seq,
                    state="registration_failed",
                )
                raise
            log_runtime_event(
                logger,
                "history.read_staged",
                level=logging.DEBUG,
                agent_id=self.agent_id,
                turn_id=active_turn_id,
                provider_session_id=provider_session_id,
                correlation_key=correlation_key,
                envelope_id=through_envelope_id,
                space_id=space_id,
                channel_id=channel_id,
                latest_seq=through_seq,
                state="staged",
            )
        return {
            "state": state,
            "correlation_key": correlation_key,
            "correlation_receipt": correlation_receipt,
            "space_id": space_id,
            "channel_id": channel_id,
            "through_seq": through_seq,
            "through_envelope_id": through_envelope_id,
        }

    async def _add_visible_message_ids(
        self,
        envelope_ids: list[str] | tuple[str, ...],
        *,
        space_id: str | None = None,
        channel_id: str | None = None,
        through_seq: int | None = None,
        mutate: bool = True,
    ) -> None:
        """Validate local rows first, then append first-seen IDs in order."""
        ids = list(envelope_ids)
        if any(not isinstance(item, str) or not item for item in ids):
            raise RuntimeError("visible message IDs must be non-empty strings")
        rows = [await self.store.get_message_by_envelope(item) for item in ids]
        for row in rows:
            if row is None:
                raise RuntimeError("visible message ID does not resolve locally")
            if space_id is not None and (
                row.envelope_kind == "dm" or row.space_id != space_id
                or row.channel_id != channel_id or not isinstance(row.server_seq, int)
                or isinstance(row.server_seq, bool) or through_seq is None
                or row.server_seq > through_seq
            ):
                raise RuntimeError("visible message ID is incompatible with watermark")
        if mutate:
            seen = set(self.active.visible_message_ids)
            self.active.visible_message_ids.extend(
                item for item in ids if not (item in seen or seen.add(item))
            )

    async def read_inbox(
        self,
        *,
        target: str = "",
        cursor: str = "",
        limit: int = 50,
        tool_arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one content page and admit it at the adapter's read boundary."""
        active_turn_id = self.active.turn_id
        provider_session_id = self.active.provider_session_id
        if not active_turn_id or not provider_session_id:
            raise RuntimeError("no active provider turn for Inbox read")
        page = await self.store.read_inbox_page(
            target=target, cursor=cursor, limit=limit
        )
        blocks: list[str] = []
        selected: list[StoredMessage] = []
        byte_count = 0
        for item in page.items:
            block = self.formatter(item)
            block_bytes = len(block.encode("utf-8"))
            if byte_count + block_bytes > MAX_FORMATTED_BYTES:
                break
            blocks.append(block)
            selected.append(item)
            byte_count += block_bytes
        truncated = len(selected) < len(page.items)
        if truncated:
            # Re-page from the last actually returned item while retaining the
            # store-pinned ceiling/generation.
            if not selected:
                raise RuntimeError("oldest Inbox message exceeds the page byte guard")
            decoded = self.store._decode_inbox_cursor(
                page.next_cursor
                or self.store._encode_inbox_cursor(
                    {
                        "v": 1,
                        "target": page.target,
                        "generation": page.snapshot_generation,
                        "ceiling": list(
                            self.store._inbox_order(page.items[-1])
                        ),
                        "last": [-1, -1, -1, ""],
                    }
                )
            )
            decoded["last"] = list(self.store._inbox_order(selected[-1]))
            next_cursor = self.store._encode_inbox_cursor(decoded)
            has_more = True
            remaining_count = page.remaining_count + len(page.items) - len(selected)
        else:
            next_cursor = page.next_cursor
            has_more = page.has_more
            remaining_count = page.remaining_count

        prior_context: list[str] = []
        prior_context_ids: list[str] = []
        if selected:
            anchors: dict[tuple[str, ...], StoredMessage] = {}
            for item in selected:
                anchors.setdefault(route_for(item).target, item)
            prior_rows: dict[str, StoredMessage] = {}
            for anchor in anchors.values():
                for item in await self.store.get_prior_context(
                    anchor,
                    limit=PRIOR_CONTEXT_MAX_ITEMS,
                    max_bytes=PRIOR_CONTEXT_MAX_BYTES,
                ):
                    prior_rows.setdefault(item.envelope_id, item)
            prior_byte_count = 0
            for item in sorted(prior_rows.values(), key=self.store._inbox_order):
                block = self.formatter(item)
                block_bytes = len(block.encode("utf-8"))
                if (
                    len(prior_context) >= PRIOR_CONTEXT_MAX_ITEMS
                    or prior_byte_count + block_bytes > PRIOR_CONTEXT_MAX_BYTES
                ):
                    continue
                prior_context.append(block)
                prior_context_ids.append(item.envelope_id)
                prior_byte_count += block_bytes

        correlation_receipt = ""
        if selected:
            correlation_key = f"inbox_read_{uuid.uuid4().hex}"
            correlation_receipt = uuid.uuid4().hex
            fired = False
            ids = tuple(item.envelope_id for item in selected)
            routes = tuple(route_for(item) for item in selected)

            async def admit_read(event: ProviderAdmissionEvent) -> None:
                nonlocal fired
                if fired or event.planning_cycle_key != correlation_key:
                    return
                if (
                    self.active.turn_id != active_turn_id
                    or self.active.provider_session_id != provider_session_id
                    or event.provider_session_id != provider_session_id
                ):
                    raise RuntimeError("Inbox read admission crossed the active Turn")
                run = await self.store.admit_messages(
                    ids,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                )
                fired = True
                self.active.message_ids[:] = list(run.message_ids)
                await self._add_visible_message_ids(list(ids) + prior_context_ids)
                self.active.routes.extend(routes)
                turn_rows = await self.store.get_in_turn_messages(
                    active_turn_id,
                    provider_session_id,
                )
                if turn_rows:
                    self._write_current_turn(
                        self._reconstruct_exact_turn(
                            turn_id=active_turn_id,
                            rows=turn_rows,
                        )
                    )
                for item, route in zip(selected, routes):
                    if item.server_seq is not None and route.kind != "dm":
                        key = (route.space_id, route.channel_id)
                        proven = await self.store.get_model_visible_through_seq(
                            active_turn_id, route.space_id, route.channel_id
                        )
                        if proven is not None:
                            self.active.through_by_channel[key] = proven
                    log_runtime_event(
                        logger,
                        "inbox.row_in_turn",
                        agent_id=self.agent_id,
                        turn_id=active_turn_id,
                        provider_session_id=provider_session_id,
                        provider_turn_id=event.provider_turn_id,
                        tool_call_id=event.tool_call_id,
                        correlation_key=correlation_key,
                        message_id=item.envelope_id,
                        server_seq=item.server_seq,
                        target=self.store.target_projection(item),
                        notice_generation=page.snapshot_generation,
                        outcome="in_turn",
                    )
                log_runtime_event(
                    logger,
                    "inbox.read_admitted",
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    provider_turn_id=event.provider_turn_id,
                    correlation_key=correlation_key,
                    notice_generation=page.snapshot_generation,
                    message_count=len(ids),
                    outcome="admitted",
                )

            arguments = dict(
                tool_arguments
                if tool_arguments is not None
                else {"target": target, "cursor": cursor, "limit": limit}
            )
            if self._admits_tool_results_on_return:
                log_runtime_event(
                    logger,
                    "inbox.read_staged",
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    correlation_key=correlation_key,
                    notice_generation=page.snapshot_generation,
                    message_count=len(ids),
                    remaining_count=remaining_count,
                    outcome="tool_return",
                )
                await self._admit_returned_tool_result(
                    admit_read,
                    planning_cycle_key=correlation_key,
                    provider_session_id=provider_session_id,
                )
            else:
                register = getattr(
                    self.adapter, "register_continuation_callback", None
                )
                if not callable(register):
                    raise RuntimeError(
                        "provider cannot correlate Inbox tool results"
                    )
                register(
                    admit_read,
                    correlation_key,
                    tool_names=("read_inbox",),
                    tool_arguments=arguments,
                    correlation_receipt=correlation_receipt,
                )
                log_runtime_event(
                    logger,
                    "inbox.read_staged",
                    agent_id=self.agent_id,
                    turn_id=active_turn_id,
                    provider_session_id=provider_session_id,
                    correlation_key=correlation_key,
                    notice_generation=page.snapshot_generation,
                    message_count=len(ids),
                    remaining_count=remaining_count,
                    outcome="staged",
                )
        return {
            "messages": blocks,
            "prior_context": prior_context,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "remaining_count": remaining_count,
            "snapshot_generation": page.snapshot_generation,
            "correlation_receipt": correlation_receipt,
        }

    async def run(self) -> None:
        await self.recover_current_turn()
        await self.recover_orphaned_turns()
        if await self.store.rearm_stranded_notice():
            log_runtime_event(
                logger,
                "notice.armed",
                agent_id=self.agent_id,
                mode="startup",
                outcome="rearmed",
            )
        if self._defer_requeued_resume:
            # Consume the recovery wake without immediately feeding the same
            # failed durable union through the initial-turn path.
            await self.coalescer.wait_for_burst()
        elif await self.store.get_pending(limit=1):
            notice = await self.store.get_notice_state()
            remaining = (
                max(
                    0.0,
                    (notice.first_pending_deadline_ms - int(time.time() * 1000))
                    / 1000,
                )
                if notice.first_pending_deadline_ms is not None
                else 0.0
            )
            self.coalescer.notify(delay_seconds=remaining)
        while not self._stopping:
            await self.coalescer.wait_for_burst()
            if self._stopping:
                break
            await self.process_once()

    async def recover_orphaned_turns(self) -> int:
        """Requeue active DB Turns left without a resumable crash join."""
        recovered = 0
        for run in await self.store.get_active_turn_runs():
            if run.message_ids:
                await self.store.requeue_messages(
                    run.message_ids,
                    turn_id=run.turn_id,
                )
            else:
                await self.store.finalize_empty_turn(
                    turn_id=run.turn_id,
                    state="requeued",
                    rearm_notice=True,
                )
            log_runtime_event(
                logger,
                "turn.requeued",
                agent_id=self.agent_id,
                turn_id=run.turn_id,
                provider_session_id=run.provider_session_id,
                state="requeued",
                mode="startup_orphan_recovery",
                message_count=len(run.message_ids),
                outcome="requeued",
            )
            recovered += 1
        return recovered

    def stop(self) -> None:
        self._stopping = True
        self.coalescer.notify()

    def _target_summary(
        self,
        targets: tuple[tuple[str, ...], ...],
        count: int,
        *,
        more_pending: bool,
        pending_targets: tuple[tuple[str, ...], ...],
    ) -> str:
        return json.dumps(
            {
                "version": 2,
                "message_count": count,
                "more_pending": more_pending,
                "pending_targets": pending_targets,
                "targets": targets,
            },
            separators=(",", ":"),
        )

    def _from_batch(
        self,
        batch: PlannedBatch,
        *,
        turn_id: str | None = None,
        planning_cycle_key: str | None = None,
    ) -> PlannedTurn | None:
        if not batch.items:
            return None
        routes = tuple(route_for(item) for item in batch.items)
        targets: list[tuple[str, ...]] = []
        for route in routes:
            if route.target not in targets:
                targets.append(route.target)
        target_tuple = tuple(targets)
        summary = self._target_summary(
            target_tuple,
            len(batch.items),
            more_pending=batch.more_available,
            pending_targets=batch.pending_target_projections,
        )
        prefix = f"<global_inbox_turn>\n{summary}\n"
        suffix = "\n</global_inbox_turn>"
        provider_input = prefix + "\n".join(batch.formatted_messages) + suffix
        wrapper_bytes = len((prefix + suffix).encode("utf-8"))
        wrapper_tokens = self.estimator(prefix + suffix)
        return PlannedTurn(
            turn_id=turn_id or f"turn_{uuid.uuid4().hex}",
            planning_cycle_key=planning_cycle_key or f"plan_{uuid.uuid4().hex}",
            message_ids=batch.message_ids,
            items=batch.items,
            routes=routes,
            targets=target_tuple,
            pending_targets=batch.pending_target_projections,
            target_summary=summary,
            formatted_blocks=batch.formatted_messages,
            provider_input=provider_input,
            formatted_tokens=batch.estimated_tokens,
            wrapper_overhead_tokens=wrapper_tokens,
            formatted_bytes=batch.formatted_bytes,
            wrapper_overhead_bytes=wrapper_bytes,
            more_available=batch.more_available,
        )

    async def plan_pending(
        self,
        *,
        items: tuple[StoredMessage, ...] | None = None,
        max_items: int | None = None,
        turn_id: str | None = None,
        planning_cycle_key: str | None = None,
    ) -> PlannedTurn | None:
        pending_universe = (
            items if items is not None else await self.store.get_pending()
        )
        if items is None:
            notice = await self.store.get_notice_state()
            if not pending_universe or not notice.delivery_pending:
                return None
            routes = tuple(route_for(item) for item in pending_universe)
            targets: list[tuple[str, ...]] = []
            normalized_counts: dict[str, int] = {}
            for route in routes:
                if route.target not in targets:
                    targets.append(route.target)
            for item in pending_universe:
                projection = self.store.target_projection(item)
                normalized_counts[projection] = (
                    normalized_counts.get(projection, 0) + 1
                )
            latest_seq = max(
                (
                    item.server_seq
                    for item in pending_universe
                    if item.server_seq is not None
                ),
                default=None,
            )
            summary = json.dumps(
                {
                    "version": 3,
                    "content_included": False,
                    "read_tool": "read_inbox",
                    "generation": notice.generation,
                    "message_count": notice.pending_count,
                    "targets": [
                        {"target": target, "count": count}
                        for target, count in normalized_counts.items()
                    ],
                    "latest_seq": latest_seq,
                },
                separators=(",", ":"),
            )
            provider_input = (
                "<global_inbox_notice>\n"
                + summary
                + "\n</global_inbox_notice>"
            )
            log_runtime_event(
                logger,
                "notice.due",
                agent_id=self.agent_id,
                notice_generation=notice.generation,
                requires_encryption=any(
                    item.is_encrypted for item in pending_universe
                ),
                message_count=notice.pending_count,
                target_count=len(targets),
                latest_seq=latest_seq,
                outcome="planned",
            )
            return PlannedTurn(
                turn_id=turn_id or f"turn_{uuid.uuid4().hex}",
                planning_cycle_key=planning_cycle_key or f"notice_{uuid.uuid4().hex}",
                message_ids=(),
                items=(),
                routes=(),
                targets=tuple(targets),
                pending_targets=tuple(targets),
                target_summary=summary,
                formatted_blocks=(),
                provider_input=provider_input,
                formatted_tokens=0,
                wrapper_overhead_tokens=self.estimator(provider_input),
                formatted_bytes=0,
                wrapper_overhead_bytes=len(provider_input.encode("utf-8")),
                more_available=True,
                notice_generation=notice.generation,
            )
        pending = pending_universe
        if max_items is not None:
            pending = pending[:max_items]
        while pending:
            batch = self.planner.plan(
                pending, formatter=self.formatter, estimator=self.estimator
            )
            if batch.unfit_head_id:
                changed = await self.planner.resolve_unfit_head(
                    batch,
                    policy=self.unfit_policy,
                    quarantine=self.store.quarantine_pending,
                )
                if changed:
                    pending_universe = await self.store.get_pending()
                    pending = pending_universe
                    if max_items is not None:
                        pending = pending[:max_items]
                    continue
                return None
            selected_ids = set(batch.message_ids)
            remaining_targets: list[tuple[str, ...]] = []
            seen_remaining_ids: set[str] = set()
            seen_remaining_targets: set[tuple[str, ...]] = set()
            for item in pending_universe:
                if (
                    item.envelope_id in selected_ids
                    or item.envelope_id in seen_remaining_ids
                ):
                    continue
                seen_remaining_ids.add(item.envelope_id)
                target = self.planner.target_projection(item)
                if target not in seen_remaining_targets:
                    seen_remaining_targets.add(target)
                    remaining_targets.append(target)
            batch = replace(
                batch,
                pending_target_projections=tuple(remaining_targets),
                more_available=bool(seen_remaining_ids),
            )
            planned = self._from_batch(
                batch,
                turn_id=turn_id,
                planning_cycle_key=planning_cycle_key,
            )
            if planned is None:
                return None
            if (
                planned.formatted_bytes + planned.wrapper_overhead_bytes
                <= MAX_FORMATTED_BYTES
                and planned.formatted_tokens + planned.wrapper_overhead_tokens
                <= MAX_ESTIMATED_TOKENS
            ):
                mode = (
                    "continuation"
                    if planned.planning_cycle_key.startswith("continuation_")
                    else "recovery"
                    if planned.planning_cycle_key.startswith("recovery_")
                    else "initial"
                )
                log_runtime_event(
                    logger,
                    "batch.planned",
                    level=logging.DEBUG,
                    agent_id=self.agent_id,
                    turn_id=planned.turn_id,
                    batch_id=planned.planning_cycle_key,
                    correlation_key=planned.planning_cycle_key,
                    envelope_id=(
                        planned.message_ids[0]
                        if len(planned.message_ids) == 1
                        else None
                    ),
                    mode=mode,
                    state="planned",
                    message_count=len(planned.message_ids),
                    target_count=len(planned.targets),
                    envelope_count=len(planned.message_ids),
                    envelope_ids=list(planned.message_ids[:16]),
                    routes=self._batch_route_projection(planned)[:16],
                    first_seq=next(
                        (
                            item.server_seq
                            for item in planned.items
                            if item.server_seq is not None
                        ),
                        None,
                    ),
                    last_seq=next(
                        (
                            item.server_seq
                            for item in reversed(planned.items)
                            if item.server_seq is not None
                        ),
                        None,
                    ),
                    formatted_bytes=(
                        planned.formatted_bytes + planned.wrapper_overhead_bytes
                    ),
                )
                return planned
            # Wrapper/container is charged: remove a real FIFO suffix and replan.
            pending = pending[: len(batch.items) - 1]
        return None

    async def _replacement_candidate(
        self, previous: AdmissionCandidate
    ) -> AdmissionCandidate:
        old = previous.payload
        replacement = await self.plan_pending(
            turn_id=getattr(old, "turn_id", None),
            planning_cycle_key=previous.planning_cycle_key,
        )
        return replacement.candidate if replacement else previous

    async def _admit(
        self, planned: PlannedTurn, event: ProviderAdmissionEvent
    ) -> None:
        if event.planning_cycle_key != planned.planning_cycle_key:
            raise RuntimeError("provider admission did not correlate to planned turn")
        # Durable Inbox recovery is keyed by the provider session.  Stateless
        # local adapters must not create a run that can never be authenticated
        # on recovery (or expose identity-required MCP history tools).
        if not event.provider_session_id:
            raise RuntimeError("provider does not support durable Inbox admission")
        if planned.message_ids:
            run = await self.store.admit_messages(
                planned.message_ids,
                turn_id=planned.turn_id,
                provider_session_id=event.provider_session_id,
            )
        else:
            state = await self.store.get_notice_state()
            if (
                state.generation != planned.notice_generation
                or state.pending_count == 0
            ):
                raise RuntimeError("Inbox notice became stale before admission")
            run = await self.store.start_turn(
                turn_id=planned.turn_id,
                provider_session_id=event.provider_session_id,
            )
            if not await self.store.mark_notice_delivered(
                planned.notice_generation
            ):
                raise RuntimeError("Inbox notice generation was already delivered")
        self.active.turn_id = run.turn_id
        self.active.message_ids[:] = list(run.message_ids)
        await self._add_visible_message_ids(list(run.message_ids))
        self.active.provider_session_id = event.provider_session_id
        self.active.provider_turn_id = event.provider_turn_id
        self.active.routes[:] = list(planned.routes)
        if self.coordinator is not None:
            self.coordinator.provider_session_id = event.provider_session_id
        log_runtime_event(
            logger,
            "turn.admitted",
            level=logging.DEBUG,
            agent_id=self.agent_id,
            turn_id=run.turn_id,
            batch_id=event.planning_cycle_key,
            provider_session_id=event.provider_session_id,
            provider_turn_id=event.provider_turn_id,
            correlation_key=event.planning_cycle_key,
            envelope_id=run.message_ids[0] if len(run.message_ids) == 1 else None,
            state=run.state,
            message_count=len(run.message_ids),
        )
        if planned.notice_generation:
            log_runtime_event(
                logger,
                "notice.admitted",
                agent_id=self.agent_id,
                turn_id=run.turn_id,
                provider_session_id=event.provider_session_id,
                provider_turn_id=event.provider_turn_id,
                notice_generation=planned.notice_generation,
                outcome="admitted",
            )

    def _write_current_turn(self, planned: PlannedTurn) -> None:
        path = self.current_turn_path
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "version": CURRENT_TURN_VERSION,
            "turn_id": planned.turn_id,
            "message_ids": list(planned.message_ids),
            "targets": [list(target) for target in planned.targets],
            "routes": [asdict(route) for route in planned.routes],
            "provider_session_id": self.active.provider_session_id,
            "provider_turn_id": self.active.provider_turn_id,
        }
        if self.runtime_event_outbox is not None:
            outbox_state = self.runtime_event_outbox.state()
            body.update({
                "logical_session_ref": outbox_state.get("session_ref", ""),
                "logical_turn_ref": outbox_state.get("active_turn_ref", ""),
                "native_session_id": outbox_state.get("native_session_id", ""),
            })
        if len(planned.targets) == 1 and planned.routes:
            route = planned.routes[0]
            body["channel_id"] = route.channel_id
            body["root_id"] = route.thread_root_id
            body["triggering_post_id"] = planned.message_ids[0]
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(body, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    async def process_once(self) -> bool:
        if self._degraded:
            return False
        async with self._boundary:
            process_started = time.monotonic()
            planned = await self.plan_pending()
            if planned is None:
                self.health = RuntimeHealth()
                return False

            rollover_seen = False
            decisions = 0
            while True:
                decisions += 1
                if decisions > self.max_context_decisions:
                    self.health = RuntimeHealth(
                        "degraded", "context decision budget exhausted"
                    )
                    self._degraded = True
                    return False
                decision = await self.context_controller.decide(
                    planned.candidate, self._replacement_candidate
                )
                log_runtime_event(
                    logger,
                    "context.checked",
                    agent_id=self.agent_id,
                    turn_id=planned.turn_id,
                    notice_generation=planned.notice_generation,
                    projected_tokens=decision.projected_tokens,
                    used_tokens_before=decision.snapshot.used_tokens,
                    context_window=decision.snapshot.context_window,
                    outcome=decision.outcome.value,
                )
                if decision.outcome is DecisionOutcome.REPLAN:
                    log_runtime_event(
                        logger,
                        "context.compacted",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        notice_generation=planned.notice_generation,
                        outcome="completed",
                    )
                    replacement = decision.candidate.payload
                    if not isinstance(replacement, PlannedTurn):
                        self.health = RuntimeHealth(
                            "degraded", "context replan did not return a planned turn"
                        )
                        self._degraded = True
                        return False
                    planned = replacement
                    continue
                if decision.outcome is DecisionOutcome.SHRINK:
                    if len(planned.items) <= 1:
                        self.health = RuntimeHealth("degraded", decision.diagnostic)
                        self._degraded = True
                        return False
                    planned = await self.plan_pending(
                        items=planned.items[:-1],
                        turn_id=planned.turn_id,
                        planning_cycle_key=planned.planning_cycle_key,
                    )
                    if planned is None:
                        return False
                    continue
                if decision.outcome is DecisionOutcome.ROLLOVER:
                    log_runtime_event(
                        logger,
                        "context.rollover",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        notice_generation=planned.notice_generation,
                        outcome="completed",
                    )
                    if rollover_seen:
                        self.health = RuntimeHealth(
                            "degraded", "provider rollover re-evaluation was exhausted"
                        )
                        self._degraded = True
                        return False
                    rollover_seen = True
                    rolled_session = self.adapter.get_provider_session_id()
                    if self.coordinator is not None:
                        self.coordinator.provider_session_id = rolled_session
                    self.active.provider_session_id = rolled_session
                    continue
                if decision.outcome is DecisionOutcome.DEGRADED:
                    self.health = RuntimeHealth("degraded", decision.diagnostic)
                    self._degraded = True
                    return False
                break

            if planned.notice_generation:
                current_notice = await self.store.get_notice_state()
                if (
                    current_notice.pending_count == 0
                    or current_notice.generation != planned.notice_generation
                    or not current_notice.delivery_pending
                ):
                    self.health = RuntimeHealth()
                    return False
            # The crash join must exist before provider admission is possible.
            self._write_current_turn(planned)
            self.attempts.reset()
            self.adapter.register_admission_callback(
                lambda event: self._admit(planned, event),
                planned.planning_cycle_key,
            )
            self.health = RuntimeHealth("in_progress", "")
            terminal = False
            admitted = False
            from . import send_mode

            send_mode.note_turn_bundle(
                list(self.send_mode_keys),
                planned.requires_encryption
                or any(item.is_encrypted for item in planned.items),
            )
            try:
                retries = 0
                while True:
                    try:
                        result = await (
                            self.run_turn(planned)
                            if retries == 0
                            else self._run_retry(planned)
                        )
                        break
                    except Exception as exc:
                        from .core import AgentAPIError

                        is_api_error = isinstance(exc, AgentAPIError)
                        is_auth = is_api_error and exc.is_auth
                        if (
                            is_api_error
                            and not is_auth
                            and self.active.turn_id == planned.turn_id
                            and retries < self.max_api_retries
                            and hasattr(self.run_turn, "handle_global_inbox_retry")
                        ):
                            retries += 1
                            await self.retry_sleep(min(2 ** (retries - 1), 4))
                            self.active.provider_turn_id = None
                            self.adapter.register_admission_callback(
                                lambda event: self._admit_retry_attempt(
                                    planned, event
                                ),
                                planned.planning_cycle_key,
                            )
                            continue
                        raise
                admitted = self.active.turn_id == planned.turn_id
                if admitted:
                    if self.active.message_ids:
                        await self.store.mark_processed(
                            tuple(self.active.message_ids), turn_id=planned.turn_id
                        )
                    else:
                        await self.store.finalize_empty_turn(
                            turn_id=planned.turn_id,
                            rearm_notice=True,
                        )
                    for item_id in self.active.message_ids:
                        row = await self.store.get_message_by_envelope(item_id)
                        log_runtime_event(
                            logger,
                            "inbox.row_processed",
                            agent_id=self.agent_id,
                            turn_id=planned.turn_id,
                            provider_session_id=self.active.provider_session_id,
                            message_id=item_id,
                            server_seq=row.server_seq if row is not None else None,
                            outcome="processed",
                        )
                    log_runtime_event(
                        logger,
                        "turn.processed",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        provider_session_id=self.active.provider_session_id,
                        provider_turn_id=self.active.provider_turn_id,
                        envelope_id=(
                            self.active.message_ids[0]
                            if len(self.active.message_ids) == 1
                            else None
                        ),
                        state=ProcessingState.PROCESSED.value,
                        message_count=len(self.active.message_ids),
                        duration_ms=int(
                            (time.monotonic() - process_started) * 1000
                        ),
                    )
                    terminal = True
                else:
                    self.health = RuntimeHealth(
                        "degraded", "provider returned without correlated admission"
                    )
                    self._degraded = True
            except asyncio.CancelledError:
                if self.active.turn_id == planned.turn_id:
                    if self.active.message_ids:
                        await self.store.requeue_messages(
                            tuple(self.active.message_ids), turn_id=planned.turn_id
                        )
                    else:
                        await self.store.finalize_empty_turn(
                            turn_id=planned.turn_id,
                            state="requeued",
                            rearm_notice=True,
                        )
                    for item_id in self.active.message_ids:
                        row = await self.store.get_message_by_envelope(item_id)
                        log_runtime_event(
                            logger,
                            "inbox.row_requeued",
                            agent_id=self.agent_id,
                            turn_id=planned.turn_id,
                            provider_session_id=self.active.provider_session_id,
                            message_id=item_id,
                            server_seq=row.server_seq if row is not None else None,
                            outcome="requeued",
                        )
                    log_runtime_event(
                        logger,
                        "turn.requeued",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        provider_session_id=self.active.provider_session_id,
                        provider_turn_id=self.active.provider_turn_id,
                        state="requeued",
                        message_count=len(self.active.message_ids),
                        duration_ms=int(
                            (time.monotonic() - process_started) * 1000
                        ),
                        error_category="cancelled",
                    )
                terminal = True
                raise
            except Exception:
                if self.active.turn_id == planned.turn_id:
                    if self.active.message_ids:
                        await self.store.requeue_messages(
                            tuple(self.active.message_ids), turn_id=planned.turn_id
                        )
                    else:
                        await self.store.finalize_empty_turn(
                            turn_id=planned.turn_id,
                            state="requeued",
                            rearm_notice=True,
                        )
                    for item_id in self.active.message_ids:
                        row = await self.store.get_message_by_envelope(item_id)
                        log_runtime_event(
                            logger,
                            "inbox.row_requeued",
                            agent_id=self.agent_id,
                            turn_id=planned.turn_id,
                            provider_session_id=self.active.provider_session_id,
                            message_id=item_id,
                            server_seq=row.server_seq if row is not None else None,
                            outcome="requeued",
                        )
                    log_runtime_event(
                        logger,
                        "turn.requeued",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        provider_session_id=self.active.provider_session_id,
                        provider_turn_id=self.active.provider_turn_id,
                        state="requeued",
                        message_count=len(self.active.message_ids),
                        duration_ms=int(
                            (time.monotonic() - process_started) * 1000
                        ),
                        error_category="provider_error",
                    )
                    terminal = True
                self.health = RuntimeHealth("degraded", "turn failed and was requeued")
            finally:
                send_mode.clear_turn_bundle(list(self.send_mode_keys))
                self.adapter.register_admission_callback(None, "")
                was_active = self.active.turn_id == planned.turn_id
                if terminal:
                    log_runtime_event(
                        logger,
                        "turn.finalized",
                        agent_id=self.agent_id,
                        turn_id=planned.turn_id,
                        provider_session_id=self.active.provider_session_id,
                        provider_turn_id=self.active.provider_turn_id,
                        notice_generation=planned.notice_generation,
                        message_count=len(self.active.message_ids),
                        outcome=(
                            "processed"
                            if self.health.state == "in_progress"
                            else "requeued"
                        ),
                    )
                if terminal or not was_active:
                    self.active.clear()
                if self.coordinator is not None:
                    self.coordinator.provider_session_id = None
                if terminal or not was_active:
                    try:
                        self.current_turn_path.unlink()
                    except FileNotFoundError:
                        pass
                if self.health.state == "in_progress":
                    self.health = RuntimeHealth()
            if not self._degraded and await self.store.get_pending(limit=1):
                self.notify()
            return True

    async def _run_retry(self, planned: PlannedTurn) -> Any:
        retry = getattr(self.run_turn, "handle_global_inbox_retry", None)
        if retry is None:
            raise RuntimeError("global runtime retry is unavailable")
        return await retry(planned)

    async def _admit_retry_attempt(
        self, planned: PlannedTurn, event: ProviderAdmissionEvent
    ) -> None:
        """Correlate a real retry turn without re-admitting the durable union."""
        if (
            event.planning_cycle_key != planned.planning_cycle_key
            or self.active.turn_id != planned.turn_id
        ):
            raise RuntimeError("provider retry admission crossed the active turn")
        self.active.provider_session_id = event.provider_session_id
        self.active.provider_turn_id = event.provider_turn_id
        if self.coordinator is not None:
            self.coordinator.provider_session_id = event.provider_session_id

    def _clear_terminal_turn(self) -> None:
        self.adapter.register_admission_callback(None, "")
        self.active.clear()
        if self.coordinator is not None:
            self.coordinator.provider_session_id = None
        try:
            self.current_turn_path.unlink()
        except OSError:
            pass

    def _reconstruct_exact_turn(
        self,
        *,
        turn_id: str,
        rows: tuple[StoredMessage, ...],
    ) -> PlannedTurn:
        blocks = tuple(self.formatter(row) for row in rows)
        routes = tuple(route_for(row) for row in rows)
        targets: list[tuple[str, ...]] = []
        for route in routes:
            if route.target not in targets:
                targets.append(route.target)
        target_tuple = tuple(targets)
        summary = self._target_summary(
            target_tuple,
            len(rows),
            more_pending=False,
            pending_targets=(),
        )
        prefix = f"<global_inbox_turn>\n{summary}\n"
        suffix = "\n</global_inbox_turn>"
        return PlannedTurn(
            turn_id=turn_id,
            planning_cycle_key=f"recovery_{turn_id}",
            message_ids=tuple(row.envelope_id for row in rows),
            items=rows,
            routes=routes,
            targets=target_tuple,
            pending_targets=(),
            target_summary=summary,
            formatted_blocks=blocks,
            provider_input=prefix + "\n".join(blocks) + suffix,
            formatted_tokens=sum(self.estimator(block) for block in blocks),
            wrapper_overhead_tokens=self.estimator(prefix + suffix),
            formatted_bytes=sum(len(block.encode("utf-8")) for block in blocks),
            wrapper_overhead_bytes=len((prefix + suffix).encode("utf-8")),
        )

    async def recover_current_turn(self) -> bool:
        """Finish or unwind a durable crash join before normal planning."""
        recovery_started = time.monotonic()
        raw: Any = None
        try:
            raw = json.loads(self.current_turn_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            self.health = RuntimeHealth("degraded", "invalid crash join")
            self._clear_terminal_turn()
            return False

        turn_id = raw.get("turn_id") if isinstance(raw, dict) else None
        run = (
            await self.store.get_turn_run(turn_id)
            if isinstance(turn_id, str) and turn_id
            else None
        )
        durable_ids = (
            tuple(run.message_ids)
            if run is not None and run.state == ProcessingState.IN_TURN.value
            else ()
        )

        async def unwind(
            diagnostic: str,
            state: str = "degraded",
            *,
            defer_requeued_resume: bool = False,
        ) -> bool:
            requeued = False
            if run is not None and durable_ids:
                # Driver-backed harnesses cannot resume an in-flight provider
                # turn. Persist the terminal boundary before making the exact
                # Inbox union eligible for a replacement attempt.
                if self.runtime_event_outbox is not None:
                    from .runtime_events import RuntimeEvent

                    outbox_state = self.runtime_event_outbox.state()
                    public_turn_ref = outbox_state.get("active_turn_ref", "")
                    session_ref = outbox_state.get("session_ref", "")
                    native_session_id = outbox_state.get(
                        "native_session_id", ""
                    )
                    join_matches_outbox = (
                        public_turn_ref
                        and session_ref
                        and native_session_id == run.provider_session_id
                        and (
                            not isinstance(raw, dict)
                            or (
                                raw.get("provider_session_id")
                                in {None, run.provider_session_id}
                                and raw.get("native_session_id", native_session_id)
                                == native_session_id
                                and raw.get("logical_session_ref", session_ref)
                                == session_ref
                                and raw.get("logical_turn_ref", public_turn_ref)
                                == public_turn_ref
                            )
                        )
                    )
                    if join_matches_outbox:
                        occurred_at = (
                            datetime.fromtimestamp(
                                run.started_at / 1000, tz=timezone.utc
                            )
                            .isoformat(timespec="milliseconds")
                            .replace("+00:00", "Z")
                        )
                        await self.runtime_event_outbox.enqueue(RuntimeEvent(
                            agent_id=self.agent_id,
                            session_ref=session_ref,
                            turn_ref=public_turn_ref,
                            type="turn.finished",
                            payload={"outcome": "abandoned"},
                            event_id=(
                                f"evt_abandoned_{self.agent_id}_"
                                f"{session_ref}_{public_turn_ref}_{run.turn_id}"
                            ),
                            occurred_at=occurred_at,
                        ), terminal=True)
                        self.runtime_event_outbox.set_active_turn(
                            None, session_ref=session_ref,
                            native_session_id=native_session_id,
                        )
                # The MessageStore run is authoritative across restarts.  An
                # in-memory ActiveExactUnion may be empty or stale after the
                # process boundary and must never narrow the durable union.
                exact_ids = durable_ids
                await self.store.requeue_messages(exact_ids, turn_id=run.turn_id)
                log_runtime_event(
                    logger,
                    "turn.requeued",
                    agent_id=self.agent_id,
                    turn_id=run.turn_id,
                    provider_session_id=run.provider_session_id,
                    state="requeued",
                    mode="recovery",
                    message_count=len(exact_ids),
                    duration_ms=int(
                        (time.monotonic() - recovery_started) * 1000
                    ),
                    error_category=state,
                )
                requeued = True
            elif (
                run is not None
                and run.state == ProcessingState.IN_TURN.value
                and not durable_ids
            ):
                await self.store.finalize_empty_turn(
                    turn_id=run.turn_id,
                    state="requeued",
                    rearm_notice=True,
                )
                log_runtime_event(
                    logger,
                    "turn.requeued",
                    agent_id=self.agent_id,
                    turn_id=run.turn_id,
                    provider_session_id=run.provider_session_id,
                    state="requeued",
                    mode="recovery",
                    message_count=0,
                    duration_ms=int(
                        (time.monotonic() - recovery_started) * 1000
                    ),
                    error_category=state,
                )
                requeued = True
            self.health = RuntimeHealth(state, diagnostic)
            self._defer_requeued_resume = defer_requeued_resume and requeued
            self._clear_terminal_turn()
            if requeued:
                self.notify()
            return False

        if (
            not isinstance(raw, dict)
            or raw.get("version") != CURRENT_TURN_VERSION
            or run is None
            or run.state != ProcessingState.IN_TURN.value
            or not durable_ids
        ):
            return await unwind("invalid or stale crash join")

        if (
            self.runtime_event_outbox is not None
            and self.runtime_event_outbox.state().get("active_turn_ref")
        ):
            outbox_state = self.runtime_event_outbox.state()
            if (
                outbox_state.get("native_session_id")
                != run.provider_session_id
                or raw.get(
                    "provider_session_id", run.provider_session_id
                ) != run.provider_session_id
                or raw.get(
                    "native_session_id",
                    outbox_state.get("native_session_id"),
                ) != outbox_state.get("native_session_id")
                or raw.get(
                    "logical_session_ref", outbox_state.get("session_ref")
                ) != outbox_state.get("session_ref")
                or raw.get(
                    "logical_turn_ref", outbox_state.get("active_turn_ref")
                ) != outbox_state.get("active_turn_ref")
            ):
                return await unwind(
                    "crash join and Runtime Event identity mismatch"
                )
            return await unwind(
                "Driver does not support in-flight turn recovery"
            )

        session = run.provider_session_id
        rows = (
            await self.store.get_in_turn_messages(turn_id, session)
            if session
            else ()
        )
        planned = self._reconstruct_exact_turn(turn_id=turn_id, rows=rows) if rows else None
        expected_routes = (
            [asdict(route) for route in planned.routes] if planned is not None else []
        )
        expected_targets = (
            [list(target) for target in planned.targets] if planned is not None else []
        )
        raw_ids = raw.get("message_ids")
        raw_routes = raw.get("routes")
        raw_targets = raw.get("targets")
        current_session = self.adapter.get_provider_session_id()
        if (
            session is None
            or current_session != session
            or planned is None
            or planned.message_ids != durable_ids
            or not isinstance(raw_ids, list)
            or tuple(raw_ids) != durable_ids
            or raw_routes != expected_routes
            or raw_targets != expected_targets
        ):
            return await unwind("crash join identity, route, or target mismatch")

        self.active.turn_id = turn_id
        self.active.message_ids[:] = list(durable_ids)
        self.active.provider_session_id = session
        self.active.routes[:] = list(planned.routes)
        if self.coordinator is not None:
            self.coordinator.provider_session_id = session
        self.attempts.reset()
        self.health = RuntimeHealth("in_progress", "resuming durable crash join")
        if not hasattr(self.run_turn, "handle_global_inbox_retry"):
            return await unwind(
                "crash resume retry unavailable", defer_requeued_resume=True
            )
        try:
            retries = 0
            while True:
                self.active.provider_turn_id = None
                self.adapter.register_admission_callback(
                    lambda event: self._admit_retry_attempt(planned, event),
                    planned.planning_cycle_key,
                )
                try:
                    await self._run_retry(planned)
                    break
                except Exception as exc:
                    from .core import AgentAPIError

                    if isinstance(exc, AgentAPIError) and exc.is_auth:
                        return await unwind(
                            "crash resume auth failure",
                            "auth_failed",
                            defer_requeued_resume=True,
                        )
                    if not isinstance(exc, AgentAPIError):
                        return await unwind(
                            f"crash resume unsafe failure: {type(exc).__name__}",
                            defer_requeued_resume=True,
                        )
                    if retries >= self.max_api_retries:
                        return await unwind(
                            "crash resume retry budget exhausted",
                            "api_error_abandoned",
                            defer_requeued_resume=True,
                        )
                    retries += 1
                    await self.retry_sleep(min(2 ** (retries - 1), 4))
            await self.store.mark_processed(
                tuple(self.active.message_ids), turn_id=turn_id
            )
            log_runtime_event(
                logger,
                "turn.processed",
                agent_id=self.agent_id,
                turn_id=turn_id,
                provider_session_id=self.active.provider_session_id,
                provider_turn_id=self.active.provider_turn_id,
                state=ProcessingState.PROCESSED.value,
                mode="recovery",
                message_count=len(self.active.message_ids),
                duration_ms=int(
                    (time.monotonic() - recovery_started) * 1000
                ),
            )
            self.health = RuntimeHealth()
            self._clear_terminal_turn()
            return True
        except asyncio.CancelledError:
            await self.store.requeue_messages(
                tuple(self.active.message_ids), turn_id=turn_id
            )
            log_runtime_event(
                logger,
                "turn.requeued",
                agent_id=self.agent_id,
                turn_id=turn_id,
                provider_session_id=self.active.provider_session_id,
                provider_turn_id=self.active.provider_turn_id,
                state="requeued",
                mode="recovery",
                message_count=len(self.active.message_ids),
                duration_ms=int(
                    (time.monotonic() - recovery_started) * 1000
                ),
                error_category="cancelled",
            )
            self.health = RuntimeHealth("degraded", "crash resume cancelled and requeued")
            self._clear_terminal_turn()
            self.notify()
            raise
        except Exception as exc:
            return await unwind(
                f"crash resume terminal failure: {type(exc).__name__}",
                defer_requeued_resume=True,
            )
