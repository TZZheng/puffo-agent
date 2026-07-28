"""Serial orchestration for the durable global agent Inbox.

Storage, prefix selection, provider context control, and sending deliberately
remain in their leaf modules.  This module joins those contracts and owns only
the mutable state of the one active turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .context_controller import (
    AdmissionCandidate,
    ContextController,
    DecisionOutcome,
    ProviderAdmissionEvent,
)
from .inbox_scheduler import (
    MAX_ESTIMATED_TOKENS,
    MAX_FORMATTED_BYTES,
    InboxCoalescer,
    InboxPlanner,
    PlannedBatch,
)
from .message_store import MessageStore, ProcessingState, StoredMessage

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
    provider_session_id: str | None = None
    through_by_channel: dict[tuple[str, str], int] = field(default_factory=dict)

    def clear(self) -> None:
        self.turn_id = ""
        self.message_ids.clear()
        self.provider_session_id = None
        self.through_by_channel.clear()


@dataclass(frozen=True)
class RuntimeHealth:
    state: str = "idle"
    diagnostic: str = ""


@dataclass
class HeldStaging:
    message_ids: tuple[str, ...] = ()
    latest_seq: int | None = None
    latest_envelope_id: str | None = None
    recovered_through_seq: int | None = None
    more_pending: bool = False
    synchronized: bool = False
    diagnostic: str = ""
    correlation_key: str = ""


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
    if item.thread_root_id:
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


def format_stored_message(item: StoredMessage) -> str:
    """Exact per-message model view; no target or speaker policy."""
    route = route_for(item)
    content = item.content
    attachments: list[str] = []
    if isinstance(content, Mapping):
        text = str(content.get("text") or "")
        raw_attachments = content.get("attachment_paths", content.get("attachments", ()))
        if isinstance(raw_attachments, Sequence) and not isinstance(
            raw_attachments, (str, bytes)
        ):
            attachments = [str(value) for value in raw_attachments]
        content_metadata = {
            key: content.get(key)
            for key in (
                "mentions",
                "sender_display_name",
                "sender_owner_slug",
                "is_from_operator",
                "sender_is_agent",
                "is_visible_to_human",
                "space_name",
                "channel_name",
            )
        }
    else:
        text = str(content or "")
        content_metadata = {}
    metadata = {
        "envelope_id": item.envelope_id,
        "server_seq": item.server_seq,
        "route": asdict(route),
        "sender_slug": item.sender_slug,
        "recipient_slug": item.recipient_slug,
        "sent_at": item.sent_at,
        "is_encrypted": item.is_encrypted,
        "attachments": attachments,
        "reply_to_id": item.reply_to_id,
        **content_metadata,
    }
    return (
        "<inbox_message>\n"
        f"{json.dumps(metadata, separators=(',', ':'), sort_keys=True)}\n"
        f"{text}\n"
        "</inbox_message>"
    )


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
        advanced = self.active.through_by_channel.get((space_id, channel_id))
        values = [value for value in (persisted, advanced) if value is not None]
        return max(values) if values else None

    async def advance_active_turn_through_seq(
        self, space_id: str, channel_id: str, seq: int
    ) -> None:
        if not self.active.turn_id:
            return
        key = (space_id, channel_id)
        self.active.through_by_channel[key] = max(
            seq, self.active.through_by_channel.get(key, seq)
        )


class TrackingSendDelegate:
    """Mark semantic attempts before awaiting the worker-owned coordinator."""

    def __init__(self, coordinator: Any, attempts: SendAttemptState):
        self.coordinator = coordinator
        self.attempts = attempts

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
        self.attempts.record(destination, "attempting")
        result = await self.coordinator.send(request, **kwargs)
        state = str(result.get("state", "failed"))
        self.attempts.states[-1] = state
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
        return result

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        return await self.send(kwargs)


class HeldRecoverySource:
    """Live durable held catch-up for the one active exact turn.

    This deliberately does not invent a remote catch-up API.  It waits for
    receipt commits, proves the exact terminal watermark exists locally, plans
    a bounded whole-message channel prefix, and stages that prefix behind a
    correlated continuation admission callback.
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
        boundary = await ActiveBoundaryAdapter(
            self.runtime.store, active
        ).get_active_turn_through_seq(space_id, channel_id)
        page = await self.runtime.store.get_channel_pending(
            space_id,
            channel_id,
            after_seq=boundary,
            through_seq=latest_seq,
            limit=50,
        )
        watermark = await self.runtime.store.get_message_by_envelope(
            latest_envelope_id
        )
        if (
            not page.items
            or watermark is None
            or watermark.server_seq != latest_seq
            or watermark.space_id != space_id
            or watermark.channel_id != channel_id
        ):
            self.runtime.held = HeldStaging(
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                diagnostic="exact held watermark mismatch",
            )
            return ()
        planned = await self.runtime.plan_pending(
            items=page.items,
            turn_id=active.turn_id,
            planning_cycle_key=f"continuation_{uuid.uuid4().hex}",
        )
        if planned is None:
            return ()
        decision = await self.runtime.context_controller.decide(
            planned.candidate, self.runtime._replacement_candidate
        )
        if decision.outcome is not DecisionOutcome.ADMIT:
            self.runtime.held = HeldStaging(
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                diagnostic=f"held context admission unavailable: {decision.outcome.value}",
            )
            return ()
        staged_ids = planned.message_ids
        correlation_key = planned.planning_cycle_key
        planned_server_seqs = [
            row.server_seq
            for row in planned.items
            if row.server_seq is not None
        ]
        recovered_through_seq = (
            max(planned_server_seqs) if planned_server_seqs else None
        )
        reached_latest = any(
            row.envelope_id == latest_envelope_id and row.server_seq == latest_seq
            for row in planned.items
        )
        more_pending = page.more_available or not reached_latest
        fired = False

        async def admit_continuation(event: ProviderAdmissionEvent) -> None:
            nonlocal fired
            if fired or event.planning_cycle_key != planned.planning_cycle_key:
                return
            fired = True
            existing_rows = await self.runtime.store.get_in_turn_messages(
                active.turn_id,
                provider_session_id,
            )
            existing_ids = {row.envelope_id for row in existing_rows}
            new_ids = tuple(
                envelope_id
                for envelope_id in staged_ids
                if envelope_id not in existing_ids
            )
            if new_ids:
                run = await self.runtime.store.admit_messages(
                    new_ids,
                    turn_id=active.turn_id,
                    provider_session_id=provider_session_id,
                )
                active.message_ids[:] = list(run.message_ids)
            else:
                active.message_ids[:] = [
                    row.envelope_id for row in existing_rows
                ]
            rows = await self.runtime.store.get_in_turn_messages(
                active.turn_id,
                provider_session_id,
            )
            if rows:
                self.runtime._write_current_turn(
                    self.runtime._reconstruct_exact_turn(
                        turn_id=active.turn_id,
                        rows=rows,
                    )
                )
            self.runtime.held = HeldStaging(
                message_ids=staged_ids,
                latest_seq=latest_seq,
                latest_envelope_id=latest_envelope_id,
                recovered_through_seq=recovered_through_seq,
                more_pending=more_pending,
                synchronized=reached_latest,
                diagnostic=(
                    "additional held context remains pending"
                    if more_pending
                    else ""
                ),
                correlation_key=correlation_key,
            )

        register_continuation = getattr(
            self.runtime.adapter, "register_continuation_callback", None
        )
        if callable(register_continuation):
            register_continuation(
                admit_continuation,
                planned.planning_cycle_key,
                channel_id=channel_id,
            )
        else:
            self.runtime.adapter.register_admission_callback(
                admit_continuation, planned.planning_cycle_key
            )
        self.runtime.held = HeldStaging(
            message_ids=staged_ids,
            latest_seq=latest_seq,
            latest_envelope_id=latest_envelope_id,
            recovered_through_seq=recovered_through_seq,
            more_pending=more_pending,
            diagnostic="continuation staged; awaiting correlated admission",
            correlation_key=correlation_key,
        )
        return tuple(
            {
                "envelope_id": row.envelope_id,
                "server_seq": row.server_seq,
                "content": row.content,
                "latest_seq": latest_seq,
                "latest_envelope_id": latest_envelope_id,
                "recovered_through_seq": recovered_through_seq,
                "more_pending": more_pending,
                "provider_session_id": provider_session_id,
                "synchronized": False,
                "continuation_correlation_key": correlation_key,
            }
            for row in planned.items
        )


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
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.run_turn = run_turn
        self.workspace = Path(workspace)
        self.context_controller = context_controller or ContextController(adapter)
        self.planner = planner or InboxPlanner()
        self.coalescer = coalescer or InboxCoalescer()
        self.formatter = formatter
        self.estimator = estimator
        self.unfit_policy = unfit_policy or (lambda *_args, **_kwargs: True)
        self.coordinator = coordinator
        self.active = ActiveExactUnion()
        self.attempts = SendAttemptState()
        self.health = RuntimeHealth()
        self.held = HeldStaging()
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
        self.send_delegate: TrackingSendDelegate | None = None
        self.held_recovery_source = HeldRecoverySource(
            self,
            catchup_pending=held_catchup,
        )

    @property
    def current_turn_path(self) -> Path:
        return self.workspace / ".puffo-agent" / "current_turn.json"

    def notify(self) -> None:
        self._degraded = False
        self.coalescer.notify()

    def notify_delivery(self) -> None:
        self.held_recovery_source.notify_delivery()

    async def run(self) -> None:
        await self.recover_current_turn()
        if self._defer_requeued_resume:
            # Consume the recovery wake without immediately feeding the same
            # failed durable union through the initial-turn path.
            await self.coalescer.wait_for_burst()
        elif await self.store.get_pending(limit=1):
            self.notify()
        while not self._stopping:
            await self.coalescer.wait_for_burst()
            if self._stopping:
                break
            await self.process_once()

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
        run = await self.store.admit_messages(
            planned.message_ids,
            turn_id=planned.turn_id,
            provider_session_id=event.provider_session_id,
        )
        self.active.turn_id = run.turn_id
        self.active.message_ids[:] = list(run.message_ids)
        self.active.provider_session_id = event.provider_session_id
        if self.coordinator is not None:
            self.coordinator.provider_session_id = event.provider_session_id

    def _write_current_turn(self, planned: PlannedTurn) -> None:
        path = self.current_turn_path
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "version": CURRENT_TURN_VERSION,
            "turn_id": planned.turn_id,
            "message_ids": list(planned.message_ids),
            "targets": [list(target) for target in planned.targets],
            "routes": [asdict(route) for route in planned.routes],
        }
        if len(planned.targets) == 1:
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
                if decision.outcome is DecisionOutcome.REPLAN:
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
                any(item.is_encrypted for item in planned.items),
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
                            continue
                        raise
                admitted = self.active.turn_id == planned.turn_id
                if admitted:
                    await self.store.mark_processed(
                        tuple(self.active.message_ids), turn_id=planned.turn_id
                    )
                    terminal = True
                else:
                    self.health = RuntimeHealth(
                        "degraded", "provider returned without correlated admission"
                    )
                    self._degraded = True
            except asyncio.CancelledError:
                if self.active.turn_id == planned.turn_id:
                    await self.store.requeue_messages(
                        tuple(self.active.message_ids), turn_id=planned.turn_id
                    )
                terminal = True
                raise
            except Exception:
                if self.active.turn_id == planned.turn_id:
                    await self.store.requeue_messages(
                        tuple(self.active.message_ids), turn_id=planned.turn_id
                    )
                    terminal = True
                self.health = RuntimeHealth("degraded", "turn failed and was requeued")
            finally:
                send_mode.clear_turn_bundle(list(self.send_mode_keys))
                self.adapter.register_admission_callback(None, "")
                was_active = self.active.turn_id == planned.turn_id
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
                exact_ids = (
                    tuple(self.active.message_ids)
                    if self.active.turn_id == run.turn_id
                    else durable_ids
                )
                await self.store.requeue_messages(exact_ids, turn_id=run.turn_id)
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
            self.health = RuntimeHealth()
            self._clear_terminal_turn()
            return True
        except asyncio.CancelledError:
            await self.store.requeue_messages(
                tuple(self.active.message_ids), turn_id=turn_id
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
