"""Runtime Manager: sole Driver-event consumer and logical ID owner."""

from __future__ import annotations

import asyncio
import inspect
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..adapters.base import Adapter, TurnContext, TurnResult
from ..context_controller import (
    CompactionResult,
    ContextCapabilities,
    ContextSnapshot,
    ProviderAdmissionEvent,
    ToolResultAdmission,
    normalize_context_snapshot,
)
from .driver import (
    CancelReceipt,
    CompactRequest,
    HarnessDriver,
    HarnessEvent,
    HarnessEventType,
    PermissionDecision,
    PermissionReceipt,
    PermissionRef,
    RuntimeOpened,
    RuntimeSpec,
    SessionRef,
    TurnInput,
    TurnRef,
    UnsupportedCapability,
)


class RuntimeStateError(RuntimeError):
    pass


class RuntimeManager:
    def __init__(
        self, driver: HarnessDriver, spec: RuntimeSpec, *,
        agent_id: str = "", session_ref: SessionRef | None = None,
        native_session_id: str = "",
        event_sink: Callable[[HarnessEvent], Awaitable[None]] | None = None,
        before_start: Callable[[], Awaitable[None] | None] | None = None,
    ):
        self.driver = driver
        self.spec = spec
        self.agent_id = agent_id
        self.event_sink = event_sink
        self.before_start = before_start
        self.session_ref = session_ref or SessionRef(
            f"session_{uuid.uuid4().hex}"
        )
        self.native_session_id = native_session_id
        self.opened: RuntimeOpened | None = None
        self.active_turn_ref: TurnRef | None = None
        self.native_turn_id = ""
        self._active_driver_turn_ref: TurnRef | None = None
        self._turn_refs: dict[TurnRef, TurnRef] = {}
        self._reader: asyncio.Task | None = None
        self._subscribers: set[asyncio.Queue[HarnessEvent | None]] = set()
        self._terminal: dict[TurnRef, asyncio.Future[HarnessEvent]] = {}
        self._closed = False
        self._command_lock = asyncio.Lock()
        self._permission_refs: set[PermissionRef] = set()
        self._continuation_admissions: list[ToolResultAdmission] = []

    async def open(self, *, resume: bool = True) -> RuntimeOpened:
        if self.opened is not None:
            return self.opened
        native_resume = (
            SessionRef(self.native_session_id)
            if resume and self.native_session_id
            else None
        )
        opened = await self.driver.open(self.spec, native_resume)
        self.native_session_id = opened.native_session_id
        # Preserve the durable Puffo logical reference independently of the
        # native provider session ID.
        self.opened = replace(opened, session_ref=self.session_ref)
        self._reader = asyncio.create_task(self._consume_events())
        if self.agent_id:
            register_runtime_manager(self.agent_id, self)
        return self.opened

    async def start_turn(self, input: TurnInput) -> Any:
        async with self._command_lock:
            if self._closed:
                raise RuntimeStateError("runtime is closed")
            if self.opened is None:
                await self.open()
            if self.active_turn_ref is not None:
                raise RuntimeStateError("one provider turn is already active")
            if self.before_start is not None:
                admitted = self.before_start()
                if inspect.isawaitable(admitted):
                    await admitted
            logical = TurnRef(f"turn_{uuid.uuid4().hex}")
            self.active_turn_ref = logical
            loop = asyncio.get_running_loop()
            self._terminal[logical] = loop.create_future()
            try:
                receipt = await self.driver.start_turn(input)
            except BaseException:
                self.active_turn_ref = None
                self._active_driver_turn_ref = None
                self._terminal.pop(logical, None)
                raise
            if isinstance(receipt, UnsupportedCapability) or not receipt.accepted:
                self.active_turn_ref = None
                self._active_driver_turn_ref = None
                self._terminal.pop(logical, None)
                return receipt
            self._active_driver_turn_ref = receipt.turn_ref
            self._turn_refs[receipt.turn_ref] = logical
            self.native_turn_id = receipt.native_turn_id
            return replace(receipt, turn_ref=logical)

    async def steer_turn(self, turn: TurnRef, input: TurnInput) -> Any:
        self._validate_active(turn)
        assert self._active_driver_turn_ref is not None
        receipt = await self.driver.steer_turn(
            self._active_driver_turn_ref, input
        )
        if isinstance(receipt, UnsupportedCapability):
            return receipt
        return replace(receipt, turn_ref=turn)

    async def cancel_turn(self, turn: TurnRef) -> Any:
        self._validate_active(turn)
        assert self._active_driver_turn_ref is not None
        receipt = await self.driver.cancel_turn(self._active_driver_turn_ref)
        if isinstance(receipt, UnsupportedCapability):
            return receipt
        return replace(receipt, turn_ref=turn)

    async def resolve_permission(
        self, turn: TurnRef, permission: PermissionRef,
        decision: PermissionDecision,
    ) -> PermissionReceipt | UnsupportedCapability:
        self._validate_active(turn)
        if permission not in self._permission_refs:
            raise RuntimeStateError("unknown or stale permission reference")
        return await self.driver.resolve_permission(permission, decision)

    async def wait_terminal(self, turn: TurnRef) -> HarnessEvent:
        future = self._terminal.get(turn)
        if future is None:
            raise RuntimeStateError("unknown turn reference")
        try:
            return await asyncio.shield(future)
        finally:
            if future.done():
                self._terminal.pop(turn, None)

    async def reload_resources(self, *, preserve_session: bool) -> None:
        if self.active_turn_ref is not None:
            raise RuntimeStateError("cannot reload while a turn is active")
        await self._stop_reader()
        await self.driver.close()
        self.opened = None
        if not preserve_session:
            self.session_ref = SessionRef(f"session_{uuid.uuid4().hex}")
        await self.open(resume=preserve_session)

    def events(self) -> AsyncIterator[HarnessEvent]:
        queue: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._subscribers.add(queue)

        async def iterate() -> AsyncIterator[HarnessEvent]:
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        return
                    yield event
            finally:
                self._subscribers.discard(queue)

        return iterate()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.agent_id:
            unregister_runtime_manager(self.agent_id, self)
        await self.driver.close()
        await self._stop_reader()
        for queue in tuple(self._subscribers):
            queue.put_nowait(None)

    async def _stop_reader(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None

    async def _consume_events(self) -> None:
        try:
            async for native in self.driver.events():
                logical_turn = (
                    self._turn_refs.get(native.turn_ref)
                    if native.turn_ref is not None
                    else None
                )
                # A provider may push turn.started before its start response.
                # During that narrow window there is exactly one pending
                # logical Turn, so attribution is still unambiguous.
                if (
                    logical_turn is None
                    and native.turn_ref is not None
                    and self.active_turn_ref is not None
                    and self._active_driver_turn_ref is None
                ):
                    logical_turn = self.active_turn_ref
                event = replace(
                    native, session_ref=self.session_ref, turn_ref=logical_turn
                )
                await self._admit_matching_tool_result(native)
                if event.type in {
                    HarnessEventType.PERMISSION_REQUESTED,
                    "turn.permission_requested",
                }:
                    value = event.data.get("permission_ref")
                    if value:
                        self._permission_refs.add(PermissionRef(str(value)))
                if self.event_sink is not None:
                    await self.event_sink(event)
                for queue in tuple(self._subscribers):
                    queue.put_nowait(event)
                if event.type in {
                    HarnessEventType.TURN_COMPLETED,
                    HarnessEventType.TURN_ABANDONED,
                    "turn.completed",
                    "turn.abandoned",
                } and logical_turn is not None:
                    future = self._terminal.get(logical_turn)
                    if future is not None and not future.done():
                        future.set_result(event)
                    self.active_turn_ref = None
                    self._active_driver_turn_ref = None
                    self.native_turn_id = ""
                    self._permission_refs.clear()
                    self._continuation_admissions.clear()
        except asyncio.CancelledError:
            raise

    def register_continuation(
        self,
        callback,
        planning_cycle_key: str,
        *,
        channel_id: str = "",
        tool_names: tuple[str, ...] = (),
        tool_arguments: dict[str, object] | None = None,
        correlation_receipt: str = "",
    ) -> None:
        if self.active_turn_ref is None or not self.native_turn_id:
            raise RuntimeStateError(
                "no active provider turn for tool-result admission"
            )
        self._continuation_admissions.append(ToolResultAdmission.build(
            callback,
            planning_cycle_key,
            self.native_turn_id,
            channel_id=channel_id,
            tool_names=tool_names,
            tool_arguments=tool_arguments,
            correlation_receipt=correlation_receipt,
        ))

    async def _admit_matching_tool_result(self, event: HarnessEvent) -> None:
        fact = event.native_diagnostic
        if (
            not isinstance(fact, dict)
            or fact.get("_puffo_internal") != "tool_result"
            or fact.get("is_error") is True
        ):
            return
        tool_name = str(fact.get("tool_name") or "")
        arguments = fact.get("arguments")
        if not isinstance(arguments, dict):
            return
        candidates = [
            (index, admission)
            for index, admission in enumerate(self._continuation_admissions)
            if admission.provider_turn_id == event.native_turn_id
            and admission.matches(tool_name, arguments)
            and (
                not admission.receipt_marker
                or admission.receipt_marker in repr(fact.get("result"))
            )
        ]
        if not candidates:
            return
        index, admission = max(
            candidates, key=lambda value: value[1].match_specificity
        )
        self._continuation_admissions.pop(index)
        await admission.callback(ProviderAdmissionEvent(
            planning_cycle_key=admission.planning_cycle_key,
            provider_session_id=self.native_session_id,
            provider_turn_id=event.native_turn_id,
            tool_call_id=str(fact.get("tool_call_id") or ""),
            admitted_at=datetime.now(timezone.utc),
        ))

    def _validate_active(self, turn: TurnRef) -> None:
        if not isinstance(turn, TurnRef) or self.active_turn_ref != turn:
            raise RuntimeStateError("stale or foreign turn reference")


class RuntimeManagerAdapter(Adapter):
    """Blocking compatibility facade over the event-driven Runtime Manager."""

    def __init__(self, manager: RuntimeManager):
        self.manager = manager
        self.assistant_text_parts: list[str] = []

    async def run_turn(self, ctx: TurnContext) -> TurnResult:
        message = "\n".join(
            str(item.get("content", ""))
            for item in ctx.messages if item.get("role") == "user"
        )
        stream = self.manager.events()
        started = await self.manager.start_turn(TurnInput(message))
        if isinstance(started, UnsupportedCapability) or not started.accepted:
            return TurnResult(reply="", metadata={"accepted": False})
        turn = started.turn_ref
        callback = getattr(self, "_context_admission_callback", None)
        if callback is not None:
            await self._fire_admission_callback(ProviderAdmissionEvent(
                planning_cycle_key=getattr(
                    self, "_context_admission_planning_cycle_key", ""
                ),
                provider_session_id=self.get_provider_session_id(),
                provider_turn_id=started.native_turn_id or None,
                admitted_at=datetime.now(timezone.utc),
            ))
        self.assistant_text_parts = []
        metadata: dict[str, Any] = {"turn_ref": str(turn)}
        async for event in stream:
            if event.turn_ref != turn:
                continue
            event_type = (
                event.type.value
                if isinstance(event.type, HarnessEventType) else event.type
            )
            if event_type == "turn.assistant_delta":
                text = event.data.get("text")
                if isinstance(text, str):
                    self.assistant_text_parts.append(text)
                    if ctx.on_progress is not None:
                        await ctx.on_progress(text)
            if event_type in {"turn.completed", "turn.abandoned"}:
                outcome = str(event.data.get("outcome") or "succeeded")
                if event_type == "turn.abandoned" or outcome != "succeeded":
                    raise RuntimeStateError(
                        f"provider turn ended with outcome {outcome}"
                    )
                metadata.update({
                    key: value for key, value in event.data.items()
                    if key in {
                        "input_tokens", "output_tokens", "tool_calls",
                        "provider_session_id", "send_message_targets",
                    }
                })
                return TurnResult(
                    reply="".join(self.assistant_text_parts),
                    input_tokens=int(metadata.get("input_tokens", 0)),
                    output_tokens=int(metadata.get("output_tokens", 0)),
                    tool_calls=int(metadata.get("tool_calls", 0)),
                    metadata=metadata,
                )
        return TurnResult(reply="", metadata={"stream_error": "runtime_exited"})

    def register_continuation_callback(
        self,
        callback,
        planning_cycle_key: str = "",
        *,
        channel_id: str = "",
        tool_names: tuple[str, ...] = (),
        tool_arguments: dict[str, object] | None = None,
        correlation_receipt: str = "",
    ) -> None:
        if callback is None:
            self.manager._continuation_admissions.clear()
            return
        self.manager.register_continuation(
            callback,
            planning_cycle_key,
            channel_id=channel_id,
            tool_names=tool_names,
            tool_arguments=tool_arguments,
            correlation_receipt=correlation_receipt,
        )

    async def warm(self, system_prompt: str) -> None:
        await self.manager.open()

    async def reload(
        self, new_system_prompt: str, *, with_session: bool = False
    ) -> None:
        await self.manager.reload_resources(preserve_session=not with_session)

    async def aclose(self) -> None:
        await self.manager.close()

    def get_provider_session_id(self) -> str | None:
        return (
            self.manager.opened.native_session_id
            if self.manager.opened is not None else None
        )

    async def get_context_snapshot(self) -> ContextSnapshot:
        status = await self.manager.driver.context_status()
        if isinstance(status, UnsupportedCapability):
            return await super().get_context_snapshot()
        return normalize_context_snapshot(
            used_tokens=status.used_tokens or 0,
            provider_context_window=status.context_window,
            measured_at=datetime.now(timezone.utc),
        )

    def get_context_capabilities(self) -> ContextCapabilities:
        capabilities = (
            self.manager.opened.capabilities
            if self.manager.opened is not None else None
        )
        if capabilities is None:
            return super().get_context_capabilities()
        compact = getattr(capabilities.compact, "value", capabilities.compact)
        context_status = getattr(
            capabilities.context_status, "value", capabilities.context_status
        )
        return ContextCapabilities(
            native_compaction=compact != "none",
            rollover=False,
            native_measurement=context_status != "none",
            diagnostic="Harness Driver capabilities",
        )

    async def compact_context(self) -> CompactionResult:
        receipt = await self.manager.driver.compact(CompactRequest())
        return CompactionResult(
            # Driver command acceptance is not completion; the normalized
            # compaction.completed event is the later confirmation.
            completed=False,
            provider_session_id=self.get_provider_session_id(),
            diagnostic=(
                "compaction accepted; awaiting canonical completion event"
                if not isinstance(receipt, UnsupportedCapability)
                else receipt.diagnostic
            ),
        )


_RUNTIME_MANAGERS: weakref.WeakValueDictionary[str, RuntimeManager] = (
    weakref.WeakValueDictionary()
)


def register_runtime_manager(agent_id: str, manager: RuntimeManager) -> None:
    _RUNTIME_MANAGERS[agent_id] = manager


def unregister_runtime_manager(agent_id: str, manager: RuntimeManager) -> None:
    if _RUNTIME_MANAGERS.get(agent_id) is manager:
        _RUNTIME_MANAGERS.pop(agent_id, None)


def get_runtime_manager(agent_id: str) -> RuntimeManager | None:
    return _RUNTIME_MANAGERS.get(agent_id)
