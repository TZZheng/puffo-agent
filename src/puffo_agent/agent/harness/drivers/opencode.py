"""OpenCode Driver using one ``run --format json`` child per turn."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from ....tasks import spawn
from ...cli_bin import normalize_launch_argv
from ..support.cleanup_errors import (
    CLEANUP_TIMEOUT_SECONDS,
    collect_cleanup_errors,
    raise_collected_errors,
)
from ..driver import (
    BusyDelivery,
    CancelCapability,
    CancelReceipt,
    CompactCapability,
    CompactRequest,
    ContextStatusCapability,
    Driver,
    DriverCapabilities,
    HarnessEvent,
    HarnessEventType,
    PermissionDecision,
    PermissionRef,
    ProtocolDiagnostics,
    RuntimeLifecycle,
    RuntimeOpened,
    RuntimeRef,
    RuntimeSpec,
    SessionRef,
    SteerCapability,
    TurnInput,
    TurnRef,
    TurnStarted,
    UnsupportedCapability,
)
from .opencode_protocol import (
    build_opencode_run_command,
    normalize_opencode_frame,
)
from ..support.subprocess_io import (
    drain_subprocess_stream_keeping_tail,
    process_group_spawn_kwargs,
    shutdown_process_tree,
    signal_process_tree,
)


OPENCODE_CAPABILITIES = DriverCapabilities(
    session_resume=True,
    inflight_turn_recovery=False,
    steer=SteerCapability.NONE,
    cancel=CancelCapability.TYPED,
    context_status=ContextStatusCapability.NONE,
    compact=CompactCapability.NONE,
    permission_bridge=False,
    lifecycle=RuntimeLifecycle.PER_TURN_CHILD,
    busy_delivery=BusyDelivery.REJECT,
)

_SHUTDOWN_GRACE_SECONDS = 5.0


class OpenCodeDriver(Driver):
    """Logical session whose only native process exists during a turn.

    This driver is not a trusted ``puffo-v0`` identity-binding entrypoint.
    Profiles requiring that binding must use the ACP driver's validated
    launch seam.
    """

    static_steer_capability = SteerCapability.NONE

    def __init__(
        self,
        process_factory: Callable[..., Any] | None = None,
        *,
        executable_version: str = "",
    ) -> None:
        self.process_factory = process_factory
        self.executable_version = executable_version
        self._spec: RuntimeSpec | None = None
        self._runtime_ref = RuntimeRef(f"runtime_{uuid.uuid4().hex}")
        self._session_ref = SessionRef(f"session_{uuid.uuid4().hex}")
        self._native_session_id = ""
        self._resumed = False
        self._session_announced = False
        self._proc: Any = None
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._turn_generation = 0
        self._turn_task: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[bytes] | None = None
        self._accepted: asyncio.Future[tuple[str, str]] | None = None
        self._events: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._terminal_reason = ""
        self._provider_failed = False
        self._closed = False

    def current_capabilities(self) -> DriverCapabilities:
        return OPENCODE_CAPABILITIES

    async def open(
        self, spec: RuntimeSpec, resume: SessionRef | None = None
    ) -> RuntimeOpened:
        if self._spec is not None:
            raise RuntimeError("driver is already open")
        if self._closed:
            self._closed = False
            self._events = asyncio.Queue()
        self._spec = spec
        self._resumed = resume is not None
        self._native_session_id = str(resume or "")
        self._session_announced = False
        return RuntimeOpened(
            self._runtime_ref,
            self._session_ref,
            self._native_session_id,
            self._resumed,
            OPENCODE_CAPABILITIES,
            ProtocolDiagnostics(
                executable_version=self.executable_version,
                schema_source="documented-jsonl",
                native_capabilities=("session_resume", "process_cancel"),
            ),
        )

    async def start_turn(self, input: TurnInput):
        if self._spec is None or self._closed:
            raise RuntimeError("driver is not open")
        if self._active.value or self._turn_task is not None:
            raise RuntimeError("one turn is already active")
        turn = TurnRef(f"turn_{uuid.uuid4().hex}")
        self._active = turn
        self._active_native_turn_id = ""
        self._provider_failed = False
        self._terminal_reason = ""
        self._turn_generation += 1
        generation = self._turn_generation
        self._accepted = asyncio.get_running_loop().create_future()
        try:
            self._proc = await self._spawn(self._spec, input.content)
            self._stderr_reader = spawn(
                drain_subprocess_stream_keeping_tail(
                    getattr(self._proc, "stderr", None)
                ),
                name="opencode.stderr",
            )
            self._turn_task = spawn(
                self._drive_turn(self._proc, turn, generation),
                name="opencode.turn",
            )
            native_session_id, native_turn_id = await asyncio.shield(
                self._accepted
            )
        except BaseException as exc:
            errors: list[BaseException] = [exc]
            await collect_cleanup_errors(
                self._abort_failed_start(generation),
                errors,
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
            raise_collected_errors("OpenCode turn start cleanup failed", errors)
        return TurnStarted(
            turn,
            native_turn_id=native_turn_id,
            accepted=True,
            delivery="first_json_frame",
        )

    async def _spawn(self, spec: RuntimeSpec, prompt: str) -> Any:
        command = build_opencode_run_command(
            spec,
            prompt=prompt,
            native_session_id=self._native_session_id,
        )
        if self.process_factory is not None:
            # One call with the declared signature. Retrying on TypeError
            # could create a second child when the factory itself failed.
            proc = self.process_factory(command, spec)
            return await proc if asyncio.iscoroutine(proc) else proc
        executable, *arguments = command
        env = dict(spec.environment)
        return await asyncio.create_subprocess_exec(
            *normalize_launch_argv(executable),
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.workspace_dir or None,
            env=env,
            limit=16 * 1024 * 1024,
            **process_group_spawn_kwargs(),
        )

    async def steer_turn(self, turn: TurnRef, input: TurnInput):
        return UnsupportedCapability("steer")

    async def cancel_turn(self, turn: TurnRef):
        self._require_active(turn)
        proc = self._proc
        if proc is None or getattr(proc, "returncode", None) is not None:
            return CancelReceipt(False, turn)
        self._terminal_reason = "cancelled"
        await signal_process_tree(
            proc, force=False, timeout=_SHUTDOWN_GRACE_SECONDS
        )
        return CancelReceipt(True, turn)

    async def context_status(self):
        return UnsupportedCapability("context_status")

    async def compact(self, request: CompactRequest):
        return UnsupportedCapability("compact")

    async def resolve_permission(
        self, request: PermissionRef, decision: PermissionDecision
    ):
        return UnsupportedCapability("permission_bridge")

    def events(self) -> AsyncIterator[HarnessEvent]:
        async def iterate():
            while True:
                event = await self._events.get()
                if event is None:
                    return
                yield event

        return iterate()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminal_reason = "runtime_closed"
        proc = self._proc
        errors: list[BaseException] = []
        await collect_cleanup_errors(
            self._settle_turn_task(proc),
            errors,
            timeout=CLEANUP_TIMEOUT_SECONDS,
        )
        if self._stderr_reader is not None:
            self._stderr_reader.cancel()
            await collect_cleanup_errors(
                asyncio.gather(self._stderr_reader, return_exceptions=True),
                errors,
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
            self._stderr_reader = None
        self._proc = None
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._accepted = None
        self._turn_task = None
        self._spec = None
        await collect_cleanup_errors(
            self._events.put(None), errors, timeout=CLEANUP_TIMEOUT_SECONDS
        )
        raise_collected_errors("OpenCode driver close failed", errors)

    async def _drive_turn(
        self, proc: Any, turn: TurnRef, generation: int
    ) -> None:
        read_task = spawn(
            self._read_turn_frames(proc, turn, generation),
            name="opencode.frames",
        )
        wait_task = spawn(proc.wait(), name="opencode.wait")
        try:
            done, _ = await asyncio.wait(
                {read_task, wait_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            if (
                read_task in done
                and not read_task.cancelled()
                and read_task.exception() is not None
                and not wait_task.done()
                and getattr(proc, "returncode", None) is None
            ):
                await signal_process_tree(
                    proc, force=False, timeout=_SHUTDOWN_GRACE_SECONDS
                )
            results = await asyncio.gather(
                read_task, wait_task, return_exceptions=True
            )
            read_result, wait_result = results
            returncode = (
                wait_result
                if isinstance(wait_result, int)
                else getattr(proc, "returncode", None)
            )
            if isinstance(read_result, BaseException):
                self._provider_failed = True
                await self._emit(
                    HarnessEventType.RUNTIME_WARNING,
                    turn=turn,
                    data={"code": "opencode_stream_read"},
                )
            await self._finish_turn(turn, generation, returncode)
        finally:
            for task in (read_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                read_task, wait_task, return_exceptions=True
            )

    async def _read_turn_frames(
        self, proc: Any, turn: TurnRef, generation: int
    ) -> None:
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            try:
                frame = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                await self._emit(
                    HarnessEventType.RUNTIME_WARNING,
                    turn=turn,
                    data={"code": "protocol_parse"},
                )
                continue
            if not isinstance(frame, dict):
                await self._emit(
                    HarnessEventType.RUNTIME_WARNING,
                    turn=turn,
                    data={"code": "protocol_shape"},
                )
                continue
            if generation != self._turn_generation:
                return
            part = frame.get("part")
            part = part if isinstance(part, dict) else {}
            native_session_id = str(
                frame.get("sessionID") or part.get("sessionID") or ""
            )
            native_turn_id = str(part.get("messageID") or "")
            if not self._accepted or not self._accepted.done():
                if not native_session_id:
                    continue
                if (
                    self._native_session_id
                    and native_session_id != self._native_session_id
                ):
                    raise RuntimeError("OpenCode resumed a different native session")
                was_new = not self._native_session_id
                self._native_session_id = native_session_id
                self._active_native_turn_id = native_turn_id
                if not self._session_announced:
                    await self._emit(
                        (
                            HarnessEventType.SESSION_OPENED
                            if was_new
                            else HarnessEventType.SESSION_RESUMED
                        ),
                        native_payload=frame,
                    )
                    self._session_announced = True
                self._accepted.set_result((native_session_id, native_turn_id))
            for event in normalize_opencode_frame(
                frame,
                session_ref=self._session_ref,
                turn_ref=turn,
            ):
                if event.type is HarnessEventType.RUNTIME_FAILED:
                    self._provider_failed = True
                await self._events.put(event)

    async def _finish_turn(
        self, turn: TurnRef, generation: int, returncode: int | None
    ) -> None:
        if generation != self._turn_generation:
            return
        accepted = self._accepted
        if accepted is not None and not accepted.done():
            accepted.set_exception(
                RuntimeError(
                    "OpenCode exited before accepting the turn "
                    f"(returncode={returncode})"
                )
            )
        reason = self._terminal_reason
        if reason:
            type_ = HarnessEventType.TURN_ABANDONED
            data = {
                "outcome": "abandoned",
                "error_code": reason,
                "retryable": reason != "cancelled",
            }
        elif returncode not in (None, 0) or self._provider_failed:
            type_ = HarnessEventType.TURN_COMPLETED
            data = {
                "outcome": "failed",
                "error_code": (
                    "opencode_run_error"
                    if self._provider_failed
                    else "opencode_process_exit"
                ),
            }
        else:
            type_ = HarnessEventType.TURN_COMPLETED
            data = {"outcome": "succeeded"}
        terminal: HarnessEvent | None = None
        if accepted is not None and accepted.done() and not accepted.cancelled():
            try:
                accepted.result()
            except Exception:
                pass
            else:
                terminal = HarnessEvent.normalized(
                    type=type_,
                    driver="opencode",
                    session_ref=self._session_ref,
                    turn_ref=turn,
                    native_session_id=self._native_session_id,
                    native_turn_id=self._active_native_turn_id,
                    data=data,
                )
        if self._stderr_reader is not None:
            await asyncio.gather(self._stderr_reader, return_exceptions=True)
            self._stderr_reader = None
        self._proc = None
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._accepted = None
        self._turn_task = None
        # Publish only after the Driver is idle.  A fast consumer may start
        # the next turn as soon as it sees this boundary.
        if terminal is not None:
            await self._events.put(terminal)

    async def _abort_failed_start(self, generation: int) -> None:
        proc = self._proc
        task = self._turn_task
        try:
            if task is not None and task is not asyncio.current_task():
                await self._settle_turn_task(proc)
        finally:
            if generation == self._turn_generation:
                self._proc = None
                self._active = TurnRef("")
                self._active_native_turn_id = ""
                self._accepted = None
                self._turn_task = None

    async def _settle_turn_task(self, proc: Any) -> None:
        task = self._turn_task
        if task is asyncio.current_task():
            return
        await shutdown_process_tree(
            proc,
            waiter=task,
            timeout=_SHUTDOWN_GRACE_SECONDS,
            task_name="opencode.shutdown_wait",
        )

    async def _emit(
        self,
        type_: HarnessEventType,
        *,
        turn: TurnRef | None = None,
        data: dict[str, Any] | None = None,
        native_payload: Any = None,
    ) -> None:
        await self._events.put(
            HarnessEvent.normalized(
                type=type_,
                driver="opencode",
                session_ref=self._session_ref,
                turn_ref=turn,
                native_session_id=self._native_session_id,
                native_turn_id=self._active_native_turn_id,
                data=data or {},
                native_payload=native_payload,
            )
        )

    def _require_active(self, turn: TurnRef) -> None:
        if turn != self._active or not self._active.value:
            raise RuntimeError("stale or foreign active turn")


OpenCodeCliDriver = OpenCodeDriver
