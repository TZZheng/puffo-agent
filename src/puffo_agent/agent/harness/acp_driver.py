"""Generic ACP v1 Driver backed by the official Python SDK."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.connection import StreamDirection, StreamEvent
from acp.exceptions import RequestError
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    CreateTerminalResponse,
    DeniedOutcome,
    FileSystemCapabilities,
    Implementation,
    EnvVariable,
    McpServerStdio,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    WaitForTerminalExitResponse,
)

from ..._proc import spawn_framed_child
from ..cli_bin import normalize_launch_argv
from .driver import (
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
    McpServerSpec,
    PermissionDecision,
    PermissionReceipt,
    PermissionRef,
    ProtocolDiagnostics,
    RuntimeLifecycle,
    RuntimeOpened,
    RuntimeRef,
    RuntimeSpec,
    SessionRef,
    SteerCapability,
    TurnInput,
    runtime_exited_data,
    TurnRef,
    TurnStarted,
    UnsupportedCapability,
)
from .driver_authority_server import (
    DRIVER_AUTHORITY_FD_ENV,
    DriverAuthorityServer,
)
from .subprocess_io import drain_subprocess_stream_keeping_tail


def acp_capabilities(*, session_resume: bool) -> DriverCapabilities:
    return DriverCapabilities(
        session_resume=session_resume,
        inflight_turn_recovery=False,
        steer=SteerCapability.NONE,
        cancel=CancelCapability.TYPED,
        context_status=ContextStatusCapability.PUSH,
        compact=CompactCapability.NONE,
        permission_bridge=True,
        lifecycle=RuntimeLifecycle.PERSISTENT_CHILD,
        busy_delivery=BusyDelivery.REJECT,
    )


class _PuffoAcpClient:
    def __init__(self, driver: AcpDriver) -> None:
        self.driver = driver

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        return await self.driver._request_permission(
            options, session_id, tool_call
        )

    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any
    ) -> None:
        await self.driver._session_update(session_id, update)

    async def write_text_file(self, **kwargs: Any) -> None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(self, **kwargs: Any) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(self, **kwargs: Any) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, **kwargs: Any) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, **kwargs: Any) -> ReleaseTerminalResponse:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, **kwargs: Any) -> None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        await self.driver._unsupported_extension(method)
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        await self.driver._unsupported_extension(method)

    def on_connect(self, conn: Any) -> None:
        return None


class AcpDriver(Driver):
    """Persistent ACP agent process with negotiated session semantics."""

    static_steer_capability = SteerCapability.NONE

    def __init__(
        self,
        process_factory: Callable[..., Any] | None = None,
        *,
        connection_factory: Callable[..., Any] = connect_to_agent,
        executable_version: str = "",
    ) -> None:
        self.process_factory = process_factory
        self.connection_factory = connection_factory
        self.executable_version = executable_version
        self._proc: Any = None
        self._conn: Any = None
        self._watcher: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[bytes] | None = None
        self._runtime_ref = RuntimeRef(f"runtime_{uuid.uuid4().hex}")
        self._session_ref = SessionRef("")
        self._native_session_id = ""
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._prompt_task: asyncio.Task[None] | None = None
        self._prompt_sent: asyncio.Future[None] | None = None
        self._events: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._permissions: dict[
            PermissionRef,
            tuple[asyncio.Future[PermissionDecision], list[PermissionOption]],
        ] = {}
        self._output_blocks: set[str] = set()
        self._fallback_block_id = ""
        self._capabilities = acp_capabilities(session_resume=False)
        self._driver_authority: DriverAuthorityServer | None = None
        self._closed = False

    def current_capabilities(self) -> DriverCapabilities:
        return self._capabilities

    async def open(
        self, spec: RuntimeSpec, resume: SessionRef | None = None
    ) -> RuntimeOpened:
        if self._proc is not None:
            raise RuntimeError("driver is already open")
        if not spec.executable:
            raise RuntimeError("ACP driver requires an explicit executable")
        if self._closed:
            self._closed = False
            self._events = asyncio.Queue()
        self._proc = await self._spawn(spec)
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("ACP child did not expose stdio pipes")
        self._stderr_reader = asyncio.create_task(
            drain_subprocess_stream_keeping_tail(self._proc.stderr)
        )
        client = _PuffoAcpClient(self)
        self._conn = self.connection_factory(
            client,
            self._proc.stdin,
            self._proc.stdout,
            observers=[self._observe_stream],
        )
        initialized = await self._conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(
                    read_text_file=False,
                    write_text_file=False,
                ),
                terminal=False,
            ),
            client_info=Implementation(name="puffo-agent", version="2"),
        )
        if initialized.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                "ACP agent negotiated unsupported protocol version "
                f"{initialized.protocol_version}"
            )
        native_caps = initialized.agent_capabilities
        can_load = bool(native_caps and native_caps.load_session)
        self._capabilities = acp_capabilities(session_resume=can_load)
        mcp_servers = [self._acp_mcp_server(server) for server in spec.mcp_servers]
        if resume is not None:
            if not can_load:
                raise RuntimeError("ACP agent does not support session/load")
            await self._conn.load_session(
                cwd=spec.workspace_dir,
                session_id=str(resume),
                mcp_servers=mcp_servers,
            )
            native_session_id = str(resume)
            resumed = True
        else:
            session = await self._conn.new_session(
                cwd=spec.workspace_dir,
                mcp_servers=mcp_servers,
            )
            native_session_id = session.session_id
            resumed = False
        if not native_session_id:
            raise RuntimeError("ACP session response omitted sessionId")
        self._native_session_id = native_session_id
        self._session_ref = SessionRef(native_session_id)
        self._watcher = asyncio.create_task(self._watch_process())
        await self._emit(
            HarnessEventType.SESSION_RESUMED
            if resumed
            else HarnessEventType.SESSION_OPENED
        )

        capability_names = ["cancel", "permission_bridge"]
        if can_load:
            capability_names.append("session/load")
        return RuntimeOpened(
            self._runtime_ref,
            self._session_ref,
            native_session_id,
            resumed,
            self._capabilities,
            ProtocolDiagnostics(
                executable_version=self.executable_version,
                schema_source="agent-client-protocol==0.10.1/protocol-v1",
                native_capabilities=tuple(capability_names),
            ),
        )

    @staticmethod
    def _acp_mcp_server(server: McpServerSpec) -> McpServerStdio:
        return McpServerStdio(
            name=server.name,
            command=server.command,
            args=list(server.args),
            env=[
                EnvVariable(name=name, value=value)
                for name, value in sorted(server.environment.items())
            ],
        )

    async def _spawn(self, spec: RuntimeSpec) -> Any:
        command = (*normalize_launch_argv(spec.executable), *spec.launch_args)
        environment = dict(spec.environment)
        # This is a Driver-owned carrier, never caller-supplied ambient state.
        environment.pop(DRIVER_AUTHORITY_FD_ENV, None)
        uses_driver_authority = _uses_lingtai_driver_authority(command)
        if self.process_factory is not None:
            if uses_driver_authority:
                raise RuntimeError(
                    "constrained LingTai ACP requires the POSIX local spawn path"
                )
            # One call with the declared signature. Retrying on TypeError
            # cannot tell "wrong arity" from "the factory raised TypeError
            # internally", and the retry would spawn a second child after the
            # first already exists.
            proc = self.process_factory(
                command, replace(spec, environment=environment)
            )
            return await proc if asyncio.iscoroutine(proc) else proc
        if not uses_driver_authority:
            return await spawn_framed_child(
                command,
                env=environment,
                cwd=spec.workspace_dir or None,
            )

        authority = DriverAuthorityServer()
        endpoint = authority.issue_root(launch_id=str(self._runtime_ref))
        endpoint_fd = endpoint.fileno()
        environment[DRIVER_AUTHORITY_FD_ENV] = str(endpoint_fd)
        try:
            proc = await spawn_framed_child(
                command,
                env=environment,
                cwd=spec.workspace_dir or None,
                pass_fds=(endpoint_fd,),
            )
        except BaseException:
            authority.close()
            raise
        finally:
            endpoint.close()
        self._driver_authority = authority
        return proc

    async def start_turn(self, input: TurnInput):
        if self._conn is None:
            raise RuntimeError("driver is not open")
        if self._active.value:
            raise RuntimeError("one turn is already active")
        turn = TurnRef(f"turn_{uuid.uuid4().hex}")
        native_turn = input.client_correlation_id or str(uuid.uuid4())
        self._active = turn
        self._active_native_turn_id = native_turn
        self._output_blocks.clear()
        self._fallback_block_id = f"assistant_{uuid.uuid4().hex}"
        self._prompt_sent = asyncio.get_running_loop().create_future()
        prompt_sent = self._prompt_sent
        prompt_task = asyncio.create_task(
            self._run_prompt(turn, input.content)
        )
        self._prompt_task = prompt_task
        sent_wait = asyncio.ensure_future(asyncio.shield(prompt_sent))
        done, _ = await asyncio.wait(
            {sent_wait, prompt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if prompt_task in done and not prompt_sent.done():
            await prompt_task
            raise RuntimeError("ACP prompt completed before request admission")
        await sent_wait
        return TurnStarted(
            turn,
            native_turn_id=native_turn,
            accepted=True,
            delivery="jsonrpc_request_written",
        )

    async def _run_prompt(self, turn: TurnRef, content: str) -> None:
        try:
            response = await self._conn.prompt(
                session_id=self._native_session_id,
                prompt=[text_block(content)],
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._finish_turn(
                turn,
                HarnessEventType.TURN_ABANDONED,
                {
                    "outcome": "abandoned",
                    "error_code": "acp_prompt_failed",
                    "retryable": True,
                },
            )
            return
        stop_reason = str(response.stop_reason)
        outcome = "succeeded" if stop_reason == "end_turn" else "failed"
        if stop_reason == "cancelled":
            event_type = HarnessEventType.TURN_ABANDONED
            data = {
                "outcome": "abandoned",
                "error_code": "cancelled",
                "retryable": False,
            }
        else:
            event_type = HarnessEventType.TURN_COMPLETED
            data = {"outcome": outcome, "stop_reason": stop_reason}
        await self._finish_turn(turn, event_type, data)

    async def _finish_turn(
        self,
        turn: TurnRef,
        type_: HarnessEventType,
        data: dict[str, Any],
    ) -> None:
        if turn != self._active:
            return
        for block_id in tuple(self._output_blocks):
            await self._emit(
                HarnessEventType.ASSISTANT_COMPLETED,
                turn=turn,
                data={"block_id": block_id},
            )
        for future, _ in self._permissions.values():
            if not future.done():
                future.set_result(PermissionDecision.DENY)
        self._permissions.clear()
        terminal = self._event(type_, turn=turn, data=data)
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._prompt_sent = None
        self._prompt_task = None
        self._output_blocks.clear()
        await self._events.put(terminal)

    async def steer_turn(self, turn: TurnRef, input: TurnInput):
        return UnsupportedCapability("steer")

    async def cancel_turn(self, turn: TurnRef):
        self._require_active(turn)
        await self._conn.cancel(session_id=self._native_session_id)
        return CancelReceipt(True, turn)

    async def context_status(self):
        return UnsupportedCapability("context_status", "ACP context is push-only")

    async def compact(self, request: CompactRequest):
        return UnsupportedCapability("compact")

    async def resolve_permission(
        self, request: PermissionRef, decision: PermissionDecision
    ):
        pending = self._permissions.get(request)
        if pending is None:
            raise RuntimeError("unknown or stale permission reference")
        future, _ = pending
        if not future.done():
            future.set_result(decision)
        return PermissionReceipt(True, request)

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
        for future, _ in self._permissions.values():
            if not future.done():
                future.set_result(PermissionDecision.DENY)
        self._permissions.clear()
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None
        proc, self._proc = self._proc, None
        if proc is not None and getattr(proc, "returncode", None) is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        current = asyncio.current_task()
        for task in (self._prompt_task, self._watcher, self._stderr_reader):
            if task is not None and task is not current:
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (self._prompt_task, self._watcher, self._stderr_reader)
                if task is not None and task is not current
            ),
            return_exceptions=True,
        )
        self._prompt_task = None
        self._watcher = None
        self._stderr_reader = None
        authority, self._driver_authority = self._driver_authority, None
        if authority is not None:
            authority.close()
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        await self._events.put(None)

    async def _watch_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        returncode = await proc.wait()
        if not self._closed:
            await self._emit(
                HarnessEventType.RUNTIME_EXITED,
                turn=self._active if self._active.value else None,
                data=runtime_exited_data(returncode),
            )

    async def _observe_stream(self, event: StreamEvent) -> None:
        if event.direction is not StreamDirection.OUTGOING:
            return
        if event.message.get("method") != "session/prompt":
            return
        future = self._prompt_sent
        if future is not None and not future.done():
            future.set_result(None)

    async def _session_update(self, session_id: str, update: Any) -> None:
        if session_id != self._native_session_id:
            await self._emit(
                HarnessEventType.RUNTIME_WARNING,
                data={"code": "foreign_session_update"},
            )
            return
        turn = self._active if self._active.value else None
        if isinstance(update, AgentMessageChunk):
            text = getattr(update.content, "text", None)
            if not isinstance(text, str):
                return
            block_id = update.message_id or self._fallback_block_id
            self._output_blocks.add(block_id)
            await self._emit(
                HarnessEventType.ASSISTANT_DELTA,
                turn=turn,
                data={"block_id": block_id, "delta": text},
                native_payload=update,
            )
            return
        if isinstance(update, ToolCallStart):
            await self._emit(
                HarnessEventType.TOOL_STARTED,
                turn=turn,
                data={
                    "tool_call_ref": update.tool_call_id,
                    "label": update.title,
                },
                native_payload=update,
            )
            return
        if isinstance(update, ToolCallProgress):
            status = str(update.status or "in_progress")
            if status in {"completed", "failed"}:
                type_ = HarnessEventType.TOOL_COMPLETED
                data = {
                    "tool_call_ref": update.tool_call_id,
                    "label": update.title or "",
                    "outcome": "succeeded" if status == "completed" else "failed",
                }
            else:
                type_ = HarnessEventType.TOOL_UPDATED
                data = {
                    "tool_call_ref": update.tool_call_id,
                    "label": update.title or "",
                    "state": status,
                }
            await self._emit(type_, turn=turn, data=data, native_payload=update)
            return
        if isinstance(update, UsageUpdate):
            await self._emit(
                HarnessEventType.CONTEXT_UPDATED,
                turn=turn,
                data={
                    "context_tokens": max(0, update.used),
                    "context_window": max(0, update.size),
                },
                native_payload=update,
            )
            return
        await self._emit(
            HarnessEventType.SESSION_UPDATED,
            turn=turn,
            data={"record_type": getattr(update, "session_update", "unknown")},
            native_payload=update,
        )

    async def _request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: Any,
    ) -> RequestPermissionResponse:
        if session_id != self._native_session_id or not self._active.value:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        ref = PermissionRef(f"permission_{uuid.uuid4().hex}")
        future = asyncio.get_running_loop().create_future()
        self._permissions[ref] = (future, options)
        await self._emit(
            HarnessEventType.PERMISSION_REQUESTED,
            turn=self._active,
            data={
                "permission_ref": str(ref),
                "tool_call_ref": str(getattr(tool_call, "tool_call_id", "")),
                "label": str(getattr(tool_call, "title", "")),
                "options": tuple(
                    {
                        "id": option.option_id,
                        "name": option.name,
                        "kind": option.kind,
                    }
                    for option in options
                ),
            },
            native_payload=tool_call,
        )
        decision = await future
        self._permissions.pop(ref, None)
        if decision is PermissionDecision.DENY:
            return RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )
        selected = _preferred_permission(options)
        if selected is None:
            return RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )
        return RequestPermissionResponse(
            outcome=AllowedOutcome(
                outcome="selected",
                option_id=selected.option_id,
            )
        )

    async def _unsupported_extension(self, method: str) -> None:
        await self._emit(
            HarnessEventType.RUNTIME_WARNING,
            turn=self._active if self._active.value else None,
            data={"code": "unsupported_extension", "method": f"_{method}"},
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
            self._event(
                type_,
                turn=turn,
                data=data,
                native_payload=native_payload,
            )
        )

    def _event(
        self,
        type_: HarnessEventType,
        *,
        turn: TurnRef | None = None,
        data: dict[str, Any] | None = None,
        native_payload: Any = None,
    ) -> HarnessEvent:
        return HarnessEvent.normalized(
            type=type_,
            driver="acp",
            session_ref=self._session_ref,
            turn_ref=turn,
            native_session_id=self._native_session_id,
            native_turn_id=self._active_native_turn_id,
            data=data or {},
            native_payload=native_payload,
        )

    def _require_active(self, turn: TurnRef) -> None:
        if turn != self._active or not self._active.value:
            raise RuntimeError("stale or foreign active turn")


def _preferred_permission(
    options: list[PermissionOption],
) -> PermissionOption | None:
    for kind in ("allow_once", "allow_always"):
        for option in options:
            if option.kind == kind:
                return option
    return options[0] if options else None


GenericAcpDriver = AcpDriver


def _uses_lingtai_driver_authority(command: tuple[str, ...]) -> bool:
    """Select only LingTai's constrained ACP profile, independent of argv[0]."""

    try:
        acp_index = command.index("acp")
    except ValueError:
        return False
    profile_args = command[acp_index + 1 :]
    return any(
        (
            arg == "--profile"
            and index + 1 < len(profile_args)
            and profile_args[index + 1] == "puffo-v0"
        )
        or arg == "--profile=puffo-v0"
        for index, arg in enumerate(profile_args)
    )
