"""Claude Code CLI Driver with replay-based idle input acceptance."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from .driver import (
    CancelCapability,
    CompactCapability,
    CompactReceipt,
    CompactRequest,
    ContextStatusCapability,
    DriverCapabilities,
    HarnessDriver,
    HarnessEvent,
    HarnessEventType,
    PermissionDecision,
    PermissionRef,
    ProtocolDiagnostics,
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


class AmbiguousDeliveryError(RuntimeError):
    """Claude may have accepted input before Puffo observed its replay."""


def claude_capabilities(compact_advertised: bool = False) -> DriverCapabilities:
    return DriverCapabilities(
        session_resume=True,
        inflight_turn_recovery=False,
        steer=SteerCapability.NONE,
        cancel=CancelCapability.NONE,
        context_status=ContextStatusCapability.NONE,
        compact=(
            CompactCapability.SESSION_COMMAND
            if compact_advertised else CompactCapability.NONE
        ),
        permission_bridge=False,
    )


class ClaudeCodeCliDriver(HarnessDriver):
    def __init__(
        self, process_factory: Callable[..., Any] | None = None,
        *, executable_version: str = "", replay_timeout: float = 30,
    ):
        self.process_factory = process_factory
        self.executable_version = executable_version
        self.replay_timeout = replay_timeout
        self._proc: Any = None
        self._reader: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._events: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._init: asyncio.Future[dict[str, Any]] | None = None
        self._pending_replay: asyncio.Future[str] | None = None
        self._pending_content = ""
        self._pending_uuid = ""
        self._runtime_ref = RuntimeRef(f"runtime_{uuid.uuid4().hex}")
        self._session_ref = SessionRef("")
        self._native_session_id = ""
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._closed = False
        self._compact_advertised = False
        self._output_block_counter = 0
        self._tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

    async def open(
        self, spec: RuntimeSpec, resume: SessionRef | None = None
    ) -> RuntimeOpened:
        if self._proc is not None:
            raise RuntimeError("driver is already open")
        if self._closed:
            self._closed = False
            self._events = asyncio.Queue()
        args = [
            spec.executable or "claude", *spec.launch_args, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--replay-user-messages", "--verbose",
        ]
        if spec.model:
            args.extend(["--model", spec.model])
        if resume is not None:
            args.extend(["--resume", str(resume)])
        if self.process_factory is None:
            env = os.environ.copy()
            env.update(spec.environment)
            self._proc = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=spec.workspace_dir or None, env=env,
                limit=16 * 1024 * 1024,
            )
        else:
            try:
                self._proc = self.process_factory(args, spec)
            except TypeError:
                self._proc = self.process_factory(args)
            if asyncio.iscoroutine(self._proc):
                self._proc = await self._proc
        self._init = asyncio.get_running_loop().create_future()
        self._reader = asyncio.create_task(self._read_loop())
        init = await asyncio.wait_for(self._init, timeout=self.replay_timeout)
        native = str(init.get("session_id") or "")
        if not native:
            raise RuntimeError("Claude system/init omitted session_id")
        self._native_session_id = native
        self._session_ref = SessionRef(native)
        commands = init.get("slash_commands") or ()
        self._compact_advertised = "/compact" in commands or "compact" in commands
        resumed = resume is not None
        await self._emit(
            HarnessEventType.SESSION_RESUMED if resumed
            else HarnessEventType.SESSION_OPENED,
            native_payload=init,
        )
        return RuntimeOpened(
            self._runtime_ref, self._session_ref, native, resumed,
            claude_capabilities(self._compact_advertised),
            ProtocolDiagnostics(
                executable_version=self.executable_version,
                schema_source="documented",
                native_capabilities=tuple(str(value) for value in commands),
            ),
        )

    async def start_turn(self, input: TurnInput):
        if self._active.value or self._pending_replay is not None:
            raise RuntimeError("one turn is already active")
        local = TurnRef(f"turn_{uuid.uuid4().hex}")
        replay_uuid = input.client_correlation_id or str(uuid.uuid4())
        self._active = local
        self._pending_content = _normalize_content(input.content)
        self._pending_uuid = replay_uuid
        self._active_native_turn_id = replay_uuid
        self._pending_replay = asyncio.get_running_loop().create_future()
        frame = {
            "type": "user",
            "session_id": self._native_session_id,
            "parent_tool_use_id": None,
            "uuid": replay_uuid,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": input.content}],
            },
        }
        await self._write(frame)
        try:
            observed_uuid = await asyncio.wait_for(
                asyncio.shield(self._pending_replay), self.replay_timeout
            )
        except (asyncio.TimeoutError, AmbiguousDeliveryError):
            self._active = TurnRef("")
            self._active_native_turn_id = ""
            return TurnStarted(
                local, accepted=False, delivery="ambiguous_at_least_once"
            )
        finally:
            self._pending_replay = None
            self._pending_content = ""
            self._pending_uuid = ""
        return TurnStarted(
            local, native_turn_id=replay_uuid,
            accepted=True, delivery="replay_acknowledged",
            replay_id=observed_uuid,
        )

    async def steer_turn(self, turn: TurnRef, input: TurnInput):
        return UnsupportedCapability("steer")

    async def cancel_turn(self, turn: TurnRef):
        return UnsupportedCapability("cancel")

    async def context_status(self):
        return UnsupportedCapability("context_status")

    async def compact(self, request: CompactRequest):
        if not self._compact_advertised:
            return UnsupportedCapability("compact")
        if self._active.value:
            raise RuntimeError("Claude /compact requires an idle session")
        text = "/compact"
        if request.instructions:
            text += " " + request.instructions
        frame = {
            "type": "user", "session_id": self._native_session_id,
            "parent_tool_use_id": None,
            "message": {"role": "user", "content": text},
        }
        await self._write(frame)
        return CompactReceipt(True)

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
        proc, self._proc = self._proc, None
        if proc is not None and getattr(proc, "returncode", None) is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        await self._events.put(None)

    async def _write(self, frame: dict[str, Any]) -> None:
        encoded = json.dumps(
            frame, ensure_ascii=False, separators=(",", ":")
        ).encode() + b"\n"
        async with self._write_lock:
            self._proc.stdin.write(encoded)
            await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    await self._emit(
                        HarnessEventType.RUNTIME_WARNING,
                        data={"code": "protocol_parse"},
                    )
                    continue
                await self._handle(frame)
        finally:
            error = AmbiguousDeliveryError(
                "Claude exited before replay acceptance was observed"
            )
            if self._pending_replay is not None and not self._pending_replay.done():
                self._pending_replay.set_exception(error)
            if self._init is not None and not self._init.done():
                self._init.set_exception(RuntimeError("Claude exited before init"))
            if not self._closed:
                await self._emit(
                    HarnessEventType.RUNTIME_EXITED,
                    turn_ref=self._active if self._active.value else None,
                )

    async def _handle(self, frame: dict[str, Any]) -> None:
        type_ = str(frame.get("type") or "")
        subtype = str(frame.get("subtype") or "")
        if type_ == "system" and subtype == "init":
            if self._init is not None and not self._init.done():
                self._init.set_result(frame)
            return
        if self._is_exact_replay(frame):
            assert self._pending_replay is not None
            self._pending_replay.set_result(str(frame.get("uuid")))
            await self._emit(
                HarnessEventType.TURN_STARTED, turn_ref=self._active,
                native_payload=frame,
            )
            return
        if type_ == "assistant":
            if not self._active.value:
                await self._emit(
                    HarnessEventType.SESSION_UPDATED,
                    data={"record_type": "assistant"},
                    native_payload=frame,
                )
                return
            content = (frame.get("message") or {}).get("content") or ()
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    self._output_block_counter += 1
                    block_id = str(
                        block.get("id")
                        or frame.get("uuid")
                        or f"result_{self._output_block_counter}"
                    )
                    await self._emit(
                        HarnessEventType.ASSISTANT_DELTA,
                        turn_ref=self._active,
                        data={
                            "text": str(block.get("text") or ""),
                            "block_id": block_id,
                        }, native_payload=frame,
                    )
                    await self._emit(
                        HarnessEventType.ASSISTANT_COMPLETED,
                        turn_ref=self._active,
                        data={"block_id": block_id},
                        native_payload=frame,
                    )
                elif block.get("type") == "tool_use":
                    tool_call_id = str(block.get("id") or "")
                    arguments = block.get("input") or {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    self._tool_calls[tool_call_id] = (
                        str(block.get("name") or ""), arguments
                    )
                    await self._emit(
                        HarnessEventType.TOOL_STARTED,
                        turn_ref=self._active,
                        data={
                            "tool_call_ref": str(block.get("id") or ""),
                            "label": str(block.get("name") or "Tool"),
                        }, native_payload=frame,
                    )
            return
        if type_ == "user" and self._active.value:
            content = (frame.get("message") or {}).get("content") or ()
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_call_id = str(block.get("tool_use_id") or "")
                tool_name, arguments = self._tool_calls.pop(
                    tool_call_id, ("", {})
                )
                await self._emit(
                    HarnessEventType.TOOL_COMPLETED,
                    turn_ref=self._active,
                    data={
                        "tool_call_ref": tool_call_id,
                        "label": tool_name or "Tool",
                        "outcome": (
                            "failed" if block.get("is_error") else "succeeded"
                        ),
                    },
                    native_payload={
                        "_puffo_internal": "tool_result",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": block.get("content"),
                        "is_error": bool(block.get("is_error")),
                    },
                )
            return
        if type_ == "result":
            if not self._active.value:
                await self._emit(
                    HarnessEventType.SESSION_UPDATED,
                    data={"record_type": "result"},
                    native_payload=frame,
                )
                return
            active = self._active
            failed = subtype not in {"success", ""}
            usage = frame.get("usage") or {}
            await self._emit(
                HarnessEventType.TURN_COMPLETED, turn_ref=active,
                data={
                    "outcome": "failed" if failed else "succeeded",
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                }, native_payload=frame,
            )
            # Reader stays alive. Records after result are session-scoped.
            self._active = TurnRef("")
            self._active_native_turn_id = ""
            return
        await self._emit(
            HarnessEventType.SESSION_UPDATED,
            data={"record_type": type_ or "unknown"},
            native_payload=frame,
        )

    def _is_exact_replay(self, frame: dict[str, Any]) -> bool:
        if self._pending_replay is None or self._pending_replay.done():
            return False
        if frame.get("type") != "user" or frame.get("isReplay") is not True:
            return False
        if str(frame.get("session_id") or "") != self._native_session_id:
            return False
        if frame.get("parent_tool_use_id") is not None:
            return False
        if str(frame.get("uuid") or "") != self._pending_uuid:
            return False
        return _normalize_content(
            (frame.get("message") or {}).get("content")
        ) == self._pending_content

    async def _emit(
        self, type_: HarnessEventType, *, turn_ref: TurnRef | None = None,
        data: dict[str, Any] | None = None, native_payload: Any = None,
    ) -> None:
        await self._events.put(HarnessEvent.normalized(
            type=type_, driver="claude-code",
            session_ref=self._session_ref, turn_ref=turn_ref,
            native_session_id=self._native_session_id,
            native_turn_id=self._active_native_turn_id,
            data=data or {}, native_payload=native_payload,
        ))


def _normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\r\n", "\n")
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ).replace("\r\n", "\n")
    return ""


ClaudeDriver = ClaudeCodeCliDriver
