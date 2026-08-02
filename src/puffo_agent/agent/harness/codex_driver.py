"""Codex app-server Driver using the exercised stable v2 JSONL shapes."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from .driver import (
    CancelCapability,
    CancelReceipt,
    CompactCapability,
    CompactReceipt,
    CompactRequest,
    ContextStatus,
    ContextStatusCapability,
    DriverCapabilities,
    HarnessDriver,
    HarnessEvent,
    HarnessEventType,
    InputReceipt,
    PermissionDecision,
    PermissionReceipt,
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
)


CODEX_CAPABILITIES = DriverCapabilities(
    session_resume=True,
    inflight_turn_recovery=False,
    steer=SteerCapability.CURRENT_TURN,
    cancel=CancelCapability.TYPED,
    context_status=ContextStatusCapability.PUSH,
    compact=CompactCapability.TYPED,
    permission_bridge=True,
)

logger = logging.getLogger(__name__)


class CodexAppServerDriver(HarnessDriver):
    def __init__(
        self, process_factory: Callable[[RuntimeSpec], Any] | None = None,
        *, executable_version: str = "",
    ):
        self.process_factory = process_factory
        self.executable_version = executable_version
        self._proc: Any = None
        self._reader: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._request_id = 0
        self._events: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._runtime_ref = RuntimeRef(f"runtime_{uuid.uuid4().hex}")
        self._session_ref = SessionRef("")
        self._native_session_id = ""
        self._active = TurnRef("")
        self._active_native_turn_id = ""
        self._context = ContextStatus(stale=True)
        self._permission_requests: dict[PermissionRef, int] = {}
        self._open_output_blocks: set[str] = set()
        self._closed = False

    async def open(
        self, spec: RuntimeSpec, resume: SessionRef | None = None
    ) -> RuntimeOpened:
        if self._proc is not None:
            raise RuntimeError("driver is already open")
        if self._closed:
            self._closed = False
            self._events = asyncio.Queue()
        if self.process_factory is None:
            executable = spec.executable or "codex"
            self._proc = await asyncio.create_subprocess_exec(
                executable, *spec.launch_args, "app-server",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=spec.workspace_dir or None,
                env=dict(spec.environment) or None,
            )
        else:
            self._proc = self.process_factory(spec)
            if asyncio.iscoroutine(self._proc):
                self._proc = await self._proc
        self._reader = asyncio.create_task(self._read_loop())
        await self._request("initialize", {
            "clientInfo": {"name": "puffo-agent", "version": "1"},
            "capabilities": {},
        })
        await self._write({"method": "initialized", "params": {}})
        if resume is None:
            result = await self._request("thread/start", {
                "cwd": spec.workspace_dir,
                "approvalPolicy": (
                    "never"
                    if spec.permission_mode == "bypassPermissions"
                    else "untrusted"
                ),
                "sandbox": spec.sandbox,
                **({"model": spec.model} if spec.model else {}),
            })
            resumed = False
        else:
            result = await self._request("thread/resume", {
                "threadId": str(resume),
            })
            resumed = True
        thread = result.get("thread", result) if isinstance(result, dict) else {}
        native = str(
            thread.get("id") or thread.get("threadId") or str(resume or "")
        )
        if not native:
            raise RuntimeError("Codex thread response omitted native thread id")
        self._native_session_id = native
        self._session_ref = SessionRef(native)
        await self._emit(
            HarnessEventType.SESSION_RESUMED if resumed
            else HarnessEventType.SESSION_OPENED,
            native_payload=result,
        )
        return RuntimeOpened(
            self._runtime_ref, self._session_ref, native, resumed,
            CODEX_CAPABILITIES,
            ProtocolDiagnostics(
                executable_version=self.executable_version,
                schema_source="generated",
                native_capabilities=(
                    "turn/steer", "turn/interrupt",
                    "thread/compact/start", "permission_bridge",
                ),
            ),
        )

    async def start_turn(self, input: TurnInput):
        if self._active.value:
            raise RuntimeError("one turn is already active")
        local = TurnRef(f"turn_{uuid.uuid4().hex}")
        self._active = local
        params: dict[str, Any] = {
            "threadId": self._native_session_id,
            "input": [{"type": "text", "text": input.content}],
        }
        if input.client_correlation_id:
            params["clientUserMessageId"] = input.client_correlation_id
        try:
            result = await self._request("turn/start", params)
        except BaseException:
            self._active = TurnRef("")
            raise
        turn = result.get("turn", result) if isinstance(result, dict) else {}
        native = str(turn.get("id") or turn.get("turnId") or "")
        if not native:
            self._active = TurnRef("")
            raise RuntimeError("Codex turn/start omitted native turn id")
        self._active_native_turn_id = native
        return TurnStarted(local, native)

    async def steer_turn(self, turn: TurnRef, input: TurnInput):
        self._require_active(turn)
        await self._request("turn/steer", {
            "threadId": self._native_session_id,
            "expectedTurnId": self._active_native_turn_id,
            "input": [{"type": "text", "text": input.content}],
        })
        return InputReceipt(True, turn, input.client_correlation_id)

    async def cancel_turn(self, turn: TurnRef):
        self._require_active(turn)
        await self._request("turn/interrupt", {
            "threadId": self._native_session_id,
            "turnId": self._active_native_turn_id,
        })
        return CancelReceipt(True, turn)

    async def context_status(self):
        return self._context

    async def compact(self, request: CompactRequest):
        if self._active_native_turn_id:
            raise RuntimeError("compaction requires an idle session")
        result = await self._request("thread/compact/start", {
            "threadId": self._native_session_id,
        })
        operation = str(
            result.get("id") if isinstance(result, dict) else ""
        )
        return CompactReceipt(True, operation)

    async def resolve_permission(
        self, request: PermissionRef, decision: PermissionDecision
    ):
        request_id = self._permission_requests.pop(request, None)
        if request_id is None:
            raise RuntimeError("unknown or stale permission reference")
        await self._write({
            "id": request_id,
            "result": {"decision": decision.value},
        })
        await self._emit(
            HarnessEventType.PERMISSION_UPDATED,
            turn_ref=self._active,
            data={
                "permission_ref": str(request),
                "state": decision.value,
            },
        )
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

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"id": request_id, "method": method, "params": params})
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _write(self, frame: dict[str, Any]) -> None:
        encoded = json.dumps(
            frame, separators=(",", ":"), ensure_ascii=False
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
                if "id" in frame and ("result" in frame or "error" in frame):
                    future = self._pending.get(frame["id"])
                    if future is not None and not future.done():
                        if "error" in frame:
                            future.set_exception(RuntimeError("Codex request failed"))
                        else:
                            future.set_result(frame.get("result"))
                    continue
                if "id" in frame and "method" in frame:
                    await self._server_request(frame)
                    continue
                await self._notification(frame)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Codex app-server exited"))
            if not self._closed:
                await self._emit(HarnessEventType.RUNTIME_EXITED)

    async def _server_request(self, frame: dict[str, Any]) -> None:
        method = str(frame.get("method") or "")
        if "approval" in method.lower() or "permission" in method.lower():
            ref = PermissionRef(f"perm_{uuid.uuid4().hex}")
            self._permission_requests[ref] = int(frame["id"])
            await self._emit(
                HarnessEventType.PERMISSION_REQUESTED,
                turn_ref=self._active,
                data={
                    "permission_ref": str(ref), "state": "pending",
                    "title": "Permission required",
                },
                native_payload=frame,
            )
            return
        await self._write({
            "id": frame["id"],
            "error": {"code": -32601, "message": "unsupported request"},
        })

    async def _notification(self, frame: dict[str, Any]) -> None:
        method = str(frame.get("method") or "")
        params = frame.get("params") if isinstance(frame.get("params"), dict) else {}
        diagnostic_item = params.get("item")
        if isinstance(diagnostic_item, dict) and (
            "tool" in method.lower()
            or "tool" in str(diagnostic_item.get("type") or "").lower()
            or any(
                key in diagnostic_item
                for key in ("tool", "server", "namespace")
            )
        ):
            logger.debug(
                "codex tool notification shape method=%s type=%s tool=%s "
                "server=%s namespace=%s status=%s keys=%s",
                method,
                str(diagnostic_item.get("type") or ""),
                str(
                    diagnostic_item.get("tool")
                    or diagnostic_item.get("name")
                    or ""
                ),
                str(diagnostic_item.get("server") or ""),
                str(diagnostic_item.get("namespace") or ""),
                str(diagnostic_item.get("status") or ""),
                sorted(str(key) for key in diagnostic_item),
            )
        if method == "turn/started":
            await self._emit(
                HarnessEventType.TURN_STARTED, turn_ref=self._active,
                native_payload=frame,
            )
        elif method == "item/agentMessage/delta":
            block_id = str(params.get("itemId") or "result")
            self._open_output_blocks.add(block_id)
            await self._emit(
                HarnessEventType.ASSISTANT_DELTA, turn_ref=self._active,
                data={
                    "text": str(params.get("delta") or ""),
                    "block_id": block_id,
                }, native_payload=frame,
            )
        elif method in {"item/completed", "item/agentMessage/completed"}:
            item = params.get("item")
            if isinstance(item, dict):
                item_id = str(item.get("id") or params.get("itemId") or "")
                item_type = str(item.get("type") or "")
                if item_type in {
                    "mcpToolCall",
                    "dynamicToolCall",
                    "functionCall",
                    "toolCall",
                }:
                    name = str(item.get("name") or item.get("tool") or "")
                    if name.startswith("mcp__") and "__" in name:
                        name = name.rsplit("__", 1)[-1]
                    arguments = item.get("arguments") or item.get("input") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    result = item.get(
                        "result",
                        item.get("output", item.get("contentItems")),
                    )
                    status = str(item.get("status") or "").lower()
                    is_error = bool(item.get("error")) or status == "failed"
                    if item_type == "dynamicToolCall":
                        is_error = is_error or item.get("success") is False
                    await self._emit(
                        HarnessEventType.TOOL_COMPLETED,
                        turn_ref=self._active,
                        data={
                            "tool_call_ref": item_id,
                            "label": name or "Tool",
                            "outcome": (
                                "failed" if is_error else "succeeded"
                            ),
                        },
                        native_payload={
                            "_puffo_internal": "tool_result",
                            "tool_call_id": item_id,
                            "tool_name": name,
                            "arguments": arguments,
                            "result": result,
                            "result_omitted": (
                                item_type == "dynamicToolCall"
                                and result is None
                            ),
                            "is_error": is_error,
                        },
                    )
            block_id = str(
                (item.get("id") if isinstance(item, dict) else None)
                or params.get("itemId") or "result"
            )
            if block_id in self._open_output_blocks:
                self._open_output_blocks.discard(block_id)
                await self._emit(
                    HarnessEventType.ASSISTANT_COMPLETED,
                    turn_ref=self._active,
                    data={"block_id": block_id},
                    native_payload=frame,
                )
        elif method == "thread/tokenUsage/updated":
            usage = params.get("tokenUsage", params)
            self._context = ContextStatus(
                used_tokens=_integer(usage.get("totalTokens")),
                context_window=_integer(usage.get("modelContextWindow")),
                stale=False,
            )
            await self._emit(
                HarnessEventType.CONTEXT_UPDATED, turn_ref=self._active,
                data={
                    "used_tokens": self._context.used_tokens,
                    "context_window": self._context.context_window,
                }, native_payload=frame,
            )
        elif method == "turn/completed":
            status = str(
                (params.get("turn") or {}).get("status")
                if isinstance(params.get("turn"), dict)
                else params.get("status") or "completed"
            )
            for block_id in tuple(sorted(self._open_output_blocks)):
                await self._emit(
                    HarnessEventType.ASSISTANT_COMPLETED,
                    turn_ref=self._active,
                    data={"block_id": block_id},
                    native_payload=frame,
                )
            self._open_output_blocks.clear()
            await self._emit(
                HarnessEventType.TURN_COMPLETED, turn_ref=self._active,
                data={"outcome": (
                    "cancelled" if "interrupt" in status else
                    "failed" if "fail" in status else "succeeded"
                )}, native_payload=frame,
            )
            self._active, self._active_native_turn_id = TurnRef(""), ""
        else:
            await self._emit(
                HarnessEventType.RUNTIME_WARNING,
                data={"code": "unknown_notification", "method": method},
                native_payload=frame,
            )

    async def _emit(
        self, type_: HarnessEventType, *, turn_ref: TurnRef | None = None,
        data: dict[str, Any] | None = None, native_payload: Any = None,
    ) -> None:
        await self._events.put(HarnessEvent.normalized(
            type=type_, driver="codex", session_ref=self._session_ref,
            turn_ref=(turn_ref if turn_ref and turn_ref.value else None),
            native_session_id=self._native_session_id,
            native_turn_id=self._active_native_turn_id,
            data=data or {}, native_payload=native_payload,
        ))

    def _require_active(self, turn: TurnRef) -> None:
        if not self._active_native_turn_id or turn != self._active:
            raise RuntimeError("stale or foreign active turn")


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


CodexDriver = CodexAppServerDriver
