from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
from puffo_agent.agent.core import PuffoAgent
from puffo_agent.agent.global_inbox_runtime import GlobalInboxRuntime
from puffo_agent.agent.message_store import (
    MessageStore,
    ProcessingState,
    ReceiptDisposition,
)
from puffo_agent.agent.harness import UnsupportedDriver, build_driver
from puffo_agent.agent.harness.claude_code_driver import (
    ClaudeCodeCliDriver,
    claude_capabilities,
)
from puffo_agent.agent.harness.codex_driver import (
    CODEX_CAPABILITIES,
    CodexAppServerDriver,
)
from puffo_agent.agent.harness.driver import (
    CancelReceipt,
    CompactRequest,
    HarnessDriver,
    HarnessEvent,
    PermissionDecision,
    PermissionRef,
    RuntimeOpened,
    RuntimeRef,
    RuntimeSpec,
    SessionRef,
    TurnInput,
    TurnRef,
    TurnStarted,
    UnsupportedCapability,
)
from puffo_agent.agent.harness.runtime_manager import (
    RuntimeManager,
    RuntimeManagerAdapter,
    RuntimeStateError,
    register_runtime_manager,
    unregister_runtime_manager,
)
from puffo_agent.agent.runtime_event_outbox import RuntimeEventOutbox, RuntimeEventProjectingSink
from puffo_agent.agent.runtime_events import RuntimeEventProjector
from puffo_agent.portal.control.client import execute_command


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_name, shape", [
    ("codex", "failed_tool"),
    ("codex", "empty_assistant"),
    ("claude", "failed_tool"),
    ("claude", "empty_assistant"),
])
async def test_provider_events_project_valid_terminal(tmp_path, driver_name, shape):
    """Real provider frames must produce a valid durable terminal stream."""
    outbox = RuntimeEventOutbox(tmp_path / f"{driver_name}-{shape}.db")
    sink = RuntimeEventProjectingSink(
        outbox, RuntimeEventProjector(agent_id="agent", session_ref="session"),
    )
    turn = TurnRef(f"{driver_name}-{shape}-turn")
    if driver_name == "codex":
        driver = CodexAppServerDriver()
        driver._session_ref = SessionRef("native")
        driver._active = turn
        driver._active_native_turn_id = "native-turn"
        await driver._notification({"method": "turn/started", "params": {}})
        if shape == "failed_tool":
            await driver._notification({
                "method": "item/completed",
                "params": {"item": {
                    "id": "tool", "type": "mcpToolCall", "name": "read",
                    "status": "failed", "error": "provider failure",
                }},
            })
        else:
            await driver._notification({
                "method": "item/agentMessage/completed",
                "params": {"item": {"id": "empty", "type": "agentMessage"}},
            })
        await driver._notification({
            "method": "turn/completed",
            "params": {"turn": {"status": "failed" if shape == "failed_tool" else "completed"}},
        })
        expected_outcome = "failed" if shape == "failed_tool" else "succeeded"
    else:
        driver = ClaudeCodeCliDriver()
        driver._session_ref = SessionRef("native")
        driver._native_session_id = "native-session"
        driver._active = turn
        driver._active_native_turn_id = "replay-id"
        driver._pending_content = "start"
        driver._pending_uuid = "replay-id"
        driver._pending_replay = asyncio.get_running_loop().create_future()
        await driver._handle({
            "type": "user", "isReplay": True, "session_id": "native-session",
            "parent_tool_use_id": None, "uuid": "replay-id",
            "message": {"content": [{"type": "text", "text": "start"}]},
        })
        if shape == "failed_tool":
            await driver._handle({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool", "name": "read", "input": {}},
            ]}})
            await driver._handle({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tool", "is_error": True, "content": "failed"},
            ]}})
        else:
            await driver._handle({"type": "assistant", "message": {"content": []}})
        await driver._handle({
            "type": "result", "subtype": "error" if shape == "failed_tool" else "success",
            "message": {},
        })
        expected_outcome = "failed" if shape == "failed_tool" else "succeeded"

    events = []
    while not driver._events.empty():
        events.append(driver._events.get_nowait())
    for value in events:
        await sink(value)
    rows = [row.event for row in outbox.prefix()]
    terminals = [row for row in rows if row["type"] == "turn.finished"]
    assert len(terminals) == 1
    assert terminals[0]["payload"]["outcome"] == expected_outcome
    if shape == "failed_tool":
        assert any(row["type"] == "tool.updated" and row["payload"].get("state") == "failed" for row in rows)
    else:
        assert not any(row["type"] == "output.updated" for row in rows)
    outbox.close()


class _FakeStdin:
    def __init__(self, on_frame=None):
        self.writes: list[bytes] = []
        self.on_frame = on_frame

    def write(self, value: bytes) -> None:
        # A write must contain exactly one complete JSONL command. This catches
        # command/request-reply byte interleaving, not merely JSON validity.
        assert value.endswith(b"\n") and value.count(b"\n") == 1
        json.loads(value)
        self.writes.append(value)
        if self.on_frame is not None:
            self.on_frame(json.loads(value))

    async def drain(self) -> None:
        await asyncio.sleep(0)


class _FakeProcess:
    def __init__(self, on_frame=None):
        self.stdout = asyncio.StreamReader()
        self.stdin = _FakeStdin(on_frame)
        self.stderr = asyncio.StreamReader()
        self.returncode = None
        self.terminated = 0

    def feed(self, frame: dict) -> None:
        self.stdout.feed_data(json.dumps(frame).encode() + b"\n")

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = 0
        self.stdout.feed_eof()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        return int(self.returncode or 0)


async def _next_matching(stream, type_: str):
    async for value in stream:
        kind = getattr(value.type, "value", value.type)
        if kind == type_:
            return value
    raise AssertionError(f"event stream ended before {type_}")


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.001)


def test_contract_has_full_common_method_surface():
    expected = {
        "open", "start_turn", "steer_turn", "cancel_turn",
        "context_status", "compact", "resolve_permission", "events", "close",
    }
    assert expected <= set(dir(HarnessDriver))
    assert all(
        inspect.iscoroutinefunction(getattr(HarnessDriver, name))
        for name in expected - {"events"}
    )


def test_typed_refs_are_not_interchangeable():
    assert RuntimeRef("x") != SessionRef("x")
    assert SessionRef("x") != TurnRef("x")
    assert TurnRef("x") != PermissionRef("x")


def test_codex_effective_capabilities():
    assert CODEX_CAPABILITIES.session_resume is True
    assert CODEX_CAPABILITIES.inflight_turn_recovery is False
    assert CODEX_CAPABILITIES.steer == "current_turn"
    assert CODEX_CAPABILITIES.cancel == "typed"
    assert CODEX_CAPABILITIES.context_status == "push"
    assert CODEX_CAPABILITIES.compact == "typed"
    assert CODEX_CAPABILITIES.permission_bridge is True


def test_claude_effective_capabilities():
    baseline = claude_capabilities()
    compact = claude_capabilities(True)
    assert baseline.session_resume is True
    assert baseline.inflight_turn_recovery is False
    assert baseline.steer == baseline.cancel == baseline.context_status == "none"
    assert baseline.compact == "none"
    assert compact.compact == "session_command"
    assert baseline.permission_bridge is False


def test_driver_factory_is_closed_without_affecting_legacy_harness_factory():
    assert not isinstance(build_driver("codex"), UnsupportedDriver)
    assert not isinstance(build_driver("claude-code"), UnsupportedDriver)
    assert isinstance(build_driver("hermes"), UnsupportedDriver)
    assert isinstance(build_driver("gemini-cli"), UnsupportedDriver)


@pytest.mark.asyncio
async def test_encrypted_control_cancel_is_agent_scoped_and_idempotent():
    calls = []

    class Manager:
        session_ref = SessionRef("session_a")

        async def cancel_turn(self, turn):
            calls.append(turn)
            return SimpleNamespace(accepted=True)

    manager = Manager()
    register_runtime_manager("agent_a", manager)
    try:
        params = {"session_ref": "session_a", "turn_ref": "turn_a"}
        first = await execute_command(
            "runtime.cancel_turn", "agent_a", params, command_id="command_a"
        )
        second = await execute_command(
            "runtime.cancel_turn", "agent_a", params, command_id="command_a"
        )
        assert first == second == {
            "ok": True, "delivered": True, "completed": False,
        }
        assert calls == [TurnRef("turn_a")]
        rejected = await execute_command(
            "runtime.cancel_turn", "agent_other", params,
            command_id="command_other",
        )
        assert rejected["error_code"] == "runtime_unavailable"
    finally:
        unregister_runtime_manager("agent_a", manager)


@pytest.mark.asyncio
async def test_runtime_manager_one_active_turn_translates_refs_and_close_is_idempotent():
    class FakeDriver(HarnessDriver):
        def __init__(self):
            self.queue = asyncio.Queue()
            self.cancelled = []
            self.close_calls = 0
            self.resume_values = []

        async def open(self, spec, resume=None):
            self.resume_values.append(resume)
            return RuntimeOpened(
                RuntimeRef("native_runtime"), SessionRef("native_session"),
                "provider-session", False, CODEX_CAPABILITIES,
                SimpleNamespace(),
            )

        async def start_turn(self, input):
            return TurnStarted(TurnRef("driver_turn"), "native-turn")

        async def steer_turn(self, turn, input):
            return UnsupportedCapability("unused")

        async def cancel_turn(self, turn):
            self.cancelled.append(turn)
            return CancelReceipt(True, turn)

        async def context_status(self):
            return UnsupportedCapability("context_status")

        async def compact(self, request):
            return UnsupportedCapability("compact")

        async def resolve_permission(self, request, decision):
            return UnsupportedCapability("permission")

        def events(self):
            async def iterate():
                while True:
                    value = await self.queue.get()
                    if value is None:
                        return
                    yield value
            return iterate()

        async def close(self):
            self.close_calls += 1
            await self.queue.put(None)

    driver = FakeDriver()
    manager = RuntimeManager(
        driver, RuntimeSpec("/tmp"), session_ref=SessionRef("logical_session")
    )
    await manager.open()
    assert driver.resume_values == [None]
    started = await manager.start_turn(TurnInput("hello"))
    assert started.turn_ref != TurnRef("driver_turn")
    with pytest.raises(RuntimeStateError, match="one provider turn"):
        await manager.start_turn(TurnInput("overlap"))
    receipt = await manager.cancel_turn(started.turn_ref)
    assert receipt.turn_ref == started.turn_ref
    assert driver.cancelled == [TurnRef("driver_turn")]
    await driver.queue.put(HarnessEvent(
        type="turn.completed", driver="fake",
        session_ref=SessionRef("native_session"),
        turn_ref=TurnRef("driver_turn"), data={"outcome": "cancelled"},
    ))
    terminal = await manager.wait_terminal(started.turn_ref)
    assert terminal.turn_ref == started.turn_ref
    await manager.close()
    await manager.close()
    assert driver.close_calls == 1


@pytest.mark.asyncio
async def test_runtime_manager_correlates_private_tool_result_and_rejects_terminal_failure():
    class FakeDriver(HarnessDriver):
        def __init__(self):
            self.queue = asyncio.Queue()
            self.turn = TurnRef("driver-turn")

        async def open(self, spec, resume=None):
            return RuntimeOpened(
                RuntimeRef("runtime"), SessionRef("native-session"),
                "native-session", False, CODEX_CAPABILITIES, SimpleNamespace(),
            )

        async def start_turn(self, input):
            return TurnStarted(self.turn, "native-turn")

        async def steer_turn(self, turn, input):
            return UnsupportedCapability("steer")

        async def cancel_turn(self, turn):
            return UnsupportedCapability("cancel")

        async def context_status(self):
            return UnsupportedCapability("context_status")

        async def compact(self, request):
            return UnsupportedCapability("compact")

        async def resolve_permission(self, request, decision):
            return UnsupportedCapability("permission")

        def events(self):
            async def iterate():
                while True:
                    yield await self.queue.get()
            return iterate()

        async def close(self):
            return None

    driver = FakeDriver()
    manager = RuntimeManager(driver, RuntimeSpec("/tmp"))
    adapter = RuntimeManagerAdapter(manager)
    await manager.open()
    started = await manager.start_turn(TurnInput("notice"))
    admitted = []

    async def admit(event):
        admitted.append(event)

    adapter.register_continuation_callback(
        admit,
        "read-page-1",
        tool_names=("read_inbox",),
        tool_arguments={"limit": 1},
        correlation_receipt="receipt-1",
    )
    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-1", "label": "read_inbox",
            "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-1",
            "tool_name": "read_inbox",
            "arguments": {"limit": 1},
            "result": "[puffo:model-visible-read:receipt-1]",
            "is_error": False,
        },
    ))
    for _ in range(20):
        if admitted:
            break
        await asyncio.sleep(0)
    assert len(admitted) == 1
    assert admitted[0].planning_cycle_key == "read-page-1"
    assert admitted[0].provider_session_id == "native-session"
    assert admitted[0].provider_turn_id == "native-turn"
    assert admitted[0].tool_call_id == "call-1"

    admitted.clear()
    adapter.register_continuation_callback(
        admit,
        "read-page-without-result",
        tool_names=("read_inbox",),
        tool_arguments={"cursor": "next"},
        correlation_receipt="receipt-omitted",
    )
    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-2", "label": "read_inbox",
            "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-2",
            "tool_name": "read_inbox",
            "arguments": {"cursor": "next"},
            "result": None,
            "is_error": False,
        },
    ))
    for _ in range(20):
        if admitted:
            break
        await asyncio.sleep(0)
    assert admitted == []

    adapter.register_continuation_callback(
        admit,
        "read-page-provider-omitted-result",
        tool_names=("read_inbox",),
        tool_arguments={"cursor": "provider-omitted"},
        correlation_receipt="receipt-provider-omitted",
    )
    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-provider-omitted",
            "label": "read_inbox", "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-provider-omitted",
            "tool_name": "read_inbox",
            "arguments": {"cursor": "provider-omitted"},
            "result": None,
            "result_omitted": True,
            "is_error": False,
        },
    ))
    for _ in range(20):
        if admitted:
            break
        await asyncio.sleep(0)
    assert len(admitted) == 1
    assert admitted[0].planning_cycle_key == "read-page-provider-omitted-result"
    assert admitted[0].tool_call_id == "call-provider-omitted"

    admitted.clear()
    for cycle in ("ambiguous-a", "ambiguous-b"):
        adapter.register_continuation_callback(
            admit,
            cycle,
            tool_names=("read_inbox",),
            tool_arguments={"cursor": "ambiguous"},
            correlation_receipt=f"receipt-{cycle}",
        )
    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-ambiguous", "label": "read_inbox",
            "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-ambiguous",
            "tool_name": "read_inbox",
            "arguments": {"cursor": "ambiguous"},
            "result": None,
            "result_omitted": True,
            "is_error": False,
        },
    ))
    await asyncio.sleep(0)
    assert admitted == []

    adapter.register_continuation_callback(
        admit,
        "read-page-foreign-session",
        tool_names=("read_inbox",),
        tool_arguments={"cursor": "foreign"},
        correlation_receipt="receipt-2",
    )
    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("foreign-session"), turn_ref=driver.turn,
        native_session_id="foreign-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-foreign", "label": "read_inbox",
            "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-foreign",
            "tool_name": "read_inbox",
            "arguments": {"cursor": "foreign"},
            "result": "[puffo:model-visible-read:receipt-2]",
            "is_error": False,
        },
    ))
    await asyncio.sleep(0)
    assert admitted == []

    await driver.queue.put(HarnessEvent.normalized(
        type="turn.tool_completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={
            "tool_call_ref": "call-2", "label": "read_inbox",
            "outcome": "succeeded",
        },
        native_payload={
            "_puffo_internal": "tool_result",
            "tool_call_id": "call-2",
            "tool_name": "read_inbox",
            "arguments": {"cursor": "foreign"},
            "result": "[puffo:model-visible-read:receipt-2]",
            "is_error": False,
        },
    ))
    for _ in range(20):
        if admitted:
            break
        await asyncio.sleep(0)
    assert len(admitted) == 1
    assert admitted[0].planning_cycle_key == "read-page-foreign-session"
    assert admitted[0].tool_call_id == "call-2"

    await driver.queue.put(HarnessEvent(
        type="turn.completed", driver="fake",
        session_ref=SessionRef("native-session"), turn_ref=driver.turn,
        native_session_id="native-session", native_turn_id="native-turn",
        data={"outcome": "failed"},
    ))
    terminal = await manager.wait_terminal(started.turn_ref)
    assert terminal.data["outcome"] == "failed"
    await manager.close()

    class FailedManager:
        async def start_turn(self, input):
            return TurnStarted(TurnRef("logical-turn"), "native-turn")

        def events(self):
            async def iterate():
                yield HarnessEvent(
                    type="turn.completed", driver="fake",
                    session_ref=SessionRef("logical-session"),
                    turn_ref=TurnRef("logical-turn"),
                    data={"outcome": "cancelled"},
                )
            return iterate()

    with pytest.raises(RuntimeStateError, match="outcome cancelled"):
        await RuntimeManagerAdapter(FailedManager()).run_turn(
            SimpleNamespace(
                messages=[{"role": "user", "content": "notice"}],
                on_progress=None,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "outcome", "succeeds"),
    [
        ("turn.completed", "succeeded", True),
        ("turn.completed", "failed", False),
        ("turn.completed", "cancelled", False),
        ("turn.abandoned", "abandoned", False),
    ],
)
async def test_manager_adapter_only_accepts_succeeded_canonical_terminal(
    event_type, outcome, succeeds,
):
    class TerminalManager:
        async def start_turn(self, input):
            return TurnStarted(TurnRef("logical-turn"), "native-turn")

        def events(self):
            async def iterate():
                yield HarnessEvent(
                    type=event_type,
                    driver="fake",
                    session_ref=SessionRef("logical-session"),
                    turn_ref=TurnRef("logical-turn"),
                    data={"outcome": outcome},
                )
            return iterate()

    call = RuntimeManagerAdapter(TerminalManager()).run_turn(
        SimpleNamespace(
            messages=[{"role": "user", "content": "notice"}],
            on_progress=None,
        )
    )
    if succeeds:
        assert (await call).metadata["turn_ref"] == "logical-turn"
    else:
        with pytest.raises(RuntimeStateError, match=f"outcome {outcome}"):
            await call


@pytest.mark.asyncio
async def test_manager_adapter_submits_only_current_semantic_input():
    """A resumed provider session must not receive historical notice text."""
    historical_notice = (
        "<global_inbox_notice>historical-notice-sentinel</global_inbox_notice>"
    )
    current_notice = (
        "<global_inbox_notice>current-notice-sentinel</global_inbox_notice>"
    )
    captured = {}

    class Manager:
        async def start_turn(self, input):
            captured["content"] = input.content
            return TurnStarted(TurnRef("logical-turn"), "native-turn")

        def events(self):
            async def iterate():
                yield HarnessEvent(
                    type="turn.completed",
                    driver="fake",
                    session_ref=SessionRef("logical-session"),
                    turn_ref=TurnRef("logical-turn"),
                    data={"outcome": "succeeded"},
                )
            return iterate()

    await RuntimeManagerAdapter(Manager()).run_turn(SimpleNamespace(
        messages=[
            {"role": "user", "content": historical_notice},
            {"role": "assistant", "content": "already handled"},
            {"role": "user", "content": current_notice},
        ],
        on_progress=None,
    ))

    assert captured["content"] == current_notice
    assert "historical-notice-sentinel" not in captured["content"]
    assert captured["content"].count("current-notice-sentinel") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["codex", "claude-code"])
@pytest.mark.parametrize("terminal_outcome", ["succeeded", "failed"])
async def test_metadata_notice_through_real_manager_reads_paginated_exact_union(
    tmp_path, provider, terminal_outcome,
):
    holder = {}
    provider_inputs = []

    def on_codex_frame(frame):
        proc = holder["proc"]
        method = frame.get("method")
        if method == "initialize":
            proc.feed({"id": frame["id"], "result": {"server": "fake"}})
        elif method == "thread/start":
            proc.feed({
                "id": frame["id"],
                "result": {"thread": {"id": "native-session"}},
            })
        elif method == "turn/start":
            provider_inputs.append(frame["params"]["input"][0]["text"])
            proc.feed({
                "id": frame["id"],
                "result": {"turn": {"id": "native-turn"}},
            })

    if provider == "codex":
        proc = _FakeProcess(on_codex_frame)
        holder["proc"] = proc
        driver = CodexAppServerDriver(lambda _spec: proc)
    else:
        def replay_claude_frame(frame):
            if frame.get("type") == "user":
                provider_inputs.append(frame["message"]["content"][0]["text"])
                proc.feed({**frame, "isReplay": True})

        proc = _FakeProcess(replay_claude_frame)
        proc.feed({
            "type": "system",
            "subtype": "init",
            "session_id": "native-session",
            "slash_commands": [],
        })
        driver = ClaudeCodeCliDriver(
            lambda *_args: proc, replay_timeout=0.5
        )

    manager = RuntimeManager(
        driver,
        RuntimeSpec(str(tmp_path)),
        session_ref=SessionRef("logical-session"),
    )
    adapter = RuntimeManagerAdapter(manager)
    await adapter.warm("system")
    store = MessageStore(tmp_path / "messages.db", now_ms=lambda: 1_000)
    await store.open()
    for seq in (1, 2):
        await store.store_receipt(
            {
                "envelope_id": f"message-{seq}",
                "envelope_kind": "channel",
                "sender_slug": "alice",
                "channel_id": "channel",
                "space_id": "space",
                "content": f"secret-{seq}",
                "content_type": "text/plain",
                "sent_at": seq,
                "is_encrypted": True,
            },
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="test",
            received_at=1_000,
        )
        notice = await store.get_notice_state()
        assert notice.first_pending_deadline_ms == 4_000
    assert (await store.get_notice_state()).pending_count == 2

    agent = PuffoAgent(
        adapter,
        "system",
        str(tmp_path / "memory"),
        workspace_dir=str(tmp_path),
        agent_id="agent",
    )
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=agent.handle_global_inbox_turn,
        workspace=tmp_path,
    )
    expected_notice = await runtime.plan_pending()
    assert expected_notice is not None
    expected_provider_input = expected_notice.provider_input
    task = asyncio.create_task(runtime.process_once())

    for _ in range(500):
        if runtime.active.turn_id:
            break
        if task.done():
            await task
        await asyncio.sleep(0.001)
    assert runtime.active.provider_session_id == "native-session"
    assert runtime.active.message_ids == []
    assert provider_inputs == [expected_provider_input]
    assert all("<global_inbox_notice>" not in entry["content"] for entry in agent.log)

    cursor = ""
    admitted = []
    for page_number in (1, 2):
        arguments = {"cursor": cursor, "limit": 1}
        page = await runtime.read_inbox(
            cursor=cursor, limit=1, tool_arguments=arguments
        )
        assert len(page["messages"]) == 1
        receipt_marker = (
            f"[puffo:model-visible-read:{page['correlation_receipt']}]"
        )
        tool_id = f"tool-{page_number}"
        if provider == "codex":
            item = {
                "id": tool_id,
                "type": "mcpToolCall",
                "tool": "read_inbox",
                "arguments": arguments,
                "result": receipt_marker,
            }
            if page_number == 2:
                item.update({
                    "type": "dynamicToolCall",
                    "status": "completed",
                    "success": True,
                    "contentItems": None,
                })
                item.pop("result")
            proc.feed({
                "method": "item/completed",
                "params": {"item": item},
            })
        else:
            proc.feed({
                "type": "assistant",
                "message": {"content": [{
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "read_inbox",
                    "input": arguments,
                }]},
            })
            proc.feed({
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": receipt_marker,
                }]},
            })
        admitted.append(f"message-{page_number}")
        def admission_is_persisted():
            if runtime.active.message_ids != admitted:
                return False
            try:
                persisted = json.loads(runtime.current_turn_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            return persisted.get("message_ids") == admitted

        await _wait_until(admission_is_persisted)
        assert runtime.active.message_ids == admitted
        persisted = json.loads(runtime.current_turn_path.read_text())
        assert persisted["message_ids"] == admitted
        assert persisted["provider_session_id"] == "native-session"
        cursor = page["next_cursor"]

    if provider == "codex":
        proc.feed({
            "method": "turn/completed",
            "params": {"turn": {
                "status": (
                    "completed"
                    if terminal_outcome == "succeeded"
                    else "failed"
                )
            }},
        })
    else:
        proc.feed({
            "type": "result",
            "subtype": (
                "success" if terminal_outcome == "succeeded" else "error"
            ),
            "usage": {},
        })
    assert await asyncio.wait_for(task, timeout=1)
    states = [
        (await store.get_message_by_envelope(f"message-{seq}"))
        .processing_state
        for seq in (1, 2)
    ]
    expected = (
        [ProcessingState.PROCESSED, ProcessingState.PROCESSED]
        if terminal_outcome == "succeeded"
        else [ProcessingState.PENDING, ProcessingState.PENDING]
    )
    assert states == expected
    assert not runtime.current_turn_path.exists()
    await adapter.aclose()
    await store.close()


@pytest.mark.asyncio
async def test_codex_driver_fake_app_server_full_protocol_and_concurrency():
    holder = {}

    def on_frame(frame):
        proc = holder["proc"]
        method = frame.get("method")
        if method == "initialize":
            proc.feed({"id": frame["id"], "result": {"server": "fake"}})
        elif method == "thread/start":
            proc.feed({"id": frame["id"], "result": {"thread": {"id": "th_1"}}})
        elif method == "turn/start":
            assert frame["params"]["clientUserMessageId"] == "client-1"
            proc.feed({"id": frame["id"], "result": {"turn": {"id": "native-1"}}})
        elif method in {"turn/steer", "turn/interrupt"}:
            assert frame["params"].get("expectedTurnId", "native-1") == "native-1"
            proc.feed({"id": frame["id"], "result": {}})
        elif method == "thread/compact/start":
            proc.feed({"id": frame["id"], "result": {"id": "compact-1"}})

    proc = _FakeProcess(on_frame)
    holder["proc"] = proc
    driver = CodexAppServerDriver(lambda _spec: proc)
    opened = await driver.open(RuntimeSpec("/workspace", model="gpt"))
    assert opened.native_session_id == "th_1"
    methods = [json.loads(value).get("method") for value in proc.stdin.writes]
    assert methods[:3] == ["initialize", "initialized", "thread/start"]

    stream = driver.events()
    await _next_matching(stream, "session.opened")
    started = await driver.start_turn(
        TurnInput("hello", client_correlation_id="client-1")
    )
    proc.feed({"method": "turn/started", "params": {"turn": {"id": "native-1"}}})
    proc.feed({
        "method": "thread/tokenUsage/updated",
        "params": {"tokenUsage": {
            "totalTokens": 12, "modelContextWindow": 100,
        }},
    })
    proc.feed({
        "id": 900, "method": "item/commandExecution/requestApproval",
        "params": {"turnId": "native-1", "command": "SECRET"},
    })
    proc.feed({"method": "future/notification", "params": {"secret": "SECRET"}})

    await _next_matching(stream, "turn.started")
    await _next_matching(stream, "turn.context_updated")
    permission = await _next_matching(stream, "turn.permission_requested")
    warning = await _next_matching(stream, "runtime.warning")
    assert warning.data == {
        "code": "unknown_notification", "method": "future/notification",
    }

    await asyncio.gather(
        driver.steer_turn(started.turn_ref, TurnInput("more")),
        driver.cancel_turn(started.turn_ref),
        driver.resolve_permission(
            PermissionRef(permission.data["permission_ref"]),
            PermissionDecision.APPROVE,
        ),
    )
    permission_updated = await _next_matching(
        stream, "turn.permission_updated"
    )
    assert permission_updated.data["state"] == "approved"
    proc.feed({
        "method": "item/completed",
        "params": {"item": {
            "id": "tool-1", "type": "mcpToolCall",
            "tool": "read_inbox", "arguments": {"limit": 1},
            "result": "[puffo:model-visible-read:receipt-1]",
        }},
    })
    tool_completed = await _next_matching(stream, "turn.tool_completed")
    assert tool_completed.data == {
        "tool_call_ref": "tool-1",
        "label": "read_inbox",
        "outcome": "succeeded",
    }
    assert "arguments" not in tool_completed.data
    assert "result" not in tool_completed.data
    proc.feed({
        "method": "item/completed",
        "params": {"item": {
            "id": "tool-2", "type": "dynamicToolCall",
            "namespace": "mcp__puffo", "tool": "read_inbox",
            "status": "completed", "success": True,
            "arguments": {"target": "", "limit": 50},
            # Current live Codex may omit contentItems even though the model
            # received the tool output.
            "contentItems": None,
        }},
    })
    dynamic_completed = await _next_matching(stream, "turn.tool_completed")
    assert dynamic_completed.data == {
        "tool_call_ref": "tool-2",
        "label": "read_inbox",
        "outcome": "succeeded",
    }
    decoded = [json.loads(value) for value in proc.stdin.writes]
    assert any(value.get("method") == "turn/steer" for value in decoded)
    assert any(value.get("method") == "turn/interrupt" for value in decoded)
    assert any(value.get("id") == 900 and "result" in value for value in decoded)
    assert (await driver.context_status()).used_tokens == 12

    proc.feed({
        "method": "turn/completed",
        "params": {"turn": {"status": "interrupted"}},
    })
    await _next_matching(stream, "turn.completed")
    compact = await driver.compact(CompactRequest())
    assert compact.operation_ref == "compact-1"
    await driver.close()
    await driver.close()
    assert proc.terminated == 1


@pytest.mark.asyncio
async def test_codex_driver_resumes_with_native_session_id_after_handshake():
    holder = {}

    def on_frame(frame):
        proc = holder["proc"]
        if frame.get("method") == "initialize":
            proc.feed({"id": frame["id"], "result": {}})
        elif frame.get("method") == "thread/resume":
            assert frame["params"] == {"threadId": "native-thread"}
            proc.feed({
                "id": frame["id"],
                "result": {"thread": {"id": "native-thread"}},
            })

    proc = _FakeProcess(on_frame)
    holder["proc"] = proc
    driver = CodexAppServerDriver(lambda _spec: proc)
    opened = await driver.open(
        RuntimeSpec("/workspace"), SessionRef("native-thread")
    )
    assert opened.resumed and opened.native_session_id == "native-thread"
    assert [
        json.loads(value).get("method") for value in proc.stdin.writes
    ] == ["initialize", "initialized", "thread/resume"]
    await driver.close()


@pytest.mark.asyncio
async def test_claude_driver_exact_replay_trailing_records_and_unsupported_zero_writes():
    captured_args = []
    holder = {}

    def factory(args, _spec):
        captured_args.extend(args)

        def on_frame(frame):
            proc = holder["proc"]
            if frame.get("uuid"):
                proc.feed({
                    **frame, "isReplay": True,
                    "message": {"role": "user", "content": [
                        {"type": "text", "text": "hello\nworld"}
                    ]},
                })
                proc.feed({
                    "type": "assistant",
                    "message": {"content": [
                        {"type": "text", "text": "visible"},
                        {
                            "type": "tool_use", "id": "tool-claude",
                            "name": "read_inbox", "input": {"limit": 1},
                        },
                    ]},
                })
                proc.feed({
                    "type": "user", "message": {"content": [{
                        "type": "tool_result", "tool_use_id": "tool-claude",
                        "content": "[puffo:model-visible-read:receipt-claude]",
                    }]},
                })
                proc.feed({"type": "result", "subtype": "success", "usage": {}})
                proc.feed({"type": "rate_limit_event", "secret": "not-public"})

        proc = _FakeProcess(on_frame)
        holder["proc"] = proc
        proc.feed({
            "type": "system", "subtype": "init", "session_id": "claude-1",
            "slash_commands": ["/compact"],
        })
        return proc

    driver = ClaudeCodeCliDriver(factory, replay_timeout=1)
    opened = await driver.open(RuntimeSpec("/workspace"))
    assert "--replay-user-messages" in captured_args
    assert opened.capabilities.compact == "session_command"
    stream = driver.events()
    await _next_matching(stream, "session.opened")
    started = await driver.start_turn(TurnInput(
        "hello\r\nworld", client_correlation_id="replay-1"
    ))
    assert started.accepted and started.delivery == "replay_acknowledged"
    assert started.replay_id == "replay-1"
    await _next_matching(stream, "turn.started")
    await _next_matching(stream, "turn.assistant_delta")
    await _next_matching(stream, "turn.tool_started")
    tool_completed = await _next_matching(stream, "turn.tool_completed")
    assert tool_completed.data == {
        "tool_call_ref": "tool-claude",
        "label": "read_inbox",
        "outcome": "succeeded",
    }
    assert "arguments" not in tool_completed.data
    assert "result" not in tool_completed.data
    await _next_matching(stream, "turn.completed")
    trailing = await _next_matching(stream, "session.updated")
    assert trailing.turn_ref is None

    before = len(holder["proc"].stdin.writes)
    assert isinstance(
        await driver.steer_turn(started.turn_ref, TurnInput("x")),
        UnsupportedCapability,
    )
    assert isinstance(
        await driver.cancel_turn(started.turn_ref), UnsupportedCapability
    )
    assert isinstance(await driver.context_status(), UnsupportedCapability)
    assert isinstance(
        await driver.resolve_permission(
            PermissionRef("p"), PermissionDecision.DENY
        ),
        UnsupportedCapability,
    )
    assert len(holder["proc"].stdin.writes) == before
    assert (await driver.compact(CompactRequest("now"))).accepted
    await driver.close()


@pytest.mark.asyncio
async def test_claude_driver_resume_flag_maps_to_resumed_system_init():
    captured = []

    def factory(args, _spec):
        captured.extend(args)
        proc = _FakeProcess()
        proc.feed({
            "type": "system", "subtype": "init",
            "session_id": "native-claude-session",
        })
        return proc

    driver = ClaudeCodeCliDriver(factory, replay_timeout=1)
    opened = await driver.open(
        RuntimeSpec("/workspace"), SessionRef("native-claude-session")
    )
    resume_index = captured.index("--resume")
    assert captured[resume_index + 1] == "native-claude-session"
    assert opened.resumed
    await driver.close()


@pytest.mark.asyncio
async def test_claude_driver_pre_replay_process_loss_is_ambiguous():
    holder = {}

    def factory(_args, _spec):
        def on_frame(_frame):
            holder["proc"].stdout.feed_eof()

        proc = _FakeProcess(on_frame)
        holder["proc"] = proc
        proc.feed({
            "type": "system", "subtype": "init", "session_id": "claude-loss",
        })
        return proc

    driver = ClaudeCodeCliDriver(factory, replay_timeout=1)
    await driver.open(RuntimeSpec("/workspace"))
    receipt = await driver.start_turn(TurnInput("accepted maybe"))
    assert not receipt.accepted
    assert receipt.delivery == "ambiguous_at_least_once"
    await driver.close()


@pytest.mark.asyncio
async def test_control_permission_stale_and_cross_agent_references_fail_closed():
    calls = []

    class Manager:
        session_ref = SessionRef("session_a")

        async def resolve_permission(self, turn, permission, decision):
            calls.append((turn, permission, decision))
            if turn != TurnRef("turn_a") or permission != PermissionRef("perm_a"):
                raise RuntimeStateError("stale")
            return SimpleNamespace(accepted=True)

    manager = Manager()
    register_runtime_manager("agent_permission", manager)
    try:
        params = {
            "session_ref": "session_a", "turn_ref": "turn_a",
            "permission_ref": "perm_a", "decision": "approved",
        }
        delivered = await execute_command(
            "runtime.resolve_permission", "agent_permission", params,
            command_id="permission-command",
        )
        assert delivered == {
            "ok": True, "delivered": True, "completed": False,
        }
        stale = await execute_command(
            "runtime.resolve_permission", "agent_permission",
            {**params, "turn_ref": "old"}, command_id="permission-stale",
        )
        assert stale["error_code"] == "stale_runtime_reference"
        cross = await execute_command(
            "runtime.resolve_permission", "different-agent", params,
            # Idempotency keys are scoped by Agent and operation; replaying a
            # valid key against a different Agent must not reuse its result.
            command_id="permission-command",
        )
        assert cross["error_code"] == "runtime_unavailable"
    finally:
        unregister_runtime_manager("agent_permission", manager)
