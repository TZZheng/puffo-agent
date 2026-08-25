import asyncio
from types import SimpleNamespace

import pytest

from acp import PROTOCOL_VERSION
from acp.connection import StreamDirection, StreamEvent
from acp.exceptions import RequestError
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)

from puffo_agent.agent.harness import build_driver
from puffo_agent.agent.harness.acp_driver import AcpDriver
from puffo_agent.agent.harness.driver import (
    HarnessEventType,
    McpServerSpec,
    PermissionDecision,
    PermissionRef,
    RuntimeLifecycle,
    RuntimeSpec,
    SessionRef,
    TurnInput,
)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = object()
        self.stdout = object()
        self.stderr = None
        self.returncode = None
        self._exit = asyncio.get_running_loop().create_future()

    async def wait(self) -> int:
        return await self._exit

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        if not self._exit.done():
            self._exit.set_result(returncode)

    def terminate(self) -> None:
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)


class _FakeConnection:
    def __init__(self, client, observer, *, can_load=True) -> None:
        self.client = client
        self.observer = observer
        self.can_load = can_load
        self.prompt_result = asyncio.get_running_loop().create_future()
        self.calls = []
        self.closed = False

    async def initialize(self, **kwargs):
        self.calls.append(("initialize", kwargs))
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(load_session=self.can_load),
        )

    async def new_session(self, **kwargs):
        self.calls.append(("new_session", kwargs))
        return NewSessionResponse(session_id="acp_session")

    async def load_session(self, **kwargs):
        self.calls.append(("load_session", kwargs))
        return SimpleNamespace()

    async def prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        await self.observer(StreamEvent(
            StreamDirection.OUTGOING,
            {"jsonrpc": "2.0", "id": 3, "method": "session/prompt"},
        ))
        return await self.prompt_result

    async def cancel(self, **kwargs):
        self.calls.append(("cancel", kwargs))

    async def close(self):
        self.closed = True


class _Harness:
    def __init__(self, *, can_load=True) -> None:
        self.proc = _FakeProcess()
        self.conn = None
        self.client = None
        self.can_load = can_load

    def process_factory(self, command, spec):
        self.command = command
        self.spec = spec
        return self.proc

    def connection_factory(self, client, _stdin, _stdout, **kwargs):
        self.client = client
        self.conn = _FakeConnection(
            client,
            kwargs["observers"][0],
            can_load=self.can_load,
        )
        return self.conn


async def _collect_through(stream, type_):
    events = []
    async for event in stream:
        events.append(event)
        if event.type is type_:
            return events
    raise AssertionError(f"stream ended before {type_}")


@pytest.mark.asyncio
async def test_acp_open_negotiates_v1_and_loads_or_creates_session():
    harness = _Harness()
    driver = AcpDriver(
        harness.process_factory,
        connection_factory=harness.connection_factory,
    )
    opened = await driver.open(RuntimeSpec("/workspace", executable="agent"))

    assert opened.native_session_id == "acp_session"
    assert opened.capabilities.session_resume is True
    assert opened.capabilities.lifecycle is RuntimeLifecycle.PERSISTENT_CHILD
    assert opened.diagnostics.schema_source == (
        "agent-client-protocol==0.10.1/protocol-v1"
    )
    assert harness.command == ("agent",)
    await driver.close()

    resumed_harness = _Harness()
    resumed = AcpDriver(
        resumed_harness.process_factory,
        connection_factory=resumed_harness.connection_factory,
    )
    result = await resumed.open(
        RuntimeSpec("/workspace", executable="agent", launch_args=("acp",)),
        SessionRef("existing"),
    )
    assert result.resumed is True
    assert resumed_harness.conn.calls[-1][0] == "load_session"
    await resumed.close()


@pytest.mark.asyncio
async def test_acp_projects_stdio_mcp_servers_at_session_creation():
    harness = _Harness()
    driver = AcpDriver(
        harness.process_factory,
        connection_factory=harness.connection_factory,
    )
    await driver.open(RuntimeSpec(
        "/workspace",
        executable="agent",
        mcp_servers=(McpServerSpec(
            name="puffo",
            command="/opt/python",
            args=("-m", "puffo_agent.mcp.puffo_core_server"),
            environment={"PUFFO_CORE_SLUG": "bot-0001"},
        ),),
    ))

    call, kwargs = harness.conn.calls[-1]
    assert call == "new_session"
    assert len(kwargs["mcp_servers"]) == 1
    server = kwargs["mcp_servers"][0]
    assert server.name == "puffo"
    assert server.command == "/opt/python"
    assert server.args == ["-m", "puffo_agent.mcp.puffo_core_server"]
    assert [(item.name, item.value) for item in server.env] == [
        ("PUFFO_CORE_SLUG", "bot-0001"),
    ]
    await driver.close()


@pytest.mark.asyncio
async def test_prompt_admission_updates_and_response_form_one_terminal():
    harness = _Harness()
    driver = AcpDriver(
        harness.process_factory,
        connection_factory=harness.connection_factory,
    )
    await driver.open(RuntimeSpec("/workspace", executable="agent"))
    stream = driver.events()
    started = await driver.start_turn(TurnInput("hello"))
    assert started.delivery == "jsonrpc_request_written"

    await harness.client.session_update(
        "acp_session",
        AgentMessageChunk(
            session_update="agent_message_chunk",
            message_id="message_1",
            content=TextContentBlock(type="text", text="answer"),
        ),
    )
    await harness.client.session_update(
        "acp_session",
        ToolCallStart(
            session_update="tool_call",
            tool_call_id="tool_1",
            title="read_inbox",
            status="in_progress",
        ),
    )
    await harness.client.session_update(
        "acp_session",
        ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tool_1",
            title="read_inbox",
            status="completed",
            raw_output="never public",
        ),
    )
    await harness.client.session_update(
        "acp_session",
        UsageUpdate(session_update="usage_update", used=10, size=100),
    )
    harness.conn.prompt_result.set_result(PromptResponse(stop_reason="end_turn"))

    events = await asyncio.wait_for(
        _collect_through(stream, HarnessEventType.TURN_COMPLETED), timeout=1
    )
    assert [event.type for event in events].count(
        HarnessEventType.TURN_COMPLETED
    ) == 1
    delta = next(
        event for event in events
        if event.type is HarnessEventType.ASSISTANT_DELTA
    )
    assert delta.data == {"block_id": "message_1", "delta": "answer"}
    tool = next(
        event for event in events
        if event.type is HarnessEventType.TOOL_COMPLETED
    )
    assert "never public" not in repr(tool.data)
    await driver.close()


@pytest.mark.asyncio
async def test_permission_request_waits_for_typed_driver_resolution():
    harness = _Harness()
    driver = AcpDriver(
        harness.process_factory,
        connection_factory=harness.connection_factory,
    )
    await driver.open(RuntimeSpec("/workspace", executable="agent"))
    stream = driver.events()
    await driver.start_turn(TurnInput("hello"))
    permission = asyncio.create_task(harness.client.request_permission(
        [
            PermissionOption(
                option_id="deny",
                name="Deny",
                kind="reject_once",
            ),
            PermissionOption(
                option_id="allow",
                name="Allow once",
                kind="allow_once",
            ),
        ],
        "acp_session",
        ToolCallStart(
            session_update="tool_call",
            tool_call_id="tool_1",
            title="shell",
        ),
    ))
    events = await asyncio.wait_for(
        _collect_through(stream, HarnessEventType.PERMISSION_REQUESTED),
        timeout=1,
    )
    request = events[-1]
    ref = PermissionRef(str(request.data["permission_ref"]))
    receipt = await driver.resolve_permission(
        ref, PermissionDecision.APPROVE
    )
    assert receipt.accepted is True
    response = await asyncio.wait_for(permission, timeout=1)
    assert response.outcome.outcome == "selected"
    assert response.outcome.option_id == "allow"
    harness.conn.prompt_result.set_result(PromptResponse(stop_reason="cancelled"))
    await driver.close()


@pytest.mark.asyncio
async def test_unknown_extension_is_explicitly_unsupported_and_observable():
    harness = _Harness()
    driver = AcpDriver(
        harness.process_factory,
        connection_factory=harness.connection_factory,
    )
    await driver.open(RuntimeSpec("/workspace", executable="agent"))
    stream = driver.events()
    with pytest.raises(RequestError) as exc_info:
        await harness.client.ext_method("vendor/steer", {})
    assert exc_info.value.code == -32601
    events = await asyncio.wait_for(
        _collect_through(stream, HarnessEventType.RUNTIME_WARNING), timeout=1
    )
    assert events[-1].data == {
        "code": "unsupported_extension",
        "method": "_vendor/steer",
    }
    await driver.close()


def test_acp_factory_is_closed_and_explicit():
    assert isinstance(build_driver("acp"), AcpDriver)
