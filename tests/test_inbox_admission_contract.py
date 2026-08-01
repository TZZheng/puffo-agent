from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer
from mcp.server.fastmcp import FastMCP

from puffo_agent.agent.adapters.base import TurnContext
from puffo_agent.agent.global_inbox_runtime import (
    ActiveBoundaryAdapter,
    GlobalInboxRuntime,
)
from puffo_agent.agent.harness.driver import (
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
)
from puffo_agent.agent.message_store import (
    MessageStore,
    ProcessingState,
    ReceiptDisposition,
)
from puffo_agent.mcp._host_mcp import PuffoRpcClient
from puffo_agent.mcp.puffo_core_tools import (
    PuffoCoreToolsConfig,
    register_core_tools,
)
from puffo_agent.portal import rpc_service
from puffo_agent.portal.host_mcp_handler import HostMcpContext


class ContractDriver(HarnessDriver):
    """Deterministic provider event source; all admission remains real."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[HarnessEvent] = asyncio.Queue()
        self.turn = TurnRef("driver-turn")

    async def open(self, spec, resume=None):
        return RuntimeOpened(
            RuntimeRef("runtime"),
            SessionRef("native-session"),
            "native-session",
            False,
            SimpleNamespace(),
            SimpleNamespace(),
        )

    async def start_turn(self, input: TurnInput):
        return TurnStarted(self.turn, "native-turn")

    async def steer_turn(self, turn, input):
        return UnsupportedCapability("steer")

    async def cancel_turn(self, turn):
        return UnsupportedCapability("cancel")

    async def context_status(self):
        return UnsupportedCapability("context_status")

    async def compact(self, request: CompactRequest):
        return UnsupportedCapability("compact")

    async def resolve_permission(self, request: PermissionRef, decision: PermissionDecision):
        return UnsupportedCapability("permission")

    def events(self):
        async def iterate():
            while True:
                yield await self.queue.get()

        return iterate()

    async def close(self):
        return None


async def _wait_until(predicate, *, steps: int = 100) -> None:
    for _ in range(steps):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def _tool_text(result) -> str:
    blocks = result[0] if isinstance(result, tuple) else result
    if not isinstance(blocks, list):
        return str(blocks)
    return "".join(
        getattr(block, "text", str(block))
        for block in blocks
    )


async def _seed(store: MessageStore, *, envelope_id: str, channel_id: str, seq: int, text: str):
    await store.store_receipt(
        {
            "envelope_id": envelope_id,
            "envelope_kind": "channel",
            "sender_slug": "peer-0001",
            "space_id": "sp_1",
            "channel_id": channel_id,
            "content": text,
            "content_type": "text/plain",
            "sent_at": seq,
            "is_encrypted": True,
        },
        server_seq=seq,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="contract test",
        received_at=seq,
    )


class BoundaryHarness:
    def __init__(
        self,
        *,
        store: MessageStore,
        driver: ContractDriver,
        manager: RuntimeManager,
        adapter: RuntimeManagerAdapter,
        runtime: GlobalInboxRuntime,
        rpc: PuffoRpcClient,
        mcp: FastMCP,
        turn_task: asyncio.Task,
        staged_responses: list[dict],
    ):
        self.store = store
        self.driver = driver
        self.manager = manager
        self.adapter = adapter
        self.runtime = runtime
        self.rpc = rpc
        self.mcp = mcp
        self.turn_task = turn_task
        self.staged_responses = staged_responses

    @property
    def session_id(self) -> str:
        return "native-session"

    @property
    def native_turn_id(self) -> str:
        return "native-turn"

    @property
    def active_turn_id(self) -> str:
        return self.runtime.active.turn_id

    async def active_boundary(self, channel_id: str) -> int | None:
        return await ActiveBoundaryAdapter(
            self.store, self.runtime.active
        ).get_active_turn_through_seq("sp_1", channel_id)

    async def emit_tool_result(
        self,
        *,
        tool_name: str,
        arguments: dict,
        result,
        native_session_id: str = "native-session",
        native_turn_id: str = "native-turn",
        is_error: bool = False,
    ) -> None:
        await self.driver.queue.put(HarnessEvent.normalized(
            type="turn.tool_completed",
            driver="contract",
            session_ref=SessionRef(native_session_id),
            turn_ref=self.driver.turn,
            native_session_id=native_session_id,
            native_turn_id=native_turn_id,
            data={
                "tool_call_ref": "contract-tool-call",
                "label": tool_name,
                "outcome": "failed" if is_error else "succeeded",
            },
            native_payload={
                "_puffo_internal": "tool_result",
                "tool_call_id": "contract-tool-call",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "is_error": is_error,
            },
        ))

    async def finish(self, outcome: str = "succeeded") -> None:
        await self.driver.queue.put(HarnessEvent(
            type="turn.completed",
            driver="contract",
            session_ref=SessionRef("native-session"),
            turn_ref=self.driver.turn,
            native_session_id="native-session",
            native_turn_id="native-turn",
            data={"outcome": outcome},
        ))
        if outcome == "succeeded":
            await self.turn_task
        else:
            with pytest.raises(RuntimeStateError):
                await self.turn_task


@pytest_asyncio.fixture
async def boundary_harness(tmp_path: Path):
    store = MessageStore(tmp_path / "messages.db")
    await store.open()
    await _seed(
        store,
        envelope_id="history-peer",
        channel_id="ch-history",
        seq=2,
        text="history peer body",
    )
    await _seed(
        store,
        envelope_id="inbox-peer",
        channel_id="ch-inbox",
        seq=3,
        text="inbox peer body",
    )

    driver = ContractDriver()
    manager = RuntimeManager(driver, RuntimeSpec(str(tmp_path)))
    adapter = RuntimeManagerAdapter(manager)
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )

    turn_task = asyncio.create_task(adapter.run_turn(TurnContext(
        system_prompt="contract",
        messages=[{"role": "user", "content": "active turn"}],
    )))
    await _wait_until(lambda: manager.active_turn_ref is not None)
    runtime.active.turn_id = str(manager.active_turn_ref)
    runtime.active.provider_session_id = adapter.get_provider_session_id()
    runtime.active.provider_turn_id = manager.native_turn_id

    ctx = HostMcpContext(
        agent_id="agent-contract",
        slug="agent-contract",
        operator_slug="operator",
        host_home=tmp_path,
        agent_home=tmp_path,
        harness="claude-code",
        keystore=None,
        http_client=None,
        message_client=SimpleNamespace(global_runtime=runtime),
    )
    rpc_service.set_rpc_resolver(
        lambda agent_id: ctx if agent_id == "agent-contract" else None
    )
    server = TestServer(rpc_service.build_app(
        rpc_service.RpcServiceConfig(enabled=True, port=0)
    ))
    await server.start_server()
    rpc = PuffoRpcClient(str(server.make_url("/")), "agent-contract")
    staged_responses: list[dict] = []
    real_stage = rpc.stage_model_visible_read

    async def record_stage(**kwargs):
        result = await real_stage(**kwargs)
        staged_responses.append(result)
        return result

    rpc.stage_model_visible_read = record_stage
    cfg = PuffoCoreToolsConfig(
        slug="agent-contract",
        device_id="device-contract",
        keystore=None,
        http_client=SimpleNamespace(keyless=False),
        data_client=store,
        agent_id="agent-contract",
        rpc_client=rpc,
    )
    mcp = FastMCP("contract")
    register_core_tools(mcp, cfg)

    harness = BoundaryHarness(
        store=store,
        driver=driver,
        manager=manager,
        adapter=adapter,
        runtime=runtime,
        rpc=rpc,
        mcp=mcp,
        turn_task=turn_task,
        staged_responses=staged_responses,
    )
    try:
        yield harness
    finally:
        if not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
        await rpc.close()
        await server.close()
        rpc_service.set_rpc_resolver(None)
        await manager.close()
        await store.close()


@pytest.mark.asyncio
async def test_real_rpc_history_and_inbox_admission_wait_for_provider_completion(
    boundary_harness: BoundaryHarness,
):
    h = boundary_harness

    history_call = await h.mcp.call_tool(
        "get_channel_history", {"channel": "ch-history", "limit": 50}
    )
    history_text = _tool_text(history_call)
    assert "history peer body" in history_text
    assert re.search(r"\[puffo:model-visible-read:[^]]+\]", history_text)
    assert h.staged_responses[-1]["state"] == "staged"

    history_row = await h.store.get_message_by_envelope("history-peer")
    assert history_row is not None
    assert history_row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-history") is None

    history_receipt = re.search(
        r"(\[puffo:model-visible-read:[^]]+\])", history_text
    ).group(1)
    await h.emit_tool_result(
        tool_name="get_channel_history",
        arguments={"channel": "ch-history", "limit": 50},
        result=history_text,
    )
    await _wait_until(
        lambda: h.runtime.active.through_by_channel.get(
            ("sp_1", "ch-history")
        ) == 2
    )
    assert history_receipt in history_text
    assert await h.active_boundary("ch-history") == 2

    inbox_call = await h.mcp.call_tool(
        "read_inbox",
        {"target": "channel:sp_1:ch-inbox", "limit": 1},
    )
    inbox_page = inbox_call[1]
    assert "inbox peer body" in inbox_page["messages"][0]
    assert inbox_page["admission_receipt"].startswith(
        "[puffo:model-visible-read:"
    )
    inbox_row = await h.store.get_message_by_envelope("inbox-peer")
    assert inbox_row is not None
    assert inbox_row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-inbox") is None

    await h.emit_tool_result(
        tool_name="read_inbox",
        arguments={"target": "channel:sp_1:ch-inbox", "limit": 1},
        result=json.dumps(inbox_page),
    )
    await _wait_until(
        lambda: h.runtime.active.through_by_channel.get(
            ("sp_1", "ch-inbox")
        ) == 3
    )
    inbox_row = await h.store.get_message_by_envelope("inbox-peer")
    assert inbox_row.processing_state is ProcessingState.IN_TURN
    assert await h.active_boundary("ch-inbox") == 3

    await h.finish()


@pytest.mark.asyncio
async def test_rpc_response_without_provider_completion_keeps_pending_and_unseen(
    boundary_harness: BoundaryHarness,
):
    h = boundary_harness
    history_call = await h.mcp.call_tool(
        "get_channel_history", {"channel": "ch-history", "limit": 50}
    )
    history_text = _tool_text(history_call)
    assert "history peer body" in history_text
    assert h.staged_responses[-1]["state"] == "staged"

    row = await h.store.get_message_by_envelope("history-peer")
    assert row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-history") is None
    await h.finish("failed")
    row = await h.store.get_message_by_envelope("history-peer")
    assert row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-history") is None


@pytest.mark.asyncio
async def test_real_provider_correlation_rejects_empty_failed_and_mismatched_results(
    boundary_harness: BoundaryHarness,
):
    h = boundary_harness
    history_call = await h.mcp.call_tool(
        "get_channel_history", {"channel": "ch-history", "limit": 50}
    )
    history_text = _tool_text(history_call)
    receipt = re.search(
        r"(\[puffo:model-visible-read:[^]]+\])", history_text
    ).group(1)
    arguments = {"channel": "ch-history", "limit": 50}
    cases = [
        ("get_thread_history", arguments, history_text, "native-session", "native-turn", False),
        ("get_channel_history", {"channel": "wrong"}, history_text, "native-session", "native-turn", False),
        ("get_channel_history", arguments, "content without receipt", "native-session", "native-turn", False),
        ("get_channel_history", arguments, history_text, "native-session", "wrong-turn", False),
        ("get_channel_history", arguments, history_text, "wrong-session", "native-turn", False),
        ("get_channel_history", arguments, history_text, "native-session", "native-turn", True),
    ]
    for tool_name, tool_args, result, session_id, turn_id, is_error in cases:
        await h.emit_tool_result(
            tool_name=tool_name,
            arguments=tool_args,
            result=result,
            native_session_id=session_id,
            native_turn_id=turn_id,
            is_error=is_error,
        )
        await asyncio.sleep(0)
        assert await h.active_boundary("ch-history") is None
        row = await h.store.get_message_by_envelope("history-peer")
        assert row.processing_state is ProcessingState.PENDING

    await h.emit_tool_result(
        tool_name="get_channel_history",
        arguments=arguments,
        result=history_text.replace(receipt, receipt),
    )
    await _wait_until(
        lambda: h.runtime.active.through_by_channel.get(
            ("sp_1", "ch-history")
        ) == 2
    )
    assert await h.active_boundary("ch-history") == 2
    await h.finish()


@pytest.mark.asyncio
async def test_empty_or_failed_rpc_read_has_no_admission_receipt_or_lifecycle_change(
    boundary_harness: BoundaryHarness,
):
    h = boundary_harness
    empty = await h.mcp.call_tool(
        "read_inbox",
        {"target": "channel:sp_1:does-not-exist", "limit": 1},
    )
    empty_page = empty[1]
    assert empty_page["messages"] == []
    assert "admission_receipt" not in empty_page
    assert await h.active_boundary("ch-inbox") is None

    await h.emit_tool_result(
        tool_name="send_message",
        arguments={"channel": "ch-inbox", "text": "content-free"},
        result={"state": "sent"},
    )
    await asyncio.sleep(0)
    row = await h.store.get_message_by_envelope("inbox-peer")
    assert row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-inbox") is None

    with pytest.raises(RuntimeError):
        await h.rpc.stage_model_visible_read(
            space_id="sp_1",
            channel_id="ch-inbox",
            through_seq=99,
            through_envelope_id="missing",
            tool_name="get_channel_history",
            tool_arguments={"channel": "ch-inbox"},
        )
    row = await h.store.get_message_by_envelope("inbox-peer")
    assert row.processing_state is ProcessingState.PENDING
    assert await h.active_boundary("ch-inbox") is None
    await h.finish()
