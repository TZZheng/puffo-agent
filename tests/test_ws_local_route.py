"""``serve_attached`` end to end: real session + bridge + endpoint, with
only the client / reporter / agent-config / auth faked.

Drives a full attach: handshake → Connected(profile) → a consumer batch
becomes a bundle → tool acks (consumer advances) → tool replies (relayed
to the client) → disconnect tears everything down.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from puffo_agent.portal.ws_local import route as route_mod
from puffo_agent.portal.ws_local.auth import AuthedAgent
from puffo_agent.portal.ws_local.bridge import WsLocalBridge
from puffo_agent.portal.ws_local.bundles import Bundle
from puffo_agent.portal.ws_local.hub import AttachPoint, WsLocalHub
from puffo_agent.portal.ws_local.route import serve_attached


class FakeTransport:
    def __init__(self) -> None:
        self._inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self):
        return await self._inbound.get()

    async def close(self) -> None:
        self.closed = True
        self._inbound.put_nowait(None)

    def feed(self, frame: dict) -> None:
        self._inbound.put_nowait(json.dumps(frame))

    def by_type(self, t: str) -> list:
        return [f for f in self.sent if f["type"] == t]


@pytest.mark.asyncio
async def test_ws_local_adapter_requires_exact_read_admission_correlation():
    adapter = route_mod._WsLocalContextAdapter()
    seen = []

    async def callback(event):
        seen.append(event)

    adapter.register_continuation_callback(
        callback,
        "inbox-read-key",
        tool_names=("read_inbox",),
        tool_arguments={"limit": 7},
        correlation_receipt="read-receipt",
    )
    with pytest.raises(RuntimeError, match="correlation failed"):
        await adapter.emit_admission(
            turn_id="turn-1", correlation_key="wrong-receipt"
        )
    assert seen == []
    # WS frames keep their existing opaque-receipt wire shape, but the
    # private bridge handoff must supply the real tool and normalized semantic
    # arguments before a continuation can be admitted.
    with pytest.raises(RuntimeError, match="tool correlation failed"):
        await adapter.emit_admission(
            turn_id="turn-1", correlation_key="read-receipt",
            tool_name="get_thread_history", tool_arguments={"limit": 7},
        )
    with pytest.raises(RuntimeError, match="argument correlation failed"):
        await adapter.emit_admission(
            turn_id="turn-1", correlation_key="read-receipt",
            tool_name="read_inbox", tool_arguments={"limit": 8},
        )
    assert seen == []
    await adapter.emit_admission(
        turn_id="turn-1", correlation_key="read-receipt",
        tool_name="read_inbox", tool_arguments={"limit": 7},
    )
    assert len(seen) == 1
    assert seen[0].planning_cycle_key == "inbox-read-key"
    assert seen[0].provider_session_id == "ws-local:turn-1"
    with pytest.raises(RuntimeError, match="correlation failed"):
        await adapter.emit_admission(
            turn_id="turn-1", correlation_key="read-receipt"
        )


@pytest.mark.asyncio
async def test_ws_local_bridge_records_actual_tool_before_opaque_admission():
    adapter = route_mod._WsLocalContextAdapter()
    seen = []

    async def callback(event):
        seen.append(event)

    adapter.register_continuation_callback(
        callback,
        "held-key",
        correlation_receipt="held-receipt",
        tool_names=("send_message",),
        tool_arguments={"channel": "ch-1", "text": "draft"},
    )
    runtime = type("Runtime", (), {"adapter": adapter})()
    bridge = WsLocalBridge(runtime=runtime)
    bundle = Bundle("bundle", "root", {}, turn_id="turn-1")
    runtime._ws_local_planned = type("Planned", (), {
        "turn_id": "turn-1", "planning_cycle_key": "initial-key",
    })()

    # The public Admitted frame contains only the receipt.  Without the
    # private observed result, it cannot consume the continuation.
    with pytest.raises(RuntimeError, match="tool result unavailable"):
        await bridge.on_admitted(
            bundle, type("Frame", (), {"correlation_key": "held-receipt"})(),
        )

    await bridge.on_tool_result(
        "get_post", {"channel": "ch-1", "text": "draft"},
        {"tool_result_admission": "[puffo:model-visible-read:held-receipt]"},
    )
    with pytest.raises(RuntimeError, match="tool correlation failed"):
        await bridge.on_admitted(
            bundle, type("Frame", (), {"correlation_key": "held-receipt"})(),
        )
    assert seen == []

    await bridge.on_tool_result(
        "send_message", {"channel": "ch-1", "text": "other"},
        {"tool_result_admission": "[puffo:model-visible-read:held-receipt]"},
    )
    with pytest.raises(RuntimeError, match="argument correlation failed"):
        await bridge.on_admitted(
            bundle, type("Frame", (), {"correlation_key": "held-receipt"})(),
        )
    assert seen == []

    await bridge.on_tool_result(
        "send_message", {"channel": "ch-1", "text": "draft"},
        {"tool_result_admission": "[puffo:model-visible-read:held-receipt]"},
    )
    await bridge.on_admitted(
        bundle, type("Frame", (), {"correlation_key": "held-receipt"})(),
    )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_ws_local_history_callback_updates_visible_registry_only_after_admission(
    tmp_path,
):
    from puffo_agent.agent.global_inbox_runtime import GlobalInboxRuntime
    from puffo_agent.agent.message_store import (
        MessageStore,
        ProcessingState,
        ReceiptDisposition,
    )

    store = MessageStore(tmp_path / "messages.db")
    await store.open()
    await store.store_receipt(
        {
            "envelope_id": "ws-history",
            "envelope_kind": "channel",
            "sender_slug": "alice",
            "space_id": "sp-1",
            "channel_id": "ch-1",
            "content": "history",
            "content_type": "text/plain",
            "sent_at": 4,
        },
        server_seq=4,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    adapter = route_mod._WsLocalContextAdapter()
    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "ws-runtime-turn"
    runtime.active.provider_session_id = "ws-local:turn-1"
    runtime.active.provider_turn_id = "turn-1"
    staged = await runtime.stage_model_visible_read(
        space_id="sp-1", channel_id="ch-1", through_seq=4,
        through_envelope_id="ws-history", tool_name="get_post",
        tool_arguments={"channel": "ch-1", "seq": 4},
        visible_message_ids=["ws-history"],
    )
    receipt = staged["correlation_receipt"]

    # A failed/no-admission result and a wrong receipt leave the continuation
    # unconsumed and all state untouched.
    assert runtime.active.visible_message_ids == []
    with pytest.raises(RuntimeError, match="correlation failed"):
        await adapter.emit_admission(turn_id="turn-1", correlation_key="wrong")
    row = await store.get_message_by_envelope("ws-history")
    assert runtime.active.visible_message_ids == []
    assert runtime.active.message_ids == []
    assert runtime.active.through_by_channel == {}
    assert row.processing_state == ProcessingState.PENDING.value

    await adapter.emit_admission(
        turn_id="turn-1", correlation_key=receipt,
        tool_name="get_post", tool_arguments={"channel": "ch-1", "seq": 4},
    )
    assert runtime.active.visible_message_ids == ["ws-history"]
    assert runtime.active.message_ids == []
    assert runtime.active.through_by_channel == {("sp-1", "ch-1"): 4}
    assert (await store.get_message_by_envelope("ws-history")).processing_state == (
        ProcessingState.PENDING.value
    )
    with pytest.raises(RuntimeError, match="correlation failed"):
        await adapter.emit_admission(turn_id="turn-1", correlation_key=receipt)
    assert runtime.active.visible_message_ids == ["ws-history"]
    assert runtime.active.through_by_channel == {("sp-1", "ch-1"): 4}

    # An admission retained from the old provider turn must not cross into a
    # replacement runtime turn, even if its receipt is otherwise exact.
    stale = await runtime.stage_model_visible_read(
        space_id="sp-1", channel_id="ch-1", through_seq=4,
        through_envelope_id="ws-history", tool_name="get_post",
        tool_arguments={"channel": "ch-1", "seq": 4},
        visible_message_ids=["ws-history"],
    )
    runtime.active.turn_id = "ws-replacement-turn"
    runtime.active.provider_session_id = "ws-local:turn-2"
    with pytest.raises(RuntimeError, match="crossed the active provider turn"):
        await adapter.emit_admission(
            turn_id="turn-2", correlation_key=stale["correlation_receipt"],
            tool_name="get_post", tool_arguments={"channel": "ch-1", "seq": 4},
        )
    assert runtime.active.visible_message_ids == ["ws-history"]
    assert runtime.active.through_by_channel == {("sp-1", "ch-1"): 4}
    assert (await store.get_message_by_envelope("ws-history")).processing_state == (
        ProcessingState.PENDING.value
    )
    await store.close()


class FakeReporter:
    def __init__(self) -> None:
        self.heartbeats = 0
        self.stopped = False

    async def run_heartbeat_loop(self):
        self.heartbeats += 1
        await asyncio.Event().wait()  # runs until cancelled

    def stop(self):
        self.stopped = True

    async def begin_turn(self, message_id):
        return "run_x"

    async def end_turn_batch(self, runs):
        pass


class FakeClient:
    def __init__(self) -> None:
        self.replies: list = []
        self._on_message = None
        self._release = asyncio.Event()
        # Surface attrs ``_build_tool_dispatch`` reads off the worker
        # client. None of these are exercised by the test's WS frames
        # (no tool_call is sent), so trivial stand-ins are enough.
        self.slug = "alice"
        self.device_id = "dev_test"
        self.keystore = object()
        self.http = object()
        self.store = object()
        self.space_id = None
        self.workspace = None

    async def listen(self, on_message):
        # One batch, then stay parked until the connection tears us down.
        await on_message("r1", [{"envelope_id": "a", "text": "hi"}], {"channel_id": "c"})
        await self._release.wait()

    def set_profile(self, *_a, **_kw):
        pass


class FakeCfg:
    display_name = "Puffo Test"

    def resolve_profile_path(self):
        return "/nonexistent/profile.md"  # OSError → empty profile_md branch

    def resolve_workspace_dir(self):
        return "/tmp/puffo-ws-local-test"


@pytest.mark.asyncio
async def test_full_attach_flow(monkeypatch):
    hub = WsLocalHub()
    client = FakeClient()
    reporter = FakeReporter()
    hub.register(AttachPoint(
        slug="puffotest", agent_id="puffotest-1", agent_cfg=FakeCfg(),
        client=client, reporter=reporter, ack_timeout_s=180.0, ping_interval_s=30.0,
    ))
    monkeypatch.setattr(
        route_mod, "authenticate_bundle",
        lambda blob, pw: AuthedAgent("puffotest-1", "puffotest", "Puffo Test"),
    )

    t = FakeTransport()
    t.feed({"type": "connect", "bundle": "Yg==", "password": "pw"})
    served = asyncio.ensure_future(serve_attached(t, hub))

    # Let the handshake + first consumer batch land, then ack + reply.
    for _ in range(6):
        await asyncio.sleep(0)
    connected = t.by_type("connected")
    assert connected and connected[0]["agent"]["display_name"] == "Puffo Test"
    bundle = t.by_type("bundle")[0]
    assert bundle["messages"][0]["envelope_id"] == "a"

    t.feed({"type": "tool_call", "command_id": "cmd_1", "tool": "send_message",
            "params": {"channel": "c", "target_root_id": "r1", "text": "done",
                       "is_visible_to_human": True}})
    t.feed({"type": "ack", "bundle_id": bundle["bundle_id"]})
    for _ in range(4):
        await asyncio.sleep(0)
    # The tool_call ran against the real send_message handler, which
    # depends on http_client + keystore — neither plumbed on FakeClient.
    # The session emits a tool_result with ok=false carrying the
    # AttributeError from inside the handler. That's enough to prove
    # the dispatch path is wired; the handler itself is exercised by
    # the existing mcp tool tests.
    results = t.by_type("tool_result")
    assert results and results[0]["command_id"] == "cmd_1"
    assert reporter.heartbeats == 1  # online while attached

    # Tool disconnects → session ends → consumer + heartbeat torn down.
    t.feed({"type": "__close__"}) if False else t._inbound.put_nowait(None)
    await served
    assert reporter.stopped
    assert hub.registry.current("puffotest") is None  # slot freed


@pytest.mark.asyncio
async def test_unservable_slug_rejected(monkeypatch):
    hub = WsLocalHub()  # nothing registered
    monkeypatch.setattr(
        route_mod, "authenticate_bundle",
        lambda blob, pw: AuthedAgent("ghost-1", "ghost", "Ghost"),
    )
    t = FakeTransport()
    t.feed({"type": "connect", "bundle": "Yg==", "password": "pw"})
    await serve_attached(t, hub)
    errors = t.by_type("error")
    assert errors and "not a ws-local agent" in errors[0]["reason"]


class _V2Coordinator:
    def __init__(self, **kwargs):
        self.provider_session_id = None
        self.held_recovery_source = kwargs["held_recovery_source"]

    async def send(self, *_args, **_kwargs):
        return {"state": "failed"}


class _V2Client:
    def __init__(self, store, tmp_path):
        self.slug = "puffotest"
        self.device_id = "dev"
        self.keystore = object()
        self.http = object()
        self.store = store
        self.space_id = None
        self.workspace = str(tmp_path)
        self.global_runtime = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def listen(self, _on_message):
        from puffo_agent.agent.message_store import ReceiptDisposition

        await self.store.store_receipt(
            {
                "envelope_id": "global-one", "envelope_kind": "channel",
                "sender_slug": "alice", "space_id": "sp", "channel_id": "ch",
                "content": "hello", "content_type": "text/plain", "sent_at": 1,
            },
            server_seq=1, disposition=ReceiptDisposition.ELIGIBLE,
            reason="test receipt",
        )
        self.global_runtime.notify()
        self.started.set()
        await self.release.wait()

    def set_profile(self, *_args, **_kwargs):
        return None


async def _start_v2_attached(monkeypatch, tmp_path, store):
    import puffo_agent.agent.send_coordinator as coordinator_mod

    monkeypatch.setattr(coordinator_mod, "SendCoordinator", _V2Coordinator)
    monkeypatch.setattr(route_mod, "_build_tool_dispatch", lambda _point: {})
    class V2Cfg(FakeCfg):
        def resolve_workspace_dir(self):
            return tmp_path

    client = _V2Client(store, tmp_path)
    reporter = FakeReporter()
    hub = WsLocalHub()
    hub.register(AttachPoint(
        slug="puffotest",
        agent_id="puffotest-1",
        agent_cfg=V2Cfg(),
        client=client,
        reporter=reporter,
        ack_timeout_s=180,
        ping_interval_s=30,
    ))
    monkeypatch.setattr(
        route_mod,
        "authenticate_bundle",
        lambda _blob, _pw: AuthedAgent(
            "puffotest-1", "puffotest", "Puffo Test",
        ),
    )
    transport = FakeTransport()
    transport.feed({
        "type": "connect",
        "bundle": "Yg==",
        "password": "pw",
        "capabilities": ["multi-target-v2", "explicit-admission-v2"],
    })
    served = asyncio.create_task(serve_attached(transport, hub))
    await asyncio.wait_for(client.started.wait(), 1)
    return client, transport, served


async def _wait_for_v2_bundle(transport):
    for _ in range(300):
        bundles = transport.by_type("bundle")
        if bundles:
            return bundles[0]
        await asyncio.sleep(0.02)
    return transport.by_type("bundle")[0]


async def _admit_and_end_v2_bundle(transport, store, bundle):
    from puffo_agent.agent.message_store import ProcessingState

    assert bundle["turn_id"]
    transport.feed({"type": "ack", "bundle_id": bundle["bundle_id"]})
    await asyncio.sleep(0)
    assert [row.envelope_id for row in await store.get_pending()] == ["global-one"]
    transport.feed({
        "type": "admitted",
        "version": 2,
        "bundle_id": bundle["bundle_id"],
        "turn_id": bundle["turn_id"],
    })
    for _ in range(20):
        run = await store.get_turn_run(bundle["turn_id"])
        if run is not None and run.state == ProcessingState.IN_TURN.value:
            break
        await asyncio.sleep(0.01)
    assert run.state == ProcessingState.IN_TURN.value
    transport.feed({"type": "end", "bundle_id": bundle["bundle_id"]})
    for _ in range(20):
        run = await store.get_turn_run(bundle["turn_id"])
        if run.state == ProcessingState.PROCESSED.value:
            break
        await asyncio.sleep(0.01)
    assert run.state == ProcessingState.PROCESSED.value


@pytest.mark.asyncio
async def test_v2_receipt_to_global_bundle_admitted_end_and_owned_runtime_cleanup(
    monkeypatch, tmp_path,
):
    from puffo_agent.agent.message_store import MessageStore

    store = MessageStore(tmp_path / "messages.db")
    await store.open()
    client, transport, served = await _start_v2_attached(monkeypatch, tmp_path, store)
    bundle = await _wait_for_v2_bundle(transport)
    await _admit_and_end_v2_bundle(transport, store, bundle)
    transport._inbound.put_nowait(None)
    await served
    assert client.global_runtime is None
    assert not hasattr(client, "_ws_local_owned_runtime")
    await store.close()
