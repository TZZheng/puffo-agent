from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from puffo_agent.agent.core import AgentAPIError, PuffoAgent
from puffo_agent.agent.adapters.base import TurnResult
from puffo_agent.agent.context_controller import (
    ContextDecision,
    ContextCapabilities,
    ContextController,
    ContextSnapshot,
    DecisionOutcome,
    ProviderAdmissionEvent,
)
from puffo_agent.agent.global_inbox_runtime import (
    ActiveBoundaryAdapter,
    ActiveExactUnion,
    BaselineAdapter,
    GlobalInboxRuntime,
    HeldRecoverySource,
    TrackingSendDelegate,
)
from puffo_agent.agent.message_store import (
    MessageStore,
    ProcessingState,
    ReceiptDisposition,
    ReceiptResult,
    ReceiptWriteStatus,
)
from puffo_agent.crypto.message import MessagePayload
from puffo_agent.crypto.ws_client import TransportOutcome


class Adapter:
    def __init__(self):
        self.callback = None
        self.key = ""
        self.session = "provider-1"
        self.inputs = []

    async def get_context_snapshot(self):
        return ContextSnapshot(0, 200_000, "test", datetime.now(timezone.utc))

    def get_context_capabilities(self):
        return ContextCapabilities()

    async def compact_context(self):
        raise AssertionError("not expected")

    async def rollover_context(self):
        raise AssertionError("not expected")

    def get_provider_session_id(self):
        return self.session

    def register_admission_callback(self, callback, planning_cycle_key=""):
        self.callback = callback
        self.key = planning_cycle_key

    async def admit(self, session: str | None = "provider-1"):
        callback, self.callback = self.callback, None
        assert callback is not None
        await callback(ProviderAdmissionEvent(
            planning_cycle_key=self.key,
            provider_session_id=session,
            provider_turn_id="provider-turn",
            admitted_at=datetime.now(timezone.utc),
        ))


async def make_store(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = MessageStore(tmp_path / "messages.db")
    await store.open()
    return store


async def receipt(
    store,
    envelope_id,
    seq,
    *,
    kind="channel",
    channel="ch-1",
    space="sp-1",
    sender="alice",
    disposition=ReceiptDisposition.ELIGIBLE,
    content=None,
):
    return await store.store_receipt(
        {
            "envelope_id": envelope_id,
            "envelope_kind": kind,
            "sender_slug": sender,
            "recipient_slug": "agent" if kind == "dm" else None,
            "channel_id": channel if kind != "dm" else None,
            "space_id": space if kind != "dm" else None,
            "content": content if content is not None else f"text-{envelope_id}",
            "content_type": "text/plain",
            "sent_at": seq,
        },
        server_seq=seq,
        disposition=disposition,
        reason="test",
    )


async def listen_delivery(
    monkeypatch,
    tmp_path,
    *,
    payload: MessagePayload,
    seq: int,
    blocked: bool = False,
    gate_foreign_dm: bool = False,
    setup=None,
):
    """Drive the production listen callback with a complete wrapper."""
    import puffo_agent.agent.puffo_core_client as client_mod
    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    store = await make_store(tmp_path)
    events: list[str] = []
    original_store_receipt = store.store_receipt

    async def ordered_store(*args, **kwargs):
        result = await original_store_receipt(*args, **kwargs)
        events.append("committed")
        return result

    store.store_receipt = ordered_store

    class Contacts:
        async def is_blocked(self, _slug):
            return blocked

        async def is_allowed(self, _slug):
            return False

    class KeyCache:
        async def get_signing_keys(self, _slug):
            return [b"key"]

        def invalidate(self, _slug):
            return None

    identity = SimpleNamespace(
        kem_secret_key="ignored",
        identity_cert_json="{}",
        server_url="https://example.test",
    )

    class KeyStore:
        def load_identity(self, _slug):
            return identity

    delivery = {"seq": seq, "envelope": {
        "envelope_id": payload.envelope_id,
        "sender_slug": payload.sender_slug,
        "type": "encrypted_message_envelope",
    }}

    class FakeWs:
        def __init__(self, **_kwargs):
            self.on_message = None
            self.on_event = None
            self.on_connect = None

        async def run(self):
            outcome = await self.on_message(delivery)
            # The callback may only release ACK/HOLD after the durable write
            # decision (commit, idempotency, or conflict).
            assert events
            events.append(outcome.value)

        async def dispatch_delivery(self, item):
            outcome = await self.on_message(item)
            return SimpleNamespace(outcome=outcome)

    client = PuffoCoreMessageClient.__new__(PuffoCoreMessageClient)
    client.slug = "agent"
    client.device_id = "dev"
    client.operator_slug = "operator"
    client.auto_accept_dm = False
    client.workspace = str(tmp_path)
    client.store = store
    client.keystore = KeyStore()
    client.http = SimpleNamespace()
    client._key_cache = KeyCache()
    client._contacts = Contacts()
    client._log = logging.getLogger("receipt-listen")
    client._warm_task = None
    client._operator_root_pubkey = None
    client._channel_space = {}
    client._catchup_stale_ms = 0
    client._max_inline_chars = 100_000
    client._segment_chars = 10_000
    client._pending_dm_approvals = {}
    client.global_runtime = SimpleNamespace(
        notify=lambda: events.append("work-wake"),
        notify_delivery=lambda: events.append("delivery-wake"),
    )
    client._processed_invite_ids = set()
    client._processed_membership_event_ids = set()

    async def none(*_args, **_kwargs):
        return None

    async def false(*_args, **_kwargs):
        return False

    async def empty(*_args, **_kwargs):
        return {}

    client._resolve_incoming_thread_root = none
    client._validate_incoming_parent_id = none
    client._maybe_allowlist_outbound_dm = none
    client._apply_invite_replies = empty
    client._maybe_handle_leave_reply = false
    client._maybe_handle_permission_reply = false
    client._maybe_handle_dm_approval_reply = false
    client._is_stale_for_catchup = lambda _sent_at: False
    client._get_space_members = empty
    client._resolve_space_name = lambda _space: asyncio.sleep(0, result="Space")
    client._resolve_channel_name = (
        lambda *_args, **_kwargs: asyncio.sleep(0, result="Channel")
    )
    client._fetch_display_name = (
        lambda slug: asyncio.sleep(0, result=slug.title())
    )
    client._fetch_owner_slug = lambda _slug: asyncio.sleep(0, result="")
    client._is_foreign_dm_sender = (
        lambda _slug: asyncio.sleep(0, result=gate_foreign_dm)
    )
    client._ensure_trusted_contact = none
    client._maybe_send_dm_notice = none
    client._shares_space_with = false
    client._maybe_gate_foreign_dm = (
        lambda **_kwargs: asyncio.sleep(0, result=gate_foreign_dm)
    )
    client._invite_poll_loop = lambda: asyncio.sleep(3600)
    client._on_ws_connect = none
    client._handle_event = none

    monkeypatch.setattr(client_mod, "PuffoCoreWsClient", FakeWs)
    monkeypatch.setattr(client_mod, "decrypt_message", lambda *_args: payload)
    monkeypatch.setattr(client_mod, "decode_secret", lambda _value: b"secret")
    monkeypatch.setattr(
        client_mod.KemKeyPair, "from_secret_bytes", lambda _value: object(),
    )
    setup_result = await setup(client, store, events, delivery) if setup else None
    await client.listen(lambda *_args: asyncio.sleep(0))
    if setup is None:
        return client, store, events, delivery
    return client, store, events, delivery, setup_result


def payload_for(
    envelope_id: str,
    *,
    kind: str = "channel",
    sender: str = "alice",
    content: str = "hello",
) -> MessagePayload:
    return MessagePayload(
        payload_type="message_payload",
        version=2,
        envelope_id=envelope_id,
        envelope_kind=kind,
        sender_slug=sender,
        sender_subkey_id="subkey",
        sent_at=1,
        message_nonce="nonce",
        content_type="text/plain",
        content=content,
        is_visible_to_human=True,
        space_id="sp-1" if kind == "channel" else None,
        channel_id="ch-1" if kind == "channel" else None,
        recipient_slug="agent" if kind == "dm" else None,
    )


@pytest.mark.asyncio
async def test_receipt_commit_before_ack_and_wake_without_admission(tmp_path):
    store = await make_store(tmp_path)
    result = await receipt(store, "m1", 1)
    assert result.acknowledge
    pending = await store.get_pending()
    assert [item.envelope_id for item in pending] == ["m1"]
    assert pending[0].model_visible_at is None
    await store.close()


@pytest.mark.asyncio
async def test_receipt_listen_eligible_commit_before_ack_and_scheduler_wake(
    monkeypatch, tmp_path,
):
    _client, store, events, _delivery = await listen_delivery(
        monkeypatch, tmp_path, payload=payload_for("eligible"), seq=11,
    )
    assert events == [
        "committed",
        "delivery-wake",
        "work-wake",
        TransportOutcome.ACK.value,
    ]
    row = (await store.get_pending())[0]
    assert row.envelope_id == "eligible"
    assert row.model_visible_at is None
    await store.close()


@pytest.mark.asyncio
async def test_receipt_listen_blocked_tombstone_never_persists_plaintext(
    monkeypatch, tmp_path,
):
    secret = "blocked-secret-that-must-not-persist"
    _client, store, events, _delivery = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload_for("blocked", content=secret),
        seq=12,
        blocked=True,
    )
    assert events == [
        "committed",
        "delivery-wake",
        TransportOutcome.ACK.value,
    ]
    raw = (tmp_path / "messages.db").read_bytes()
    assert secret.encode() not in raw
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_terminal_channel_delivery_wake_releases_exact_watermark_waiter(
    monkeypatch, tmp_path,
):
    async def setup(client, store, _events, _delivery):
        runtime = GlobalInboxRuntime(
            store=store,
            adapter=Adapter(),
            run_turn=lambda _planned: None,
            workspace=tmp_path,
        )
        runtime.held_recovery_source.wait_timeout_s = 0.5
        client.global_runtime = runtime
        return asyncio.create_task(
            runtime.held_recovery_source.wait_for_held_delivery(
                "sp-1", "ch-1", 14, "terminal"
            )
        )

    _client, store, events, _delivery, waiter = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload_for("terminal", content="secret"),
        seq=14,
        blocked=True,
        setup=setup,
    )
    assert await waiter
    assert "delivery-wake" not in events  # real runtime wake, not the event spy
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_idempotent_terminal_channel_delivery_wake_releases_waiter(
    monkeypatch, tmp_path,
):
    async def setup(client, store, events, _delivery):
        runtime = GlobalInboxRuntime(
            store=store,
            adapter=Adapter(),
            run_turn=lambda _planned: None,
            workspace=tmp_path,
        )
        runtime.held_recovery_source.wait_timeout_s = 0.5
        client.global_runtime = runtime
        waiter = asyncio.create_task(
            runtime.held_recovery_source.wait_for_held_delivery(
                "sp-1", "ch-1", 15, "idempotent-terminal"
            )
        )
        await asyncio.sleep(0)
        await store.store_receipt(
            {
                "envelope_id": "idempotent-terminal",
                "envelope_kind": "channel",
                "sender_slug": "agent",
                "channel_id": "ch-1",
                "space_id": "sp-1",
                "recipient_slug": None,
                "content_type": "text/plain",
                "content": "hello",
                "sent_at": 1,
                "thread_root_id": None,
                "reply_to_id": None,
                "is_encrypted": True,
            },
            server_seq=15,
            disposition=ReceiptDisposition.TERMINAL,
            reason="self echo",
        )
        events.clear()
        assert not waiter.done()
        return waiter

    _client, store, events, _delivery, waiter = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload_for(
            "idempotent-terminal", sender="agent", content="hello"
        ),
        seq=15,
        setup=setup,
    )
    assert await waiter
    assert events == ["committed", TransportOutcome.ACK.value]
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "expected_events"),
    [
        (
            "eligible_dm",
            ["committed", "work-wake", TransportOutcome.ACK.value],
        ),
        (
            "conflict",
            ["write-conflict", TransportOutcome.HOLD.value],
        ),
    ],
)
async def test_notification_matrix_dm_and_conflict(
    monkeypatch, tmp_path, kind, expected_events,
):
    async def setup(_client, store, events, _delivery):
        if kind == "conflict":
            async def conflict(*_args, **_kwargs):
                events.append("write-conflict")
                return ReceiptResult(
                    ReceiptWriteStatus.CONFLICT,
                    ReceiptDisposition.ELIGIBLE,
                    "conflict",
                    False,
                )

            store.store_receipt = conflict

    payload = payload_for(
        f"matrix-{kind}",
        kind="dm" if kind == "eligible_dm" else "channel",
    )
    _client, store, events, _delivery, _setup = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload,
        seq=16,
        setup=setup,
    )
    assert events == expected_events
    await store.close()


@pytest.mark.asyncio
async def test_notification_matrix_raised_write_wakes_neither(
    monkeypatch, tmp_path,
):
    captured = {}

    async def setup(_client, store, events, _delivery):
        captured["store"] = store
        captured["events"] = events

        async def fail(*_args, **_kwargs):
            events.append("write-raised")
            raise OSError("disk full")

        store.store_receipt = fail

    with pytest.raises(OSError, match="disk full"):
        await listen_delivery(
            monkeypatch,
            tmp_path,
            payload=payload_for("matrix-raised"),
            seq=17,
            setup=setup,
        )
    assert captured["events"] == ["write-raised"]
    await captured["store"].close()


@pytest.mark.asyncio
async def test_gated_receipt_listen_holds_then_exact_wrapper_promotion_acks_once(
    monkeypatch, tmp_path,
):
    client, store, events, delivery = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload_for("gated", kind="dm", sender="foreign"),
        seq=13,
        gate_foreign_dm=True,
    )
    assert events == ["committed", TransportOutcome.HOLD.value]
    assert await store.get_pending() == ()

    client._is_foreign_dm_sender = lambda _slug: asyncio.sleep(0, result=True)
    client._maybe_gate_foreign_dm = lambda **_kwargs: asyncio.sleep(0, result=False)
    client._contacts.is_allowed = lambda _slug: asyncio.sleep(0, result=True)
    posts = []

    class ApprovalHttp:
        async def get(self, _path):
            return {"messages": [delivery]}

        async def post(self, path, body):
            posts.append((path, body))
            return {}

    client.http = ApprovalHttp()
    await client._drain_pending_from_sender("foreign")
    assert [row.envelope_id for row in await store.get_pending()] == ["gated"]
    assert posts == [("/messages/ack", {"envelope_ids": ["gated"]})]
    await client._drain_pending_from_sender("foreign")
    assert [row.envelope_id for row in await store.get_pending()] == ["gated"]
    await store.close()


@pytest.mark.asyncio
async def test_local_event_introduction_has_no_server_seq_and_wakes_scheduler(
    tmp_path,
):
    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    store = await make_store(tmp_path)
    client = PuffoCoreMessageClient.__new__(PuffoCoreMessageClient)
    client.store = store
    client._log = logging.getLogger("local-event")
    wakes = []
    client.global_runtime = SimpleNamespace(notify=lambda: wakes.append("wake"))
    client._resolve_space_name = lambda _space: asyncio.sleep(0, result="Space")
    client._resolve_channel_name = (
        lambda *_args, **_kwargs: asyncio.sleep(0, result="General")
    )
    await client._enqueue_channel_intro_nudge(space_id="sp", channel_id="ch")
    rows = await store.get_pending()
    assert len(rows) == 1 and rows[0].server_seq is None
    assert wakes == ["wake"]
    await store.close()


@pytest.mark.asyncio
async def test_gated_promotion_exact_wrapper_ack_only_after_promotion(tmp_path):
    store = await make_store(tmp_path)
    gated = await receipt(
        store, "dm1", 7, kind="dm",
        disposition=ReceiptDisposition.FOREIGN_DM_GATED,
    )
    assert not gated.acknowledge
    assert await store.get_pending() == ()
    promoted = await store.promote_gated_receipt("dm1", 7, reason="approved")
    assert promoted.acknowledge
    assert [m.envelope_id for m in await store.get_pending()] == ["dm1"]
    await store.close()


@pytest.mark.asyncio
async def test_local_event_has_no_fabricated_server_seq_and_global_order(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "m1", 1)
    local = await store.store_local_event(
        {
            "envelope_id": "local-1",
            "envelope_kind": "channel",
            "sender_slug": "runtime",
            "channel_id": "ch-1",
            "space_id": "sp-1",
            "content": "membership changed",
            "sent_at": 2,
        },
        reason="local event",
    )
    await receipt(store, "m2", 2)
    assert local.server_seq is None
    assert [m.envelope_id for m in await store.get_pending()] == [
        "m1", "local-1", "m2",
    ]
    await store.close()


@pytest.mark.asyncio
async def test_multi_target_global_order_route_metadata_and_current_turn(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "c1", 1)
    await receipt(store, "d1", 2, kind="dm", sender="bob")
    adapter = Adapter()
    turn_ids = []

    async def run(planned):
        adapter.inputs.append(planned.provider_input)
        data = json.loads(
            (tmp_path / ".puffo-agent/current_turn.json").read_text()
        )
        assert "channel_id" not in data
        assert data["message_ids"] == ["c1", "d1"]
        turn_ids.append(data["turn_id"])
        await adapter.admit()

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path,
    )
    assert await runtime.process_once()
    sent = adapter.inputs[0]
    assert sent.index("text-c1") < sent.index("text-d1")
    assert '"message_count":2' in sent
    assert '"route":' in sent and '"sender_slug":"bob"' in sent
    assert (await store.get_turn_run(turn_ids[0])).state == ProcessingState.PROCESSED.value
    assert not (tmp_path / ".puffo-agent/current_turn.json").exists()
    await store.close()


@pytest.mark.asyncio
async def test_multi_target_real_provider_input_preserves_every_sender_metadata(
    tmp_path,
):
    store = await make_store(tmp_path)
    first_content = {
        "text": "channel body",
        "attachment_paths": ["/workspace/a.txt"],
        "mentions": [{"username": "agent", "is_agent": True, "is_self": True}],
        "sender_display_name": "Alice A",
        "sender_owner_slug": "",
        "is_from_operator": True,
        "sender_is_agent": False,
        "is_visible_to_human": True,
        "space_name": "Space One",
        "channel_name": "General",
    }
    second_content = {
        "text": "dm body",
        "attachment_paths": ["/workspace/b.png"],
        "mentions": [{"username": "helper", "is_agent": True, "is_self": False}],
        "sender_display_name": "Helper Bot",
        "sender_owner_slug": "operator",
        "is_from_operator": False,
        "sender_is_agent": True,
        "is_visible_to_human": False,
        "space_name": "",
        "channel_name": "Direct message",
    }
    await receipt(store, "channel-message", 1, content=first_content)
    await store.store_receipt(
        {
            "envelope_id": "dm-message",
            "envelope_kind": "dm",
            "sender_slug": "helper",
            "recipient_slug": "agent",
            "content": second_content,
            "content_type": "text/plain",
            "sent_at": 2,
            "reply_to_id": "dm-parent",
            "is_encrypted": False,
        },
        server_seq=2,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    planned = await runtime.plan_pending()

    class RecordingAdapter:
        def __init__(self):
            self.input = ""

        async def run_turn(self, ctx):
            self.input = ctx.messages[-1]["content"]
            return TurnResult(
                reply="[SILENT]",
                metadata={"assistant_text_parts": ["[SILENT]"]},
            )

    recording = RecordingAdapter()
    agent = PuffoAgent(
        adapter=recording,
        system_prompt="system",
        memory_dir=str(tmp_path / "memory"),
        workspace_dir=str(tmp_path),
        agent_id="test",
    )
    await agent.handle_global_inbox_turn(planned)
    sent = recording.input
    assert sent.index("channel body") < sent.index("dm body")
    for expected in (
        '"kind":"channel"',
        '"kind":"dm"',
        '"mentions":[{"is_agent":true,"is_self":true,"username":"agent"}]',
        '"sender_display_name":"Alice A"',
        '"sender_owner_slug":"operator"',
        '"is_from_operator":true',
        '"sender_is_agent":true',
        '"is_visible_to_human":false',
        '"space_name":"Space One"',
        '"channel_name":"Direct message"',
        '"/workspace/a.txt"',
        '"/workspace/b.png"',
        '"is_encrypted":false',
        '"reply_to_id":"dm-parent"',
    ):
        assert expected in sent
    await store.close()


@pytest.mark.asyncio
async def test_pre_admission_failure_leaves_pending_and_crash_join_exists(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "m1", 1)
    adapter = Adapter()

    async def fail(_planned):
        assert (tmp_path / ".puffo-agent/current_turn.json").exists()
        raise RuntimeError("provider failed before admission")

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=fail, workspace=tmp_path,
    )
    await runtime.process_once()
    assert [m.envelope_id for m in await store.get_pending()] == ["m1"]
    assert not (tmp_path / ".puffo-agent/current_turn.json").exists()
    await store.close()


@pytest.mark.asyncio
async def test_admission_failure_requeues_exact_union_and_provider_session_clears(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "m1", 1)
    adapter = Adapter()

    class Coordinator:
        provider_session_id = None

    coordinator = Coordinator()

    async def fail(_planned):
        await adapter.admit("actual-session")
        assert coordinator.provider_session_id == "actual-session"
        raise RuntimeError("unsafe recovery")

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=fail, workspace=tmp_path,
        coordinator=coordinator,
    )
    await runtime.process_once()
    assert coordinator.provider_session_id is None
    assert [m.envelope_id for m in await store.get_pending()] == ["m1"]
    await store.close()


@pytest.mark.asyncio
async def test_baseline_boundary_stateless_and_same_channel_advance(tmp_path):
    store = await make_store(tmp_path)
    assert await BaselineAdapter(store).get_context_baseline_seq("sp", "ch") is None
    await store.set_context_baseline("sp", "ch", 4)
    assert await BaselineAdapter(store).get_context_baseline_seq("sp", "ch") == 4
    active = ActiveExactUnion(turn_id="turn")
    boundary = ActiveBoundaryAdapter(store, active)
    assert await boundary.get_active_turn_through_seq("sp", "ch") is None
    await boundary.advance_active_turn_through_seq("sp", "ch", 8)
    await boundary.advance_active_turn_through_seq("sp2", "other", 3)
    assert await boundary.get_active_turn_through_seq("sp", "ch") == 8
    assert await boundary.get_active_turn_through_seq("sp2", "other") == 3
    await store.close()


@pytest.mark.asyncio
async def test_failed_held_and_attachment_send_attempts_are_tracked():
    class Coordinator:
        async def send(self, request=None, **kwargs):
            destination = (
                request.get("destination") if isinstance(request, dict) else ""
            )
            return {"state": "held" if destination == "ch-held" else "failed"}

    from puffo_agent.agent.global_inbox_runtime import SendAttemptState
    attempts = SendAttemptState()
    delegate = TrackingSendDelegate(Coordinator(), attempts)
    assert (await delegate.send({"destination": "ch-held"}))["state"] == "held"
    assert (await delegate.send({
        "destination": "ch-failed", "attachment_paths": ["/tmp/a"],
    }))["state"] == "failed"
    assert attempts.attempts == 2
    assert attempts.states == ["held", "failed"]


@pytest.mark.asyncio
async def test_coordinator_worker_host_context_shares_failed_held_attachment_attempts(
    tmp_path, monkeypatch,
):
    from puffo_agent.agent.global_inbox_runtime import SendAttemptState
    from puffo_agent.portal.worker import Worker
    import puffo_agent.portal.state as state_mod

    class Coordinator:
        async def send(self, request=None, **_kwargs):
            destination = request.get("destination", "")
            return {"state": "held" if destination == "held" else "failed"}

    attempts = SendAttemptState()
    delegate = TrackingSendDelegate(Coordinator(), attempts)
    client = SimpleNamespace(
        slug="agent",
        operator_slug="operator",
        keystore=object(),
        http=object(),
        send_delegate=delegate,
    )
    worker = Worker.__new__(Worker)
    worker._client = client
    worker.agent_cfg = SimpleNamespace(
        id="agent-id", runtime=SimpleNamespace(harness="claude-code"),
    )
    monkeypatch.setattr(state_mod, "agent_home_dir", lambda _id: tmp_path)
    context = worker.host_mcp_context()
    assert context.send_coordinator is delegate
    await context.send_coordinator.send({
        "destination": "failed", "text": "text",
    })
    await context.send_coordinator.send({
        "destination": "held", "attachment_paths": ["/tmp/file"],
    })
    assert context.send_coordinator.attempts is attempts
    assert attempts.states == ["failed", "held"]


@pytest.mark.asyncio
async def test_startup_pending_and_content_neutral_coalescing(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "m1", 1)
    adapter = Adapter()
    completed = asyncio.Event()

    async def run(_planned):
        await adapter.admit()
        completed.set()

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path,
    )
    task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(completed.wait(), 1)
    runtime.stop()
    await task
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_limit_wrapper_bytes_shrinks_real_fifo_suffix(tmp_path):
    store = await make_store(tmp_path)
    # Leaf byte cap counts message bodies; runtime additionally counts wrapper.
    await receipt(store, "m1", 1, content="a" * 47_900)
    await receipt(store, "m2", 2, content="b" * 47_900)
    runtime = GlobalInboxRuntime(
        store=store, adapter=Adapter(), run_turn=lambda _p: None,
        workspace=tmp_path, estimator=lambda text: len(text.encode()) // 4,
    )
    planned = await runtime.plan_pending()
    assert planned is not None
    assert planned.message_ids == ("m1",)
    assert planned.more_available
    await store.close()


@pytest.mark.asyncio
async def test_crash_join_mismatched_or_stateless_session_requeues(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "m1", 1)
    await store.admit_messages(
        ["m1"], turn_id="turn-1", provider_session_id="old-session",
    )
    path = tmp_path / ".puffo-agent/current_turn.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "version": 2, "turn_id": "turn-1", "message_ids": ["m1"],
        "targets": [["channel", "sp-1", "ch-1"]], "routes": [],
    }))
    adapter = Adapter()
    adapter.session = None
    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=lambda _p: None,
        workspace=tmp_path,
    )
    assert not await runtime.recover_current_turn()
    assert [m.envelope_id for m in await store.get_pending()] == ["m1"]
    assert not path.exists()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "invalid_version",
        "id_mismatch",
        "route_mismatch",
        "target_mismatch",
        "stateless",
        "session_mismatch",
    ],
)
async def test_crash_join_invalid_exact_union_metadata_requeues(tmp_path, case):
    store = await make_store(tmp_path)
    await receipt(store, "first", 1, sender="alice")
    await receipt(store, "second", 2, sender="bob")
    adapter = Adapter()
    seed = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    planned = await seed.plan_pending()
    assert planned is not None
    seed._write_current_turn(planned)
    session = None if case == "stateless" else adapter.session
    await store.admit_messages(
        planned.message_ids,
        turn_id=planned.turn_id,
        provider_session_id=session,
    )
    raw = json.loads(seed.current_turn_path.read_text())
    if case == "invalid_version":
        raw["version"] = 999
    elif case == "id_mismatch":
        raw["message_ids"] = ["first"]
    elif case == "route_mismatch":
        raw["routes"][0]["channel_id"] = "wrong-channel"
    elif case == "target_mismatch":
        raw["targets"] = [["channel", "wrong-space", "wrong-channel"]]
    elif case == "session_mismatch":
        adapter.session = "different-provider-session"
    seed.current_turn_path.write_text(json.dumps(raw))

    coordinator = SimpleNamespace(provider_session_id="stale")
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        coordinator=coordinator,
    )
    assert not await runtime.recover_current_turn()
    assert [row.envelope_id for row in await store.get_pending()] == [
        "first",
        "second",
    ]
    assert not runtime.current_turn_path.exists()
    assert runtime.active.turn_id == ""
    assert coordinator.provider_session_id is None
    assert adapter.callback is None
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["invalid_shape", "no_run", "stale_run"])
async def test_crash_join_invalid_or_stale_file_is_removed(tmp_path, case):
    store = await make_store(tmp_path)
    await receipt(store, "message", 1)
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.current_turn_path.parent.mkdir(parents=True)
    if case == "invalid_shape":
        runtime.current_turn_path.write_text(json.dumps(["not", "a", "join"]))
    elif case == "no_run":
        runtime.current_turn_path.write_text(json.dumps({
            "version": 2,
            "turn_id": "missing-turn",
            "message_ids": ["message"],
            "routes": [],
            "targets": [],
        }))
    else:
        planned = await runtime.plan_pending()
        assert planned is not None
        runtime._write_current_turn(planned)
        await store.admit_messages(
            planned.message_ids,
            turn_id=planned.turn_id,
            provider_session_id=adapter.session,
        )
        await store.mark_processed(planned.message_ids, turn_id=planned.turn_id)

    assert not await runtime.recover_current_turn()
    assert not runtime.current_turn_path.exists()
    if case == "stale_run":
        assert await store.get_pending() == ()
    else:
        assert [row.envelope_id for row in await store.get_pending()] == ["message"]
    assert runtime.active.turn_id == ""
    assert adapter.callback is None
    await store.close()


@pytest.mark.asyncio
async def test_startup_recovery_resumes_exact_crash_join_before_planning(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "first", 1, sender="alice")
    await receipt(store, "second", 2, sender="bob")
    seed_adapter = Adapter()
    seed_runtime = GlobalInboxRuntime(
        store=store,
        adapter=seed_adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    planned = await seed_runtime.plan_pending()
    assert planned is not None
    seed_runtime._write_current_turn(planned)
    await store.admit_messages(
        planned.message_ids,
        turn_id=planned.turn_id,
        provider_session_id=seed_adapter.session,
    )

    retry_seen = asyncio.Event()

    class Runner:
        initial_calls = 0
        retry_calls = 0
        recovered = None

        async def __call__(self, _planned):
            self.initial_calls += 1
            raise AssertionError("startup recovery must not invoke initial turn")

        async def handle_global_inbox_retry(self, recovered):
            self.retry_calls += 1
            self.recovered = recovered
            retry_seen.set()

    class NoPlanner:
        def plan(self, *_args, **_kwargs):
            raise AssertionError("startup recovery must not call the planner")

    runner = Runner()
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=runner,
        workspace=tmp_path,
        planner=NoPlanner(),
    )
    task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(retry_seen.wait(), timeout=1)
    for _ in range(20):
        if not runtime.current_turn_path.exists():
            break
        await asyncio.sleep(0)
    runtime.stop()
    await asyncio.wait_for(task, timeout=1)

    assert runner.initial_calls == 0
    assert runner.retry_calls == 1
    assert runner.recovered.message_ids == planned.message_ids
    assert runner.recovered.routes == planned.routes
    assert runner.recovered.targets == planned.targets
    assert runner.recovered.provider_input == planned.provider_input
    assert await store.get_pending() == ()
    assert not runtime.current_turn_path.exists()
    assert runtime.active.turn_id == ""
    assert adapter.callback is None
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "health_state", "diagnostic", "expected_retry_calls"),
    [
        ("unavailable", "degraded", "retry unavailable", 0),
        ("auth", "auth_failed", "auth failure", 1),
        ("unsafe", "degraded", "unsafe failure", 1),
        ("exhausted", "api_error_abandoned", "budget exhausted", 3),
    ],
)
async def test_resume_failure_requeues_exact_union_and_cleans_identity(
    tmp_path, failure, health_state, diagnostic, expected_retry_calls,
):
    store = await make_store(tmp_path)
    await receipt(store, "first", 1)
    await receipt(store, "second", 2)
    adapter = Adapter()
    seed = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    planned = await seed.plan_pending()
    assert planned is not None
    seed._write_current_turn(planned)
    await store.admit_messages(
        planned.message_ids,
        turn_id=planned.turn_id,
        provider_session_id=adapter.session,
    )

    class Runner:
        initial_calls = 0
        retry_calls = 0

        async def __call__(self, _planned):
            self.initial_calls += 1
            raise AssertionError("failed resume must not enter initial delivery")

        async def handle_global_inbox_retry(self, _planned):
            self.retry_calls += 1
            if failure == "auth":
                raise AgentAPIError("auth", is_auth=True)
            if failure == "unsafe":
                raise RuntimeError("unsafe")
            raise AgentAPIError("rate limited", is_auth=False)

    runner = Runner()
    if failure == "unavailable":
        del Runner.handle_global_inbox_retry
    coordinator = SimpleNamespace(provider_session_id="stale")
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=runner,
        workspace=tmp_path,
        coordinator=coordinator,
        max_api_retries=2,
        retry_sleep=lambda _delay: asyncio.sleep(0),
    )
    task = asyncio.create_task(runtime.run())

    async def wait_for_cleanup():
        while runtime.current_turn_path.exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_for_cleanup(), timeout=1)
    assert not runtime.current_turn_path.exists()
    runtime.stop()
    await asyncio.wait_for(task, timeout=1)

    assert runner.initial_calls == 0
    assert runner.retry_calls == expected_retry_calls
    assert [row.envelope_id for row in await store.get_pending()] == [
        "first",
        "second",
    ]
    assert runtime.health.state == health_state
    assert diagnostic in runtime.health.diagnostic
    assert runtime.active.turn_id == ""
    assert coordinator.provider_session_id is None
    assert adapter.callback is None
    await store.close()


@pytest.mark.asyncio
async def test_busy_provider_keeps_arrival_pending_until_next_boundary(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "first", 1)
    adapter = Adapter()
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def run(planned):
        seen.append(planned.message_ids)
        await adapter.admit()
        entered.set()
        await release.wait()

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path,
    )
    first_turn = asyncio.create_task(runtime.process_once())
    await entered.wait()
    await receipt(store, "during-busy", 2)
    assert [row.envelope_id for row in await store.get_pending()] == ["during-busy"]
    release.set()
    await first_turn
    assert [row.envelope_id for row in await store.get_pending()] == ["during-busy"]
    assert await runtime.process_once()
    assert seen == [("first",), ("during-busy",)]
    await store.close()


@pytest.mark.asyncio
async def test_limit_count_token_byte_and_wrapper_overhead_are_independent(tmp_path):
    store = await make_store(tmp_path)
    for index in range(51):
        await receipt(store, f"count-{index}", index + 1, content="x")
    count_runtime = GlobalInboxRuntime(
        store=store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        formatter=lambda item: str(item.content),
        estimator=lambda _text: 1,
    )
    assert len((await count_runtime.plan_pending()).message_ids) == 50
    await store.close()

    token_store = await make_store(tmp_path / "tokens")
    for index in range(3):
        await receipt(token_store, f"token-{index}", index + 1, content="token")
    token_runtime = GlobalInboxRuntime(
        store=token_store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path / "tokens",
        formatter=lambda item: str(item.content),
        estimator=lambda text: 0 if text.startswith("<global") else 16_000,
    )
    assert (await token_runtime.plan_pending()).message_ids == (
        "token-0", "token-1",
    )
    await token_store.close()

    byte_store = await make_store(tmp_path / "bytes")
    await receipt(byte_store, "byte-1", 1, content="a" * 48_000)
    await receipt(byte_store, "byte-2", 2, content="b" * 48_000)
    byte_runtime = GlobalInboxRuntime(
        store=byte_store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path / "bytes",
        formatter=lambda item: str(item.content),
        estimator=lambda _text: 1,
    )
    byte_plan = await byte_runtime.plan_pending()
    assert byte_plan.message_ids == ("byte-1",)
    assert byte_plan.wrapper_overhead_bytes > 0
    await byte_store.close()


@pytest.mark.asyncio
async def test_unfit_head_policy_quarantines_once_without_starvation(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "oversize", 1, content="x" * 100_000)
    await receipt(store, "next", 2, content="ok")
    calls = []

    async def policy(envelope_id, reason):
        calls.append((envelope_id, reason))
        return True

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        formatter=lambda item: str(item.content),
        estimator=lambda _text: 1,
        unfit_policy=policy,
    )
    planned = await runtime.plan_pending()
    assert planned.message_ids == ("next",)
    assert len(calls) == 1 and calls[0][0] == "oversize"
    assert [row.envelope_id for row in await store.get_pending()] == ["next"]
    await store.close()


class ScriptedContext:
    def __init__(self, adapter, outcomes, on_replan=None):
        self.adapter = adapter
        self.outcomes = list(outcomes)
        self.on_replan = on_replan
        self.calls = 0
        self.snapshot = ContextSnapshot(
            0, 200_000, "scripted", datetime.now(timezone.utc),
        )

    async def decide(self, candidate, replan):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        replacement = candidate
        if outcome is DecisionOutcome.REPLAN:
            if self.on_replan is not None:
                await self.on_replan()
            replacement = await replan(candidate)
        if outcome is DecisionOutcome.ROLLOVER:
            self.adapter.session = "rolled-session"
        return ContextDecision(
            outcome=outcome,
            candidate=replacement,
            snapshot=self.snapshot,
            projected_tokens=replacement.projected_tokens,
            diagnostic=outcome.value,
        )


@pytest.mark.asyncio
async def test_context_replan_authoritative_arrival_shrink_rollover_and_admit(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "one", 1)
    await receipt(store, "two", 2)
    adapter = Adapter()
    coordinator = SimpleNamespace(provider_session_id=None)

    async def arrival():
        await receipt(store, "arrived-during-compaction", 3)

    controller = ScriptedContext(
        adapter,
        [
            DecisionOutcome.REPLAN,
            DecisionOutcome.SHRINK,
            DecisionOutcome.ROLLOVER,
            DecisionOutcome.ADMIT,
        ],
        on_replan=arrival,
    )
    seen = []

    async def run(planned):
        seen.append(planned.message_ids)
        assert coordinator.provider_session_id == "rolled-session"
        await adapter.admit("rolled-session")

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        context_controller=controller,
        coordinator=coordinator,
    )
    await runtime.process_once()
    # Replan includes the arrival; SHRINK removes a real FIFO suffix.
    assert seen == [("one", "two")]
    assert [row.envelope_id for row in await store.get_pending()] == [
        "arrived-during-compaction",
    ]
    assert coordinator.provider_session_id is None
    await store.close()


@pytest.mark.asyncio
async def test_context_degraded_and_budget_exhaustion_do_not_poll_until_notify(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "one", 1)
    adapter = Adapter()
    controller = ScriptedContext(
        adapter,
        [DecisionOutcome.DEGRADED, DecisionOutcome.ADMIT],
    )

    async def run(_planned):
        await adapter.admit()

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        context_controller=controller,
    )
    task = asyncio.create_task(runtime.run())
    runtime.notify()
    await asyncio.sleep(0.15)
    assert runtime.health.state == "degraded"
    assert controller.calls == 1
    await asyncio.sleep(0.15)
    assert controller.calls == 1
    runtime.notify()
    await asyncio.sleep(0.15)
    assert controller.calls == 2
    runtime.stop()
    await task

    await receipt(store, "budget", 2)
    adversarial = ScriptedContext(
        adapter, [DecisionOutcome.SHRINK] * 4,
    )
    exhausted = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        context_controller=adversarial,
        max_context_decisions=2,
    )
    await exhausted.process_once()
    assert exhausted.health.state == "degraded"
    assert await store.get_pending()
    await store.close()


class RetryingRunner:
    def __init__(self, adapter, retry_effects):
        self.adapter = adapter
        self.retry_effects = list(retry_effects)
        self.initial_calls = 0
        self.retry_calls = 0

    async def __call__(self, _planned):
        self.initial_calls += 1
        await self.adapter.admit()
        raise AgentAPIError("rate limit", is_auth=False)

    async def handle_global_inbox_retry(self, _planned):
        self.retry_calls += 1
        effect = self.retry_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


@pytest.mark.asyncio
async def test_admission_retry_success_preserves_union_without_reappend(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "retry", 1)
    adapter = Adapter()
    runner = RetryingRunner(adapter, [None])
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=runner,
        workspace=tmp_path,
        retry_sleep=lambda _delay: asyncio.sleep(0),
    )
    await runtime.process_once()
    assert runner.initial_calls == 1 and runner.retry_calls == 1
    assert await store.get_pending() == ()
    assert not runtime.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
async def test_retry_core_uses_exact_provider_input_without_duplicate_log_append(
    tmp_path,
):
    captured = {}

    class RetryAdapter:
        async def run_retry_turn(self, kick, fallback, ctx):
            captured.update(kick=kick, fallback=fallback, messages=list(ctx.messages))
            return TurnResult(
                reply="[SILENT]",
                metadata={"assistant_text_parts": ["[SILENT]"]},
            )

    planned = SimpleNamespace(provider_input="<exact-global-input>")
    agent = PuffoAgent(
        adapter=RetryAdapter(),
        system_prompt="system",
        memory_dir=str(tmp_path / "memory"),
    )
    agent.log.append({"role": "user", "content": "<exact-global-input>"})
    before = list(agent.log)
    await agent.handle_global_inbox_retry(planned)
    assert agent.log == before
    assert captured["fallback"] == "<exact-global-input>"
    assert captured["messages"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        AgentAPIError("auth", is_auth=True),
        RuntimeError("unsafe"),
        AgentAPIError("rate", is_auth=False),
    ],
)
async def test_admission_retry_auth_unsafe_or_exhaustion_requeues(
    tmp_path, error,
):
    store = await make_store(tmp_path)
    await receipt(store, "retry", 1)
    adapter = Adapter()
    if isinstance(error, AgentAPIError) and error.is_auth:
        class Runner(RetryingRunner):
            async def __call__(self, _planned):
                await self.adapter.admit()
                raise error
        runner = Runner(adapter, [])
    elif isinstance(error, RuntimeError):
        class Runner(RetryingRunner):
            async def __call__(self, _planned):
                await self.adapter.admit()
                raise error
        runner = Runner(adapter, [])
    else:
        runner = RetryingRunner(adapter, [error, error])
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=runner,
        workspace=tmp_path,
        max_api_retries=2,
        retry_sleep=lambda _delay: asyncio.sleep(0),
    )
    await runtime.process_once()
    assert [row.envelope_id for row in await store.get_pending()] == ["retry"]
    assert not runtime.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
async def test_silence_without_correlated_admission_degrades_without_self_wake(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "silent", 1)
    adapter = Adapter()
    calls = 0

    async def run(_planned):
        nonlocal calls
        calls += 1
        return None

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path,
    )
    await runtime.process_once()
    assert runtime.health.state == "degraded"
    assert [row.envelope_id for row in await store.get_pending()] == ["silent"]
    assert calls == 1
    assert not runtime.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
async def test_held_watermark_waits_through_unrelated_notification_and_continuation(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "initial", 1)
    await store.admit_messages(
        ["initial"], turn_id="turn", provider_session_id="provider-1",
    )
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn"
    runtime.active.message_ids[:] = ["initial"]
    runtime.active.provider_session_id = "provider-1"
    source = HeldRecoverySource(runtime, wait_timeout_s=0.5)
    waiter = asyncio.create_task(
        source.wait_for_held_delivery("sp-1", "ch-1", 3, "watermark"),
    )
    source.notify_delivery()
    await asyncio.sleep(0)
    await receipt(store, "watermark", 3)
    source.notify_delivery()
    assert await waiter

    rows = await source.query_held_messages(
        "sp-1", "ch-1", 3, "watermark", "provider-1",
    )
    assert [row["envelope_id"] for row in rows] == ["watermark"]
    assert rows[0]["continuation_correlation_key"]
    assert [row.envelope_id for row in await store.get_pending()] == ["watermark"]
    await adapter.admit("provider-1")
    assert await store.get_pending() == ()
    in_turn = await store.get_in_turn_messages("turn", "provider-1")
    assert [row.envelope_id for row in in_turn] == ["initial", "watermark"]
    assert runtime.held.synchronized
    await store.close()


@pytest.mark.asyncio
async def test_held_timeout_mismatch_stateless_and_context_rejection_stage_nothing(
    tmp_path,
):
    store = await make_store(tmp_path)
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn"
    runtime.active.provider_session_id = "provider-1"
    source = HeldRecoverySource(runtime, wait_timeout_s=0.01)
    assert not await source.wait_for_held_delivery("sp", "ch", 9, "missing")
    assert await source.query_held_messages(
        "sp", "ch", 9, "missing", None,
    ) == ()
    assert runtime.held.message_ids == ()
    await receipt(store, "rejected", 10)
    runtime.active.provider_session_id = "provider-1"
    runtime.context_controller = ScriptedContext(
        adapter, [DecisionOutcome.DEGRADED],
    )
    assert await source.query_held_messages(
        "sp-1", "ch-1", 10, "rejected", "provider-1",
    ) == ()
    assert runtime.held.message_ids == ()
    await store.close()


@pytest.mark.asyncio
async def test_held_continuation_uses_whole_message_fifty_item_limit(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "initial", 1)
    await store.admit_messages(
        ["initial"], turn_id="turn", provider_session_id="provider-1",
    )
    for seq in range(2, 53):
        await receipt(store, f"held-{seq}", seq, content="small")
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn"
    runtime.active.message_ids[:] = ["initial"]
    runtime.active.provider_session_id = "provider-1"
    rows = await runtime.held_recovery_source.query_held_messages(
        "sp-1", "ch-1", 52, "held-52", "provider-1",
    )
    assert len(rows) == 50
    assert rows[0]["envelope_id"] == "held-2"
    assert rows[-1]["envelope_id"] == "held-51"
    assert "held-52" not in runtime.held.message_ids
    await store.close()
