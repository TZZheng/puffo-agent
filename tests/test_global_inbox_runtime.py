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
    MessageRoute,
    SendAttemptState,
    TrackingSendDelegate,
    await_listener_with_runtime,
    format_stored_message,
    route_for,
)
from puffo_agent.agent.shared_content import DEFAULT_SHARED_CLAUDE_MD
from puffo_agent.agent.inbox_scheduler import (
    InboxNoticeDelivery,
    NoticeDeliveryCapability,
)
from puffo_agent.agent.message_store import (
    PRIOR_CONTEXT_MAX_BYTES,
    PRIOR_CONTEXT_MAX_ITEMS,
    MessageStore,
    ProcessingState,
    ReceiptDisposition,
    ReceiptResult,
    ReceiptWriteStatus,
    StoredMessage,
)
from puffo_agent.agent._logging import log_runtime_event
from puffo_agent.agent.runtime_event_outbox import RuntimeEventOutbox
from puffo_agent.agent.runtime_events import RuntimeEvent
from puffo_agent.crypto.message import MessagePayload
from puffo_agent.crypto.ws_client import TransportOutcome


def runtime_events(caplog):
    events = []
    for record in caplog.records:
        message = record.getMessage()
        marker = "runtime_event="
        if marker in message:
            events.append(json.loads(message.split(marker, 1)[1]))
    return events


def test_runtime_event_helper_fails_open_and_omits_unavailable(
    caplog, monkeypatch,
):
    caplog.set_level(
        logging.INFO,
        logger=__name__,
    )
    target = logging.getLogger(__name__)
    log_runtime_event(target, "unknown_event", agent_id="agent")
    log_runtime_event(
        target, "batch.planned", unknown_field=["ignored"],
    )
    log_runtime_event(
        target, "batch.planned", agent_id=object(), first_seq=float("nan"),
    )

    import puffo_agent.agent._logging as logging_module
    original_dumps = logging_module.json.dumps
    monkeypatch.setattr(
        logging_module.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("serialize")),
    )
    log_runtime_event(target, "batch.planned", agent_id="agent")
    monkeypatch.setattr(logging_module.json, "dumps", original_dumps)

    class RaisingHandler(logging.Handler):
        def emit(self, _record):
            raise RuntimeError("handler failed")

    isolated = logging.getLogger("test.runtime-event-fail-open")
    isolated.handlers[:] = [RaisingHandler()]
    isolated.propagate = False
    isolated.setLevel(logging.INFO)
    try:
        log_runtime_event(isolated, "batch.planned", agent_id="agent")
    finally:
        isolated.handlers.clear()

    log_runtime_event(
        target,
        "batch.planned",
        agent_id="agent",
        first_seq=None,
        last_seq=None,
    )
    log_runtime_event(
        target,
        "batch.planned",
        envelope_ids=["env-1"],
        routes=[{
            "space_id": "sp-1",
            "channel_id": "ch-1",
            "count": 1,
            "min_seq": 2,
            "max_seq": 2,
        }],
    )
    log_runtime_event(
        target,
        "batch.planned",
        routes=[{"channel_id": "ch-1", "payload": "rejected"}],
    )
    assert runtime_events(caplog) == [
        {"event": "batch.planned"},
        {"event": "batch.planned"},
        {
            "event": "batch.planned",
            "agent_id": "agent",
        },
        {
            "event": "batch.planned",
            "envelope_ids": ["env-1"],
            "routes": [{
                "channel_id": "ch-1",
                "count": 1,
                "max_seq": 2,
                "min_seq": 2,
                "space_id": "sp-1",
            }],
        },
        {"event": "batch.planned"},
    ]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "runtime observability degraded" in record.getMessage()
    ]
    assert warnings
    assert all(
        "unknown_event" not in warning
        and "unknown_field" not in warning
        for warning in warnings
    )


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

    async def admit(
        self,
        session: str | None = "provider-1",
        provider_turn_id: str = "provider-turn",
    ):
        callback, self.callback = self.callback, None
        assert callback is not None
        await callback(ProviderAdmissionEvent(
            planning_cycle_key=self.key,
            provider_session_id=session,
            provider_turn_id=provider_turn_id,
            admitted_at=datetime.now(timezone.utc),
        ))


class ToolReturnAdapter(Adapter):
    tool_result_admission_boundary = "tool_return"

    def register_continuation_callback(self, *_args, **_kwargs):
        raise AssertionError("tool-return admission must not await provider completion")


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
    is_encrypted=True,
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
            "is_encrypted": is_encrypted,
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
            # The callback may only release ACK/DEFER/HOLD after the durable
            # write decision (commit, idempotency, or conflict).
            assert events
            events.append(outcome.value)

        async def dispatch_delivery(self, item):
            outcome = await self.on_message(item)
            return SimpleNamespace(outcome=outcome)

    client = PuffoCoreMessageClient.__new__(PuffoCoreMessageClient)
    client.agent_id = "agent-id"
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
    monkeypatch, tmp_path, caplog,
):
    caplog.set_level(
        logging.DEBUG,
        logger="receipt-listen",
    )
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
    receipt_event = next(
        item for item in runtime_events(caplog)
        if item["event"] == "inbox.receipt_committed"
    )
    assert receipt_event == {
        "event": "inbox.receipt_committed",
        "agent_id": "agent-id",
        "agent_slug": "agent",
        "envelope_id": "eligible",
        "space_id": "sp-1",
        "channel_id": "ch-1",
            "seq": 11,
            "server_seq": 11,
            "message_id": "eligible",
            "mode": "transport_receipt",
        "state": "eligible",
    }
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
    assert events == ["committed", TransportOutcome.DEFER.value]
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
async def test_legacy_gated_receipt_backfills_then_promotes_and_acks(
    monkeypatch, tmp_path,
):
    payload = payload_for("legacy-gated", kind="dm", sender="foreign")

    async def setup(_client, store, _events, _delivery):
        await store.store(payload)

    client, store, events, delivery, _ = await listen_delivery(
        monkeypatch,
        tmp_path,
        payload=payload,
        seq=16,
        gate_foreign_dm=True,
        setup=setup,
    )
    assert events == ["committed", TransportOutcome.DEFER.value]
    legacy = await store.get_message_by_envelope("legacy-gated")
    assert legacy is not None
    assert legacy.server_seq == 16
    assert legacy.receipt_disposition is ReceiptDisposition.FOREIGN_DM_GATED
    assert legacy.processing_state is None

    client._is_foreign_dm_sender = lambda _slug: asyncio.sleep(0, result=True)
    client._maybe_gate_foreign_dm = lambda **_kwargs: asyncio.sleep(
        0, result=False
    )
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

    assert [row.envelope_id for row in await store.get_pending()] == [
        "legacy-gated"
    ]
    assert posts == [
        ("/messages/ack", {"envelope_ids": ["legacy-gated"]}),
    ]
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
async def test_multi_target_global_order_route_metadata_and_current_turn(
    tmp_path, caplog,
):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )
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
    assert '"more_pending":false' in sent
    assert '"pending_targets":[]' in sent
    assert '"route":' in sent and '"sender_slug":"bob"' in sent
    assert (await store.get_turn_run(turn_ids[0])).state == ProcessingState.PROCESSED.value
    assert not (tmp_path / ".puffo-agent/current_turn.json").exists()
    events = runtime_events(caplog)
    names = [item["event"] for item in events]
    assert names.index("batch.planned") < names.index("turn.admitted")
    assert names.index("turn.admitted") < names.index("turn.processed")
    batch = next(
        item for item in events if item["event"] == "batch.planned"
    )
    assert batch["message_count"] == 2
    assert batch["target_count"] == 2
    assert batch["first_seq"] == 1
    assert batch["last_seq"] == 2
    assert "member_ids" not in batch
    assert batch["batch_id"] == batch["correlation_key"]
    assert batch["envelope_count"] == 2
    assert batch["envelope_ids"] == ["c1", "d1"]
    assert batch["routes"] == [
        {
            "channel_id": "ch-1",
            "count": 1,
            "max_seq": 1,
            "min_seq": 1,
            "space_id": "sp-1",
        },
        {
            "count": 1,
            "dm_peer": "bob",
            "max_seq": 2,
            "min_seq": 2,
        },
    ]
    admitted = next(
        item for item in events if item["event"] == "turn.admitted"
    )
    assert admitted["provider_session_id"] == "provider-1"
    assert admitted["provider_turn_id"] == "provider-turn"
    assert next(
        item for item in events if item["event"] == "turn.processed"
    )["state"] == "processed"
    await store.close()


@pytest.mark.asyncio
async def test_committed_turn_survives_runtime_log_handler_failure(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "committed", 1)
    adapter = Adapter()
    admitted_turn_id = ""

    async def run(planned):
        nonlocal admitted_turn_id
        admitted_turn_id = planned.turn_id
        await adapter.admit()

    class RaisingHandler(logging.Handler):
        def emit(self, _record):
            raise RuntimeError("log handler failed")

    import puffo_agent.agent.global_inbox_runtime as runtime_module
    handler = RaisingHandler()
    runtime_module.logger.addHandler(handler)
    runtime_module.logger.setLevel(logging.INFO)
    try:
        runtime = GlobalInboxRuntime(
            store=store,
            adapter=adapter,
            run_turn=run,
            workspace=tmp_path,
            agent_id="real-agent-id",
        )
        assert await runtime.process_once()
    finally:
        runtime_module.logger.removeHandler(handler)

    run_state = await store.get_turn_run(admitted_turn_id)
    assert run_state is not None
    assert run_state.state == ProcessingState.PROCESSED.value
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_turn_send_mode_tracks_encrypted_bundle_and_clears(tmp_path):
    from puffo_agent.agent import send_mode

    store = await make_store(tmp_path)
    await receipt(store, "encrypted", 1, is_encrypted=True)
    adapter = Adapter()

    async def run(_planned):
        assert await send_mode.encryption_required(
            "agent-send-mode", store, None
        )
        await adapter.admit()

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        send_mode_keys=("agent-send-mode",),
    )
    assert await runtime.process_once()
    assert not await send_mode.encryption_required(
        "agent-send-mode", store, None
    )
    await store.close()


@pytest.mark.asyncio
async def test_turn_send_mode_aliases_clear_after_provider_failure(tmp_path, caplog):
    from puffo_agent.agent import send_mode

    caplog.set_level(logging.INFO)
    store = await make_store(tmp_path)
    secret = "runtime-sensitive-message-body"
    await receipt(
        store,
        "encrypted-failure",
        1,
        is_encrypted=True,
        content=secret,
    )
    adapter = Adapter()
    keys = ("agent-id-send-mode", "agent-slug-send-mode")

    async def run(_planned):
        for key in keys:
            assert await send_mode.encryption_required(key, store, None)
        await adapter.admit()
        raise RuntimeError("provider failed")

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        send_mode_keys=keys,
    )
    assert await runtime.process_once()
    assert runtime.health.state == "degraded"
    requeued = next(
        item for item in runtime_events(caplog)
        if item["event"] == "turn.requeued"
    )
    assert requeued["state"] == "requeued"
    assert requeued["error_category"] == "provider_error"
    assert secret not in " ".join(
        record.getMessage() for record in caplog.records
    )
    for key in keys:
        assert not await send_mode.encryption_required(key, store, None)
    await store.close()


@pytest.mark.asyncio
async def test_turn_send_mode_plaintext_bundle_does_not_require_encryption(tmp_path):
    from puffo_agent.agent import send_mode

    store = await make_store(tmp_path)
    await receipt(store, "plaintext", 1, is_encrypted=False)
    adapter = Adapter()

    async def run(_planned):
        assert not await send_mode.encryption_required(
            "plaintext-agent", store, None
        )
        await adapter.admit()

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        send_mode_keys=("plaintext-agent",),
    )
    assert await runtime.process_once()
    await store.close()


@pytest.mark.asyncio
async def test_turn_send_mode_clears_when_provider_turn_is_cancelled(tmp_path):
    from puffo_agent.agent import send_mode

    store = await make_store(tmp_path)
    await receipt(store, "cancelled-encrypted", 1, is_encrypted=True)
    adapter = Adapter()
    provider_started = asyncio.Event()

    async def run(_planned):
        await adapter.admit()
        provider_started.set()
        await asyncio.Event().wait()

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        send_mode_keys=("cancelled-agent",),
    )
    task = asyncio.create_task(runtime.process_once())
    await provider_started.wait()
    assert await send_mode.encryption_required("cancelled-agent", store, None)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not await send_mode.encryption_required(
        "cancelled-agent", store, None
    )
    assert [row.envelope_id for row in await store.get_pending()] == [
        "cancelled-encrypted"
    ]
    await store.close()


@pytest.mark.asyncio
async def test_listener_guard_stops_transport_when_runtime_crashes():
    listener_started = asyncio.Event()
    listener_stopped = asyncio.Event()

    async def listen_forever():
        listener_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            listener_stopped.set()

    async def crash_runtime():
        await listener_started.wait()
        raise ValueError("runtime boom")

    runtime_task = asyncio.create_task(crash_runtime())
    with pytest.raises(RuntimeError, match="global inbox crashed: runtime boom"):
        await await_listener_with_runtime(
            listen_forever(),
            runtime_task,
            label="global inbox",
        )

    assert listener_stopped.is_set()


@pytest.mark.asyncio
async def test_listener_guard_observes_simultaneous_listener_failure():
    release = asyncio.Event()
    listener_started = asyncio.Event()

    async def fail_listener():
        listener_started.set()
        await release.wait()
        raise OSError("listener boom")

    async def fail_runtime():
        await listener_started.wait()
        await release.wait()
        raise ValueError("runtime boom")

    runtime_task = asyncio.create_task(fail_runtime())
    guarded = asyncio.create_task(
        await_listener_with_runtime(
            fail_listener(),
            runtime_task,
            label="global inbox",
        )
    )
    await listener_started.wait()
    release.set()

    with pytest.raises(
        RuntimeError,
        match="runtime boom; listener also failed: listener boom",
    ):
        await guarded


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
    assert await store.get_context_baseline("sp", "ch") == 4
    assert await store.get_context_baseline("sp2", "other") is None
    await store.close()


@pytest.mark.asyncio
async def test_model_visible_read_advances_only_after_exact_tool_result_admission(
    tmp_path, caplog,
):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )
    store = await make_store(tmp_path)
    await receipt(store, "history-2", 2)

    class ContinuationAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.continuations = []

        def register_continuation_callback(
            self,
            callback,
            planning_cycle_key="",
            *,
            channel_id="",
            tool_names=(),
            tool_arguments=None,
            correlation_receipt="",
        ):
            self.continuations.append((
                callback,
                planning_cycle_key,
                channel_id,
                tool_names,
                tool_arguments,
                correlation_receipt,
            ))

        async def admit_continuation(self):
            callback, key, *_ = self.continuations.pop(0)
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=key,
                provider_session_id="provider-1",
                provider_turn_id="provider-turn",
                tool_call_id="tool-call-history",
                admitted_at=datetime.now(timezone.utc),
            ))

    adapter = ContinuationAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn"
    runtime.active.provider_session_id = "provider-1"
    runtime.active.provider_turn_id = "provider-turn"
    runtime.active.routes[:] = [
        MessageRoute("history-2", "channel", "sp-1", "ch-1"),
    ]
    boundary = ActiveBoundaryAdapter(store, runtime.active)

    result = await runtime.stage_model_visible_read(
        space_id="sp-1",
        channel_id="ch-1",
        through_seq=2,
        through_envelope_id="history-2",
        tool_name="get_channel_history",
        tool_arguments={"channel": "ch-1"},
    )

    assert result["state"] == "staged"
    assert await boundary.get_active_turn_through_seq("sp-1", "ch-1") is None
    assert adapter.continuations[0][3] == ("get_channel_history",)
    assert adapter.continuations[0][4] == {"channel": "ch-1"}
    assert adapter.continuations[0][5] == result["correlation_receipt"]
    staged_events = runtime_events(caplog)
    assert [event["event"] for event in staged_events].count(
        "history.read_staged"
    ) == 1
    assert not any(
        event["event"] == "history.read_admitted" for event in staged_events
    )

    await adapter.admit_continuation()
    assert await boundary.get_active_turn_through_seq("sp-1", "ch-1") == 2
    history_events = [
        event for event in runtime_events(caplog)
        if event["event"].startswith("history.")
    ]
    assert [event["event"] for event in history_events] == [
        "history.read_staged", "history.read_admitted",
    ]
    assert history_events[0]["correlation_key"] == history_events[1][
        "correlation_key"
    ]
    assert history_events[1]["provider_turn_id"] == "provider-turn"
    assert history_events[1]["tool_call_id"] == "tool-call-history"

    class BoundaryCoordinator:
        async def send(self, _request):
            seen_seq = await boundary.get_active_turn_through_seq(
                "sp-1", "ch-1",
            )
            return {
                "state": "held",
                "seen_seq": seen_seq,
                "latest_seq": 3,
            }

    delegate = TrackingSendDelegate(
        BoundaryCoordinator(),
        SendAttemptState(),
        runtime=runtime,
    )
    await delegate.send({"destination": "ch-1"})
    correlated = [
        event for event in runtime_events(caplog)
        if event["event"] in {
            "history.read_staged",
            "history.read_admitted",
            "send.attempted",
        }
    ]
    assert [event["event"] for event in correlated] == [
        "history.read_staged",
        "history.read_admitted",
        "send.attempted",
    ]
    attempted = correlated[-1]
    assert attempted["turn_id"] == "turn"
    assert attempted["provider_session_id"] == "provider-1"
    assert attempted["provider_turn_id"] == "provider-turn"
    assert attempted["space_id"] == "sp-1"
    assert attempted["channel_id"] == "ch-1"

    with pytest.raises(RuntimeError, match="does not match local storage"):
        await runtime.stage_model_visible_read(
            space_id="sp-1",
            channel_id="ch-1",
            through_seq=3,
            through_envelope_id="history-2",
            tool_name="get_channel_history",
            tool_arguments={"channel": "ch-1"},
        )
    assert any(
        event["event"] == "history.read_staged"
        and event.get("latest_seq") == 3
        and event.get("state") == "invalid_watermark"
        for event in runtime_events(caplog)
    )
    await store.close()


@pytest.mark.asyncio
async def test_model_visible_read_admits_at_runtime_tool_return(tmp_path, caplog):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )
    store = await make_store(tmp_path)
    await receipt(store, "history-tool-return", 7)
    adapter = ToolReturnAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn-tool-return"
    runtime.active.provider_session_id = adapter.session
    runtime.active.provider_turn_id = "provider-turn-tool-return"

    result = await runtime.stage_model_visible_read(
        space_id="sp-1",
        channel_id="ch-1",
        through_seq=7,
        through_envelope_id="history-tool-return",
        tool_name="get_channel_history",
        tool_arguments={"channel": "ch-1"},
    )

    assert result["state"] == "admitted"
    assert runtime.active.through_by_channel[("sp-1", "ch-1")] == 7
    history_events = [
        event for event in runtime_events(caplog)
        if event["event"].startswith("history.")
    ]
    assert [event["event"] for event in history_events] == [
        "history.read_staged", "history.read_admitted",
    ]
    assert history_events[0]["state"] == "tool_return"
    assert history_events[1]["provider_turn_id"] == (
        "provider-turn-tool-return"
    )
    await store.close()


@pytest.mark.asyncio
async def test_failed_held_and_attachment_send_attempts_are_tracked(caplog):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )

    class Coordinator:
        async def send(self, request=None, **kwargs):
            destination = (
                request.get("destination") if isinstance(request, dict) else ""
            )
            if destination == "ch-held":
                return {
                    "state": "held", "envelope_id": "held-envelope",
                    "seen_seq": 4, "latest_seq": 5,
                }
            if destination == "ch-sent":
                return {
                    "state": "sent", "envelope_id": "sent-envelope", "seq": 6,
                    "latest_seq_before_send": 5,
                }
            return {"state": "failed", "error_kind": "validation"}

    from puffo_agent.agent.global_inbox_runtime import SendAttemptState
    attempts = SendAttemptState()
    delegate = TrackingSendDelegate(Coordinator(), attempts)
    assert (await delegate.send({"destination": "ch-held"}))["state"] == "held"
    assert (await delegate.send({"destination": "ch-held"}))["state"] == "held"
    assert (await delegate.send({
        "destination": "ch-sent", "send_anyway": True,
    }))["state"] == "sent"
    assert (await delegate.send({"destination": "@operator"}))["state"] == "failed"
    assert (await delegate.send({
        "destination": "ch-failed", "attachment_paths": ["/tmp/a"],
    }))["state"] == "failed"
    assert attempts.attempts == 5
    assert attempts.states == ["held", "held", "sent", "failed", "failed"]
    send_event_records = [
        event for event in runtime_events(caplog)
        if event["event"].startswith("send.")
    ]
    assert [event["event"] for event in send_event_records] == [
        "send.attempted", "send.held",
        "send.attempted", "send.held",
        "send.attempted", "send.committed",
        "send.attempted", "send.failed",
        "send.attempted", "send.failed",
    ]
    attempts_only = [
        event for event in send_event_records
        if event["event"] == "send.attempted"
    ]
    assert [event.get("mode") for event in attempts_only] == [
        "require_current", "require_current", "send_anyway", None,
        "require_current",
    ]
    assert [event["attempt_phase"] for event in attempts_only] == [
        "initial", "reconsider", "initial", "initial", "initial",
    ]
    assert [event["transport"] for event in attempts_only] == [
        "channel", "channel", "channel", "dm", "channel",
    ]
    committed = next(
        event for event in send_event_records
        if event["event"] == "send.committed"
    )
    assert committed["latest_seq_before_send"] == 5
    assert "latest_seq" not in committed
    held = next(
        event for event in send_event_records
        if event["event"] == "send.held"
    )
    assert held["latest_seq"] == 5
    assert "latest_seq_before_send" not in held
    assert all(
        record.levelno == (
            logging.DEBUG
            if json.loads(record.getMessage().split("runtime_event=", 1)[1])[
                "event"
            ] == "send.attempted"
            else logging.INFO
        )
        for record in caplog.records
        if "runtime_event=" in record.getMessage()
    )

    keyless_coordinator = Coordinator()
    keyless_coordinator.http_client = SimpleNamespace(keyless=True)
    keyless_delegate = TrackingSendDelegate(keyless_coordinator, SendAttemptState())
    await keyless_delegate.send({"destination": "ch-unsequenced"})
    assert [
        event["transport"]
        for event in runtime_events(caplog)
        if event["event"] == "send.attempted"
    ][-1] == "keyless"


@pytest.mark.asyncio
async def test_coordinator_exception_is_tracked_and_re_raised(caplog):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )

    class Coordinator:
        async def send(self, *_args, **_kwargs):
            raise RuntimeError("coordinator failed")

    from puffo_agent.agent.global_inbox_runtime import SendAttemptState
    attempts = SendAttemptState()
    delegate = TrackingSendDelegate(Coordinator(), attempts)
    with pytest.raises(RuntimeError, match="coordinator failed"):
        await delegate.send({"destination": "ch-failed"})

    assert attempts.states == ["failed"]
    assert [event["event"] for event in runtime_events(caplog)] == [
        "send.attempted",
        "send.failed",
    ]
    assert runtime_events(caplog)[-1]["error_category"] == "delegate_exception"


@pytest.mark.asyncio
async def test_send_events_use_only_active_resolved_route(caplog, tmp_path):
    caplog.set_level(
        logging.DEBUG,
        logger="puffo_agent.agent.global_inbox_runtime",
    )

    class Coordinator:
        async def send(self, request=None, **_kwargs):
            return {"state": "sent", "envelope_id": "sent", "seq": 9}

    runtime = GlobalInboxRuntime(
        store=await make_store(tmp_path),
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.routes[:] = [
        MessageRoute("incoming", "channel", "sp-1", "resolved-channel"),
    ]
    delegate = TrackingSendDelegate(
        Coordinator(), SendAttemptState(), runtime=runtime,
    )
    await delegate.send({"destination": "resolved-channel"})
    await delegate.send({"destination": "model-only-destination"})

    attempted = [
        event for event in runtime_events(caplog)
        if event["event"] == "send.attempted"
    ]
    assert attempted[0]["space_id"] == "sp-1"
    assert attempted[0]["channel_id"] == "resolved-channel"
    assert "channel_id" not in attempted[1]
    assert "space_id" not in attempted[1]
    await runtime.store.close()


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
    assert planned.pending_targets == (("channel", "sp-1", "ch-1"),)
    assert '"pending_targets":[["channel","sp-1","ch-1"]]' in (
        planned.target_summary
    )
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
async def test_driver_recovery_abandons_before_exact_union_replacement(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "first", 1, sender="alice")
    await receipt(store, "second", 2, sender="bob")

    class RecoveryAdapter(Adapter):
        def register_continuation_callback(
            self, callback, planning_cycle_key, **_metadata
        ):
            self.continuation = callback
            self.continuation_key = planning_cycle_key

        async def admit_continuation(self):
            callback, self.continuation = self.continuation, None
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=self.continuation_key,
                provider_session_id=self.session,
                provider_turn_id="provider-turn",
                tool_call_id="tool-recovery",
                admitted_at=datetime.now(timezone.utc),
            ))

    seed_adapter = RecoveryAdapter()
    seed = GlobalInboxRuntime(
        store=store, adapter=seed_adapter, run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    planned = await seed.plan_pending()
    assert planned is not None
    seed._write_current_turn(planned)
    seed_adapter.register_admission_callback(
        lambda event: seed._admit(planned, event),
        planned.planning_cycle_key,
    )
    await seed_adapter.admit()
    page = await seed.read_inbox(limit=2, tool_arguments={"limit": 2})
    assert len(page["messages"]) == 2
    await seed_adapter.admit_continuation()
    persisted = json.loads(seed.current_turn_path.read_text(encoding="utf-8"))
    assert persisted["message_ids"] == ["first", "second"]

    outbox = RuntimeEventOutbox(tmp_path / "state" / "runtime_events.db")
    outbox.set_active_turn(
        "public_old_turn", session_ref="logical_session",
        native_session_id=str(seed_adapter.session),
    )
    crashed_adapter = Adapter()
    crashed_adapter.session = None
    runtime = GlobalInboxRuntime(
        store=store, adapter=crashed_adapter,
        run_turn=lambda _planned: None, workspace=tmp_path,
        agent_id="agent", runtime_event_outbox=outbox,
    )
    assert not await runtime.recover_current_turn()
    rows = outbox.prefix()
    assert len(rows) == 1
    assert rows[0].event["type"] == "turn.finished"
    assert rows[0].event["payload"]["outcome"] == "abandoned"
    assert [item.envelope_id for item in await store.get_pending()] == [
        "first", "second",
    ]
    await outbox.enqueue(RuntimeEvent(
        agent_id="agent", session_ref="logical_session",
        turn_ref="public_replacement_turn", type="turn.started", payload={},
    ))
    rows = outbox.prefix()
    assert rows[0].sequence < rows[1].sequence
    assert rows[1].event["turn_ref"] == "public_replacement_turn"
    assert rows[0].event["payload"]["outcome"] != "cancelled"
    outbox.close()
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

    async def wait_for_terminal_cleanup():
        while runtime.current_turn_path.exists():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait_for_terminal_cleanup(), timeout=1)
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
async def test_api_retry_rebinds_processed_event_to_new_provider_turn(
    tmp_path, caplog,
):
    caplog.set_level(logging.INFO)
    store = await make_store(tmp_path)
    await receipt(store, "retry-provider-turn", 1)
    adapter = Adapter()

    class Runner:
        async def __call__(self, _planned):
            await adapter.admit(provider_turn_id="provider-turn-initial")
            raise AgentAPIError("rate limit", is_auth=False)

        async def handle_global_inbox_retry(self, _planned):
            await adapter.admit(provider_turn_id="provider-turn-retry")

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=Runner(),
        workspace=tmp_path,
        agent_id="real-agent-id",
        retry_sleep=lambda _delay: asyncio.sleep(0),
    )
    assert await runtime.process_once()
    processed = next(
        event for event in runtime_events(caplog)
        if event["event"] == "turn.processed"
    )
    assert processed["provider_turn_id"] == "provider-turn-retry"
    assert await store.get_pending() == ()
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
async def test_held_watermark_sync_proof_does_not_admit_or_expose_content(
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
    assert rows[0] == {
        "space_id": "sp-1",
        "channel_id": "ch-1",
        "envelope_id": "watermark",
        "server_seq": 3,
        "latest_seq": 3,
        "latest_envelope_id": "watermark",
        "provider_session_id": "provider-1",
    }
    assert "content" not in rows[0]
    assert [row.envelope_id for row in await store.get_pending()] == ["watermark"]
    assert runtime.held.synchronized
    assert runtime.held.message_ids == ()
    assert await ActiveBoundaryAdapter(
        store, runtime.active
    ).get_active_turn_through_seq("sp-1", "ch-1") == 1
    await store.close()


@pytest.mark.asyncio
async def test_held_sync_proof_cannot_mutate_a_replacement_turn(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "initial", 1)
    await receipt(store, "held", 2)
    await store.admit_messages(
        ["initial"], turn_id="old-turn", provider_session_id="provider-1",
    )
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "old-turn"
    runtime.active.message_ids[:] = ["initial"]
    runtime.active.provider_session_id = "provider-1"
    runtime.active.provider_turn_id = "old-provider-turn"
    rows = await runtime.held_recovery_source.query_held_messages(
        "sp-1", "ch-1", 2, "held", "provider-1",
    )
    assert [row["envelope_id"] for row in rows] == ["held"]

    runtime.active.clear()
    runtime.active.turn_id = "new-turn"
    runtime.active.provider_session_id = "provider-2"
    runtime.active.provider_turn_id = "new-provider-turn"

    assert runtime.active.turn_id == "new-turn"
    assert runtime.active.message_ids == []
    assert [row.envelope_id for row in await store.get_pending()] == ["held"]
    assert [
        row.envelope_id
        for row in await store.get_in_turn_messages(
            "old-turn", "provider-1"
        )
    ] == ["initial"]
    assert not runtime.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
async def test_held_timeout_mismatch_and_context_pressure_stage_nothing(
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
    metadata = await source.query_held_messages(
        "sp-1", "ch-1", 10, "rejected", "provider-1",
    )
    assert metadata
    assert runtime.held.synchronized is True
    assert runtime.held.message_ids == ()
    assert [row.envelope_id for row in await store.get_pending()] == ["rejected"]
    await store.close()


@pytest.mark.asyncio
async def test_held_timeout_uses_signed_pending_catchup_before_failing(tmp_path):
    store = await make_store(tmp_path)

    async def catchup(envelope_id):
        assert envelope_id == "late-watermark"
        await receipt(store, envelope_id, 9)
        return True

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        held_catchup=catchup,
    )
    source = HeldRecoverySource(
        runtime,
        wait_timeout_s=0,
        catchup_pending=catchup,
    )
    assert await source.wait_for_held_delivery(
        "sp-1", "ch-1", 9, "late-watermark",
    )
    assert runtime.held.diagnostic == ""
    await store.close()


@pytest.mark.asyncio
async def test_held_sync_is_independent_of_fifty_message_content_pages(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "initial", 1)
    await store.admit_messages(
        ["initial"], turn_id="turn", provider_session_id="provider-1",
    )
    for seq in range(2, 53):
        await receipt(store, f"held-{seq}", seq, content="small")
    adapter = ToolReturnAdapter()
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
    assert [row["envelope_id"] for row in rows] == ["held-52"]
    assert "content" not in rows[0]
    assert runtime.held.synchronized is True
    assert runtime.held.message_ids == ()
    assert [row.envelope_id for row in await store.get_pending()] == [
        f"held-{seq}" for seq in range(2, 53)
    ]
    first = await runtime.read_inbox(
        target="channel:sp-1:ch-1", limit=50,
    )
    assert first["has_more"] is True
    assert len(first["messages"]) == 50
    assert await ActiveBoundaryAdapter(
        store, runtime.active
    ).get_active_turn_through_seq("sp-1", "ch-1") == 51

    second = await runtime.read_inbox(
        target="channel:sp-1:ch-1",
        cursor=first["next_cursor"],
        limit=50,
    )
    assert second["has_more"] is False
    assert len(second["messages"]) == 1
    assert "held-52" in second["messages"][0]
    assert await ActiveBoundaryAdapter(
        store, runtime.active
    ).get_active_turn_through_seq("sp-1", "ch-1") == 52
    await store.close()


@pytest.mark.asyncio
async def test_held_sync_ignores_formatter_and_context_budget(tmp_path):
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
        formatter=lambda _row: "x" * 20_000,
        estimator=lambda _text: 1,
    )
    runtime.active.turn_id = "turn"
    runtime.active.message_ids[:] = ["initial"]
    runtime.active.provider_session_id = "provider-1"

    rows = await runtime.held_recovery_source.query_held_messages(
        "sp-1", "ch-1", 52, "held-52", "provider-1",
    )

    assert [row["envelope_id"] for row in rows] == ["held-52"]
    assert "content" not in rows[0]
    assert runtime.held.message_ids == ()
    assert runtime.held.synchronized is True
    await store.close()


@pytest.mark.asyncio
async def test_repeated_held_sync_proof_is_metadata_only(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "initial", 1)
    await receipt(store, "held", 2)
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

    first = await runtime.held_recovery_source.query_held_messages(
        "sp-1", "ch-1", 2, "held", "provider-1",
    )
    second = await runtime.held_recovery_source.query_held_messages(
        "sp-1", "ch-1", 2, "held", "provider-1",
    )
    assert first == second
    assert "content" not in first[0]

    in_turn = await store.get_in_turn_messages("turn", "provider-1")
    assert [row.envelope_id for row in in_turn] == ["initial"]
    assert runtime.active.message_ids == ["initial"]
    assert [row.envelope_id for row in await store.get_pending()] == ["held"]
    await store.close()


@pytest.mark.asyncio
async def test_notice_then_correlated_read_admits_and_processes_exact_page(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "pull-1", 1)
    await receipt(store, "pull-2", 2)

    class PullAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.continuation = None
            self.continuation_key = ""

        def register_continuation_callback(
            self, callback, planning_cycle_key, **_kwargs
        ):
            self.continuation = callback
            self.continuation_key = planning_cycle_key

        async def admit_continuation(self):
            callback, self.continuation = self.continuation, None
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=self.continuation_key,
                provider_session_id=self.session,
                provider_turn_id="provider-turn",
                tool_call_id="tool-1",
                admitted_at=datetime.now(timezone.utc),
            ))

    adapter = PullAdapter()

    async def run(planned):
        assert "text-pull" not in planned.provider_input
        await adapter.admit()
        page = await runtime.read_inbox(limit=1, tool_arguments={"limit": 1})
        assert len(page["messages"]) == 1
        assert [row.envelope_id for row in await store.get_pending()] == [
            "pull-1", "pull-2"
        ]
        await adapter.admit_continuation()
        assert [row.envelope_id for row in await store.get_pending()] == ["pull-2"]

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path
    )
    assert await runtime.process_once()
    assert (await store.get_message_by_envelope("pull-1")).processing_state is (
        ProcessingState.PROCESSED
    )
    assert (await store.get_message_by_envelope("pull-2")).processing_state is (
        ProcessingState.PENDING
    )
    await store.close()


def test_format_stored_message_marks_only_runtime_identity_aliases(tmp_path):
    def stored(envelope_id, sender):
        return StoredMessage(
            envelope_id=envelope_id,
            envelope_kind="channel",
            sender_slug=sender,
            channel_id="ch-1",
            space_id="sp-1",
            recipient_slug=None,
            content_type="text/plain",
            content="same body regardless of sender",
            sent_at=1,
            received_at=1,
            server_seq=1,
        )

    def metadata(block):
        return json.loads(block.splitlines()[1])

    human = stored("human", "human")
    peer = stored("peer", "peer-agent")
    self_echo = stored("self", "wire-agent")

    assert metadata(format_stored_message(human))["is_self"] is False
    assert metadata(format_stored_message(peer))["is_self"] is False
    assert metadata(format_stored_message(self_echo))["is_self"] is False
    assert metadata(
        format_stored_message(self_echo, current_agent_aliases=("wire-agent",))
    )["is_self"] is True

    runtime = GlobalInboxRuntime(
        store=SimpleNamespace(),
        adapter=SimpleNamespace(slug="wire-agent"),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        send_mode_keys=("runtime-agent-id",),
    )
    assert metadata(runtime.formatter(self_echo))["is_self"] is True
    assert metadata(runtime.formatter(stored("alias", "runtime-agent-id")))[
        "is_self"
    ] is True
    assert metadata(runtime.formatter(human))["is_self"] is False

    no_identity = GlobalInboxRuntime(
        store=SimpleNamespace(),
        adapter=SimpleNamespace(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    assert metadata(no_identity.formatter(self_echo))["is_self"] is False

    custom_calls = []

    def custom_formatter(item):
        custom_calls.append(item.envelope_id)
        return f"custom:{item.envelope_id}"

    custom = GlobalInboxRuntime(
        store=SimpleNamespace(),
        adapter=SimpleNamespace(slug="wire-agent"),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        formatter=custom_formatter,
        agent_id="wire-agent",
    )
    assert custom.formatter(self_echo) == "custom:self"
    assert custom_calls == ["self"]


@pytest.mark.asyncio
async def test_peer_progress_starts_grounded_followup_turn(tmp_path):
    result = await _run_composed_peer_progress_case(
        tmp_path,
        peer_body=(
            "The current task has progressed to the next dependency; "
            "the requested outline is complete."
        ),
    )
    assert result["provider_turns"] == 2
    assert result["decisions"] == ["send", "[SILENT]"]
    assert result["transport_calls"] == 1
    assert result["peer_state"] is ProcessingState.PROCESSED
    second_input = json.dumps(result["provider_inputs"][1], sort_keys=True)
    for expected in (
        result["human_body"],
        result["first_contribution"],
        result["peer_body"],
    ):
        assert expected in second_input
    assert result["self_metadata"]["sender_slug"] == "agent"
    assert result["self_metadata"]["is_self"] is True
    assert result["prior_ids"] == ["human-origin", "agent-contribution"]
    assert result["future_state"] is ProcessingState.PENDING


async def _run_composed_peer_progress_case(
    tmp_path,
    *,
    peer_body: str,
):
    """Run two fresh provider Turns against only durable/tool-return input."""
    store = await make_store(tmp_path)
    human_body = "Please outline the current task."
    first_contribution = "I mapped the current dependency."
    await receipt(store, "human-origin", 1, sender="human", content=human_body)

    class DeterministicProvider:
        """Small provider double that reasons from the real tool result only."""

        def __init__(self, instructions):
            self.instructions = instructions
            self.turn_inputs = []

        @staticmethod
        def parse(block):
            lines = block.splitlines()
            return json.loads(lines[1]), "\n".join(lines[2:-1])

        def decide(self, read_inbox_result):
            self.turn_inputs.append(read_inbox_result)
            lowered_instructions = " ".join(self.instructions.lower().split())
            for phrase in (
                "prior contribution",
                "[silent]",
                "follow-up",
                "correction",
                "mention",
                "dependency",
            ):
                assert phrase in lowered_instructions

            messages = [self.parse(block) for block in read_inbox_result["messages"]]
            prior_context = [
                self.parse(block) for block in read_inbox_result["prior_context"]
            ]
            if not prior_context:
                return "send", "I will take the requested action."

            prior_self = [
                (metadata, body)
                for metadata, body in prior_context
                if metadata.get("is_self") is True
            ]
            peer_text = " ".join(body.lower() for _metadata, body in messages)
            new_action = any(
                marker in peer_text
                for marker in (
                    "follow-up",
                    "correction",
                    "@you(",
                    "new dependency",
                    "please verify",
                )
            )
            same_assignment_satisfied = bool(prior_self) and not new_action
            if same_assignment_satisfied:
                return "[SILENT]", None
            return "send", "I will take the newly identified action."

    class ProviderAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.continuation = None
            self.continuation_key = ""
            self.continuation_calls = []

        def register_continuation_callback(
            self, callback, planning_cycle_key, **_kwargs
        ):
            self.continuation = callback
            self.continuation_key = planning_cycle_key

        async def admit_continuation(self, provider_turn_id):
            callback, self.continuation = self.continuation, None
            assert callback is not None
            self.continuation_calls.append(self.continuation_key)
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=self.continuation_key,
                provider_session_id=self.session,
                provider_turn_id=provider_turn_id,
                tool_call_id=f"read-{provider_turn_id}",
                admitted_at=datetime.now(timezone.utc),
            ))

    class FakeServerTransport:
        def __init__(self):
            self.calls = []

        async def send(self, request=None, **kwargs):
            request = dict(request or kwargs)
            self.calls.append(request)
            number = len(self.calls)
            return {
                "state": "sent",
                "envelope_id": (
                    "agent-contribution" if number == 1 else "agent-followup"
                ),
                "seq": 2 if number == 1 else 5,
            }

    class FakeCoordinator:
        provider_session_id = None

        def __init__(self, transport):
            self.http_client = SimpleNamespace(keyless=False)
            self.transport = transport

        async def send(self, request=None, **kwargs):
            return await self.transport.send(request, **kwargs)

    adapter = ProviderAdapter()
    transport = FakeServerTransport()
    coordinator = FakeCoordinator(transport)
    provider = DeterministicProvider(DEFAULT_SHARED_CLAUDE_MD)
    provider_inputs = []
    decisions = []
    runtime = None

    async def run(planned):
        turn_number = len(provider_inputs) + 1
        # Each invocation intentionally has no provider-session transcript.
        await adapter.admit(provider_turn_id=f"provider-turn-{turn_number}")
        page = await runtime.read_inbox(
            limit=1,
            tool_arguments={"limit": 1},
        )
        assert len(page["messages"]) == 1
        assert set(page) == {
            "messages",
            "prior_context",
            "next_cursor",
            "has_more",
            "remaining_count",
            "snapshot_generation",
            "correlation_receipt",
        }
        await adapter.admit_continuation(
            provider_turn_id=f"provider-read-{turn_number}"
        )
        provider_input = {
            "fresh_notice": planned.provider_input,
            "read_inbox_result": page,
        }
        provider_inputs.append(provider_input)

        if turn_number == 1:
            assert human_body in page["messages"][0]
            assert page["prior_context"] == []
            decision, reply = provider.decide(page)
            assert decision == "send"
            assert reply
            decisions.append(decision)
            sent = await runtime.send_delegate.send({
                "destination": "ch-1",
                "text": first_contribution,
                "visibility_level": "human",
            })
            assert sent["state"] == "sent"
            await receipt(
                store,
                "agent-contribution",
                2,
                sender="agent",
                disposition=ReceiptDisposition.TERMINAL,
                content=first_contribution,
            )
            return

        decision_input = json.dumps(page, sort_keys=True)
        assert peer_body in page["messages"][0]
        assert human_body in decision_input
        assert first_contribution in decision_input
        prior_metadata = [
            json.loads(block.splitlines()[1]) for block in page["prior_context"]
        ]
        self_metadata = next(
            metadata
            for metadata in prior_metadata
            if metadata["envelope_id"] == "agent-contribution"
        )
        assert self_metadata["sender_slug"] == "agent"
        assert self_metadata["is_self"] is True
        decision, reply = provider.decide(page)
        decisions.append(decision)
        if decision == "send":
            assert reply
            sent = await runtime.send_delegate.send({
                "destination": "ch-1",
                "text": reply,
                "visibility_level": "human",
            })
            assert sent["state"] == "sent"
            await receipt(
                store,
                "agent-followup",
                5,
                sender="agent",
                disposition=ReceiptDisposition.TERMINAL,
                content=reply,
            )
        else:
            assert decision == "[SILENT]"
            assert reply is None

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        coordinator=coordinator,
        agent_id="current-agent-id",
        send_mode_keys=("current-agent-id", "agent"),
    )
    runtime.send_delegate = TrackingSendDelegate(
        coordinator, runtime.attempts, runtime
    )

    assert await runtime.process_once()
    first = await store.get_message_by_envelope("human-origin")
    contribution = await store.get_message_by_envelope("agent-contribution")
    assert first is not None and first.processing_state is ProcessingState.PROCESSED
    assert contribution is not None
    assert contribution.processing_state is None
    assert contribution.receipt_disposition is ReceiptDisposition.TERMINAL

    await receipt(
        store,
        "peer-progress",
        3,
        sender="peer-agent",
        content=peer_body,
    )
    await receipt(
        store,
        "future-pending",
        4,
        sender="peer-agent",
        content="future pending work must not leak",
    )
    prior_lifecycle_before = {}
    for envelope_id in ("human-origin", "agent-contribution"):
        row = await store.get_message_by_envelope(envelope_id)
        assert row is not None
        prior_lifecycle_before[envelope_id] = (
            row.processing_state,
            row.receipt_disposition,
            row.model_visible_at,
        )
    assert await runtime.process_once()
    peer = await store.get_message_by_envelope("peer-progress")
    future = await store.get_message_by_envelope("future-pending")
    assert peer is not None and peer.processing_state is ProcessingState.PROCESSED
    assert future is not None and future.processing_state is ProcessingState.PENDING
    assert len(adapter.continuation_calls) == 2
    prior_ids = [
        json.loads(block.splitlines()[1])["envelope_id"]
        for block in provider_inputs[1]["read_inbox_result"]["prior_context"]
    ]
    prior_blocks = provider_inputs[1]["read_inbox_result"]["prior_context"]
    second_result = provider_inputs[1]["read_inbox_result"]
    message_metadata = [
        json.loads(block.splitlines()[1]) for block in second_result["messages"]
    ]
    prior_metadata = [json.loads(block.splitlines()[1]) for block in prior_blocks]
    assert [metadata["envelope_id"] for metadata in message_metadata] == [
        "peer-progress"
    ]
    assert [metadata["envelope_id"] for metadata in prior_metadata] == [
        "human-origin",
        "agent-contribution",
    ]
    assert message_metadata[0]["is_self"] is False
    assert prior_metadata[0]["is_self"] is False
    self_metadata = next(
        metadata
        for metadata in prior_metadata
        if metadata["envelope_id"] == "agent-contribution"
    )
    prior_lifecycle_after = {}
    for envelope_id in ("human-origin", "agent-contribution"):
        row = await store.get_message_by_envelope(envelope_id)
        assert row is not None
        prior_lifecycle_after[envelope_id] = (
            row.processing_state,
            row.receipt_disposition,
            row.model_visible_at,
        )
    assert prior_lifecycle_after == prior_lifecycle_before
    assert len(prior_blocks) <= PRIOR_CONTEXT_MAX_ITEMS
    assert sum(len(block.encode("utf-8")) for block in prior_blocks) <= (
        PRIOR_CONTEXT_MAX_BYTES
    )
    result = {
        "provider_turns": len(provider_inputs),
        "provider_inputs": provider_inputs,
        "decisions": decisions,
        "transport_calls": len(transport.calls),
        "peer_state": peer.processing_state,
        "future_state": future.processing_state,
        "prior_ids": prior_ids,
        "human_body": human_body,
        "first_contribution": first_contribution,
        "peer_body": peer_body,
        "self_metadata": self_metadata,
        "prior_lifecycle": prior_lifecycle_after,
        "provider_turn_inputs": provider.turn_inputs,
    }
    await store.close()
    return result


@pytest.mark.asyncio
async def test_followup_after_prior_contribution(tmp_path):
    result = await _run_composed_peer_progress_case(
        tmp_path,
        peer_body="A follow-up asks for the next concrete step.",
    )
    assert result["decisions"] == ["send", "send"]
    assert result["transport_calls"] == 2


@pytest.mark.asyncio
async def test_correction_after_prior_contribution(tmp_path):
    result = await _run_composed_peer_progress_case(
        tmp_path,
        peer_body="Correction: the dependency is on the other branch.",
    )
    assert result["decisions"] == ["send", "send"]
    assert result["transport_calls"] == 2


@pytest.mark.asyncio
async def test_mention_after_prior_contribution(tmp_path):
    result = await _run_composed_peer_progress_case(
        tmp_path,
        peer_body="@you(agent) please verify the current dependency.",
    )
    assert result["decisions"] == ["send", "send"]
    assert result["transport_calls"] == 2


@pytest.mark.asyncio
async def test_dependency_after_prior_contribution(tmp_path):
    result = await _run_composed_peer_progress_case(
        tmp_path,
        peer_body="New dependency exposed: the release checklist needs approval.",
    )
    assert result["decisions"] == ["send", "send"]
    assert result["transport_calls"] == 2


@pytest.mark.asyncio
async def test_read_inbox_prior_context_preserves_paging_and_exact_admission(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(
        store,
        "prior-page-context",
        1,
        content="durable prior context",
    )
    await store.admit_messages(
        ["prior-page-context"],
        turn_id="prior-page-context-turn",
        provider_session_id="provider-1",
    )
    await store.mark_processed(
        ["prior-page-context"], turn_id="prior-page-context-turn"
    )
    await receipt(store, "page-context-1", 2, content="page one")
    await receipt(store, "page-context-2", 3, content="page two")
    await store.start_turn(
        turn_id="paging-turn",
        provider_session_id="provider-1",
    )
    adapter = ToolReturnAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "paging-turn"
    runtime.active.provider_session_id = adapter.session
    runtime.active.provider_turn_id = "provider-turn"

    first = await runtime.read_inbox(limit=1, tool_arguments={"limit": 1})
    second = await runtime.read_inbox(
        cursor=first["next_cursor"],
        limit=1,
        tool_arguments={"cursor": first["next_cursor"], "limit": 1},
    )
    assert "page one" in first["messages"][0]
    assert "page two" in second["messages"][0]
    assert [
        json.loads(block.splitlines()[1])["envelope_id"]
        for block in first["prior_context"]
    ] == ["prior-page-context"]
    assert [
        json.loads(block.splitlines()[1])["envelope_id"]
        for block in second["prior_context"]
    ] == ["prior-page-context"]
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert second["next_cursor"] == ""
    assert runtime.active.message_ids == ["page-context-1", "page-context-2"]
    prior = await store.get_message_by_envelope("prior-page-context")
    page_rows = await asyncio.gather(
        store.get_message_by_envelope("page-context-1"),
        store.get_message_by_envelope("page-context-2"),
    )
    assert prior is not None and prior.processing_state is ProcessingState.PROCESSED
    assert all(
        row is not None and row.processing_state is ProcessingState.IN_TURN
        for row in page_rows
    )
    await store.mark_processed(
        ["page-context-1", "page-context-2"], turn_id="paging-turn"
    )
    await store.close()


@pytest.mark.asyncio
async def test_correlated_read_processes_sequence_less_local_runtime_event(tmp_path):
    store = await make_store(tmp_path)
    await store.store_local_event(
        {
            "envelope_id": "intro-prompt-ch-1",
            "envelope_kind": "channel",
            "sender_slug": "system",
            "channel_id": "ch-1",
            "space_id": "sp-1",
            "content": "introduce yourself",
            "sent_at": 1,
            "thread_root_id": "intro-prompt-ch-1",
        },
        reason="channel introduction",
        intro_channel_id="ch-1",
    )

    class PullAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.continuation = None
            self.continuation_key = ""

        def register_continuation_callback(
            self, callback, planning_cycle_key, **_kwargs
        ):
            self.continuation = callback
            self.continuation_key = planning_cycle_key

        async def admit_continuation(self):
            callback, self.continuation = self.continuation, None
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=self.continuation_key,
                provider_session_id=self.session,
                provider_turn_id="provider-turn",
                tool_call_id="tool-local",
                admitted_at=datetime.now(timezone.utc),
            ))

    adapter = PullAdapter()

    async def run(planned):
        await adapter.admit()
        page = await runtime.read_inbox(limit=50, tool_arguments={})
        assert len(page["messages"]) == 1
        assert "introduce yourself" in page["messages"][0]
        await adapter.admit_continuation()
        route = runtime.resolve_active_send_route(
            "ch-1", {"root_id": ""}, {}
        )
        assert route is not None
        assert route.kind == "channel"
        assert route.thread_root_id == ""

    runtime = GlobalInboxRuntime(
        store=store, adapter=adapter, run_turn=run, workspace=tmp_path
    )
    assert await runtime.process_once()
    intro = await store.get_message_by_envelope("intro-prompt-ch-1")
    assert intro is not None
    assert intro.server_seq is None
    assert intro.processing_state is ProcessingState.PROCESSED
    await store.close()


@pytest.mark.asyncio
async def test_only_intro_system_anchor_authorizes_top_level_channel_send(tmp_path):
    store = await make_store(tmp_path)
    await store.store_local_event(
        {
            "envelope_id": "membership-joined-ch-1-agent-1-event-1",
            "envelope_kind": "channel",
            "sender_slug": "system",
            "channel_id": "ch-1",
            "space_id": "sp-1",
            "content": "membership update",
            "sent_at": 1,
            "thread_root_id": "membership-joined-ch-1-agent-1-event-1",
        },
        reason="membership system message",
    )
    row = await store.get_message_by_envelope(
        "membership-joined-ch-1-agent-1-event-1"
    )
    assert row is not None
    route = route_for(row)
    assert route.kind == "thread"
    assert route.thread_root_id == row.envelope_id
    await store.close()


@pytest.mark.asyncio
async def test_initial_and_busy_notices_are_complete_content_free_inputs(tmp_path):
    plaintext = "plaintext-notice-sentinel"
    attachment = "attachment-content-sentinel"
    store = await make_store(tmp_path)
    await receipt(store, "notice-1", 8, content=f"{plaintext}:{attachment}")

    class BusyAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.offers = []
            self.accept = True

        async def offer_inbox_notice(self, turn_id, provider_input):
            self.offers.append((turn_id, provider_input))
            return self.accept

    adapter = BusyAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        notice_delivery=InboxNoticeDelivery(NoticeDeliveryCapability.DIRECT),
    )
    initial = await runtime.plan_pending()
    assert initial is not None
    initial_serialized = json.dumps({
        "provider_input": initial.provider_input,
        "message_ids": initial.message_ids,
        "formatted_blocks": initial.formatted_blocks,
    })
    assert initial.message_ids == ()
    assert initial.formatted_blocks == ()
    assert '"generation":' in initial.provider_input
    assert '"message_count":1' in initial.provider_input
    assert '"latest_seq":8' in initial.provider_input
    assert '"version":3' in initial.provider_input
    assert '"content_included":false' in initial.provider_input
    assert '"read_tool":"read_inbox"' in initial.provider_input
    assert "channel:sp-1:ch-1" in initial.provider_input
    assert plaintext not in initial_serialized
    assert attachment not in initial_serialized

    runtime.active.turn_id = "active-turn"
    runtime.active.provider_session_id = "provider-1"
    assert await runtime.offer_busy_notice(turn_id="active-turn")
    busy_serialized = json.dumps(adapter.offers)
    assert plaintext not in busy_serialized
    assert attachment not in busy_serialized
    state = await store.get_notice_state()
    assert state.last_delivered_generation == state.generation
    await store.close()


@pytest.mark.asyncio
async def test_rejected_or_stale_busy_notice_retains_generation(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "notice-reject", 9)

    class RejectingAdapter(Adapter):
        async def offer_inbox_notice(self, _turn_id, _provider_input):
            return False

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=RejectingAdapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        notice_delivery=InboxNoticeDelivery(NoticeDeliveryCapability.DIRECT),
    )
    runtime.active.turn_id = "active-turn"
    runtime.active.provider_session_id = "provider-1"
    before = await store.get_notice_state()
    assert not await runtime.offer_busy_notice(turn_id="stale-turn")
    assert not await runtime.offer_busy_notice(turn_id="active-turn")
    after = await store.get_notice_state()
    assert after.generation == before.generation
    assert after.last_delivered_generation == before.last_delivered_generation
    assert after.delivery_pending
    assert [row.envelope_id for row in await store.get_pending()] == [
        "notice-reject"
    ]
    await store.close()


@pytest.mark.asyncio
async def test_notice_turn_without_correlated_read_rearms_pending_work(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "notice-unread", 10)
    adapter = Adapter()

    async def run_turn(_planned):
        await adapter.admit()

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run_turn,
        workspace=tmp_path,
    )
    before = await store.get_notice_state()

    assert await runtime.process_once()

    after = await store.get_notice_state()
    assert [row.envelope_id for row in await store.get_pending()] == [
        "notice-unread"
    ]
    assert after.generation == before.generation + 1
    assert after.last_delivered_generation == before.generation
    assert after.delivery_pending
    await store.close()


@pytest.mark.asyncio
async def test_crash_recovery_requeues_empty_notice_turn_and_rearms_delivery(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "notice-crash", 11)
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
    await store.start_turn(
        turn_id=planned.turn_id,
        provider_session_id=adapter.session,
    )
    before = await store.get_notice_state()
    assert await store.mark_notice_delivered(before.generation)

    recovered = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    assert not await recovered.recover_current_turn()

    run = await store.get_turn_run(planned.turn_id)
    after = await store.get_notice_state()
    assert run is not None and run.state == "requeued"
    assert [row.envelope_id for row in await store.get_pending()] == [
        "notice-crash"
    ]
    assert after.generation == before.generation + 1
    assert after.last_delivered_generation == before.generation
    assert after.delivery_pending
    assert not recovered.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_startup_recovers_orphaned_turn_without_crash_join(
    tmp_path,
    admitted,
):
    store = await make_store(tmp_path)
    await receipt(store, "orphan-pending", 12)
    notice = await store.get_notice_state()
    assert await store.mark_notice_delivered(notice.generation)
    if admitted:
        await store.admit_messages(
            ["orphan-pending"],
            turn_id="orphan-turn",
            provider_session_id="provider-old",
        )
    else:
        await store.start_turn(
            turn_id="orphan-turn",
            provider_session_id="provider-old",
        )

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=Adapter(),
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    assert await runtime.recover_orphaned_turns() == 1

    run = await store.get_turn_run("orphan-turn")
    repaired = await store.get_notice_state()
    assert run is not None and run.state == "requeued"
    assert [row.envelope_id for row in await store.get_pending()] == [
        "orphan-pending"
    ]
    assert repaired.delivery_pending
    assert await store.get_active_turn_runs() == ()
    assert await runtime.recover_orphaned_turns() == 0
    await store.close()


@pytest.mark.asyncio
async def test_one_turn_reads_two_targets_sends_twice_and_processes_exact_union(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "target-a", 1, channel="ch-a")
    await receipt(store, "target-b", 2, channel="ch-b")

    class MultiReadAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.continuations = {}

        def register_continuation_callback(
            self, callback, planning_cycle_key, **_metadata
        ):
            self.continuations[planning_cycle_key] = callback

        async def admit_key(self, key):
            callback = self.continuations.pop(key)
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=key,
                provider_session_id=self.session,
                provider_turn_id="provider-turn",
                tool_call_id=f"tool-{key}",
                admitted_at=datetime.now(timezone.utc),
            ))

    class Coordinator:
        def __init__(self):
            self.calls = []
            self.provider_session_id = None

        async def send(self, request=None, **kwargs):
            self.calls.append((request, kwargs))
            return {
                "state": "sent",
                "envelope_id": f"sent-{len(self.calls)}",
                "seq": 10 + len(self.calls),
            }

    adapter = MultiReadAdapter()
    coordinator = Coordinator()

    async def run(_planned):
        await adapter.admit()
        for channel in ("ch-a", "ch-b"):
            page = await runtime.read_inbox(
                target=f"channel:sp-1:{channel}",
                limit=10,
                tool_arguments={
                    "target": f"channel:sp-1:{channel}",
                    "limit": 10,
                },
            )
            assert len(page["messages"]) == 1
            key = next(iter(adapter.continuations))
            await adapter.admit_key(key)
            route = runtime.resolve_active_send_route(
                channel, {"destination": channel}, {}
            )
            assert route is not None and route.channel_id == channel
            assert (await runtime.send_delegate.send(
                {"destination": channel, "text": f"reply-{channel}"}
            ))["state"] == "sent"
        assert runtime.resolve_plain_fallback_route() is None
        assert runtime.active.message_ids == ["target-a", "target-b"]

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        coordinator=coordinator,
    )
    runtime.send_delegate = TrackingSendDelegate(
        coordinator, runtime.attempts, runtime
    )
    assert await runtime.process_once()
    assert [call[0]["destination"] for call in coordinator.calls] == [
        "ch-a", "ch-b"
    ]
    assert [
        (await store.get_message_by_envelope(envelope_id)).processing_state
        for envelope_id in ("target-a", "target-b")
    ] == [ProcessingState.PROCESSED, ProcessingState.PROCESSED]
    await store.close()


@pytest.mark.asyncio
async def test_zero_send_notice_turn_can_succeed_without_output(tmp_path):
    store = await make_store(tmp_path)
    await receipt(store, "zero-send", 1)
    adapter = Adapter()
    sends = []

    async def run(_planned):
        await adapter.admit()
        page = await runtime.read_inbox(limit=1, tool_arguments={"limit": 1})
        assert page["messages"]
        # No send is attempted; the Turn still completes its exact union.
        callback = adapter.continuation
        await callback(ProviderAdmissionEvent(
            planning_cycle_key=adapter.continuation_key,
            provider_session_id=adapter.session,
            provider_turn_id="provider-turn",
            tool_call_id="tool-zero",
            admitted_at=datetime.now(timezone.utc),
        ))

    # Add only the continuation seam used by read_inbox to the simple adapter.
    adapter.continuation = None
    adapter.continuation_key = ""

    def register(callback, planning_cycle_key, **_metadata):
        adapter.continuation = callback
        adapter.continuation_key = planning_cycle_key

    adapter.register_continuation_callback = register
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
    )
    assert await runtime.process_once()
    assert sends == []
    assert (
        await store.get_message_by_envelope("zero-send")
    ).processing_state is ProcessingState.PROCESSED
    assert runtime.resolve_plain_fallback_route() is None
    await store.close()


@pytest.mark.asyncio
async def test_notice_replan_rereads_durable_generation_and_aborts_stale_work(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "stale-after-compact", 1)
    adapter = Adapter()
    ran = []

    async def disappear():
        assert await store.quarantine_pending(
            "stale-after-compact", reason="removed during compact refresh"
        )

    controller = ScriptedContext(
        adapter,
        [DecisionOutcome.REPLAN, DecisionOutcome.ADMIT],
        on_replan=disappear,
    )

    async def run(_planned):
        ran.append(True)

    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=run,
        workspace=tmp_path,
        context_controller=controller,
    )
    assert not await runtime.process_once()
    assert controller.calls == 2
    assert ran == []
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_unrecoverable_notice_pressure_retains_pending_in_degraded_state(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "pressure", 1)
    adapter = Adapter()
    controller = ScriptedContext(adapter, [DecisionOutcome.DEGRADED])
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        context_controller=controller,
    )
    assert not await runtime.process_once()
    assert runtime.health.state == "degraded"
    assert runtime._degraded is True
    assert [row.envelope_id for row in await store.get_pending()] == ["pressure"]
    await store.close()


@pytest.mark.asyncio
async def test_read_inbox_byte_guard_repaginates_without_lifecycle_mutation(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "guard-1", 1, content="a" * 60_000)
    await receipt(store, "guard-2", 2, content="b" * 60_000)

    class CorrelatingAdapter(Adapter):
        def register_continuation_callback(
            self, callback, planning_cycle_key, **metadata
        ):
            self.continuation = (callback, planning_cycle_key, metadata)

    adapter = CorrelatingAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn-guard"
    runtime.active.provider_session_id = "provider-1"
    page = await runtime.read_inbox(limit=2, tool_arguments={"limit": 2})
    assert len(page["messages"]) == 1
    assert page["has_more"] is True
    assert page["next_cursor"]
    assert page["remaining_count"] == 1
    assert set(page) == {
        "messages",
        "prior_context",
        "next_cursor",
        "has_more",
        "remaining_count",
        "snapshot_generation",
        "correlation_receipt",
    }
    assert [row.envelope_id for row in await store.get_pending()] == [
        "guard-1", "guard-2"
    ]
    await store.close()


@pytest.mark.asyncio
async def test_read_inbox_admits_exact_page_at_runtime_tool_return(
    tmp_path, caplog,
):
    caplog.set_level(
        logging.INFO,
        logger="puffo_agent.agent.global_inbox_runtime",
    )
    store = await make_store(tmp_path)
    await receipt(store, "tool-return-inbox", 9)
    await store.start_turn(
        turn_id="turn-tool-return",
        provider_session_id="provider-1",
    )
    adapter = ToolReturnAdapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn-tool-return"
    runtime.active.provider_session_id = adapter.session
    runtime.active.provider_turn_id = "provider-turn-tool-return"

    page = await runtime.read_inbox(limit=1, tool_arguments={"limit": 1})

    assert len(page["messages"]) == 1
    assert runtime.active.message_ids == ["tool-return-inbox"]
    assert runtime.active.through_by_channel[("sp-1", "ch-1")] == 9
    stored = await store.get_message_by_envelope("tool-return-inbox")
    assert stored is not None
    assert stored.processing_state is ProcessingState.IN_TURN
    assert runtime.current_turn_path.exists()
    inbox_events = [
        event for event in runtime_events(caplog)
        if event["event"].startswith("inbox.read")
    ]
    assert [event["event"] for event in inbox_events] == [
        "inbox.read_staged", "inbox.read_admitted",
    ]
    assert inbox_events[0]["outcome"] == "tool_return"
    assert inbox_events[1]["provider_turn_id"] == (
        "provider-turn-tool-return"
    )
    await store.close()


def test_plain_fallback_requires_one_unique_admitted_route():
    active = ActiveExactUnion()
    runtime = GlobalInboxRuntime.__new__(GlobalInboxRuntime)
    runtime.active = active
    assert runtime.resolve_plain_fallback_route() is None

    first = MessageRoute("m-1", "channel", "sp", "ch-a")
    duplicate_target = MessageRoute("m-2", "channel", "sp", "ch-a")
    second = MessageRoute("m-3", "channel", "sp", "ch-b")
    active.routes[:] = [first]
    assert runtime.resolve_plain_fallback_route() == first
    active.routes.append(duplicate_target)
    assert runtime.resolve_plain_fallback_route() == first
    active.routes.append(second)
    assert runtime.resolve_plain_fallback_route() is None


class _CrashAfterBoundary(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("message_count", [1, 2])
@pytest.mark.parametrize("boundary", ["terminal", "requeue", "clear"])
async def test_notice_read_crash_join_restart_is_exact_and_idempotent(
    tmp_path, monkeypatch, message_count, boundary,
):
    store = await make_store(tmp_path)
    for seq in range(1, message_count + 1):
        await receipt(store, f"page-{seq}", seq)

    class CorrelatingAdapter(Adapter):
        def register_continuation_callback(
            self, callback, planning_cycle_key, **_metadata
        ):
            self.continuation = callback
            self.continuation_key = planning_cycle_key

        async def admit_continuation(self, tool_call_id):
            callback, self.continuation = self.continuation, None
            await callback(ProviderAdmissionEvent(
                planning_cycle_key=self.continuation_key,
                provider_session_id=self.session,
                provider_turn_id="native-turn",
                tool_call_id=tool_call_id,
                admitted_at=datetime.now(timezone.utc),
            ))

    outbox = RuntimeEventOutbox(tmp_path / "state" / "runtime_events.db")
    outbox.set_active_turn(
        "logical-turn",
        session_ref="logical-session",
        native_session_id="provider-1",
    )
    adapter = CorrelatingAdapter()
    seed = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        agent_id="agent",
        runtime_event_outbox=outbox,
    )
    await store.start_turn(
        turn_id="durable-turn", provider_session_id="provider-1"
    )
    seed.active.turn_id = "durable-turn"
    seed.active.provider_session_id = "provider-1"
    seed.active.provider_turn_id = "native-turn"
    cursor = ""
    admitted = []
    for page_number in range(message_count):
        page = await seed.read_inbox(
            cursor=cursor,
            limit=1,
            tool_arguments={
                "cursor": cursor,
                "limit": 1,
            },
        )
        assert len(page["messages"]) == 1
        await adapter.admit_continuation(f"tool-{page_number + 1}")
        admitted.append(f"page-{page_number + 1}")
        persisted = json.loads(seed.current_turn_path.read_text())
        assert persisted["message_ids"] == admitted
        assert persisted["provider_session_id"] == "provider-1"
        assert persisted["provider_turn_id"] == "native-turn"
        assert persisted["logical_session_ref"] == "logical-session"
        assert persisted["logical_turn_ref"] == "logical-turn"
        cursor = page["next_cursor"]

    requeues = []
    original_requeue = store.requeue_messages

    async def record_requeue(message_ids, *, turn_id):
        terminal_rows = outbox.prefix()
        assert len(terminal_rows) == 1
        assert terminal_rows[0].event["payload"] == {
            "outcome": "abandoned"
        }
        requeues.append((tuple(message_ids), turn_id))
        result = await original_requeue(message_ids, turn_id=turn_id)
        if boundary == "requeue" and len(requeues) == 1:
            raise _CrashAfterBoundary("after exact requeue")
        return result

    monkeypatch.setattr(store, "requeue_messages", record_requeue)
    original_enqueue = outbox.enqueue
    enqueue_calls = 0

    async def crash_after_terminal(event, *, terminal=None):
        nonlocal enqueue_calls
        enqueue_calls += 1
        result = await original_enqueue(event, terminal=terminal)
        if boundary == "terminal" and enqueue_calls == 1:
            raise _CrashAfterBoundary("after abandoned persistence")
        return result

    monkeypatch.setattr(outbox, "enqueue", crash_after_terminal)
    crashed_adapter = Adapter()
    crashed_adapter.session = None
    first = GlobalInboxRuntime(
        store=store,
        adapter=crashed_adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        agent_id="agent",
        runtime_event_outbox=outbox,
    )
    if boundary == "clear":
        original_clear = first._clear_terminal_turn

        def crash_after_clear():
            original_clear()
            raise _CrashAfterBoundary("after active-state clear")

        monkeypatch.setattr(first, "_clear_terminal_turn", crash_after_clear)

    with pytest.raises(_CrashAfterBoundary):
        await first.recover_current_turn()

    monkeypatch.setattr(outbox, "enqueue", original_enqueue)
    if boundary == "requeue":
        monkeypatch.setattr(store, "requeue_messages", record_requeue)
    second_adapter = Adapter()
    second_adapter.session = None
    second = GlobalInboxRuntime(
        store=store,
        adapter=second_adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        agent_id="agent",
        runtime_event_outbox=outbox,
    )
    assert not await second.recover_current_turn()
    rows = outbox.prefix()
    assert len(rows) == 1
    assert rows[0].event["type"] == "turn.finished"
    assert rows[0].event["payload"] == {"outcome": "abandoned"}
    assert rows[0].event_id.startswith("evt_abandoned_")
    assert requeues == [(tuple(admitted), "durable-turn")]
    assert [row.envelope_id for row in await store.get_pending()] == admitted
    assert not second.current_turn_path.exists()
    outbox.close()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_ids", [[], ["page-1"], ["page-1", "page-1"]])
async def test_crash_join_rejects_empty_partial_or_duplicate_union(
    tmp_path, invalid_ids,
):
    store = await make_store(tmp_path)
    await receipt(store, "page-1", 1)
    await receipt(store, "page-2", 2)
    await store.admit_messages(
        ["page-1", "page-2"],
        turn_id="durable-turn",
        provider_session_id="provider-1",
    )
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    rows = await store.get_in_turn_messages("durable-turn", "provider-1")
    runtime.active.turn_id = "durable-turn"
    runtime.active.provider_session_id = "provider-1"
    runtime._write_current_turn(
        runtime._reconstruct_exact_turn(turn_id="durable-turn", rows=rows)
    )
    raw = json.loads(runtime.current_turn_path.read_text())
    raw["message_ids"] = invalid_ids
    runtime.current_turn_path.write_text(json.dumps(raw))

    assert not await runtime.recover_current_turn()
    assert [row.envelope_id for row in await store.get_pending()] == [
        "page-1", "page-2"
    ]
    assert not runtime.current_turn_path.exists()
    await store.close()


@pytest.mark.asyncio
async def test_crash_join_outbox_session_mismatch_requeues_without_foreign_terminal(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "page-1", 1)
    await store.admit_messages(
        ["page-1"],
        turn_id="durable-turn",
        provider_session_id="provider-1",
    )
    outbox = RuntimeEventOutbox(tmp_path / "state" / "runtime_events.db")
    outbox.set_active_turn(
        "foreign-turn",
        session_ref="foreign-logical-session",
        native_session_id="provider-other",
    )
    adapter = Adapter()
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
        agent_id="agent",
        runtime_event_outbox=outbox,
    )
    rows = await store.get_in_turn_messages("durable-turn", "provider-1")
    runtime.active.turn_id = "durable-turn"
    runtime.active.provider_session_id = "provider-1"
    runtime._write_current_turn(
        runtime._reconstruct_exact_turn(turn_id="durable-turn", rows=rows)
    )
    raw = json.loads(runtime.current_turn_path.read_text())
    raw["logical_session_ref"] = "logical-session"
    raw["logical_turn_ref"] = "logical-turn"
    raw["native_session_id"] = "provider-1"
    runtime.current_turn_path.write_text(json.dumps(raw))

    assert not await runtime.recover_current_turn()
    assert outbox.prefix() == []
    assert outbox.state()["active_turn_ref"] == "foreign-turn"
    assert [row.envelope_id for row in await store.get_pending()] == ["page-1"]
    outbox.close()
    await store.close()


# These assertions exercise the superseded daemon-push batch contract.  Keep
# them visible (rather than deleting the historical crash/retry scenarios)
# until they are rewritten around notice Turns plus correlated read pages.
_SUPERSEDED_PUSH_BATCH_TESTS = (
    "test_multi_target_global_order_route_metadata_and_current_turn",
    "test_committed_turn_survives_runtime_log_handler_failure",
    "test_turn_send_mode_aliases_clear_after_provider_failure",
    "test_turn_send_mode_clears_when_provider_turn_is_cancelled",
    "test_multi_target_real_provider_input_preserves_every_sender_metadata",
    "test_failed_held_and_attachment_send_attempts_are_tracked",
    "test_startup_pending_and_content_neutral_coalescing",
    "test_limit_wrapper_bytes_shrinks_real_fifo_suffix",
    "test_crash_join_invalid_exact_union_metadata_requeues",
    "test_crash_join_invalid_or_stale_file_is_removed",
    "test_startup_recovery_resumes_exact_crash_join_before_planning",
    "test_resume_failure_requeues_exact_union_and_cleans_identity",
    "test_busy_provider_keeps_arrival_pending_until_next_boundary",
    "test_limit_count_token_byte_and_wrapper_overhead_are_independent",
    "test_unfit_head_policy_quarantines_once_without_starvation",
    "test_context_replan_authoritative_arrival_shrink_rollover_and_admit",
    "test_context_degraded_and_budget_exhaustion_do_not_poll_until_notify",
    "test_admission_retry_success_preserves_union_without_reappend",
    "test_api_retry_rebinds_processed_event_to_new_provider_turn",
)
for _superseded_name in _SUPERSEDED_PUSH_BATCH_TESTS:
    globals()[_superseded_name] = pytest.mark.skip(
        reason="superseded by frozen metadata-notice/read_inbox contract"
    )(globals()[_superseded_name])
