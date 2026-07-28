import asyncio
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from puffo_agent.agent.message_store import (
    ChannelRoot,
    LifecycleConflict,
    MessageStore,
    ProcessingState,
    ReceiptDisposition,
    ReceiptWriteStatus,
    StoredMessage,
)
from puffo_agent.crypto.ws_client import PuffoCoreWsClient, TransportOutcome


def _now_ms() -> int:
    return int(time.time() * 1000)


def _temp_store() -> MessageStore:
    d = tempfile.mkdtemp()
    return MessageStore(os.path.join(d, "messages.db"))


def _channel_payload(envelope_id: str, channel_id: str = "ch_1", sent_at: int | None = None, **kwargs):
    return {
        "envelope_id": envelope_id,
        "envelope_kind": "channel",
        "sender_slug": kwargs.get("sender_slug", "alice-0001"),
        "channel_id": channel_id,
        "space_id": kwargs.get("space_id", "sp_1"),
        "content_type": "text/plain",
        "content": kwargs.get("content", f"Message {envelope_id}"),
        "sent_at": sent_at or _now_ms(),
        "thread_root_id": kwargs.get("thread_root_id"),
        "reply_to_id": kwargs.get("reply_to_id"),
    }


def _dm_payload(envelope_id: str, sender: str, recipient: str, sent_at: int | None = None, **kwargs):
    return {
        "envelope_id": envelope_id,
        "envelope_kind": "dm",
        "sender_slug": sender,
        "recipient_slug": recipient,
        "content_type": "text/plain",
        "content": kwargs.get("content", f"DM {envelope_id}"),
        "sent_at": sent_at or _now_ms(),
    }


@pytest.mark.asyncio
async def test_concurrent_open_initializes_once():
    store = _temp_store()

    await asyncio.gather(*(store.open() for _ in range(16)))

    await store.store(_channel_payload("env_concurrent_open"))
    assert await store.has_message("env_concurrent_open")
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_open_across_instances_serializes_schema_migration():
    for round_index in range(8):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "messages.db")
            stores = [MessageStore(db_path) for _ in range(16)]
            try:
                await asyncio.gather(*(store.open() for store in stores))

                envelope_id = f"env_multi_instance_open_{round_index}"
                await stores[0].store(_channel_payload(envelope_id))
                assert await stores[-1].has_message(envelope_id)
            finally:
                await asyncio.gather(
                    *(store.close() for store in stores),
                    return_exceptions=True,
                )


@pytest.mark.asyncio
async def test_concurrent_open_across_processes_retries_wal_transition(tmp_path):
    code = """
import asyncio
import sys
from puffo_agent.agent.message_store import MessageStore

async def main():
    store = MessageStore(sys.argv[1])
    await store.open()
    await store.close()

asyncio.run(main())
"""
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src_dir, env.get("PYTHONPATH", "")) if part
    )

    for round_index in range(3):
        db_path = tmp_path / f"process-{round_index}" / "messages.db"
        processes = [
            await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                str(db_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            for _ in range(12)
        ]
        results = await asyncio.gather(
            *(process.communicate() for process in processes)
        )
        failures = [
            stderr.decode("utf-8", errors="replace")
            for process, (_stdout, stderr) in zip(processes, results, strict=True)
            if process.returncode != 0
        ]
        assert not failures

        store = MessageStore(db_path)
        await store.open()
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_open_releases_schema_write_lock(tmp_path, monkeypatch):
    original = MessageStore._execute_locked_script
    transaction_started = asyncio.Event()

    async def block_after_begin(db, _script):
        await db.execute("BEGIN IMMEDIATE")
        transaction_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        MessageStore,
        "_execute_locked_script",
        staticmethod(block_after_begin),
    )
    db_path = tmp_path / "messages.db"
    interrupted = MessageStore(db_path)
    task = asyncio.create_task(interrupted.open())
    await transaction_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    monkeypatch.setattr(
        MessageStore,
        "_execute_locked_script",
        staticmethod(original),
    )
    replacement = MessageStore(db_path)
    await asyncio.wait_for(replacement.open(), timeout=2.0)
    await replacement.close()


@pytest.mark.asyncio
async def test_store_and_has_message():
    store = _temp_store()
    await store.open()

    assert not await store.has_message("env_1")
    await store.store(_channel_payload("env_1"))
    assert await store.has_message("env_1")

    await store.close()


@pytest.mark.asyncio
async def test_duplicate_insert_ignored():
    store = _temp_store()
    await store.open()

    await store.store(_channel_payload("env_1", content="first"))
    await store.store(_channel_payload("env_1", content="second"))

    msgs = await store.get_channel_history("ch_1")
    assert len(msgs) == 1
    assert msgs[0].content == "first"

    await store.close()


@pytest.mark.asyncio
async def test_channel_history_order():
    store = _temp_store()
    await store.open()

    base = _now_ms()
    await store.store(_channel_payload("env_1", sent_at=base))
    await store.store(_channel_payload("env_2", sent_at=base + 1000))
    await store.store(_channel_payload("env_3", sent_at=base + 2000))

    msgs = await store.get_channel_history("ch_1")
    assert len(msgs) == 3
    assert msgs[0].envelope_id == "env_1"
    assert msgs[2].envelope_id == "env_3"

    await store.close()


@pytest.mark.asyncio
async def test_channel_history_limit():
    store = _temp_store()
    await store.open()

    base = _now_ms()
    for i in range(10):
        await store.store(_channel_payload(f"env_{i}", sent_at=base + i * 1000))

    msgs = await store.get_channel_history("ch_1", limit=3)
    assert len(msgs) == 3
    assert msgs[0].envelope_id == "env_7"
    assert msgs[2].envelope_id == "env_9"

    await store.close()


@pytest.mark.asyncio
async def test_channel_history_before():
    store = _temp_store()
    await store.open()

    base = 1_000_000_000_000
    await store.store(_channel_payload("env_1", sent_at=base))
    await store.store(_channel_payload("env_2", sent_at=base + 1000))
    await store.store(_channel_payload("env_3", sent_at=base + 2000))

    msgs = await store.get_channel_history("ch_1", before=base + 2000)
    assert len(msgs) == 2
    assert msgs[0].envelope_id == "env_1"
    assert msgs[1].envelope_id == "env_2"

    await store.close()


@pytest.mark.asyncio
async def test_channel_filter():
    store = _temp_store()
    await store.open()

    await store.store(_channel_payload("env_1", channel_id="ch_1"))
    await store.store(_channel_payload("env_2", channel_id="ch_2"))
    await store.store(_channel_payload("env_3", channel_id="ch_1"))

    msgs = await store.get_channel_history("ch_1")
    assert len(msgs) == 2
    assert all(m.channel_id == "ch_1" for m in msgs)

    await store.close()


@pytest.mark.asyncio
async def test_dm_history():
    store = _temp_store()
    await store.open()

    base = _now_ms()
    await store.store(_dm_payload("env_1", "alice-0001", "bob-0001", sent_at=base))
    await store.store(_dm_payload("env_2", "bob-0001", "alice-0001", sent_at=base + 1000))
    await store.store(_dm_payload("env_3", "alice-0001", "carol-0001", sent_at=base + 2000))

    msgs = await store.get_dm_history("bob-0001")
    assert len(msgs) == 2
    assert msgs[0].envelope_id == "env_1"
    assert msgs[1].envelope_id == "env_2"

    await store.close()


@pytest.mark.asyncio
async def test_dm_history_before():
    store = _temp_store()
    await store.open()

    base = 1_000_000_000_000
    await store.store(_dm_payload("env_1", "alice", "bob", sent_at=base))
    await store.store(_dm_payload("env_2", "bob", "alice", sent_at=base + 1000))

    msgs = await store.get_dm_history("bob", before=base + 1000)
    assert len(msgs) == 1
    assert msgs[0].envelope_id == "env_1"

    await store.close()


@pytest.mark.asyncio
async def test_cleanup():
    store = _temp_store()
    await store.open()

    old_time = _now_ms() - 100 * 86_400_000
    await store.store(_channel_payload("env_old", sent_at=old_time), received_at=old_time)
    await store.store(_channel_payload("env_new", sent_at=_now_ms()))

    count = await store.cleanup(retention_days=90)
    assert count == 1
    assert not await store.has_message("env_old")
    assert await store.has_message("env_new")

    await store.close()


@pytest.mark.asyncio
async def test_json_content_roundtrip():
    store = _temp_store()
    await store.open()

    payload = _channel_payload("env_1", content={"text": "hello", "attachments": [1, 2]})
    await store.store(payload)

    msgs = await store.get_channel_history("ch_1")
    assert msgs[0].content == {"text": "hello", "attachments": [1, 2]}

    await store.close()


@pytest.mark.asyncio
async def test_string_content_roundtrip():
    store = _temp_store()
    await store.open()

    payload = _channel_payload("env_1", content="plain text")
    await store.store(payload)

    msgs = await store.get_channel_history("ch_1")
    assert msgs[0].content == "plain text"

    await store.close()


@pytest.mark.asyncio
async def test_threading_fields():
    store = _temp_store()
    await store.open()

    payload = _channel_payload(
        "env_1", thread_root_id="env_root", reply_to_id="env_parent",
    )
    await store.store(payload)

    msgs = await store.get_channel_history("ch_1")
    assert msgs[0].thread_root_id == "env_root"
    assert msgs[0].reply_to_id == "env_parent"

    await store.close()


@pytest.mark.asyncio
async def test_auto_open():
    store = _temp_store()
    await store.store(_channel_payload("env_1"))
    assert await store.has_message("env_1")
    await store.close()


@pytest.mark.asyncio
async def test_for_agent_factory():
    store = MessageStore.for_agent("test-agent-123")
    assert "test-agent-123" in str(store.db_path)
    assert str(store.db_path).endswith("messages.db")


# ── get_channel_roots ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_channel_roots_excludes_replies_and_counts_them():
    """Only thread_root_id IS NULL rows are returned; the
    ``reply_count`` field is the running count of replies."""
    store = _temp_store()
    await store.open()
    # Two roots in the same channel.
    await store.store(_channel_payload("root_a", sent_at=100))
    await store.store(_channel_payload("root_b", sent_at=200))
    # Three replies on root_a, one on root_b.
    for i, rt in enumerate(["root_a", "root_a", "root_a"], start=1):
        await store.store(_channel_payload(
            f"reply_a_{i}", sent_at=100 + i, thread_root_id=rt,
        ))
    await store.store(_channel_payload(
        "reply_b_1", sent_at=210, thread_root_id="root_b",
    ))

    roots = await store.get_channel_roots("ch_1")
    assert [r.message.envelope_id for r in roots] == ["root_a", "root_b"]
    counts = {r.message.envelope_id: r.reply_count for r in roots}
    assert counts == {"root_a": 3, "root_b": 1}
    await store.close()


@pytest.mark.asyncio
async def test_channel_roots_since_envelope_id_filters_by_sent_at():
    """``since=<envelope_id>`` resolves to that envelope's sent_at
    and applies an exclusive lower bound."""
    store = _temp_store()
    await store.open()
    await store.store(_channel_payload("root_old", sent_at=100))
    await store.store(_channel_payload("root_mid", sent_at=200))
    await store.store(_channel_payload("root_new", sent_at=300))

    roots = await store.get_channel_roots(
        "ch_1", since_envelope_id="root_old",
    )
    # Strictly after root_old's sent_at, so root_mid + root_new.
    assert [r.message.envelope_id for r in roots] == ["root_mid", "root_new"]
    await store.close()


@pytest.mark.asyncio
async def test_channel_roots_before_and_after_ts():
    """``before`` / ``after`` are exclusive ms-epoch bounds."""
    store = _temp_store()
    await store.open()
    for env_id, ts in [
        ("r_1", 100), ("r_2", 200), ("r_3", 300), ("r_4", 400),
    ]:
        await store.store(_channel_payload(env_id, sent_at=ts))

    roots = await store.get_channel_roots(
        "ch_1", after_ts=100, before_ts=400,
    )
    assert [r.message.envelope_id for r in roots] == ["r_2", "r_3"]
    await store.close()


# ── get_thread_messages ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_messages_includes_root_and_replies():
    store = _temp_store()
    await store.open()
    await store.store(_channel_payload("root_x", sent_at=100))
    await store.store(_channel_payload(
        "reply_1", sent_at=110, thread_root_id="root_x",
    ))
    await store.store(_channel_payload(
        "reply_2", sent_at=120, thread_root_id="root_x",
    ))
    # An unrelated root + reply mustn't leak in.
    await store.store(_channel_payload("root_other", sent_at=130))
    await store.store(_channel_payload(
        "other_reply", sent_at=140, thread_root_id="root_other",
    ))

    msgs = await store.get_thread_messages("root_x")
    assert [m.envelope_id for m in msgs] == ["root_x", "reply_1", "reply_2"]
    await store.close()


@pytest.mark.asyncio
async def test_thread_messages_since_filter():
    store = _temp_store()
    await store.open()
    await store.store(_channel_payload("root_x", sent_at=100))
    await store.store(_channel_payload(
        "reply_1", sent_at=110, thread_root_id="root_x",
    ))
    await store.store(_channel_payload(
        "reply_2", sent_at=120, thread_root_id="root_x",
    ))

    msgs = await store.get_thread_messages(
        "root_x", since_envelope_id="reply_1",
    )
    # Strictly after reply_1's sent_at → only reply_2.
    assert [m.envelope_id for m in msgs] == ["reply_2"]
    await store.close()


@pytest.mark.asyncio
async def test_is_encrypted_defaults_true_when_absent():
    store = _temp_store()
    await store.store(_channel_payload("env_enc"))  # payload carries no is_encrypted
    msg = await store.get_message_by_envelope("env_enc")
    assert msg is not None and msg.is_encrypted is True
    await store.close()


@pytest.mark.asyncio
async def test_is_encrypted_false_for_plaintext():
    store = _temp_store()
    p = _channel_payload("env_plain")
    p["is_encrypted"] = False
    await store.store(p)
    msg = await store.get_message_by_envelope("env_plain")
    assert msg is not None and msg.is_encrypted is False
    await store.close()


@pytest.mark.asyncio
async def test_is_encrypted_migration_backfills_legacy_rows_true():
    import sqlite3

    d = tempfile.mkdtemp()
    path = os.path.join(d, "messages.db")
    # Legacy schema without is_encrypted + a pre-fix (E2EE) row.
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (envelope_id TEXT PRIMARY KEY, envelope_kind TEXT NOT NULL, "
        "sender_slug TEXT NOT NULL, channel_id TEXT, space_id TEXT, recipient_slug TEXT, "
        "content_type TEXT NOT NULL DEFAULT 'text/plain', content TEXT NOT NULL, "
        "sent_at INTEGER NOT NULL, received_at INTEGER NOT NULL, thread_root_id TEXT, reply_to_id TEXT)"
    )
    con.execute(
        "INSERT INTO messages (envelope_id, envelope_kind, sender_slug, channel_id, "
        "content_type, content, sent_at, received_at) VALUES "
        "('env_old','channel','alice-0001','ch_1','text/plain','old msg',1,1)"
    )
    con.commit()
    con.close()

    store = MessageStore(path)
    await store.open()  # runs the ALTER-TABLE migration
    msg = await store.get_message_by_envelope("env_old")
    assert msg is not None and msg.is_encrypted is True  # legacy row backfilled encrypted
    await store.close()


@pytest.mark.asyncio
async def test_has_dm_from():
    store = _temp_store()
    await store.open()

    assert not await store.has_dm_from("alice-1")
    await store.store(_channel_payload("env_ch", sender_slug="alice-1"))
    assert not await store.has_dm_from("alice-1")  # channel post doesn't count
    await store.store(_dm_payload("env_dm", "alice-1", "agent-9"))
    assert await store.has_dm_from("alice-1")
    assert not await store.has_dm_from("")

    await store.close()


# ── Agent-wide Inbox receipts, ordering, and lifecycle ───────────


@pytest.mark.asyncio
async def test_schema_migration_is_idempotent_and_checks_new_columns():
    import sqlite3

    d = tempfile.mkdtemp()
    path = os.path.join(d, "messages.db")
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (envelope_id TEXT PRIMARY KEY, envelope_kind TEXT NOT NULL, "
        "sender_slug TEXT NOT NULL, channel_id TEXT, space_id TEXT, recipient_slug TEXT, "
        "content_type TEXT NOT NULL DEFAULT 'text/plain', content TEXT NOT NULL, "
        "sent_at INTEGER NOT NULL, received_at INTEGER NOT NULL, thread_root_id TEXT, reply_to_id TEXT)"
    )
    con.execute(
        "INSERT INTO messages VALUES "
        "('legacy','channel','alice','ch','sp',NULL,'text/plain','old',1,1,NULL,NULL)"
    )
    con.commit()
    con.close()

    store = MessageStore(path)
    await store.open()
    await store.close()
    await store.open()
    db = await store._ensure_db()
    async with db.execute("SELECT * FROM messages WHERE envelope_id = 'legacy'") as cur:
        row = await cur.fetchone()
    assert row["is_encrypted"] == 1
    for column in (
        "server_seq", "receipt_disposition", "receipt_reason", "processing_state",
        "processing_turn_id", "model_visible_at", "processed_at", "local_ordinal",
        "after_server_seq",
    ):
        assert row[column] is None
    with pytest.raises(Exception):
        await db.execute(
            "UPDATE messages SET receipt_disposition = 'invalid' WHERE envelope_id = 'legacy'"
        )
    await db.rollback()
    with pytest.raises(Exception):
        await db.execute(
            "UPDATE messages SET processing_state = 'invalid' WHERE envelope_id = 'legacy'"
        )
    await db.rollback()
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='thread_processing_state'"
    ) as cur:
        assert await cur.fetchone() is not None
    await store.close()


@pytest.mark.asyncio
async def test_exact_baseline_schema_migration_keeps_inbox_fields_null():
    import sqlite3

    d = tempfile.mkdtemp()
    path = os.path.join(d, "messages.db")
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (envelope_id TEXT PRIMARY KEY, envelope_kind TEXT NOT NULL, "
        "sender_slug TEXT NOT NULL, channel_id TEXT, space_id TEXT, recipient_slug TEXT, "
        "content_type TEXT NOT NULL DEFAULT 'text/plain', content TEXT NOT NULL, "
        "sent_at INTEGER NOT NULL, received_at INTEGER NOT NULL, thread_root_id TEXT, "
        "reply_to_id TEXT, is_encrypted INTEGER NOT NULL DEFAULT 1)"
    )
    con.execute(
        "INSERT INTO messages VALUES "
        "('baseline','channel','alice','ch','sp',NULL,'text/plain','old',1,1,NULL,NULL,0)"
    )
    con.commit()
    con.close()
    store = MessageStore(path)
    await store.open()
    msg = await store.get_message_by_envelope("baseline")
    assert msg is not None and msg.is_encrypted is False
    assert msg.server_seq is None
    assert msg.receipt_disposition is None
    assert msg.processing_state is None
    await store.close()


@pytest.mark.asyncio
async def test_receipt_ack_mapping_idempotency_conflicts_and_gated_promotion():
    store = _temp_store()
    eligible = await store.store_receipt(
        _channel_payload("eligible"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="accepted",
    )
    terminal = await store.store_receipt(
        _channel_payload("terminal"),
        server_seq=2,
        disposition=ReceiptDisposition.TERMINAL,
        reason="redacted",
    )
    gated = await store.store_receipt(
        _dm_payload("gated", "foreign", "agent"),
        server_seq=3,
        disposition=ReceiptDisposition.FOREIGN_DM_GATED,
        reason="approval required",
    )
    assert eligible.acknowledge and terminal.acknowledge
    assert not gated.acknowledge
    repeated = await store.store_receipt(
        _channel_payload("eligible", content="ignored"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="ignored",
    )
    assert repeated.status is ReceiptWriteStatus.IDEMPOTENT
    assert repeated.acknowledge

    by_id = await store.store_receipt(
        _channel_payload("eligible"),
        server_seq=99,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="conflict",
    )
    by_seq = await store.store_receipt(
        _channel_payload("different"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="conflict",
    )
    assert by_id.status is ReceiptWriteStatus.CONFLICT and not by_id.acknowledge
    assert by_seq.status is ReceiptWriteStatus.CONFLICT and not by_seq.acknowledge

    promoted = await store.promote_gated_receipt("gated", 3, reason="approved")
    promoted_again = await store.promote_gated_receipt("gated", 3, reason="approved")
    assert promoted.status is ReceiptWriteStatus.COMMITTED and promoted.acknowledge
    assert promoted_again.status is ReceiptWriteStatus.IDEMPOTENT and promoted_again.acknowledge
    assert [m.envelope_id for m in await store.get_pending()] == ["eligible", "gated"]
    await store.close()


@pytest.mark.asyncio
async def test_receipt_handler_seam_maps_durable_acknowledge_to_transport_outcome():
    store = _temp_store()
    client = object.__new__(PuffoCoreWsClient)

    async def handler(delivery):
        envelope = delivery["envelope"]
        disposition = ReceiptDisposition(envelope.pop("_disposition"))
        result = await store.store_receipt(
            envelope,
            server_seq=delivery["seq"],
            disposition=disposition,
            reason="classified",
        )
        return (
            TransportOutcome.ACK
            if result.acknowledge
            else (
                TransportOutcome.DEFER
                if disposition is ReceiptDisposition.FOREIGN_DM_GATED
                else TransportOutcome.HOLD
            )
        )

    client.on_message = handler
    eligible = _channel_payload("transport-eligible")
    eligible["_disposition"] = ReceiptDisposition.ELIGIBLE.value
    gated = _dm_payload("transport-gated", "foreign", "agent")
    gated["_disposition"] = ReceiptDisposition.FOREIGN_DM_GATED.value
    assert (
        await client.dispatch_delivery({"seq": 20, "envelope": eligible})
    ).outcome is TransportOutcome.ACK
    assert (
        await client.dispatch_delivery({"seq": 21, "envelope": gated})
    ).outcome is TransportOutcome.DEFER

    conflict = _channel_payload("other-id")
    conflict["_disposition"] = ReceiptDisposition.ELIGIBLE.value
    assert (
        await client.dispatch_delivery({"seq": 20, "envelope": conflict})
    ).outcome is TransportOutcome.HOLD

    async def failing_handler(_delivery):
        raise RuntimeError("injected commit failure")

    client.on_message = failing_handler
    assert (
        await client.dispatch_delivery({
            "seq": 22,
            "envelope": _channel_payload("commit-failure"),
        })
    ).outcome is TransportOutcome.HOLD
    await store.close()


@pytest.mark.asyncio
async def test_legacy_sequence_backfill_does_not_activate_history_row():
    store = _temp_store()
    await store.store(_channel_payload("legacy"))
    result = await store.store_receipt(
        _channel_payload("legacy"),
        server_seq=8,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="real delivery",
    )
    assert result.status is ReceiptWriteStatus.COMMITTED and result.acknowledge
    msg = await store.get_message_by_envelope("legacy")
    assert msg is not None
    assert msg.server_seq == 8
    assert msg.receipt_disposition is None and msg.processing_state is None
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_legacy_gated_dm_backfill_stays_deferred_until_promotion():
    store = _temp_store()
    await store.store(_dm_payload("legacy-gated", "foreign", "agent"))

    gated = await store.store_receipt(
        _dm_payload("legacy-gated", "foreign", "agent"),
        server_seq=9,
        disposition=ReceiptDisposition.FOREIGN_DM_GATED,
        reason="foreign dm awaiting approval",
    )

    assert gated.status is ReceiptWriteStatus.COMMITTED
    assert not gated.acknowledge
    row = await store.get_message_by_envelope("legacy-gated")
    assert row is not None
    assert row.server_seq == 9
    assert row.receipt_disposition is ReceiptDisposition.FOREIGN_DM_GATED
    assert row.processing_state is None

    promoted = await store.promote_gated_receipt(
        "legacy-gated", 9, reason="approved",
    )
    assert promoted.status is ReceiptWriteStatus.COMMITTED
    assert promoted.acknowledge
    assert [item.envelope_id for item in await store.get_pending()] == [
        "legacy-gated"
    ]
    await store.close()


@pytest.mark.asyncio
async def test_local_pending_fifo_frontier_ordinal_and_channel_page_survive_reopen():
    store = _temp_store()
    await store.store_receipt(
        _channel_payload("s1"), server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE, reason="ok",
    )
    await store.store_local_event(_channel_payload("l1"), reason="runtime")
    await store.store_local_event(_channel_payload("l2"), reason="runtime")
    await store.store_receipt(
        _channel_payload("s2"), server_seq=2,
        disposition=ReceiptDisposition.ELIGIBLE, reason="ok",
    )
    await store.close()
    await store.open()
    pending = await store.get_pending()
    assert [m.envelope_id for m in pending] == ["s1", "l1", "l2", "s2"]
    assert [m.server_seq for m in pending] == [1, None, None, 2]
    assert pending[1].after_server_seq == pending[2].after_server_seq == 1
    assert pending[1].local_ordinal < pending[2].local_ordinal
    page = await store.get_channel_pending("sp_1", "ch_1", limit=3)
    assert [m.envelope_id for m in page.items] == ["s1", "l1", "l2"]
    assert page.through_seq == 1 and page.more_available
    await store.close()


@pytest.mark.asyncio
async def test_introduction_local_marker_commit_and_replay_are_atomic():
    store = _temp_store()
    payload = _channel_payload("intro")
    first = await store.store_local_event(
        payload, reason="introduction", intro_channel_id="ch_1"
    )
    replay = await store.store_local_event(
        payload, reason="introduction", intro_channel_id="ch_1"
    )
    assert first.envelope_id == replay.envelope_id == "intro"
    assert [m.envelope_id for m in await store.get_pending()] == ["intro"]
    assert await store.has_channel_intro_been_prompted("ch_1")
    with pytest.raises(LifecycleConflict):
        await store.store_local_event(
            _channel_payload("other"), reason="introduction", intro_channel_id="ch_1"
        )
    assert not await store.has_message("other")
    await store.close()


@pytest.mark.asyncio
async def test_introduction_marker_rolls_back_when_local_insert_fails():
    store = _temp_store()
    await store.store(_channel_payload("duplicate"))
    with pytest.raises(Exception):
        await store.store_local_event(
            _channel_payload("duplicate"),
            reason="introduction",
            intro_channel_id="new-channel",
        )
    assert not await store.has_channel_intro_been_prompted("new-channel")
    await store.close()


@pytest.mark.asyncio
async def test_sequence_less_server_receipt_is_never_pending_work():
    store = _temp_store()
    db = await store._ensure_db()
    values = store._payload_values(_channel_payload("missing-sequence"), None)
    await db.execute(
        """INSERT INTO messages
           (envelope_id, envelope_kind, sender_slug, channel_id, space_id,
            recipient_slug, content_type, content, sent_at, received_at,
            thread_root_id, reply_to_id, is_encrypted, receipt_disposition,
            receipt_reason, processing_state)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        values + (
            ReceiptDisposition.ELIGIBLE.value,
            "malformed legacy data",
            ProcessingState.PENDING.value,
        ),
    )
    await db.commit()
    assert await store.get_pending() == ()
    await store.close()


@pytest.mark.asyncio
async def test_pending_lifecycle_turn_recovery_baseline_through_and_cleanup():
    store = _temp_store()
    old = _now_ms() - 100 * 86_400_000
    for seq in (1, 2):
        await store.store_receipt(
            _channel_payload(f"s{seq}"),
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="ok",
            received_at=old,
        )
    before = await store.get_pending()
    with pytest.raises(LifecycleConflict):
        await store.admit_messages(
            ["s1", "missing"], turn_id="bad", provider_session_id="provider"
        )
    assert await store.get_pending() == before

    run = await store.admit_messages(
        ["s1", "s2"], turn_id="turn", provider_session_id="provider",
        model_visible_at=123,
    )
    assert run.message_ids == ("s1", "s2")
    assert [m.envelope_id for m in await store.get_in_turn_messages("turn", "provider")] == [
        "s1", "s2",
    ]
    assert await store.get_model_visible_through_seq("turn", "sp_1", "ch_1") == 2
    await store.set_context_baseline("sp_1", "ch_1", 5)
    await store.set_context_baseline("sp_1", "ch_2", 9)
    assert await store.get_context_baseline("sp_1", "ch_1") == 5
    assert await store.get_context_baseline("sp_1", "ch_2") == 9
    assert await store.cleanup(90) == 0

    await store.close()
    await store.open()
    reopened = await store.get_turn_run("turn")
    assert reopened is not None and reopened.provider_session_id == "provider"
    requeued = await store.requeue_messages(["s2", "s1"], turn_id="turn")
    assert requeued.state == "requeued"
    assert await store.get_model_visible_through_seq("turn", "sp_1", "ch_1") is None
    completed_admission = await store.admit_messages(
        ["s1", "s2"], turn_id="turn2", provider_session_id="provider2"
    )
    assert completed_admission.state == ProcessingState.IN_TURN.value
    completed = await store.mark_processed(["s2", "s1"], turn_id="turn2")
    assert completed.state == ProcessingState.PROCESSED.value
    await store.close()


@pytest.mark.asyncio
async def test_active_turn_append_uses_stable_union_for_complete_and_requeue():
    store = _temp_store()
    for seq in range(1, 7):
        await store.store_receipt(
            _channel_payload(f"s{seq}"),
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="ok",
        )

    initial = await store.admit_messages(
        ["s1"], turn_id="append-complete", provider_session_id="durable"
    )
    assert initial.message_ids == ("s1",)
    appended = await store.admit_messages(
        ["s2", "s3"], turn_id="append-complete", provider_session_id="durable"
    )
    assert appended.message_ids == ("s1", "s2", "s3")
    with pytest.raises(LifecycleConflict):
        await store.admit_messages(
            ["s4"], turn_id="append-complete", provider_session_id="other"
        )
    with pytest.raises(LifecycleConflict):
        await store.admit_messages(
            ["s2"], turn_id="append-complete", provider_session_id="durable"
        )
    with pytest.raises(LifecycleConflict):
        await store.admit_messages(
            ["s4", "missing"],
            turn_id="append-complete",
            provider_session_id="durable",
        )
    assert (await store.get_turn_run("append-complete")).message_ids == (
        "s1", "s2", "s3",
    )
    completed = await store.mark_processed(
        ["s3", "s1", "s2"], turn_id="append-complete"
    )
    assert completed.state == ProcessingState.PROCESSED.value
    with pytest.raises(LifecycleConflict):
        await store.admit_messages(
            ["s4"], turn_id="append-complete", provider_session_id="durable"
        )

    await store.admit_messages(
        ["s4"], turn_id="append-requeue", provider_session_id="durable-2"
    )
    appended_requeue = await store.admit_messages(
        ["s5", "s6"], turn_id="append-requeue", provider_session_id="durable-2"
    )
    assert appended_requeue.message_ids == ("s4", "s5", "s6")
    requeued = await store.requeue_messages(
        ["s6", "s4", "s5"], turn_id="append-requeue"
    )
    assert requeued.state == "requeued"
    await store.close()


@pytest.mark.asyncio
async def test_nullable_provider_session_is_non_resumable_and_requeue_only():
    store = _temp_store()
    for seq in (1, 2):
        await store.store_receipt(
            _channel_payload(f"stateless-{seq}"),
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="ok",
        )
    run = await store.admit_messages(
        ["stateless-1"], turn_id="stateless-complete", provider_session_id=None
    )
    assert run.provider_session_id is None
    completed = await store.mark_processed(
        ["stateless-1"], turn_id="stateless-complete"
    )
    assert completed.state == ProcessingState.PROCESSED.value

    await store.admit_messages(
        ["stateless-2"], turn_id="stateless-interrupted", provider_session_id=None
    )
    db = await store._ensure_db()
    async with db.execute("PRAGMA table_info(turn_runs)") as cursor:
        columns = {row["name"]: row for row in await cursor.fetchall()}
    assert columns["provider_session_id"]["notnull"] == 0
    await store.close()
    await store.open()
    reopened = await store.get_turn_run("stateless-interrupted")
    assert reopened is not None and reopened.provider_session_id is None
    assert await store.get_in_turn_messages("stateless-interrupted", None) == ()
    requeued = await store.requeue_messages(
        ["stateless-2"], turn_id="stateless-interrupted"
    )
    assert requeued.state == "requeued"
    await store.close()


@pytest.mark.asyncio
async def test_nonnullable_provider_session_schema_migration_preserves_turn():
    import sqlite3

    store = _temp_store()
    await store.open()
    await store.close()
    connection = sqlite3.connect(store.db_path)
    connection.executescript(
        """
        DROP TABLE turn_run_messages;
        DROP TABLE turn_runs;
        CREATE TABLE turn_runs (
            turn_id TEXT PRIMARY KEY,
            provider_session_id TEXT NOT NULL,
            state TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            completed_at INTEGER
        );
        CREATE TABLE turn_run_messages (
            turn_id TEXT NOT NULL,
            envelope_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (turn_id, envelope_id),
            UNIQUE (turn_id, ordinal),
            FOREIGN KEY (turn_id) REFERENCES turn_runs(turn_id)
        );
        INSERT INTO turn_runs VALUES ('legacy-turn', 'durable', 'in_turn', 123, NULL);
        INSERT INTO turn_run_messages VALUES ('legacy-turn', 'legacy-message', 0);
        """
    )
    connection.close()

    await store.open()
    db = await store._ensure_db()
    async with db.execute("PRAGMA table_info(turn_runs)") as cursor:
        columns = {row["name"]: row for row in await cursor.fetchall()}
    assert columns["provider_session_id"]["notnull"] == 0
    run = await store.get_turn_run("legacy-turn")
    assert run is not None
    assert run.provider_session_id == "durable"
    assert run.message_ids == ("legacy-message",)
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_inbox_receipt_local_and_lifecycle_operations_serialize():
    store = _temp_store()
    await store.open()
    await store.store_receipt(
        _channel_payload("admit-me"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="ok",
    )
    receipt, local, admitted = await asyncio.gather(
        store.store_receipt(
            _channel_payload("concurrent-receipt"),
            server_seq=2,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="ok",
        ),
        store.store_local_event(_channel_payload("concurrent-local"), reason="runtime"),
        store.admit_messages(
            ["admit-me"], turn_id="concurrent-turn", provider_session_id="durable"
        ),
    )
    assert receipt.acknowledge
    assert local.processing_state is ProcessingState.PENDING
    assert admitted.message_ids == ("admit-me",)
    completed, pending = await asyncio.gather(
        store.mark_processed(["admit-me"], turn_id="concurrent-turn"),
        store.get_pending(),
    )
    assert completed.state == ProcessingState.PROCESSED.value
    assert {item.envelope_id for item in pending} == {
        "concurrent-receipt", "concurrent-local",
    }
    await store.close()


@pytest.mark.asyncio
async def test_legacy_writer_waits_for_active_inbox_transaction(monkeypatch):
    store = _temp_store()
    await store.open()
    transaction_started = asyncio.Event()
    release_transaction = asyncio.Event()
    original = store._store_receipt_unlocked

    async def paused_receipt(*args, **kwargs):
        db = await store._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        transaction_started.set()
        await release_transaction.wait()
        await db.rollback()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "_store_receipt_unlocked", paused_receipt)
    receipt_task = asyncio.create_task(
        store.store_receipt(
            _channel_payload("locked-receipt"),
            server_seq=1,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="ok",
        )
    )
    await transaction_started.wait()
    legacy_task = asyncio.create_task(store.store(_channel_payload("legacy-write")))
    await asyncio.sleep(0)
    assert not legacy_task.done()

    db = await store._ensure_db()
    async with db.execute(
        "SELECT 1 FROM messages WHERE envelope_id = 'legacy-write'"
    ) as cursor:
        assert await cursor.fetchone() is None

    release_transaction.set()
    receipt = await receipt_task
    await legacy_task
    assert receipt.acknowledge
    assert await store.has_message("locked-receipt")
    assert await store.has_message("legacy-write")
    await store.close()
