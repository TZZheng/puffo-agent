import asyncio
import os
import sqlite3
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
async def test_due_reminder_occurrence_is_delivered_once_across_duplicate_ticks():
    """A durable due occurrence has one local Inbox event across duplicate ticks."""
    store = _temp_store()
    reminder = await store.create_reminder(
        content="review the launch notes",
        target="channel:sp_1:ch_1",
        intended_at_ms=1_000,
    )

    await asyncio.gather(
        store.deliver_due_reminders(now_ms=2_000),
        store.deliver_due_reminders(now_ms=2_000),
    )

    event_id = f"reminder-occurrence:{reminder.occurrence_id}"
    pending = await store.get_pending()
    assert [item.envelope_id for item in pending] == [event_id]
    delivered = await store.get_reminder(reminder.reminder_id)
    assert delivered is not None and delivered.state == "delivered"
    assert delivered.delivered_event_id == event_id

    await store.deliver_due_reminders(now_ms=3_000)
    await store.close()
    await store.open()
    assert [item.envelope_id for item in await store.get_pending()] == [event_id]
    assert (await store.get_reminder(reminder.reminder_id)).state == "delivered"
    await store.close()


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


@pytest.mark.asyncio
async def test_equal_timestamp_envelope_cursor_pages_both_directions(tmp_path):
    """Envelope-id cursors retain the composite timestamp/id ordering.

    All roots and replies deliberately share a timestamp: a timestamp-only
    cursor would skip the second and third records on the next page.
    """
    store = MessageStore(str(tmp_path / "equal-cursor.db"))
    await store.open()
    try:
        for envelope_id in ("aaa_root", "root_b", "root_c"):
            await store.store(_channel_payload(envelope_id, sent_at=123))
        for envelope_id in ("reply_a", "reply_b", "reply_c"):
            await store.store(_channel_payload(
                envelope_id, sent_at=123, thread_root_id="aaa_root",
            ))
        forward = ["aaa_root"]
        cursor = "aaa_root"
        while True:
            page = await store.get_channel_roots(
                "ch_1", limit=1, since_envelope_id=cursor,
            )
            if not page:
                break
            forward.extend(value.message.envelope_id for value in page)
            cursor = page[-1].message.envelope_id
        thread_forward = ["aaa_root"]
        cursor = "aaa_root"
        while True:
            page = await store.get_thread_messages(
                "aaa_root", limit=1, since_envelope_id=cursor,
            )
            if not page:
                break
            thread_forward.extend(value.envelope_id for value in page)
            cursor = page[-1].envelope_id
        assert forward == ["aaa_root", "root_b", "root_c"]
        assert thread_forward == ["aaa_root", "reply_a", "reply_b", "reply_c"]
        backward = []
        cursor = None
        while True:
            page = await store.get_channel_roots(
                "ch_1", limit=1, before_envelope_id=cursor,
            )
            if not page:
                break
            backward.extend(value.message.envelope_id for value in page)
            cursor = page[0].message.envelope_id
        thread_backward = []
        cursor = None
        while True:
            page = await store.get_thread_messages(
                "aaa_root", limit=1, before_envelope_id=cursor,
            )
            if not page:
                break
            thread_backward.extend(value.envelope_id for value in page)
            cursor = page[0].envelope_id
        assert backward == ["root_c", "root_b", "aaa_root"]
        assert thread_backward == ["reply_c", "reply_b", "reply_a", "aaa_root"]
        assert list(reversed(backward)) == forward
        assert list(reversed(thread_backward)) == thread_forward
    finally:
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
@pytest.mark.parametrize(
    "lifecycle",
    [ProcessingState.PENDING, ProcessingState.IN_TURN, ProcessingState.PROCESSED],
)
async def test_local_runtime_redelivery_promotes_in_place_and_preserves_lifecycle(
    lifecycle,
):
    store = _temp_store()
    await store.store_local_event(
        _channel_payload("rollout"), reason="legacy bridge",
    )
    if lifecycle is not ProcessingState.PENDING:
        await store.admit_messages(
            ["rollout"], turn_id="turn-1", provider_session_id="session-1",
        )
    if lifecycle is ProcessingState.PROCESSED:
        await store.mark_processed(["rollout"], turn_id="turn-1")
    before = await store.get_message_by_envelope("rollout")
    notice_before = await store.get_notice_state()
    work_before = tuple(
        item.envelope_id for item in await store.get_pending()
    )
    result = await store.store_receipt(
        _channel_payload("rollout"),
        server_seq=42,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="trusted bridge replay",
    )
    assert result.status is ReceiptWriteStatus.COMMITTED
    await store.close()
    await store.open()
    row = await store.get_message_by_envelope("rollout")
    assert row is not None and before is not None
    assert row.server_seq == 42
    assert row.receipt_disposition is ReceiptDisposition.ELIGIBLE
    assert row.processing_state is lifecycle
    assert row.processing_turn_id == before.processing_turn_id
    assert row.local_ordinal is None and row.after_server_seq is None
    notice_after = await store.get_notice_state()
    work_after = tuple(item.envelope_id for item in await store.get_pending())
    assert notice_after == notice_before
    assert work_after == work_before
    replay = await store.store_receipt(
        _channel_payload("rollout"),
        server_seq=42,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="trusted bridge replay",
    )
    assert replay.status is ReceiptWriteStatus.IDEMPOTENT
    assert await store.get_notice_state() == notice_before
    assert tuple(
        item.envelope_id for item in await store.get_pending()
    ) == work_before
    await store.close()


@pytest.mark.asyncio
async def test_local_runtime_promotion_sequence_collision_is_non_mutating():
    store = _temp_store()
    await store.store_local_event(_channel_payload("local"), reason="legacy")
    await store.store_receipt(
        _channel_payload("server"),
        server_seq=7,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="server",
    )
    result = await store.store_receipt(
        _channel_payload("local"),
        server_seq=7,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="collision",
    )
    assert result.status is ReceiptWriteStatus.CONFLICT
    row = await store.get_message_by_envelope("local")
    assert row is not None
    assert row.server_seq is None
    assert row.receipt_disposition is ReceiptDisposition.LOCAL_RUNTIME
    assert row.local_ordinal is not None
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
async def test_model_visible_boundary_skips_earlier_terminal_receipt(tmp_path):
    store = MessageStore(tmp_path / "terminal-safe-prefix.db")
    await store.store_receipt(
        _channel_payload("terminal-receipt", channel_id="ch_terminal"),
        server_seq=10,
        disposition=ReceiptDisposition.TERMINAL,
        reason="self echo",
    )
    await store.store_receipt(
        _channel_payload("active-turn", channel_id="ch_terminal"),
        server_seq=11,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.admit_messages(
        ["active-turn"], turn_id="active-turn", provider_session_id="provider",
        model_visible_at=123,
    )

    terminal = await store.get_message_by_envelope("terminal-receipt")
    assert terminal is not None
    assert terminal.receipt_disposition is ReceiptDisposition.TERMINAL
    assert terminal.processing_state is None
    assert await store.get_model_visible_through_seq(
        "active-turn", "sp_1", "ch_terminal"
    ) == 11
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


@pytest.mark.asyncio
async def test_durable_notice_deadline_is_fixed_and_survives_reopen(tmp_path):
    now = [1_000]
    path = tmp_path / "notice.db"
    store = MessageStore(path, now_ms=lambda: now[0])
    await store.store_receipt(
        _channel_payload("n1"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
        received_at=now[0],
    )
    first = await store.get_notice_state()
    assert first.first_pending_deadline_ms == 4_000
    now[0] = 2_500
    await store.store_receipt(
        _channel_payload("n2"),
        server_seq=2,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
        received_at=now[0],
    )
    later = await store.get_notice_state()
    assert later.first_pending_deadline_ms == 4_000
    assert later.generation > first.generation
    # A transport that learns its native session only from acceptance may
    # still offer an unaccepted generation once.
    assert later.is_due_for(None)
    await store.close()

    reopened = MessageStore(path, now_ms=lambda: 20_000)
    assert (await reopened.get_notice_state()).first_pending_deadline_ms == 4_000
    assert await reopened.mark_notice_delivered(later.generation, "provider-1")
    accepted = await reopened.get_notice_state()
    assert accepted.last_delivered_provider_session_id == "provider-1"
    assert not accepted.is_due_for("provider-1")
    assert not accepted.is_due_for(None)
    assert accepted.is_due_for("provider-2")
    assert not await reopened.mark_notice_delivered(later.generation, "provider-1")
    assert await reopened.mark_notice_delivered(later.generation, "provider-2")
    assert not await reopened.mark_notice_delivered(later.generation, "provider-2")
    await reopened.close()


@pytest.mark.asyncio
async def test_notice_state_session_migration_preserves_baseline_row(tmp_path):
    import sqlite3

    path = tmp_path / "baseline-notice-state.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (envelope_id TEXT PRIMARY KEY, "
        "envelope_kind TEXT NOT NULL, sender_slug TEXT NOT NULL, "
        "channel_id TEXT, space_id TEXT, recipient_slug TEXT, "
        "content_type TEXT NOT NULL DEFAULT 'text/plain', content TEXT NOT NULL, "
        "sent_at INTEGER NOT NULL, received_at INTEGER NOT NULL, "
        "thread_root_id TEXT, reply_to_id TEXT, "
        "is_encrypted INTEGER NOT NULL DEFAULT 1)"
    )
    con.execute(
        "CREATE TABLE inbox_notice_state ("
        "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
        "generation INTEGER NOT NULL, pending_count INTEGER NOT NULL, "
        "first_pending_deadline_ms INTEGER, "
        "last_delivered_generation INTEGER NOT NULL)"
    )
    con.execute(
        "INSERT INTO inbox_notice_state VALUES (1, 7, 3, 4321, 7)"
    )
    con.commit()
    con.close()

    store = MessageStore(path)
    await store.open()
    state = await store.get_notice_state()
    assert (
        state.generation,
        state.pending_count,
        state.first_pending_deadline_ms,
        state.last_delivered_generation,
        state.last_delivered_provider_session_id,
    ) == (7, 3, 4321, 7, None)
    assert state.is_due_for("replacement-session")
    db = await store._ensure_db()
    async with db.execute("PRAGMA table_info(inbox_notice_state)") as cursor:
        columns = {row["name"] for row in await cursor.fetchall()}
    assert "last_delivered_provider_session_id" in columns
    assert await store.mark_notice_delivered(7, "replacement-session")
    assert (await store.get_notice_state()).last_delivered_provider_session_id == (
        "replacement-session"
    )
    await store.close()


@pytest.mark.asyncio
async def test_empty_notice_turn_rearms_unchanged_pending_work_atomically(tmp_path):
    now = [1_000]
    store = MessageStore(tmp_path / "notice-rearm.db", now_ms=lambda: now[0])
    await store.store_receipt(
        _channel_payload("notice-pending"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    delivered = await store.get_notice_state()
    assert await store.mark_notice_delivered(delivered.generation, "provider-1")
    await store.start_turn(
        turn_id="empty-notice-turn",
        provider_session_id="provider-1",
    )

    now[0] = 2_000
    run = await store.finalize_empty_turn(
        turn_id="empty-notice-turn",
        state="requeued",
        rearm_notice=True,
    )
    rearmed = await store.get_notice_state()

    assert run.state == "requeued"
    assert rearmed.pending_count == 1
    assert rearmed.generation == delivered.generation + 1
    assert rearmed.last_delivered_generation == delivered.generation
    assert rearmed.first_pending_deadline_ms == 5_000
    assert rearmed.delivery_pending
    await store.close()


@pytest.mark.asyncio
async def test_successful_empty_notice_turn_keeps_same_session_suppressed(
    tmp_path,
):
    store = MessageStore(tmp_path / "notice-empty-success.db")
    await store.store_receipt(
        _channel_payload("notice-empty-success"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    due = await store.get_notice_state()
    await store.start_turn(
        turn_id="empty-success-turn",
        provider_session_id="provider-1",
        notice_generation=due.generation,
    )
    run = await store.finalize_empty_turn(turn_id="empty-success-turn")
    state = await store.get_notice_state()

    assert run.state == ProcessingState.PROCESSED.value
    assert state.generation == due.generation
    assert state.pending_count == 1
    assert state.last_delivered_generation == due.generation
    assert state.last_delivered_provider_session_id == "provider-1"
    assert not state.is_due_for("provider-1")
    assert state.is_due_for("provider-2")
    assert await store.get_active_turn_runs() == ()
    await store.close()


@pytest.mark.asyncio
async def test_inbox_page_snapshot_excludes_concurrent_arrival_and_is_read_only(tmp_path):
    store = MessageStore(tmp_path / "pages.db")
    for seq in range(1, 4):
        await store.store_receipt(
            _channel_payload(f"page-{seq}", channel_id="ch_1"),
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="test",
        )
    first = await store.read_inbox_page(limit=2)
    assert [item.envelope_id for item in first.items] == ["page-1", "page-2"]
    assert first.has_more and first.remaining_count == 1
    await store.store_receipt(
        _channel_payload("page-4", channel_id="ch_1"),
        server_seq=4,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    second = await store.read_inbox_page(cursor=first.next_cursor, limit=2)
    assert [item.envelope_id for item in second.items] == ["page-3"]
    assert all(
        item.processing_state is ProcessingState.PENDING
        for item in await store.get_pending()
    )
    with pytest.raises(ValueError, match="another target"):
        await store.read_inbox_page(
            target="channel:sp_1:ch_1",
            cursor=first.next_cursor,
            limit=2,
        )
    await store.close()


@pytest.mark.asyncio
async def test_prior_context_is_bounded_ordered_and_lifecycle_filtered(tmp_path):
    store = MessageStore(tmp_path / "prior-context.db")

    await store.store_receipt(
        _channel_payload("prior-human", channel_id="ch_context", content="human"),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.admit_messages(
        ["prior-human"], turn_id="prior-human-turn", provider_session_id="provider"
    )
    await store.mark_processed(["prior-human"], turn_id="prior-human-turn")
    await store.store_receipt(
        _channel_payload(
            "prior-self-echo",
            channel_id="ch_context",
            sender_slug="agent",
            content="agent contribution",
        ),
        server_seq=2,
        disposition=ReceiptDisposition.TERMINAL,
        reason="self echo",
    )
    await store.store_receipt(
        _channel_payload(
            "prior-in-turn",
            channel_id="ch_context",
            content="still in turn",
        ),
        server_seq=3,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.admit_messages(
        ["prior-in-turn"], turn_id="prior-in-turn-turn", provider_session_id="provider"
    )
    await store.store_receipt(
        _channel_payload(
            "prior-page",
            channel_id="ch_context",
            content="newly admitted page",
        ),
        server_seq=4,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.store_receipt(
        _channel_payload(
            "prior-future",
            channel_id="ch_context",
            content="future pending work",
        ),
        server_seq=5,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.store_receipt(
        _channel_payload(
            "prior-other-channel",
            channel_id="ch_other",
            content="other channel",
        ),
        server_seq=6,
        disposition=ReceiptDisposition.TERMINAL,
        reason="test",
    )
    await store.store_receipt(
        _channel_payload(
            "prior-other-thread",
            channel_id="ch_context",
            thread_root_id="other-root",
            content="other thread",
        ),
        server_seq=7,
        disposition=ReceiptDisposition.TERMINAL,
        reason="test",
    )

    anchor = await store.get_message_by_envelope("prior-page")
    assert anchor is not None
    context = await store.get_prior_context(anchor)
    assert [item.envelope_id for item in context] == [
        "prior-human",
        "prior-self-echo",
    ]
    assert all(item.server_seq < anchor.server_seq for item in context)
    assert all(
        item.processing_state is ProcessingState.PROCESSED
        or item.receipt_disposition is ReceiptDisposition.TERMINAL
        for item in context
    )
    complete_page = await store.get_prior_context_page(anchor)
    assert complete_page.items == context
    assert complete_page.has_more is False

    limited = await store.get_prior_context(anchor, limit=1)
    assert [item.envelope_id for item in limited] == ["prior-self-echo"]
    limited_page = await store.get_prior_context_page(anchor, limit=1)
    assert limited_page.items == limited
    assert limited_page.has_more is True
    bounded = await store.get_prior_context(
        anchor, limit=20, max_bytes=len("agent contribution")
    )
    assert [item.envelope_id for item in bounded] == ["prior-self-echo"]
    assert sum(len(str(item.content).encode("utf-8")) for item in bounded) <= len(
        "agent contribution"
    )
    bounded_page = await store.get_prior_context_page(
        anchor, limit=20, max_bytes=len("agent contribution")
    )
    assert bounded_page.items == bounded
    assert bounded_page.has_more is True

    in_turn = await store.get_message_by_envelope("prior-in-turn")
    future = await store.get_message_by_envelope("prior-future")
    assert in_turn is not None and in_turn.processing_state is ProcessingState.IN_TURN
    assert future is not None and future.processing_state is ProcessingState.PENDING
    await store.requeue_messages(
        ["prior-in-turn"], turn_id="prior-in-turn-turn"
    )
    await store.close()


@pytest.mark.asyncio
async def test_prior_context_dm_route_excludes_other_peers_and_future_rows(tmp_path):
    store = MessageStore(tmp_path / "prior-context-dm.db")
    await store.store_receipt(
        _dm_payload(
            "dm-prior", "peer-1", "agent", content="earlier DM"
        ),
        server_seq=1,
        disposition=ReceiptDisposition.TERMINAL,
        reason="test",
    )
    await store.store_receipt(
        _dm_payload(
            "dm-other", "peer-2", "agent", content="other DM"
        ),
        server_seq=2,
        disposition=ReceiptDisposition.TERMINAL,
        reason="test",
    )
    await store.store_receipt(
        _dm_payload(
            "dm-page", "peer-1", "agent", content="current DM"
        ),
        server_seq=3,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.store_receipt(
        _dm_payload(
            "dm-future", "peer-1", "agent", content="future DM"
        ),
        server_seq=4,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )

    anchor = await store.get_message_by_envelope("dm-page")
    assert anchor is not None
    context = await store.get_prior_context(anchor)
    assert [item.envelope_id for item in context] == ["dm-prior"]
    assert all(item.envelope_kind == "dm" for item in context)
    assert all(item.sender_slug == "peer-1" or item.recipient_slug == "peer-1" for item in context)
    assert [item.envelope_id for item in await store.get_pending()] == [
        "dm-page",
        "dm-future",
    ]
    await store.close()


@pytest.mark.asyncio
async def test_inbox_page_traverses_more_than_fifty_with_complete_metadata(tmp_path):
    store = MessageStore(tmp_path / "deep-pages.db")
    for seq in range(1, 74):
        await store.store_receipt(
            _channel_payload(f"deep-{seq:03d}", channel_id="ch_deep"),
            server_seq=seq,
            disposition=ReceiptDisposition.ELIGIBLE,
            reason="test",
        )

    ids: list[str] = []
    cursor = ""
    generation = None
    remaining = []
    while True:
        page = await store.read_inbox_page(
            target="channel:sp_1:ch_deep",
            cursor=cursor,
            limit=17,
        )
        ids.extend(item.envelope_id for item in page.items)
        generation = page.snapshot_generation if generation is None else generation
        assert page.snapshot_generation == generation
        assert isinstance(page.next_cursor, str)
        assert isinstance(page.has_more, bool)
        assert isinstance(page.remaining_count, int)
        remaining.append(page.remaining_count)
        if not page.has_more:
            assert page.next_cursor == ""
            break
        assert page.next_cursor
        cursor = page.next_cursor

    assert ids == [f"deep-{seq:03d}" for seq in range(1, 74)]
    assert remaining == [56, 39, 22, 5, 0]
    assert len(await store.get_pending()) == 73
    await store.close()


@pytest.mark.asyncio
async def test_reminder_schema_migrates_additively_without_changing_existing_inbox(tmp_path):
    import sqlite3

    path = tmp_path / "pre-reminder.db"
    store = MessageStore(path, now_ms=lambda: 1_000)
    await store.store_receipt(
        _channel_payload("server-before", sent_at=1),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await store.store_local_event(
        _channel_payload("local-before", sent_at=2), reason="test",
    )
    await store.start_turn(turn_id="empty-turn", provider_session_id="provider")
    notice_before = await store.get_notice_state()
    await store.close()

    # Simulate the exact pre-slice file: all existing Inbox tables/data stay,
    # only the additive reminder table is absent.
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE reminder_occurrences")
    connection.commit()
    connection.close()

    reopened = MessageStore(path, now_ms=lambda: 2_000)
    await reopened.open()
    db = await reopened._ensure_db()
    async with db.execute("PRAGMA table_info(reminder_occurrences)") as cursor:
        columns = {row["name"] for row in await cursor.fetchall()}
    assert columns == {
        "reminder_id", "occurrence_id", "target", "content", "intended_at_ms",
        "state", "created_at_ms", "claimed_at_ms", "actual_fire_at_ms",
        "cancelled_at_ms", "delivered_at_ms", "delivered_event_id",
        "revision", "server_ack_revision", "payload_format", "opaque_payload",
        "sync_retry_after_ms", "sync_retry_count", "sync_permanent_revision",
        "sync_permanent_code", "delivery_claim_id", "delivery_claim_acquired",
    }
    async with db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'reminder_occurrences'"
    ) as cursor:
        schema = (await cursor.fetchone())["sql"]
    assert "'scheduled','claimed','cancelled','delivered'" in schema.replace(" ", "")
    assert [item.envelope_id for item in await reopened.get_pending()] == [
        "server-before", "local-before",
    ]
    assert await reopened.get_notice_state() == notice_before
    run = await reopened.get_turn_run("empty-turn")
    assert run is not None and run.state == ProcessingState.IN_TURN.value
    await reopened.close()


@pytest.mark.asyncio
async def test_reminder_sync_schema_backfills_old_lifecycles_without_losing_facts(tmp_path):
    """The remote outbox is additive to the accepted local reminder table."""
    path = tmp_path / "old-reminders.db"
    bootstrap = MessageStore(path, now_ms=lambda: 1_000)
    await bootstrap.store_receipt(
        _channel_payload("existing-inbox", sent_at=1),
        server_seq=1,
        disposition=ReceiptDisposition.ELIGIBLE,
        reason="test",
    )
    await bootstrap.close()

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE reminder_occurrences")
    connection.execute(
        """CREATE TABLE reminder_occurrences (
            reminder_id TEXT PRIMARY KEY,
            occurrence_id TEXT NOT NULL UNIQUE,
            target TEXT NOT NULL,
            content TEXT NOT NULL,
            intended_at_ms INTEGER NOT NULL,
            state TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            claimed_at_ms INTEGER,
            actual_fire_at_ms INTEGER,
            cancelled_at_ms INTEGER,
            delivered_at_ms INTEGER,
            delivered_event_id TEXT UNIQUE
        )"""
    )
    rows = [
        ("scheduled", "occ-scheduled", "scheduled", None, None, None, None),
        ("claimed", "occ-claimed", "claimed", 20, 20, None, None),
        ("cancelled", "occ-cancelled", "cancelled", None, None, 30, None),
        ("delivered", "occ-delivered", "delivered", 40, 40, None, 50),
    ]
    for reminder_id, occurrence_id, state, claimed_at, fire_at, cancelled_at, delivered_at in rows:
        connection.execute(
            """INSERT INTO reminder_occurrences
               (reminder_id, occurrence_id, target, content, intended_at_ms,
                state, created_at_ms, claimed_at_ms, actual_fire_at_ms,
                cancelled_at_ms, delivered_at_ms, delivered_event_id)
               VALUES (?, ?, 'dm:peer', 'old exact content', 10, ?, 1, ?, ?, ?, ?, ?)""",
            (
                reminder_id, occurrence_id, state, claimed_at, fire_at,
                cancelled_at, delivered_at,
                f"reminder-occurrence:{occurrence_id}" if delivered_at else None,
            ),
        )
    connection.commit()
    connection.close()

    store = MessageStore(path, now_ms=lambda: 1_000)
    await store.open()
    db = await store._ensure_db()
    async with db.execute(
        """SELECT occurrence_id, state, revision, server_ack_revision,
                  payload_format, opaque_payload, sync_retry_count,
                  sync_permanent_code
           FROM reminder_occurrences ORDER BY occurrence_id"""
    ) as cursor:
        migrated = {row["occurrence_id"]: dict(row) for row in await cursor.fetchall()}
    assert migrated["occ-scheduled"]["revision"] == 1
    assert migrated["occ-claimed"]["revision"] == 1
    assert migrated["occ-cancelled"]["revision"] == 2
    assert migrated["occ-delivered"]["revision"] == 2
    assert all(row["server_ack_revision"] == 0 for row in migrated.values())
    assert all(row["payload_format"] is None for row in migrated.values())
    assert [item.envelope_id for item in await store.get_pending()] == ["existing-inbox"]
    assert (await store.get_reminder("delivered")).delivered_event_id == (
        "reminder-occurrence:occ-delivered"
    )
    await store.close()


@pytest.mark.asyncio
async def test_reminder_sync_revisions_and_acknowledgments_are_transactional(tmp_path):
    store = MessageStore(tmp_path / "messages.db", now_ms=lambda: 2_000)
    created = await store.create_reminder(
        reminder_id="reminder-state", occurrence_id="occurrence-state",
        target="dm:peer", content="exact private", intended_at_ms=1_000,
        created_at_ms=500,
    )
    record = await store.get_reminder_sync_record(created.occurrence_id)
    assert record is not None and (record.state, record.revision, record.server_ack_revision) == (
        "scheduled", 1, 0,
    )
    await store.claim_due_reminders(now_ms=2_000)
    record = await store.get_reminder_sync_record(created.occurrence_id)
    assert record is not None and (record.state, record.revision) == ("claimed", 1)

    await store.schedule_reminder_sync_retry(
        occurrence_id=created.occurrence_id, revision=1, retry_after_ms=9_000,
    )
    cancelled = await store.cancel_reminder(created.reminder_id, cancelled_at_ms=2_000)
    assert cancelled.state == "cancelled"
    record = await store.get_reminder_sync_record(created.occurrence_id)
    assert record is not None and (record.revision, record.sync_retry_count) == (2, 0)
    assert not await store.acknowledge_reminder_sync_revision(
        occurrence_id=created.occurrence_id, revision=1,
    )
    assert await store.acknowledge_reminder_sync_revision(
        occurrence_id=created.occurrence_id, revision=2,
    )
    await store.cancel_reminder(created.reminder_id, cancelled_at_ms=2_100)
    record = await store.get_reminder_sync_record(created.occurrence_id)
    assert record is not None and (record.revision, record.server_ack_revision) == (2, 2)

    delivered = await store.create_reminder(
        reminder_id="reminder-delivered", occurrence_id="occurrence-delivered",
        target="dm:peer", content="deliver exactly once", intended_at_ms=1_000,
        created_at_ms=500,
    )
    await store.deliver_due_reminders(now_ms=2_000)
    delivered_record = await store.get_reminder_sync_record(delivered.occurrence_id)
    assert delivered_record is not None
    assert (delivered_record.state, delivered_record.revision) == ("delivered", 2)
    assert [item.envelope_id for item in await store.get_pending()] == [
        f"reminder-occurrence:{delivered.occurrence_id}"
    ]
    await store.close()


@pytest.mark.asyncio
async def test_reminder_identity_projection_order_and_reopen_are_stable(tmp_path):
    now = [1_000]
    path = tmp_path / "reminders.db"
    store = MessageStore(path, now_ms=lambda: now[0])
    later = await store.create_reminder(
        content="later exact text",
        target="channel:sp_1:ch_1:thread:root_1",
        intended_at_ms=5_000,
    )
    earlier = await store.create_reminder(
        content="earlier exact text",
        target="dm:peer_1",
        intended_at_ms=2_000,
    )
    assert later.reminder_id and later.occurrence_id
    assert later.reminder_id != later.occurrence_id
    assert earlier.reminder_id != earlier.occurrence_id
    listed = await store.list_reminders(limit=50)
    assert [item.reminder_id for item in listed] == [
        earlier.reminder_id, later.reminder_id,
    ]
    assert later.as_dict() == {
        "reminder_id": later.reminder_id,
        "occurrence_id": later.occurrence_id,
        "state": "scheduled",
        "target": "channel:sp_1:ch_1:thread:root_1",
        "content": "later exact text",
        "intended_at": "1970-01-01T00:00:05.000Z",
        "actual_fire_at": None,
        "created_at": "1970-01-01T00:00:01.000Z",
        "cancelled_at": None,
        "delivered_at": None,
    }
    await store.close()

    reopened = MessageStore(path, now_ms=lambda: now[0])
    restored = await reopened.get_reminder(later.reminder_id)
    assert restored == later
    await reopened.close()


@pytest.mark.asyncio
async def test_reminder_restart_boundaries_and_cancellation_are_atomic(tmp_path, monkeypatch):
    now = [1_000]
    path = tmp_path / "reminder-boundaries.db"
    store = MessageStore(path, now_ms=lambda: now[0])
    scheduled = await store.create_reminder(
        content="scheduled", target="channel:sp:ch", intended_at_ms=2_000,
    )
    claimed = await store.create_reminder(
        content="claimed", target="channel:sp:ch", intended_at_ms=1_000,
    )
    assert [item.reminder_id for item in await store.claim_due_reminders(now_ms=1_000)] == [
        claimed.reminder_id
    ]
    await store.close()

    reopened = MessageStore(path, now_ms=lambda: 3_000)
    # A scheduled and a previously claimed occurrence both recover to one
    # durable event, never a second intent/occurrence.
    recovered = await reopened.deliver_due_reminders(now_ms=3_000)
    assert {item.reminder_id for item in recovered} == {
        scheduled.reminder_id, claimed.reminder_id,
    }
    assert len(await reopened.get_pending()) == 2

    rollback = await reopened.create_reminder(
        content="rollback", target="channel:sp:ch", intended_at_ms=1,
    )
    original_insert = reopened._insert_local_event_in_transaction

    async def fail_insert(*_args, **_kwargs):
        raise RuntimeError("injected reminder delivery rollback")

    monkeypatch.setattr(reopened, "_insert_local_event_in_transaction", fail_insert)
    with pytest.raises(RuntimeError, match="rollback"):
        await reopened.deliver_due_reminders(now_ms=3_000)
    incomplete = await reopened.get_reminder(rollback.reminder_id)
    assert incomplete is not None and incomplete.state == "claimed"
    assert await reopened.get_message_by_envelope(
        f"reminder-occurrence:{rollback.occurrence_id}"
    ) is None
    monkeypatch.setattr(reopened, "_insert_local_event_in_transaction", original_insert)
    await reopened.close()

    # The interrupted transaction leaves a recoverable claimed fact on disk,
    # not a partial Inbox event. Recovery after a real reopen owns delivery.
    reopened = MessageStore(path, now_ms=lambda: 3_001)
    incomplete = await reopened.get_reminder(rollback.reminder_id)
    assert incomplete is not None and incomplete.state == "claimed"
    assert await reopened.get_message_by_envelope(
        f"reminder-occurrence:{rollback.occurrence_id}"
    ) is None
    assert [item.reminder_id for item in await reopened.deliver_due_reminders(now_ms=3_001)] == [
        rollback.reminder_id
    ]

    cancelled = await reopened.create_reminder(
        content="cancel", target="channel:sp:ch", intended_at_ms=4_000,
    )
    first_cancel = await reopened.cancel_reminder(cancelled.reminder_id, cancelled_at_ms=3_100)
    second_cancel = await reopened.cancel_reminder(cancelled.reminder_id, cancelled_at_ms=3_200)
    assert first_cancel.state == second_cancel.state == "cancelled"
    assert second_cancel.cancelled_at_ms == 3_100
    assert not await reopened.deliver_due_reminders(now_ms=5_000)
    assert await reopened.get_message_by_envelope(
        f"reminder-occurrence:{cancelled.occurrence_id}"
    ) is None

    claimed_cancel = await reopened.create_reminder(
        content="claimed cancel", target="channel:sp:ch", intended_at_ms=1,
    )
    await reopened.claim_due_reminders(now_ms=3_500)
    claimed_cancelled = await reopened.cancel_reminder(claimed_cancel.reminder_id)
    assert claimed_cancelled.state == "cancelled"
    assert not await reopened.deliver_due_reminders(now_ms=5_000)
    assert await reopened.get_message_by_envelope(
        f"reminder-occurrence:{claimed_cancel.occurrence_id}"
    ) is None

    delivered = await reopened.create_reminder(
        content="history stays", target="channel:sp:ch", intended_at_ms=1,
    )
    await reopened.deliver_due_reminders(now_ms=5_000)
    after_delivery_cancel = await reopened.cancel_reminder(delivered.reminder_id)
    assert after_delivery_cancel.state == "delivered"
    assert await reopened.get_message_by_envelope(
        f"reminder-occurrence:{delivered.occurrence_id}"
    ) is not None
    await reopened.close()


@pytest.mark.asyncio
async def test_claimed_cancel_delivery_race_serializes_to_one_valid_terminal_state():
    store = _temp_store()
    reminder = await store.create_reminder(
        content="race", target="channel:sp:ch", intended_at_ms=1,
    )
    await store.claim_due_reminders(now_ms=1)
    cancel, delivered = await asyncio.gather(
        store.cancel_reminder(reminder.reminder_id, cancelled_at_ms=2),
        store.deliver_due_reminders(now_ms=2),
    )
    terminal = await store.get_reminder(reminder.reminder_id)
    assert terminal is not None
    event = await store.get_message_by_envelope(
        f"reminder-occurrence:{reminder.occurrence_id}"
    )
    if terminal.state == "cancelled":
        assert event is None and delivered == () and cancel.state == "cancelled"
    else:
        assert terminal.state == "delivered" and event is not None
        assert cancel.state == "delivered"
    await store.close()


@pytest.mark.asyncio
async def test_stale_authorization_cannot_cross_an_envelope_fence_between_connections(
    tmp_path, monkeypatch,
):
    """A second runtime may persist the fence after local authorization."""
    now = 2_000
    path = tmp_path / "reminder-envelope-fence.db"
    first = MessageStore(path, now_ms=lambda: now)
    second = MessageStore(path, now_ms=lambda: now)

    async def create_due(name: str):
        return await first.create_reminder(
            reminder_id=f"reminder-{name}",
            occurrence_id=f"occurrence-{name}",
            content="due",
            target="dm:peer",
            intended_at_ms=1_000,
            created_at_ms=500,
        )

    async def persist_fence(occurrence_id: str) -> None:
        await second.persist_reminder_envelope(
            occurrence_id=occurrence_id,
            payload_format="puffo-reminder-aead-v1",
            opaque_payload=f"fence:{occurrence_id}".encode("ascii"),
        )

    blocked_before_claim = await create_due("before-claim")
    await persist_fence(blocked_before_claim.occurrence_id)
    assert await first.claim_authorized_reminders(
        (blocked_before_claim.occurrence_id,), now_ms=now,
    ) == ()
    assert (await first.get_reminder(blocked_before_claim.reminder_id)).state == "scheduled"

    blocked_before_delivery = await create_due("before-delivery")
    assert [item.occurrence_id for item in await first.claim_authorized_reminders(
        (blocked_before_delivery.occurrence_id,), now_ms=now,
    )] == [blocked_before_delivery.occurrence_id]
    await persist_fence(blocked_before_delivery.occurrence_id)
    assert await first.deliver_authorized_reminders(
        (blocked_before_delivery.occurrence_id,), now_ms=now,
    ) == ()
    assert (await first.get_reminder(blocked_before_delivery.reminder_id)).state == "claimed"

    blocked_during_delivery = await create_due("during-delivery")
    assert [item.occurrence_id for item in await first.claim_authorized_reminders(
        (blocked_during_delivery.occurrence_id,), now_ms=now,
    )] == [blocked_during_delivery.occurrence_id]
    original_deliver = first._deliver_claimed_reminder_unlocked

    async def persist_before_final_delivery(reminder_id: str, *, now_ms: int):
        assert reminder_id == blocked_during_delivery.reminder_id
        await persist_fence(blocked_during_delivery.occurrence_id)
        return await original_deliver(reminder_id, now_ms=now_ms)

    monkeypatch.setattr(
        first, "_deliver_claimed_reminder_unlocked", persist_before_final_delivery,
    )
    assert await first.deliver_authorized_reminders(
        (blocked_during_delivery.occurrence_id,), now_ms=now,
    ) == ()
    assert (await first.get_reminder(blocked_during_delivery.reminder_id)).state == "claimed"
    for reminder in (
        blocked_before_claim,
        blocked_before_delivery,
        blocked_during_delivery,
    ):
        assert await first.get_message_by_envelope(
            f"reminder-occurrence:{reminder.occurrence_id}"
        ) is None
    await first.close()
    await second.close()
