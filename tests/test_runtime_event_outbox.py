from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from puffo_agent.agent.runtime_event_outbox import (
    APPEND_PATH,
    OutboxCapacityError,
    RuntimeEventOutbox,
    RuntimeEventProjectingSink,
    RuntimeEventUploader,
    runtime_event_outbox_path,
)
from puffo_agent.agent.harness.driver import HarnessEvent, SessionRef, TurnRef
from puffo_agent.agent.runtime_events import RuntimeEvent, RuntimeEventProjector


def event(index: int, type_: str = "turn.started") -> RuntimeEvent:
    return RuntimeEvent(
        agent_id="agent", session_ref="session", turn_ref=f"turn_{index}",
        type=type_, payload={} if type_ == "turn.started" else {
            "outcome": "succeeded"
        },
        event_id=f"evt_{index}_{type_}",
    )


@pytest.mark.asyncio
async def test_restart_retry_order_and_append(tmp_path):
    path = tmp_path / "runtime_events.db"
    first = RuntimeEventOutbox(path)
    await first.enqueue(event(1))
    first.close()
    second = RuntimeEventOutbox(path)
    attempts = []

    async def transport(path, body):
        attempts.append((path, body))
        if len(attempts) == 1:
            raise OSError("lost response")
        ids = [value["event_id"] for value in json.loads(body)["events"]]
        return 200, {"accepted": [
            {"event_id": value, "cursor": value} for value in ids
        ]}

    uploader = RuntimeEventUploader(second, transport)
    assert (await uploader.upload_once()).state == "retry"
    assert second.prefix()[0].retry_count == 1
    assert (await uploader.upload_once()).state == "uploaded"
    assert attempts[0] == attempts[1]
    assert attempts[0][0] == APPEND_PATH
    assert second.prefix() == []
    second.close()


@pytest.mark.asyncio
async def test_duplicate_event_id_is_idempotent_only_for_exact_content(tmp_path):
    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    original = event(1)
    first = await outbox.enqueue(original)
    second = await outbox.enqueue(original)
    assert second == first
    assert outbox.usage()[0] == 1
    with pytest.raises(ValueError, match="different content"):
        await outbox.enqueue(RuntimeEvent(
            agent_id="agent",
            session_ref="session",
            turn_ref="different-turn",
            type="turn.started",
            payload={},
            event_id=original.event_id,
        ))
    outbox.close()


@pytest.mark.asyncio
async def test_successful_exact_prefix_acknowledgement_leaves_suffix(tmp_path):
    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    await outbox.enqueue(event(1))
    await outbox.enqueue(event(2))
    prefix = outbox.prefix(max_rows=1)
    outbox.acknowledge(prefix, [prefix[0].event_id])
    remaining = outbox.prefix()
    assert [row.event_id for row in remaining] == [event(2).event_id]
    outbox.close()


@pytest.mark.asyncio
async def test_capacity_reserves_terminal_and_backpressures(tmp_path):
    outbox = RuntimeEventOutbox(
        tmp_path / "runtime_events.db", max_rows=2,
        terminal_reserve_rows=1, terminal_reserve_bytes=0,
    )
    await outbox.enqueue(event(1))
    assert not outbox.can_start_turn()
    with pytest.raises(OutboxCapacityError):
        await outbox.enqueue(event(2))
    await outbox.enqueue(event(1, "turn.finished"), terminal=True)
    assert outbox.usage()[0] == 2
    outbox.close()


def _initial_pair(*, occurred_at="9999-12-31T23:59:59.999999+00:00"):
    projector = RuntimeEventProjector(agent_id="agent", session_ref="session")
    return projector.project_all(HarnessEvent(
        type="turn.started", driver="fake", session_ref=SessionRef("native"),
        turn_ref=TurnRef("turn_" + "f" * 32), occurred_at=occurred_at,
    ))


@pytest.mark.asyncio
async def test_turn_capacity_requires_complete_initial_pair_and_terminal(tmp_path):
    start, activity = _initial_pair()
    terminal = event(1, "turn.finished")
    start_bytes = len(RuntimeEventOutbox.canonical_bytes(start))
    pair_bytes = sum(len(RuntimeEventOutbox.canonical_bytes(value)) for value in (start, activity))
    terminal_bytes = len(RuntimeEventOutbox.canonical_bytes(terminal))

    exact = RuntimeEventOutbox(
        tmp_path / "exact.db", max_rows=3,
        max_bytes=pair_bytes + terminal_bytes, terminal_reserve_bytes=terminal_bytes,
    )
    assert exact.can_start_turn(estimated_start_bytes=pair_bytes)
    await exact.enqueue(start)
    await exact.enqueue(activity)
    await exact.enqueue(terminal, terminal=True)
    assert [row.event_type for row in exact.prefix()] == [
        "turn.started", "activity.updated", "turn.finished",
    ]
    exact.close()

    # This is sufficient for the superseded one-row calculation, but is one
    # byte short of the lifecycle pair required by the public contract.
    short = RuntimeEventOutbox(
        tmp_path / "short.db", max_rows=3,
        max_bytes=pair_bytes + terminal_bytes - 1,
        terminal_reserve_bytes=terminal_bytes,
    )
    assert start_bytes + terminal_bytes <= short.max_bytes
    assert not short.can_start_turn(estimated_start_bytes=pair_bytes)
    with pytest.raises(OutboxCapacityError):
        short.require_turn_capacity(estimated_start_bytes=pair_bytes)
    assert short.usage() == (0, 0)
    await short.enqueue(terminal, terminal=True)
    assert [row.event_type for row in short.prefix()] == ["turn.finished"]
    short.close()


def test_maximum_timestamp_initial_estimate_covers_zero_microsecond_form():
    maximum = sum(
        len(RuntimeEventOutbox.canonical_bytes(value)) for value in _initial_pair()
    )
    zero_microseconds = sum(
        len(RuntimeEventOutbox.canonical_bytes(value))
        for value in _initial_pair(occurred_at="2026-07-30T12:00:00+00:00")
    )
    assert maximum >= zero_microseconds


@pytest.mark.asyncio
async def test_buffered_output_pressure_discards_nonterminal_but_keeps_terminal(
    tmp_path,
):
    outbox = RuntimeEventOutbox(
        tmp_path / "runtime_events.db", max_rows=3,
        terminal_reserve_rows=1, terminal_reserve_bytes=0,
    )
    sink = RuntimeEventProjectingSink(
        outbox,
        RuntimeEventProjector(agent_id="agent", session_ref="session"),
        delta_bytes=1024,
    )

    def native(type_, data=None):
        return HarnessEvent(
            type=type_, driver="fake", session_ref=SessionRef("native"),
            turn_ref=TurnRef("turn"), data=data or {},
        )

    await sink(native("turn.started"))
    await sink(native(
        "turn.assistant_delta", {"text": "buffered", "block_id": "result"}
    ))
    await sink(native("turn.completed", {"outcome": "succeeded"}))
    values = [row.event for row in outbox.prefix()]
    assert [value["type"] for value in values] == [
        "turn.started", "activity.updated", "turn.finished"
    ]
    assert values[-1]["payload"] == {"outcome": "succeeded"}
    outbox.close()


def test_location_is_daemon_agent_state_beside_messages_not_workspace(tmp_path):
    agent_state = tmp_path / "daemon-state" / "agents" / "agent"
    workspace = tmp_path / "user-workspace"
    messages = agent_state / "messages.db"
    path = runtime_event_outbox_path(agent_state)
    assert path == messages.with_name("runtime_events.db")
    assert workspace not in path.parents
    outbox = RuntimeEventOutbox(path)
    assert outbox.path == path
    outbox.close()


@pytest.mark.asyncio
async def test_byte_bound_and_terminal_capacity_reservation(tmp_path):
    start = event(1)
    terminal = event(1, "turn.finished")
    start_bytes = len(RuntimeEventOutbox.canonical_bytes(start))
    terminal_bytes = len(RuntimeEventOutbox.canonical_bytes(terminal))
    outbox = RuntimeEventOutbox(
        tmp_path / "runtime_events.db",
        max_rows=3,
        max_bytes=start_bytes + terminal_bytes,
        terminal_reserve_rows=1,
        terminal_reserve_bytes=terminal_bytes,
    )
    assert outbox.can_start_turn(estimated_start_bytes=start_bytes)
    await outbox.enqueue(start)
    assert outbox.usage() == (1, start_bytes)
    with pytest.raises(OutboxCapacityError):
        await outbox.enqueue(event(2))
    await outbox.enqueue(terminal, terminal=True)
    rows, bytes_ = outbox.usage()
    assert rows == 2
    assert bytes_ <= outbox.max_bytes
    outbox.close()


@pytest.mark.asyncio
async def test_coalescing_sink_projects_normalized_lifecycle(tmp_path):
    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    sink = RuntimeEventProjectingSink(
        outbox,
        RuntimeEventProjector(agent_id="agent", session_ref="session"),
        delta_bytes=4,
    )

    def native(type_, data=None):
        return HarnessEvent(
            type=type_, driver="fake", session_ref=SessionRef("native"),
            turn_ref=TurnRef("turn"), data=data or {},
        )

    await sink(native("turn.started"))
    for token in "abcdefghij":
        await sink(native(
            "turn.assistant_delta", {"text": token, "block_id": "result"}
        ))
    await sink(native(
        "turn.assistant_completed", {"block_id": "result"}
    ))
    await sink(native("turn.completed", {"outcome": "succeeded"}))
    values = [row.event for row in outbox.prefix()]
    deltas = [
        item["payload"]["delta"] for item in values
        if item["type"] == "output.updated"
        and item["payload"]["phase"] == "delta"
    ]
    assert "".join(deltas) == "abcdefghij"
    assert len(deltas) < 10
    assert values[0]["type"] == "turn.started"
    assert values[-1] == {
        **values[-1],
        "type": "turn.finished",
        "payload": {"outcome": "succeeded"},
    }
    outbox.close()


@pytest.mark.asyncio
async def test_order_uses_sequence_not_occurred_at(tmp_path):
    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    later_clock = RuntimeEvent(
        agent_id="agent", session_ref="session", turn_ref="turn",
        type="turn.started", payload={}, event_id="evt_inserted_first",
        occurred_at="2099-01-01T00:00:00Z",
    )
    earlier_clock = RuntimeEvent(
        agent_id="agent", session_ref="session", turn_ref="turn",
        type="turn.finished", payload={"outcome": "succeeded"},
        event_id="evt_inserted_second", occurred_at="2000-01-01T00:00:00Z",
    )
    await outbox.enqueue(later_clock)
    await outbox.enqueue(earlier_clock, terminal=True)
    assert [row.event_id for row in outbox.prefix()] == [
        "evt_inserted_first", "evt_inserted_second",
    ]
    outbox.close()


@pytest.mark.asyncio
async def test_malformed_head_is_discarded_so_later_sequence_can_progress(tmp_path):
    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    await outbox.enqueue(event(1))
    await outbox.enqueue(event(2))

    async def malformed(_path, _body):
        return 200, {"accepted": [{"event_id": "evt_2"}]}

    uploader = RuntimeEventUploader(outbox, malformed)
    # A rejected multi-row batch isolates its head before any discard.
    assert (await uploader.upload_once()).state == "retry"
    result = await uploader.upload_once()
    assert result.state == "discarded"
    assert [row.event_id for row in outbox.prefix()] == [
        "evt_2_turn.started",
    ]
    outbox.close()


@pytest.mark.asyncio
async def test_log_boundaries_are_json_safe_and_content_free(tmp_path, caplog):
    logger = logging.getLogger("test.runtime.outbox")
    caplog.set_level(logging.INFO, logger=logger.name)
    outbox = RuntimeEventOutbox(
        tmp_path / "runtime_events.db", logger=logger
    )
    secret = "sk-super-secret-payload-123456"
    value = RuntimeEvent(
        agent_id="agent", session_ref="session", turn_ref="turn",
        type="output.updated",
        payload={
            "block_id": "out", "kind": "result", "phase": "delta",
            "delta": secret,
        },
        event_id="evt_log",
    )
    await outbox.enqueue(value)

    attempts = 0

    async def transport(_path, body):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(secret)
        ids = [item["event_id"] for item in json.loads(body)["events"]]
        return 200, {"accepted": [{"event_id": item} for item in ids]}

    uploader = RuntimeEventUploader(outbox, transport)
    assert (await uploader.upload_once()).state == "retry"
    assert (await uploader.upload_once()).state == "uploaded"
    sink = RuntimeEventProjectingSink(
        outbox,
        RuntimeEventProjector(agent_id="agent", session_ref="session"),
    )
    await sink(HarnessEvent(
        type="turn.started", driver="fake",
        session_ref=SessionRef("native"), turn_ref=TurnRef("logged-turn"),
    ))
    await sink(HarnessEvent(
        type="turn.completed", driver="fake",
        session_ref=SessionRef("native"), turn_ref=TurnRef("logged-turn"),
        data={"outcome": "succeeded"},
    ))
    records = []
    for record in caplog.records:
        marker = "runtime_event="
        if marker in record.getMessage():
            records.append(json.loads(record.getMessage().split(marker, 1)[1]))
    names = {record["event"] for record in records}
    assert {
        "runtime.enqueued", "runtime.batch_attempt",
        "runtime.retry", "runtime.acknowledged",
        "runtime.normalized_event", "runtime.projected",
    } <= names
    enqueued = next(
        record for record in records if record["event"] == "runtime.enqueued"
    )
    assert enqueued == {
        "event": "runtime.enqueued", "agent_id": "agent",
        "session_ref": "session", "turn_ref": "turn",
        "event_id": "evt_log", "event_type": "output.updated",
        "outbox_sequence": 1,
    }
    assert any(
        record.get("retry_count") == 1
        for record in records if record["event"] == "runtime.retry"
    )
    assert secret not in "\n".join(record.getMessage() for record in caplog.records)
    outbox.close()
