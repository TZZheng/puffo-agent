from __future__ import annotations

import asyncio

import pytest

from puffo_agent.agent.message_store import MessageStore
from puffo_agent.agent.reminder_scheduler import ReminderScheduler


@pytest.mark.asyncio
async def test_scheduler_delivers_due_and_late_facts_without_early_fire(tmp_path):
    now = [1_000]
    store = MessageStore(tmp_path / "messages.db", now_ms=lambda: now[0])
    wakes: list[str] = []
    scheduler = ReminderScheduler(
        store=store, notify=lambda: wakes.append("wake"), now_ms=lambda: now[0],
    )
    future = await scheduler.create_reminder(
        content="future exact content",
        target="channel:sp:ch",
        intended_at="1970-01-01T00:00:02Z",
    )
    assert await scheduler.process_due_once() == ()
    assert (await store.get_reminder(future["reminder_id"])).state == "scheduled"
    assert wakes == []

    late = await scheduler.create_reminder(
        content="past exact content",
        target="dm:peer",
        intended_at="1970-01-01T00:00:00.500Z",
    )
    delivered = await scheduler.process_due_once()
    assert [item.reminder_id for item in delivered] == [late["reminder_id"]]
    late_fact = delivered[0].as_dict()
    assert late_fact["actual_fire_at"] >= late_fact["intended_at"]
    assert wakes == ["wake"]

    now[0] = 2_000
    delivered = await scheduler.process_due_once()
    assert [item.reminder_id for item in delivered] == [future["reminder_id"]]
    await store.close()


@pytest.mark.asyncio
async def test_scheduler_recovers_claimed_work_after_reopen(tmp_path):
    now = [5_000]
    path = tmp_path / "claimed.db"
    store = MessageStore(path, now_ms=lambda: now[0])
    reminder = await store.create_reminder(
        content="recover", target="channel:sp:ch", intended_at_ms=1_000,
    )
    await store.claim_due_reminders(now_ms=5_000)
    await store.close()

    reopened = MessageStore(path, now_ms=lambda: now[0])
    wakes: list[str] = []
    scheduler = ReminderScheduler(
        store=reopened, notify=lambda: wakes.append("wake"), now_ms=lambda: now[0],
    )
    delivered = await scheduler.process_due_once()
    assert [item.reminder_id for item in delivered] == [reminder.reminder_id]
    assert (await reopened.get_reminder(reminder.reminder_id)).state == "delivered"
    assert wakes == ["wake"]
    await reopened.close()


@pytest.mark.asyncio
async def test_scheduler_notifies_committed_deliveries_once_per_batch(tmp_path):
    store = MessageStore(tmp_path / "batch.db", now_ms=lambda: 2_000)
    commits: list[str] = []
    scheduler = ReminderScheduler(
        store=store,
        notify=lambda: None,
        now_ms=lambda: 2_000,
        on_deliveries_committed=lambda: commits.append("committed"),
    )
    for reminder_id in ("reminder-one", "reminder-two"):
        await store.create_reminder(
            reminder_id=reminder_id,
            occurrence_id=f"occurrence-{reminder_id}",
            content="due",
            target="dm:peer",
            intended_at_ms=1_000,
        )

    assert len(await scheduler.process_due_once()) == 2
    assert commits == ["committed"]
    assert await scheduler.process_due_once() == ()
    assert commits == ["committed"]
    await store.close()


@pytest.mark.asyncio
async def test_scheduler_earlier_create_replans_wait_and_stop_has_no_leaked_task(tmp_path):
    now = [1_000]
    waits: list[float | None] = []
    entered_wait = asyncio.Event()

    async def wait_for_change(event: asyncio.Event, timeout: float | None) -> None:
        waits.append(timeout)
        entered_wait.set()
        await event.wait()

    store = MessageStore(tmp_path / "wait.db", now_ms=lambda: now[0])
    scheduler = ReminderScheduler(
        store=store,
        notify=lambda: None,
        now_ms=lambda: now[0],
        wait_for_change=wait_for_change,
    )
    task = asyncio.create_task(scheduler.run())
    await asyncio.wait_for(entered_wait.wait(), timeout=1)
    assert waits == [None]

    async def wait_for_count(count: int) -> None:
        while len(waits) < count:
            await asyncio.sleep(0)

    await scheduler.create_reminder(
        content="later", target="channel:sp:ch",
        intended_at="1970-01-01T00:00:10Z",
    )
    await asyncio.wait_for(wait_for_count(2), timeout=1)
    assert waits[-1] == 9.0

    await scheduler.create_reminder(
        content="earlier", target="channel:sp:ch",
        intended_at="1970-01-01T00:00:02Z",
    )
    await asyncio.wait_for(wait_for_count(3), timeout=1)
    assert waits[-1] == 1.0
    scheduler.stop()
    await task
    assert task.done() and not task.cancelled()
    await store.close()
