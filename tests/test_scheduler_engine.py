"""PUF-394: SchedulerEngine tick — fire, catch-up-once, no-overlap, completion."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from puffo_agent.agent.scheduler import (
    SchedulerEngine,
    is_valid_cron,
    next_run,
)
from puffo_agent.agent.scheduler_store import SchedulerStore

# 2026-07-24 10:00:00 UTC in ms.
T0 = int(datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
HOUR = 3_600_000


def test_is_valid_cron():
    assert is_valid_cron("0 9 * * *")
    assert is_valid_cron("*/5 * * * *")
    assert not is_valid_cron("nonsense")
    assert not is_valid_cron("99 99 * * *")


def test_next_run_is_the_next_future_occurrence():
    # Daily at 09:00 UTC; from 10:00 the next is tomorrow 09:00.
    got = next_run("0 9 * * *", T0)
    assert got == int(datetime(2026, 7, 25, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # Strictly after the base.
    assert next_run("*/5 * * * *", T0) > T0


class _Fires:
    def __init__(self, *, boom: set[str] | None = None):
        self.calls: list[tuple[str, str]] = []  # (job_id, envelope_id)
        self._boom = boom or set()

    async def __call__(self, job: dict, envelope_id: str) -> None:
        self.calls.append((job["id"], envelope_id))
        if job["id"] in self._boom:
            raise RuntimeError("fire failed")


async def _store(tmp_path) -> SchedulerStore:
    s = SchedulerStore(tmp_path / "scheduler.db")
    await s.open()
    return s


@pytest.mark.asyncio
async def test_tick_fires_due_job_and_advances(tmp_path):
    store = await _store(tmp_path)
    fires = _Fires()
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)
    job = await store.create_job(
        name="daily", cron_expr="0 9 * * *", prompt="report", channel_id="ch_1",
        next_run_at=T0 - HOUR,  # due
    )

    assert await engine.tick() == 1
    assert fires.calls == [(job["id"], f"cron-{job['id']}-{T0}")]
    got = await store.get_job(job["id"])
    assert got["last_run_at"] == T0
    assert got["next_run_at"] > T0  # rescheduled forward
    assert await store.has_open_run(job["id"]) is True  # run left open until completion


@pytest.mark.asyncio
async def test_catch_up_once_not_backfill(tmp_path):
    store = await _store(tmp_path)
    fires = _Fires()
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)
    # Hourly job that was due 5 hours ago (daemon was down).
    await store.create_job(
        name="hourly", cron_expr="0 * * * *", prompt="p", channel_id=None,
        next_run_at=T0 - 5 * HOUR,
    )
    fired = await engine.tick()
    assert fired == 1  # fires ONCE, not 5×
    # Next tick: not due (rescheduled to the future) → nothing fires.
    assert await engine.tick() == 0
    assert len(fires.calls) == 1


@pytest.mark.asyncio
async def test_no_overlap_skips_when_prior_run_open(tmp_path):
    store = await _store(tmp_path)
    fires = _Fires()
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)
    job = await store.create_job(
        name="slow", cron_expr="* * * * *", prompt="p", channel_id=None,
        next_run_at=T0 - 60_000,
    )
    # Simulate a prior fire still in flight.
    await store.open_run(job["id"], "cron-prior", started_at=T0 - 60_000)

    assert await engine.tick() == 0  # skipped, not fired
    assert fires.calls == []
    got = await store.get_job(job["id"])
    assert got["last_run_status"] == "skipped"
    assert got["next_run_at"] > T0  # still rescheduled so it doesn't hot-loop


@pytest.mark.asyncio
async def test_fire_error_marks_run_and_job(tmp_path):
    store = await _store(tmp_path)
    job = await store.create_job(
        name="x", cron_expr="* * * * *", prompt="p", channel_id=None,
        next_run_at=T0 - 60_000,
    )
    fires = _Fires(boom={job["id"]})
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)

    await engine.tick()
    got = await store.get_job(job["id"])
    assert got["last_run_status"] == "error"
    assert await store.has_open_run(job["id"]) is False  # run closed on failure
    # record_fire ran BEFORE the failing fire (crash-safety ordering): a crash
    # mid-fire must leave next_run_at already advanced so the slot can't re-fire.
    assert got["next_run_at"] > T0


@pytest.mark.asyncio
async def test_record_completion_closes_run_and_stamps_status(tmp_path):
    store = await _store(tmp_path)
    fires = _Fires()
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)
    job = await store.create_job(
        name="x", cron_expr="0 9 * * *", prompt="p", channel_id=None,
        next_run_at=T0 - HOUR,
    )
    await engine.tick()
    (_, envelope_id) = fires.calls[0]

    await engine.record_completion(envelope_id, "ok")
    assert await store.has_open_run(job["id"]) is False
    assert (await store.get_job(job["id"]))["last_run_status"] == "ok"


@pytest.mark.asyncio
async def test_disabled_and_future_jobs_dont_fire(tmp_path):
    store = await _store(tmp_path)
    fires = _Fires()
    engine = SchedulerEngine(store, fires, now_fn=lambda: T0)
    future = await store.create_job(
        name="future", cron_expr="0 9 * * *", prompt="p", channel_id=None,
        next_run_at=T0 + HOUR,
    )
    off = await store.create_job(
        name="off", cron_expr="0 9 * * *", prompt="p", channel_id=None,
        next_run_at=T0 - HOUR,
    )
    await store.update_job(off["id"], enabled=0)

    assert await engine.tick() == 0
    assert fires.calls == []
    _ = future
