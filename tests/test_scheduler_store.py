"""PUF-394: SchedulerStore CRUD + due-query + no-overlap run tracking."""
from __future__ import annotations

import pytest

from puffo_agent.agent.scheduler_store import SchedulerStore


async def _fresh(tmp_path) -> SchedulerStore:
    s = SchedulerStore(tmp_path / "scheduler.db")
    await s.open()
    return s


async def _mk(store, *, name="daily", cron="0 9 * * *", next_run_at=1000) -> dict:
    return await store.create_job(
        name=name, cron_expr=cron, prompt="post the report", channel_id="ch_1",
        next_run_at=next_run_at,
    )


@pytest.mark.asyncio
async def test_create_then_get_and_list(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store)
    assert job["id"].startswith("cronjob_")
    assert job["name"] == "daily"
    assert job["enabled"] == 1
    assert job["last_run_at"] is None
    got = await store.get_job(job["id"])
    assert got == job
    assert [j["id"] for j in await store.list_jobs()] == [job["id"]]


@pytest.mark.asyncio
async def test_get_missing_returns_none(tmp_path):
    store = await _fresh(tmp_path)
    assert await store.get_job("cronjob_nope") is None


@pytest.mark.asyncio
async def test_update_mutable_fields_only(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store)
    updated = await store.update_job(
        job["id"], cron_expr="*/5 * * * *", enabled=0, name="renamed",
    )
    assert updated["cron_expr"] == "*/5 * * * *"
    assert updated["enabled"] == 0
    assert updated["name"] == "renamed"
    assert updated["updated_at"] >= job["updated_at"]
    # created_at is immutable even if passed.
    again = await store.update_job(job["id"], created_at=0)
    assert again["created_at"] == job["created_at"]


@pytest.mark.asyncio
async def test_update_missing_returns_none(tmp_path):
    store = await _fresh(tmp_path)
    assert await store.update_job("cronjob_nope", name="x") is None


@pytest.mark.asyncio
async def test_delete(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store)
    assert await store.delete_job(job["id"]) is True
    assert await store.delete_job(job["id"]) is False
    assert await store.get_job(job["id"]) is None


@pytest.mark.asyncio
async def test_due_jobs_respects_enabled_and_next_run(tmp_path):
    store = await _fresh(tmp_path)
    a = await _mk(store, name="past", next_run_at=500)
    b = await _mk(store, name="future", next_run_at=5000)
    disabled = await _mk(store, name="off", next_run_at=100)
    await store.update_job(disabled["id"], enabled=0)

    due = await store.due_jobs(now_ms=1000)
    assert [j["id"] for j in due] == [a["id"]]  # b is future, disabled excluded
    _ = b


@pytest.mark.asyncio
async def test_no_overlap_open_run_guard(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store)
    assert await store.has_open_run(job["id"]) is False
    await store.open_run(job["id"], envelope_id="cron-x-1", started_at=1000)
    assert await store.has_open_run(job["id"]) is True
    # Closing the run clears the guard.
    assert await store.close_run("cron-x-1", status="ok", ended_at=2000) is True
    assert await store.has_open_run(job["id"]) is False
    # Second close is a no-op (already closed).
    assert await store.close_run("cron-x-1", status="ok", ended_at=3000) is False


@pytest.mark.asyncio
async def test_record_fire_advances_next_run(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store, next_run_at=1000)
    await store.record_fire(job["id"], last_run_at=1000, next_run_at=90000)
    got = await store.get_job(job["id"])
    assert got["last_run_at"] == 1000
    assert got["next_run_at"] == 90000


@pytest.mark.asyncio
async def test_set_last_status(tmp_path):
    store = await _fresh(tmp_path)
    job = await _mk(store)
    await store.set_last_status(job["id"], "error")
    assert (await store.get_job(job["id"]))["last_run_status"] == "error"


@pytest.mark.asyncio
async def test_survives_reopen(tmp_path):
    path = tmp_path / "scheduler.db"
    s1 = SchedulerStore(path)
    await s1.open()
    job = await s1.create_job(
        name="persist", cron_expr="0 * * * *", prompt="p", channel_id=None,
        next_run_at=42,
    )
    await s1.close()
    # New store on the same file (models a daemon restart) sees the job.
    s2 = SchedulerStore(path)
    await s2.open()
    got = await s2.get_job(job["id"])
    assert got is not None and got["next_run_at"] == 42
    assert got["channel_id"] is None
    await s2.close()
