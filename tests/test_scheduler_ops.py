"""PUF-394: scheduler_ops CRUD + validation + formatting."""
from __future__ import annotations

import pytest

from puffo_agent.agent import scheduler_ops as ops
from puffo_agent.agent.scheduler_store import SchedulerStore


async def _store(tmp_path) -> SchedulerStore:
    s = SchedulerStore(tmp_path / "scheduler.db")
    await s.open()
    return s


@pytest.mark.asyncio
async def test_create_persists_and_confirms(tmp_path):
    store = await _store(tmp_path)
    msg = await ops.create_op(
        store, name="daily-report", cron_expr="0 9 * * *",
        prompt="post the daily report to #general", channel_id="ch_1",
    )
    assert "Created recurring job cronjob_" in msg
    assert "persists across session/daemon restart" in msg
    jobs = await store.list_jobs()
    assert len(jobs) == 1 and jobs[0]["name"] == "daily-report"
    assert jobs[0]["next_run_at"] > 0


@pytest.mark.asyncio
async def test_create_rejects_bad_cron_and_blank_fields(tmp_path):
    store = await _store(tmp_path)
    assert "invalid cron_expr" in await ops.create_op(
        store, name="x", cron_expr="not a cron", prompt="p", channel_id=None
    )
    assert "name is required" in await ops.create_op(
        store, name="  ", cron_expr="0 9 * * *", prompt="p", channel_id=None
    )
    assert "prompt is required" in await ops.create_op(
        store, name="x", cron_expr="0 9 * * *", prompt="", channel_id=None
    )
    assert await store.list_jobs() == []  # nothing persisted on any error


@pytest.mark.asyncio
async def test_list_empty_and_populated(tmp_path):
    store = await _store(tmp_path)
    assert await ops.list_op(store) == "No recurring jobs."
    await ops.create_op(store, name="j1", cron_expr="*/5 * * * *", prompt="p", channel_id="ch_9")
    listing = await ops.list_op(store)
    assert "j1" in listing and "*/5 * * * *" in listing and "→ ch_9" in listing


@pytest.mark.asyncio
async def test_update_changes_cron_and_recomputes_next(tmp_path):
    store = await _store(tmp_path)
    await ops.create_op(store, name="j", cron_expr="0 9 * * *", prompt="p", channel_id=None)
    job_id = (await store.list_jobs())[0]["id"]
    before = (await store.get_job(job_id))["next_run_at"]

    msg = await ops.update_op(store, job_id=job_id, cron_expr="*/1 * * * *", enabled=False)
    assert "Updated" in msg and "disabled" in msg
    after = await store.get_job(job_id)
    assert after["cron_expr"] == "*/1 * * * *"
    assert after["enabled"] == 0
    assert after["next_run_at"] != before  # recomputed


@pytest.mark.asyncio
async def test_update_missing_and_bad_cron(tmp_path):
    store = await _store(tmp_path)
    assert "no job" in await ops.update_op(store, job_id="cronjob_nope", name="x")
    await ops.create_op(store, name="j", cron_expr="0 9 * * *", prompt="p", channel_id=None)
    job_id = (await store.list_jobs())[0]["id"]
    assert "invalid cron_expr" in await ops.update_op(store, job_id=job_id, cron_expr="bad")


@pytest.mark.asyncio
async def test_delete(tmp_path):
    store = await _store(tmp_path)
    await ops.create_op(store, name="j", cron_expr="0 9 * * *", prompt="p", channel_id=None)
    job_id = (await store.list_jobs())[0]["id"]
    assert "Deleted recurring job" in await ops.delete_op(store, job_id=job_id)
    assert "no job" in await ops.delete_op(store, job_id=job_id)
