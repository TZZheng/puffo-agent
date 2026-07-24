"""PUF-394: cron RPC handler integration + wiring source-pins.

The host_mcp_handler cron_* fns are directly testable (they only need a ctx
carrying a real SchedulerStore). The MCP-tool / RPC-route / Worker-tick glue
runs only inside a live daemon, so it's source-pinned here (+ py_compile in CI)
and flagged for runtime QA.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import puffo_agent
from puffo_agent.agent.scheduler_store import SchedulerStore
from puffo_agent.agent.status_reporter import _LOCAL_ONLY_ENVELOPE_PREFIXES
from puffo_agent.portal import host_mcp_handler as h


async def _ctx(tmp_path):
    store = SchedulerStore(tmp_path / "scheduler.db")
    await store.open()
    return SimpleNamespace(scheduler_store=store)


@pytest.mark.asyncio
async def test_cron_handlers_crud_through_ctx(tmp_path):
    ctx = await _ctx(tmp_path)
    created = await h.cron_create(
        ctx, name="daily", cron_expr="0 9 * * *", prompt="report", channel_id="ch_1",
    )
    assert "Created recurring job cronjob_" in created

    listing = await h.cron_list(ctx)
    assert "daily" in listing and "0 9 * * *" in listing
    job_id = (await ctx.scheduler_store.list_jobs())[0]["id"]

    # Omitted update fields are left unchanged (blank → None normalization).
    updated = await h.cron_update(
        ctx, job_id=job_id, name="", cron_expr="*/5 * * * *", prompt="",
        channel_id="", enabled=False,
    )
    assert "*/5 * * * *" in updated and "disabled" in updated
    row = await ctx.scheduler_store.get_job(job_id)
    assert row["name"] == "daily"  # unchanged (blank name normalized to None)
    assert row["cron_expr"] == "*/5 * * * *"

    assert "Deleted recurring job" in await h.cron_delete(ctx, job_id=job_id)


@pytest.mark.asyncio
async def test_cron_handler_errors_when_store_missing(tmp_path):
    ctx = SimpleNamespace(scheduler_store=None)
    with pytest.raises(RuntimeError):
        await h.cron_list(ctx)


def test_cron_envelope_prefix_is_local_only():
    # Fired cron turns must skip the server /processing round-trips.
    assert "cron-" in _LOCAL_ONLY_ENVELOPE_PREFIXES


# ── source-pins for the daemon-only glue ─────────────────────────────


def _src(rel: str) -> str:
    return (Path(puffo_agent.__file__).parent / rel).read_text(encoding="utf-8")


def test_mcp_tools_registered():
    src = _src("mcp/puffo_core_tools.py")
    for tool in ("async def cron_create", "async def cron_list",
                 "async def cron_update", "async def cron_delete"):
        assert tool in src
    # both dispatch branches present (ws-local + harness RPC).
    assert "cfg.message_client.cron_create" in src
    assert "cfg.rpc_client.cron_create" in src


def test_rpc_routes_registered():
    src = _src("portal/rpc_service.py")
    for route in ("cron-create", "cron-list", "cron-update", "cron-delete"):
        assert route in src


def test_worker_wires_engine_tick_and_fire():
    src = _src("portal/worker.py")
    assert "SchedulerEngine(" in src
    assert "scheduler_task = asyncio.ensure_future(scheduler_loop())" in src
    assert "record_completion(root_id" in src  # cron turn completion hook
    assert "scheduler_store=scheduler_store" in src  # threaded into the client


def test_client_has_fire_and_cron_methods():
    src = _src("agent/puffo_core_client.py")
    assert "async def fire_cron_job" in src
    assert "_admit_thread_message" in src
    for m in ("async def cron_create", "async def cron_list",
              "async def cron_update", "async def cron_delete"):
        assert m in src
