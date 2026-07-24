"""PUF-394: agent-facing scheduler operations (create/list/update/delete).

The single place the CRUD + validation + LLM-facing formatting lives, called
by BOTH dispatch paths — the ws-local `PuffoCoreMessageClient` (in-process) and
the harness RPC handler (`host_mcp_handler`). Each returns a plain string that
becomes the MCP tool result the agent reads.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .scheduler import _now_ms, is_valid_cron, next_run
from .scheduler_store import SchedulerStore


def _fmt_ts(ms: int | None) -> str:
    if not ms:
        return "never"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _fmt_job(job: dict) -> str:
    state = "enabled" if job["enabled"] else "disabled"
    where = f" → {job['channel_id']}" if job.get("channel_id") else ""
    last = ""
    if job.get("last_run_at"):
        last = f"; last {_fmt_ts(job['last_run_at'])} ({job.get('last_run_status') or '?'})"
    return (
        f"{job['id']} \"{job['name']}\" [{job['cron_expr']}] {state}{where}; "
        f"next {_fmt_ts(job['next_run_at'])}{last}"
    )


def server_snapshot(jobs: list[dict]) -> list[dict]:
    """Map local job rows → the server-mirror wire shape (state-replace)."""
    return [
        {
            "job_id": j["id"],
            "name": j["name"],
            "cron_expr": j["cron_expr"],
            "enabled": bool(j["enabled"]),
            "last_run_at": j["last_run_at"],
            "last_run_status": j["last_run_status"],
            "next_run_at": j["next_run_at"],
        }
        for j in jobs
    ]


async def create_op(
    store: SchedulerStore,
    *,
    name: str,
    cron_expr: str,
    prompt: str,
    channel_id: str | None,
    now_ms: int | None = None,
) -> str:
    if not (name or "").strip():
        return "error: name is required"
    if not (prompt or "").strip():
        return "error: prompt is required (the instruction to run each time)"
    if not is_valid_cron(cron_expr or ""):
        return f'error: invalid cron_expr {cron_expr!r} (use standard crontab, e.g. "0 9 * * *")'
    nxt = next_run(cron_expr, now_ms if now_ms is not None else _now_ms())
    job = await store.create_job(
        name=name.strip(),
        cron_expr=cron_expr,
        prompt=prompt,
        channel_id=channel_id or None,
        next_run_at=nxt,
    )
    return (
        f"Created recurring job {job['id']} \"{job['name']}\" [{cron_expr}]; "
        f"next run {_fmt_ts(nxt)}. This persists across session/daemon restart."
    )


async def list_op(store: SchedulerStore) -> str:
    jobs = await store.list_jobs()
    if not jobs:
        return "No recurring jobs."
    return "\n".join(_fmt_job(j) for j in jobs)


async def update_op(
    store: SchedulerStore,
    *,
    job_id: str,
    name: str | None = None,
    cron_expr: str | None = None,
    prompt: str | None = None,
    channel_id: str | None = None,
    enabled: bool | None = None,
    now_ms: int | None = None,
) -> str:
    if await store.get_job(job_id) is None:
        return f"error: no job {job_id!r}"
    fields: dict = {}
    if name is not None:
        fields["name"] = name.strip()
    if prompt is not None:
        fields["prompt"] = prompt
    if channel_id is not None:
        fields["channel_id"] = channel_id
    if enabled is not None:
        fields["enabled"] = 1 if enabled else 0
    if cron_expr is not None:
        if not is_valid_cron(cron_expr):
            return f"error: invalid cron_expr {cron_expr!r}"
        fields["cron_expr"] = cron_expr
        fields["next_run_at"] = next_run(
            cron_expr, now_ms if now_ms is not None else _now_ms()
        )
    updated = await store.update_job(job_id, **fields)
    assert updated is not None
    return f"Updated {_fmt_job(updated)}"


async def delete_op(store: SchedulerStore, *, job_id: str) -> str:
    if await store.delete_job(job_id):
        return f"Deleted recurring job {job_id}."
    return f"error: no job {job_id!r}"
