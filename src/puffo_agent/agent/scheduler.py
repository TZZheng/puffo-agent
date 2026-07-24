"""PUF-394: the scheduler engine — crontab next-run compute + the tick.

Runs as a per-agent Worker background task (holds `self._client` to fire).
Pure-ish: the actual turn-injection is an injected `fire` callback so the
engine is unit-testable without a live client. openworker-modeled:

- **catch-up-once**: `next_run_at` after a fire is always computed from *now*,
  never from the stale slot — so a daemon that was down for 5 hourly slots
  fires once and reschedules to the next future slot, not five times.
- **no-overlap**: a job with an open run (prior fire still processing) is
  skipped for this occurrence (status ``skipped``) and rescheduled.

The engine tick is driven by the Worker's normal harness ``_run`` loop
(cli-local / cli-docker / codex — the FB-425 target). ws-local-idle agents get
the store (CRUD via the MCP tools works) but no tick, so their jobs won't fire
until they run under the harness path — a documented follow-up, not a bug.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from croniter import croniter

from .scheduler_store import SchedulerStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def is_valid_cron(cron_expr: str) -> bool:
    return croniter.is_valid(cron_expr)


def next_run(cron_expr: str, after_ms: int) -> int:
    """Next crontab occurrence strictly after `after_ms`, in ms-epoch (UTC)."""
    base = datetime.fromtimestamp(after_ms / 1000, tz=timezone.utc)
    return int(croniter(cron_expr, base).get_next(datetime).timestamp() * 1000)


# fire(job_dict, envelope_id) → admits the synthetic cron turn onto the queue.
FireFn = Callable[[dict, str], Awaitable[None]]


class SchedulerEngine:
    def __init__(
        self,
        store: SchedulerStore,
        fire: FireFn,
        *,
        now_fn: Callable[[], int] = _now_ms,
    ) -> None:
        self._store = store
        self._fire = fire
        self._now = now_fn

    async def tick(self) -> int:
        """Fire every due job once; return the count fired. A single tick on
        startup does the catch-up-once (due_jobs returns all past-due, each
        reschedules forward from now)."""
        now = self._now()
        fired = 0
        for job in await self._store.due_jobs(now):
            nxt = next_run(job["cron_expr"], now)
            # No-overlap: a prior fire still in flight → skip this occurrence.
            if await self._store.has_open_run(job["id"]):
                await self._store.update_job(
                    job["id"], next_run_at=nxt, last_run_status="skipped"
                )
                continue
            envelope_id = f"cron-{job['id']}-{now}"
            await self._store.open_run(job["id"], envelope_id, now)
            # Advance next_run_at BEFORE firing so a crash mid-fire can't
            # re-fire the same slot on the next tick. Trade-off (intended): a
            # hard crash (SIGKILL) between here and the fire leaves next_run_at
            # advanced with an open run that never completes — that occurrence
            # is skipped, not retried. Preferred over double-firing.
            await self._store.record_fire(job["id"], now, nxt)
            try:
                await self._fire(job, envelope_id)
                fired += 1
            except Exception:  # noqa: BLE001 — a bad fire shouldn't wedge the tick
                await self._store.close_run(envelope_id, "error", self._now())
                await self._store.set_last_status(job["id"], "error")
        return fired

    async def record_completion(self, envelope_id: str, status: str) -> None:
        """Called by the Worker when a fired cron turn finishes: close the run
        + stamp the job's last_run_status."""
        job_id = await self._store.job_for_open_envelope(envelope_id)
        await self._store.close_run(envelope_id, status, self._now())
        if job_id is not None:
            await self._store.set_last_status(job_id, status)
