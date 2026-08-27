"""Spawn helper: a worker task that dies unclaimed logs instead of vanishing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable
from weakref import WeakSet

logger = logging.getLogger(__name__)

# ``ensure_future`` returns an existing Task/Future unchanged.  Keeping the
# registration set weak makes ``spawn(existing_task)`` idempotent without
# extending the task's lifetime.
_reported_tasks: WeakSet[asyncio.Future[Any]] = WeakSet()


def _task_label(task: asyncio.Future[Any]) -> str:
    get_name = getattr(task, "get_name", None)
    return get_name() if get_name is not None else repr(task)


def _report_task_death(task: asyncio.Future[Any]) -> None:
    # cancelled: shutdown path, not a failure
    if task.cancelled():
        return
    error = task.exception()
    if error is None:
        return
    logger.error("worker task died: %s", _task_label(task), exc_info=error)


def spawn(
    coro: Awaitable[Any],
    *,
    name: str | None = None,
    report_failure: bool = True,
) -> asyncio.Future[Any]:
    """Schedule ``coro`` and optionally report an otherwise detached failure.

    Pass ``report_failure=False`` only when an owning path is guaranteed to
    await or inspect the returned future and emit exactly one alertable,
    traceback-bearing failure record.  Merely intending to inspect it, or
    logging at DEBUG, is not sufficient.
    """
    if asyncio.iscoroutine(coro):
        # running-loop only: preserves create_task's RuntimeError contract
        task: asyncio.Future[Any] = asyncio.get_running_loop().create_task(coro, name=name)
    else:
        task = asyncio.ensure_future(coro)
        if name is not None and isinstance(task, asyncio.Task):
            task.set_name(name)
    if report_failure and task not in _reported_tasks:
        _reported_tasks.add(task)
        task.add_done_callback(_report_task_death)
    return task
