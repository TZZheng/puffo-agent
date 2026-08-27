"""Preserve primary failures while making cleanup failures queryable."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from ...tasks import spawn


_CLEANUP_ERRORS_ATTR = "_puffo_cleanup_errors"


def attach_cleanup_error(
    primary: BaseException, cleanup: BaseException
) -> None:
    """Attach structured cleanup evidence without replacing the primary."""
    errors = (*cleanup_errors(primary), cleanup)
    setattr(primary, _CLEANUP_ERRORS_ATTR, errors)
    primary.add_note(
        f"puffo cleanup failure: {type(cleanup).__name__}: {cleanup}"
    )


def cleanup_errors(error: BaseException) -> tuple[BaseException, ...]:
    """Return structured cleanup failures attached to an error."""
    value = getattr(error, _CLEANUP_ERRORS_ATTR, ())
    return value if isinstance(value, tuple) else ()


async def collect_cleanup_errors(
    awaitable: Awaitable[Any], errors: list[BaseException]
) -> None:
    """Finish one cleanup operation even when its caller is cancelled."""
    async def finish() -> Any:
        return await awaitable

    task = spawn(finish(), name="harness.cleanup")
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            errors.append(exc)
            if task.done():
                return
            continue
        except BaseException as exc:
            errors.append(exc)
        return


def raise_collected_errors(
    label: str, errors: list[BaseException]
) -> None:
    """Raise failures without ever grouping or replacing cancellation."""
    if not errors:
        return
    cancelled = next(
        (error for error in errors if isinstance(error, asyncio.CancelledError)),
        None,
    )
    if cancelled is not None:
        for error in errors:
            if error is not cancelled:
                attach_cleanup_error(cancelled, error)
        raise cancelled
    non_exception = next(
        (error for error in errors if not isinstance(error, Exception)),
        None,
    )
    if non_exception is not None:
        for error in errors:
            if error is not non_exception:
                attach_cleanup_error(non_exception, error)
        raise non_exception
    if len(errors) == 1:
        raise errors[0]
    raise ExceptionGroup(label, errors)  # type: ignore[arg-type]
