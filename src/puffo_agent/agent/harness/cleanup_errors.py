"""Preserve primary failures while making cleanup failures queryable."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from ...tasks import spawn


_CLEANUP_ERRORS_ATTR = "_puffo_cleanup_errors"
_SUPPRESSED_ERRORS_ATTR = "_puffo_suppressed_primary_errors"
CLEANUP_TIMEOUT_SECONDS = 10.0


class CleanupTimeoutError(TimeoutError):
    """A supervised cleanup operation exceeded its explicit upper bound."""


def attach_cleanup_error(
    primary: BaseException, cleanup: BaseException
) -> None:
    """Attach structured cleanup evidence without replacing the primary."""
    errors = (*cleanup_errors(primary), cleanup)
    setattr(primary, _CLEANUP_ERRORS_ATTR, errors)
    primary.add_note(
        f"puffo cleanup failure: {type(cleanup).__name__}: {cleanup}"
    )


def attach_suppressed_primary_error(
    cancellation: BaseException, primary: BaseException
) -> None:
    """Preserve a real failure displaced by the raw cancellation contract."""
    value = getattr(cancellation, _SUPPRESSED_ERRORS_ATTR, ())
    if not isinstance(value, tuple):
        raise TypeError("malformed structured suppressed-error evidence")
    setattr(cancellation, _SUPPRESSED_ERRORS_ATTR, (*value, primary))
    cancellation.add_note(
        "puffo primary failure suppressed by cancellation: "
        f"{type(primary).__name__}: {primary}"
    )


def mark_cleanup_checked(primary: BaseException) -> None:
    """Record that the structured cleanup protocol ran without a failure."""
    if not hasattr(primary, _CLEANUP_ERRORS_ATTR):
        setattr(primary, _CLEANUP_ERRORS_ATTR, ())


def cleanup_errors(error: BaseException) -> tuple[BaseException, ...]:
    """Return structured cleanup failures attached to an error."""
    if not hasattr(error, _CLEANUP_ERRORS_ATTR):
        raise LookupError("error has no structured cleanup evidence")
    value = getattr(error, _CLEANUP_ERRORS_ATTR)
    if not isinstance(value, tuple) or not all(
        isinstance(item, BaseException) for item in value
    ):
        raise TypeError("malformed structured cleanup evidence")
    return value


def suppressed_primary_errors(
    error: BaseException,
) -> tuple[BaseException, ...]:
    """Return real failures displaced by cancellation, if any."""
    value = getattr(error, _SUPPRESSED_ERRORS_ATTR, ())
    if not isinstance(value, tuple) or not all(
        isinstance(item, BaseException) for item in value
    ):
        raise TypeError("malformed structured suppressed-error evidence")
    return value


async def collect_cleanup_errors(
    awaitable: Awaitable[Any],
    errors: list[BaseException],
    *,
    timeout: float,
) -> None:
    """Supervise cleanup through cancellation, but never beyond ``timeout``."""
    if timeout <= 0:
        raise ValueError("cleanup timeout must be positive")

    async def finish() -> Any:
        return await awaitable

    task = spawn(finish(), name="harness.cleanup")
    deadline = asyncio.get_running_loop().time() + timeout
    cancellation: asyncio.CancelledError | None = None

    async def cancel_without_waiting_forever() -> None:
        nonlocal cancellation
        task.cancel()
        try:
            # Give ordinary cancellation-aware coroutines one turn to run
            # their ``finally`` blocks.  A coroutine that suppresses
            # cancellation is deliberately left to the task supervisor.
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
                errors.append(exc)

    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            await cancel_without_waiting_forever()
            errors.append(
                CleanupTimeoutError(
                    f"cleanup exceeded {timeout:g} seconds"
                )
            )
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
                errors.append(exc)
            if task.done():
                return
            continue
        except TimeoutError:
            await cancel_without_waiting_forever()
            errors.append(
                CleanupTimeoutError(
                    f"cleanup exceeded {timeout:g} seconds"
                )
            )
            return
        except BaseException as exc:
            errors.append(exc)
        return


def raise_collected_errors(
    label: str, errors: list[BaseException]
) -> None:
    """Raise failures without ever grouping or replacing cancellation."""
    if not errors:
        return
    cancelled_index = next(
        (
            index
            for index, error in enumerate(errors)
            if isinstance(error, asyncio.CancelledError)
        ),
        None,
    )
    cancelled = (
        errors[cancelled_index] if cancelled_index is not None else None
    )
    if cancelled is not None:
        mark_cleanup_checked(cancelled)
        assert cancelled_index is not None
        for error in errors[:cancelled_index]:
            attach_suppressed_primary_error(cancelled, error)
        for error in errors[cancelled_index + 1 :]:
            if not isinstance(error, asyncio.CancelledError):
                attach_cleanup_error(cancelled, error)
        raise cancelled
    non_exception = next(
        (error for error in errors if not isinstance(error, Exception)),
        None,
    )
    if non_exception is not None:
        mark_cleanup_checked(non_exception)
        for error in errors:
            if error is not non_exception:
                attach_cleanup_error(non_exception, error)
        raise non_exception
    if len(errors) == 1:
        mark_cleanup_checked(errors[0])
        raise errors[0]
    group = ExceptionGroup(label, errors)  # type: ignore[arg-type]
    setattr(group, _CLEANUP_ERRORS_ATTR, tuple(errors[1:]))
    raise group
