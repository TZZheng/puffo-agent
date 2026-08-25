"""One shape for ``runtime.exited``, across every cross-process driver.

``RuntimeManager._handle_runtime_exit_locked`` republishes the event verbatim
to sinks and subscribers, so whatever payload a driver happens to build is
what every downstream consumer reads.  Before this contract there were four
shapes: ACP and Pi carried ``returncode``, Codex carried no data at all, and
Claude carried only its provider error.  "Did the child report an exit
status?" therefore had a different answer per provider, and nothing in the
suite asserted on a driver-emitted ``runtime.exited`` to notice.

The invariant is presence, not value: ``returncode`` may be ``None`` when the
status is not observable at the moment stdout ends.  A driver that omits the
key instead is indistinguishable, to a consumer using ``.get()``, from one
reporting a child that has not exited yet.
"""

from __future__ import annotations

import asyncio

import pytest

from puffo_agent.agent.harness.acp_driver import AcpDriver
from puffo_agent.agent.harness.claude_code_driver import ClaudeCodeCliDriver
from puffo_agent.agent.harness.codex_driver import CodexAppServerDriver
from puffo_agent.agent.harness.driver import (
    HarnessEvent,
    HarnessEventType,
    runtime_exited_data,
)
from puffo_agent.agent.harness.pi_driver import PiDriver


class _NullStdin:
    def write(self, _data: bytes) -> None:
        return None

    async def drain(self) -> None:
        return None


class _DeadChild:
    """A child whose stdout is already at EOF -- it died without saying so."""

    def __init__(self, returncode: int | None = 3) -> None:
        self.stdin = _NullStdin()
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()
        self.returncode = returncode

    async def wait(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _exit_event(driver) -> HarnessEvent:
    """The ``runtime.exited`` the driver put on its own event queue."""
    while not driver._events.empty():
        event = driver._events.get_nowait()
        if event is not None and event.type is HarnessEventType.RUNTIME_EXITED:
            return event
    raise AssertionError("driver emitted no runtime.exited")


def _assert_reports_an_exit_status(event: HarnessEvent) -> None:
    assert "returncode" in event.data, (
        f"driver {event.driver!r} omits returncode from runtime.exited; "
        "a consumer cannot tell that from a child with no status yet"
    )


@pytest.mark.asyncio
async def test_acp_exit_reports_a_status():
    driver = AcpDriver()
    driver._proc = _DeadChild()

    await driver._watch_process()

    _assert_reports_an_exit_status(_exit_event(driver))


@pytest.mark.asyncio
async def test_claude_exit_reports_a_status():
    driver = ClaudeCodeCliDriver()
    driver._proc = _DeadChild()

    await driver._read_loop()

    _assert_reports_an_exit_status(_exit_event(driver))


@pytest.mark.asyncio
async def test_codex_exit_reports_a_status():
    driver = CodexAppServerDriver()
    driver._proc = _DeadChild()

    await driver._read_loop()

    _assert_reports_an_exit_status(_exit_event(driver))


@pytest.mark.asyncio
async def test_pi_exit_reports_a_status():
    driver = PiDriver()
    driver._proc = _DeadChild()

    await driver._read_loop()

    _assert_reports_an_exit_status(_exit_event(driver))


@pytest.mark.asyncio
async def test_claude_exit_keeps_the_provider_error_beside_the_status():
    """A provider payload adds to the contract; it does not replace it."""
    driver = ClaudeCodeCliDriver()
    driver._proc = _DeadChild()
    driver._active_provider_error = {"error_code": "quota_exhausted"}

    await driver._read_loop()

    event = _exit_event(driver)
    _assert_reports_an_exit_status(event)
    assert event.data["error_code"] == "quota_exhausted"


def test_a_provider_payload_cannot_shadow_the_status():
    """`RuntimeManager` reads `error_code` flat, so providers merge flat.

    Merging flat is what makes shadowing possible, which is why the uniform
    field is written last rather than left to `dict.update` ordering luck.
    """
    data = runtime_exited_data(3, {"error_code": "boom", "returncode": 0})

    assert data == {"error_code": "boom", "returncode": 3}


def test_an_unobservable_status_is_reported_as_none_not_omitted():
    assert runtime_exited_data(None) == {"returncode": None}
