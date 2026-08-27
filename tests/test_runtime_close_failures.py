import asyncio
from types import SimpleNamespace

import pytest

from puffo_agent.agent.harness.cleanup_errors import cleanup_errors
from puffo_agent.agent.harness.codex_driver import CODEX_CAPABILITIES
from puffo_agent.agent.harness.driver import (
    Driver,
    HarnessEvent,
    HarnessEventType,
    RuntimeOpened,
    RuntimeRef,
    RuntimeSpec,
    SessionRef,
    TurnRef,
    UnsupportedCapability,
)
from puffo_agent.agent.harness.runtime_manager import (
    RuntimeManager,
    RuntimeManagerAdapter,
)


class _CloseErrorDriver(Driver):
    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.close_calls = 0

    async def open(self, spec, resume=None):
        return RuntimeOpened(
            RuntimeRef("runtime"),
            SessionRef("native-session"),
            "native-session",
            False,
            CODEX_CAPABILITIES,
            SimpleNamespace(),
        )

    async def start_turn(self, input):
        return UnsupportedCapability("start_turn")

    async def steer_turn(self, turn, input):
        return UnsupportedCapability("steer")

    async def cancel_turn(self, turn):
        return UnsupportedCapability("cancel")

    async def context_status(self):
        return UnsupportedCapability("context_status")

    async def compact(self, request):
        return UnsupportedCapability("compact")

    async def resolve_permission(self, request, decision):
        return UnsupportedCapability("permission")

    def events(self):
        async def iterate():
            while True:
                yield await self.queue.get()

        return iterate()

    async def close(self):
        self.close_calls += 1
        raise RuntimeError("process-tree cleanup failed")


def test_cleanup_reader_rejects_missing_protocol_marker():
    with pytest.raises(LookupError, match="no structured cleanup evidence"):
        cleanup_errors(asyncio.CancelledError())


@pytest.mark.asyncio
async def test_manager_close_finishes_state_cleanup_before_propagating_error():
    driver = _CloseErrorDriver()
    manager = RuntimeManager(driver, RuntimeSpec("/tmp"))
    await manager.open()
    stream = manager.events()

    with pytest.raises(RuntimeError, match="process-tree cleanup failed"):
        await manager.close()

    assert manager.opened is None
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_runtime_exit_preserves_primary_and_cleanup_failures():
    driver = _CloseErrorDriver()

    async def failing_sink(_event):
        raise ValueError("primary persistence failure")

    manager = RuntimeManager(
        driver,
        RuntimeSpec("/tmp"),
        event_sink=failing_sink,
    )
    await manager.open()

    with pytest.raises(ExceptionGroup) as exc_info:
        await manager._handle_runtime_exit_locked(HarnessEvent(
            type=HarnessEventType.RUNTIME_EXITED,
            driver="fake",
            session_ref=manager.session_ref,
            data={"returncode": 1},
        ))

    assert [str(exc) for exc in exc_info.value.exceptions] == [
        "primary persistence failure",
        "process-tree cleanup failed",
    ]
    assert manager.opened is None
    await manager._stop_reader()


@pytest.mark.asyncio
async def test_runtime_exit_cancellation_still_closes_and_preserves_both():
    driver = _CloseErrorDriver()

    async def cancelled_sink(_event):
        raise asyncio.CancelledError

    manager = RuntimeManager(
        driver,
        RuntimeSpec("/tmp"),
        event_sink=cancelled_sink,
    )
    await manager.open()

    task = asyncio.create_task(
        manager._handle_runtime_exit_locked(HarnessEvent(
            type=HarnessEventType.RUNTIME_EXITED,
            driver="fake",
            session_ref=manager.session_ref,
            data={"returncode": 1},
        ))
    )
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert task.cancelled()
    assert [str(exc) for exc in cleanup_errors(exc_info.value)] == [
        "process-tree cleanup failed"
    ]
    assert driver.close_calls == 1
    assert manager.opened is None
    await manager._stop_reader()


@pytest.mark.asyncio
async def test_invalid_resume_cancellation_still_closes_and_preserves_both(
    monkeypatch,
):
    driver = _CloseErrorDriver()
    manager = RuntimeManager(driver, RuntimeSpec("/tmp"))
    await manager.open()

    async def cancelled_terminal(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        manager,
        "_publish_terminal_locked",
        cancelled_terminal,
    )

    task = asyncio.create_task(
        manager._retire_invalid_resume_locked(
            HarnessEvent(
                type=HarnessEventType.TURN_ABANDONED,
                driver="fake",
                session_ref=manager.session_ref,
            ),
            TurnRef("turn_cancelled"),
        )
    )
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert task.cancelled()
    assert [str(exc) for exc in cleanup_errors(exc_info.value)] == [
        "process-tree cleanup failed"
    ]
    assert driver.close_calls == 1
    assert manager.opened is None
    await manager._stop_reader()


@pytest.mark.asyncio
async def test_open_and_close_failures_are_grouped_without_fresh_fallback():
    class OpenAndCloseErrorDriver(_CloseErrorDriver):
        def __init__(self):
            super().__init__()
            self.open_calls = 0

        async def open(self, spec, resume=None):
            self.open_calls += 1
            raise ValueError("primary open failure")

    driver = OpenAndCloseErrorDriver()
    manager = RuntimeManager(driver, RuntimeSpec("/tmp"))
    manager.native_session_id = "resume-me"

    with pytest.raises(ExceptionGroup) as exc_info:
        await manager.open()

    assert [str(exc) for exc in exc_info.value.exceptions] == [
        "primary open failure",
        "process-tree cleanup failed",
    ]
    assert driver.open_calls == 1
    assert driver.close_calls == 1


@pytest.mark.asyncio
async def test_start_and_retire_failures_are_grouped():
    class StartAndCloseErrorDriver(_CloseErrorDriver):
        async def start_turn(self, input):
            raise ValueError("primary start failure")

    driver = StartAndCloseErrorDriver()
    manager = RuntimeManager(driver, RuntimeSpec("/tmp"))
    await manager.open()

    with pytest.raises(ExceptionGroup) as exc_info:
        await manager.start_turn(SimpleNamespace(content="hello"))

    assert [str(exc) for exc in exc_info.value.exceptions] == [
        "primary start failure",
        "process-tree cleanup failed",
    ]
    assert driver.close_calls == 1
    assert manager.opened is None


@pytest.mark.asyncio
async def test_adapter_close_aggregates_manager_and_post_close_failures():
    class FailingManager:
        async def close(self):
            raise ValueError("manager cleanup failed")

    async def failing_post_close():
        raise RuntimeError("container cleanup failed")

    adapter = RuntimeManagerAdapter(
        FailingManager(),
        post_close=failing_post_close,
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        await adapter.aclose()

    assert [str(exc) for exc in exc_info.value.exceptions] == [
        "manager cleanup failed",
        "container cleanup failed",
    ]
