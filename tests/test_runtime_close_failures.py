import asyncio
from types import SimpleNamespace

import pytest

from puffo_agent.agent.harness.codex_driver import CODEX_CAPABILITIES
from puffo_agent.agent.harness.driver import (
    Driver,
    HarnessEvent,
    HarnessEventType,
    RuntimeOpened,
    RuntimeRef,
    RuntimeSpec,
    SessionRef,
    UnsupportedCapability,
)
from puffo_agent.agent.harness.runtime_manager import (
    RuntimeManager,
    RuntimeManagerAdapter,
)


class _CloseErrorDriver(Driver):
    def __init__(self) -> None:
        self.queue = asyncio.Queue()

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
        raise RuntimeError("process-tree cleanup failed")


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
