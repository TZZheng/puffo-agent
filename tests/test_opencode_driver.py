import asyncio
import json

import pytest

from puffo_agent.agent.harness import build_driver
from puffo_agent.agent.harness.driver import (
    BusyDelivery,
    HarnessEventType,
    RuntimeLifecycle,
    RuntimeSpec,
    TurnInput,
    UnsupportedCapability,
)
from puffo_agent.agent.harness.opencode_driver import (
    OPENCODE_CAPABILITIES,
    OpenCodeDriver,
)
from puffo_agent.agent.harness.runtime_manager import RuntimeManager


class _TurnProcess:
    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None
        self.terminated = 0
        self._exit = asyncio.get_running_loop().create_future()

    def feed(self, frame: dict) -> None:
        self.stdout.feed_data(json.dumps(frame).encode() + b"\n")

    def feed_raw_json(self, frame: dict) -> None:
        self.stdout.feed_data(
            json.dumps(frame, ensure_ascii=False).encode() + b"\n"
        )

    def eof(self) -> None:
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    def exit(self, returncode: int = 0) -> None:
        self.returncode = returncode
        if not self._exit.done():
            self._exit.set_result(returncode)

    def terminate(self) -> None:
        self.terminated += 1
        self.exit(-15)
        self.eof()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        return await self._exit


async def _next_matching(stream, type_: HarnessEventType):
    async for event in stream:
        if event.type is type_:
            return event
    raise AssertionError(f"event stream ended before {type_.value}")


async def _collect_through(stream, type_: HarnessEventType):
    events = []
    async for event in stream:
        events.append(event)
        if event.type is type_:
            return events
    raise AssertionError(f"event stream ended before {type_.value}")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_signal", ["exit", "eof"])
async def test_turn_terminal_waits_for_both_process_exit_and_stream_eof(
    first_signal,
):
    proc = _TurnProcess()
    commands = []

    def factory(command, _spec):
        commands.append(command)
        return proc

    driver = OpenCodeDriver(factory)
    opened = await driver.open(RuntimeSpec("/workspace"))
    assert opened.native_session_id == ""
    stream = driver.events()
    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    proc.feed({
        "type": "step_start",
        "sessionID": "ses_1",
        "part": {"messageID": "msg_1"},
    })
    receipt = await asyncio.wait_for(started, timeout=1)
    assert receipt.native_turn_id == "msg_1"
    terminal = asyncio.create_task(
        _next_matching(stream, HarnessEventType.TURN_COMPLETED)
    )

    getattr(proc, first_signal)()
    await asyncio.sleep(0)
    assert not terminal.done()
    getattr(proc, "eof" if first_signal == "exit" else "exit")()

    event = await asyncio.wait_for(terminal, timeout=1)
    assert event.data == {"outcome": "succeeded"}
    assert commands[0][-1] == "hello"
    await driver.close()


@pytest.mark.asyncio
async def test_second_turn_resumes_native_session_and_emits_one_session_boundary():
    processes = [_TurnProcess(), _TurnProcess()]
    commands = []

    def factory(command, _spec):
        commands.append(command)
        return processes[len(commands) - 1]

    driver = OpenCodeDriver(factory)
    await driver.open(RuntimeSpec("/workspace"))
    stream = driver.events()
    events = []

    for index, proc in enumerate(processes, start=1):
        started = asyncio.create_task(
            driver.start_turn(TurnInput(f"turn {index}"))
        )
        proc.feed({
            "type": "step_start",
            "sessionID": "ses_1",
            "part": {"messageID": f"msg_{index}"},
        })
        await asyncio.wait_for(started, timeout=1)
        proc.eof()
        proc.exit()
        events.extend(
            await asyncio.wait_for(
                _collect_through(stream, HarnessEventType.TURN_COMPLETED),
                timeout=1,
            )
        )

    assert "--session" not in commands[0]
    session_index = commands[1].index("--session")
    assert commands[1][session_index + 1] == "ses_1"
    await driver.close()
    assert sum(
        event.type
        in {HarnessEventType.SESSION_OPENED, HarnessEventType.SESSION_RESUMED}
        for event in events
    ) == 1


@pytest.mark.asyncio
async def test_cancel_terminates_child_and_emits_one_abandon_terminal():
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda *_args: proc)
    await driver.open(RuntimeSpec("/workspace"))
    stream = driver.events()
    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    proc.feed({
        "type": "step_start",
        "sessionID": "ses_1",
        "part": {"messageID": "msg_1"},
    })
    receipt = await started

    cancel = await driver.cancel_turn(receipt.turn_ref)
    assert cancel.accepted is True
    terminal = await asyncio.wait_for(
        _next_matching(stream, HarnessEventType.TURN_ABANDONED), timeout=1
    )
    assert terminal.data["error_code"] == "cancelled"
    assert proc.terminated == 1
    await driver.close()


@pytest.mark.asyncio
async def test_start_rejects_process_that_never_emits_valid_acceptance_frame():
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda *_args: proc)
    await driver.open(RuntimeSpec("/workspace"))
    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    proc.stdout.feed_data(b"not-json\n")
    proc.eof()
    proc.exit(2)

    with pytest.raises(RuntimeError, match="before accepting"):
        await asyncio.wait_for(started, timeout=1)
    await driver.close()


@pytest.mark.asyncio
async def test_jsonl_reader_does_not_split_valid_unicode_line_separators():
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda *_args: proc)
    await driver.open(RuntimeSpec("/workspace"))
    stream = driver.events()
    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    text = "one\u2028two\u2029three\u0085four"
    proc.feed_raw_json({
        "type": "text",
        "sessionID": "ses_1",
        "part": {"id": "part_1", "messageID": "msg_1", "text": text},
    })
    await asyncio.wait_for(started, timeout=1)
    proc.eof()
    proc.exit()

    events = await asyncio.wait_for(
        _collect_through(stream, HarnessEventType.TURN_COMPLETED), timeout=1
    )
    [delta] = [
        event for event in events
        if event.type is HarnessEventType.ASSISTANT_DELTA
    ]
    assert delta.data["delta"] == text
    await driver.close()


def test_capabilities_and_factory_declare_per_turn_reject_contract():
    assert isinstance(build_driver("opencode"), OpenCodeDriver)
    assert OPENCODE_CAPABILITIES.lifecycle is RuntimeLifecycle.PER_TURN_CHILD
    assert OPENCODE_CAPABILITIES.busy_delivery is BusyDelivery.REJECT
    assert OPENCODE_CAPABILITIES.steer == "none"
    assert isinstance(
        asyncio.run(OpenCodeDriver().steer_turn(None, TurnInput("x"))),
        UnsupportedCapability,
    )


@pytest.mark.asyncio
async def test_manager_adopts_session_learned_by_per_turn_child():
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda *_args: proc)
    manager = RuntimeManager(
        driver,
        RuntimeSpec("/workspace"),
        driver_name="opencode",
    )
    await manager.open()
    started = asyncio.create_task(manager.start_turn(TurnInput("hello")))
    proc.feed({
        "type": "step_start",
        "sessionID": "ses_discovered",
        "part": {"messageID": "msg_1"},
    })
    receipt = await asyncio.wait_for(started, timeout=1)
    assert manager.native_session_id == "ses_discovered"
    proc.eof()
    proc.exit()
    terminal = await asyncio.wait_for(
        manager.wait_terminal(receipt.turn_ref), timeout=1
    )
    assert terminal.type is HarnessEventType.TURN_COMPLETED
    await manager.close()
