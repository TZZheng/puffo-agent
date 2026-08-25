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


# -- diagnosability: a failure must say why (PUF harness acceptance) --------


@pytest.mark.asyncio
async def test_bare_model_name_fails_fast_with_the_fix_in_hand():
    """OpenCode requires '<provider>/<model>'; its own error for a bare name
    is an opaque UnknownError that a recovery loop retries forever. The
    format rule is deterministic, so the driver refuses it at open()."""
    from puffo_agent.agent.errors import ProviderFailureError

    driver = OpenCodeDriver(lambda command, _spec: _TurnProcess())

    with pytest.raises(ProviderFailureError) as excinfo:
        await driver.open(RuntimeSpec("/workspace", model="deepseek-chat"))

    assert "deepseek-chat" in str(excinfo.value)
    assert "provider" in str(excinfo.value)
    assert excinfo.value.error_code == "provider_error"


@pytest.mark.asyncio
async def test_qualified_model_name_is_accepted():
    driver = OpenCodeDriver(lambda command, _spec: _TurnProcess())

    opened = await driver.open(
        RuntimeSpec("/workspace", model="deepseek/deepseek-chat")
    )

    assert opened is not None
    await driver.close()


@pytest.mark.asyncio
async def test_error_frame_detail_reaches_the_failed_turn():
    """The error frame's message is the only thing that distinguishes a wrong
    model from a provider outage; dropping it made both look identical."""
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda command, _spec: proc)
    await driver.open(RuntimeSpec("/workspace"))
    stream = driver.events()
    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    proc.feed({
        "type": "step_start",
        "sessionID": "ses_1",
        "part": {"messageID": "msg_1"},
    })
    await asyncio.wait_for(started, timeout=1)
    proc.feed({
        "type": "error",
        "sessionID": "ses_1",
        "error": {
            "name": "UnknownError",
            "data": {"message": "Unexpected server error. ref err_3bf8"},
        },
    })
    proc.exit(1)
    proc.eof()

    events = await asyncio.wait_for(
        _collect_through(stream, HarnessEventType.TURN_COMPLETED), timeout=1
    )

    failed_frame = next(
        e for e in events if e.type is HarnessEventType.RUNTIME_FAILED
    )
    assert "Unexpected server error" in failed_frame.data.get("diagnostic", "")
    terminal = events[-1]
    assert terminal.data["outcome"] == "failed"
    assert "Unexpected server error" in terminal.data.get("diagnostic", "")
    await driver.close()


@pytest.mark.asyncio
async def test_pre_acceptance_exit_carries_the_stderr_tail():
    """A child that dies before its first JSON frame explains itself only on
    stderr ("Error: Session not found"). That text must reach the exception
    the caller sees, or the operator gets a bare returncode forever."""
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda command, _spec: proc)
    await driver.open(RuntimeSpec("/workspace"))

    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    await asyncio.sleep(0)
    proc.stderr.feed_data(b"Error: Session not found\n")
    proc.exit(1)
    proc.eof()

    with pytest.raises(RuntimeError, match="Session not found"):
        await asyncio.wait_for(started, timeout=1)
    await driver.close()


@pytest.mark.asyncio
async def test_credential_shaped_stderr_is_redacted_before_surfacing():
    """Surfacing stderr must not become a credential leak: the child may
    echo the very key whose rejection killed it."""
    proc = _TurnProcess()
    driver = OpenCodeDriver(lambda command, _spec: proc)
    await driver.open(RuntimeSpec("/workspace"))

    started = asyncio.create_task(driver.start_turn(TurnInput("hello")))
    await asyncio.sleep(0)
    proc.stderr.feed_data(
        b"rejected api_key=sk_live_abcdef1234567890 for provider\n"
    )
    proc.exit(1)
    proc.eof()

    with pytest.raises(RuntimeError) as excinfo:
        await asyncio.wait_for(started, timeout=1)

    assert "sk_live_abcdef1234567890" not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)
    await driver.close()
