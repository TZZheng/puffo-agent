from __future__ import annotations

import asyncio
import json

import pytest

from puffo_agent.agent.harness.claude_code_driver import (
    ClaudeCodeCliDriver,
    claude_capabilities,
)
from puffo_agent.agent.harness.driver import RuntimeSpec, TurnInput


class _Stdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> None:
        json.loads(value)
        self.writes.append(value)

    async def drain(self) -> None:
        await asyncio.sleep(0)


class _Process:
    def __init__(self) -> None:
        self.stdin = _Stdin()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None

    def feed(self, frame: dict) -> None:
        self.stdout.feed_data(json.dumps(frame).encode() + b"\n")

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.feed_eof()

    kill = terminate

    async def wait(self) -> int:
        return int(self.returncode or 0)


def _lifecycle(proc: _Process, command_id: str, state: str) -> None:
    proc.feed({
        "type": "command_lifecycle",
        "command_uuid": command_id,
        "state": state,
    })


def _result(proc: _Process, command_id: str, input_: int, output: int) -> None:
    proc.feed({
        "type": "result",
        "subtype": "success",
        "user_message_uuid": command_id,
        "usage": {"input_tokens": input_, "output_tokens": output},
    })


async def _next(stream, event_type: str):
    async for event in stream:
        if getattr(event.type, "value", event.type) == event_type:
            return event
    raise AssertionError(f"event stream ended before {event_type}")


async def _open_driver(*, ack_timeout: float = 0.2):
    proc = _Process()
    proc.feed({
        "type": "system",
        "subtype": "init",
        "session_id": "claude-lifecycle",
        "capabilities": ["msg_lifecycle_v1"],
    })
    driver = ClaudeCodeCliDriver(
        lambda _args, _spec: proc,
        replay_timeout=ack_timeout,
    )
    await driver.open(RuntimeSpec("/workspace"))
    stream = driver.events()
    await _next(stream, "session.opened")
    started = await driver.start_turn(
        TurnInput("first", client_correlation_id="primary")
    )
    _lifecycle(proc, "primary", "queued")
    _lifecycle(proc, "primary", "started")
    proc.feed({
        "type": "user",
        "session_id": "claude-lifecycle",
        "parent_tool_use_id": None,
        "uuid": "primary",
        "isReplay": True,
        "message": {"role": "user", "content": "first"},
    })
    assert (await _next(stream, "turn.started")).native_turn_id == "primary"
    return proc, driver, stream, started


async def _offer_gated(proc, driver, started):
    pending = asyncio.create_task(driver.steer_turn(
        started.turn_ref,
        TurnInput("new inbox", client_correlation_id="gated"),
    ))
    while len(proc.stdin.writes) < 2:
        await asyncio.sleep(0)
    _lifecycle(proc, "gated", "queued")
    return await pending


def test_claude_lifecycle_capability_is_explicit():
    assert claude_capabilities().steer == "none"
    assert claude_capabilities(message_lifecycle_v1=True).steer == "gated"


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_shape", ["folded", "separate"])
async def test_gated_commands_share_one_logical_turn(delivery_shape):
    proc, driver, stream, started = await _open_driver()
    receipt = await _offer_gated(proc, driver, started)
    assert receipt.accepted and receipt.delivery == "queued_native_command"
    assert not (await driver.steer_turn(started.turn_ref, TurnInput("third"))).accepted

    if delivery_shape == "folded":
        _lifecycle(proc, "gated", "started")
        _lifecycle(proc, "gated", "completed")
        _result(proc, "primary", 10, 1)
        _lifecycle(proc, "primary", "completed")
        expected = (10, 1, 10)
    else:
        _result(proc, "primary", 10, 1)
        _lifecycle(proc, "primary", "completed")
        _lifecycle(proc, "gated", "started")
        _result(proc, "gated", 20, 2)
        _lifecycle(proc, "gated", "completed")
        expected = (30, 3, 20)

    completed = await _next(stream, "turn.completed")
    assert completed.native_turn_id == "primary"
    assert tuple(completed.data[key] for key in (
        "input_tokens", "output_tokens", "context_tokens"
    )) == expected
    await driver.close()


@pytest.mark.asyncio
async def test_gated_command_without_queue_ack_is_not_accepted():
    proc, driver, stream, started = await _open_driver(ack_timeout=0.01)
    receipt = await driver.steer_turn(started.turn_ref, TurnInput("new inbox"))
    assert not receipt.accepted and receipt.delivery == "queue_ack_timeout"
    _result(proc, "primary", 0, 0)
    _lifecycle(proc, "primary", "completed")
    await _next(stream, "turn.completed")
    await driver.close()
