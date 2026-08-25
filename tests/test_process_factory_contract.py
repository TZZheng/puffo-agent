"""The process factory is called once, with its declared signature.

Three drivers used to call the factory as ``(command, spec)`` and, on
``TypeError``, retry as ``(command)``.  ``except TypeError`` cannot distinguish
"this factory takes fewer arguments" from "this factory raised TypeError while
running", so a factory that failed partway through was called a second time --
spawning a second child when the first may already exist.

These tests observe the difference directly: a factory that raises ``TypeError``
on its first call and succeeds on its second returns a process under the old
behaviour and propagates under the fixed one.  Both the call count and the
raised exception are asserted, so neither half can regress silently.
"""

from __future__ import annotations


import pytest

from puffo_agent.agent.harness import (
    AcpDriver,
    ClaudeCodeCliDriver,
    OpenCodeDriver,
)
from puffo_agent.agent.harness.driver import RuntimeSpec


class _RetryObservingFactory:
    """Raises TypeError once, then would succeed -- like a real second spawn."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.calls == 1:
            raise TypeError("raised from inside the factory body, not arity")
        return object()  # a "process" the retry would have handed back


def _spec() -> RuntimeSpec:
    return RuntimeSpec("/workspace", executable="agent")


@pytest.mark.asyncio
async def test_acp_spawn_does_not_retry_a_failing_factory():
    factory = _RetryObservingFactory()
    driver = AcpDriver(factory)

    with pytest.raises(TypeError):
        await driver._spawn(_spec())

    assert factory.calls == 1


@pytest.mark.asyncio
async def test_opencode_spawn_does_not_retry_a_failing_factory():
    factory = _RetryObservingFactory()
    driver = OpenCodeDriver(factory)

    with pytest.raises(TypeError):
        await driver._spawn(_spec(), "prompt")

    assert factory.calls == 1


@pytest.mark.asyncio
async def test_claude_open_does_not_retry_a_failing_factory():
    factory = _RetryObservingFactory()
    driver = ClaudeCodeCliDriver(factory)

    with pytest.raises(TypeError):
        await driver.open(_spec())

    assert factory.calls == 1
