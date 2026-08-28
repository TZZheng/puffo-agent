"""Framed-stream children are spawned in exactly one place.

asyncio's StreamReader defaults to 64 KiB. On overrun ``readline()`` does not
truncate -- it raises, which kills the read loop. The symptom is a session that
hangs, with the cause in a LimitOverrunError nobody is watching. One large tool
result is enough.

This used to be enforced by parsing each driver's source for a ``limit=``
keyword, against a hand-written list of seven modules. The list was the
defect: a new driver was exempt until someone remembered to add it, so the
test's default answer for unseen code was "fine". ``spawn_framed_child`` now
holds the limit, the whole-environment replacement and the Windows console
flags, and those are asserted by *running* it rather than by reading it.

What a scan is still the right tool for is the one thing no runtime test can
show: the *absence* of a bypass. That scan is a directory walk with a named
exemption, so a new driver is covered the day it is written and can only be
excused on purpose.
"""

from __future__ import annotations

import asyncio
import ast
import os
from pathlib import Path

import pytest

from puffo_agent import _proc
from puffo_agent._proc import STREAM_READER_LIMIT_BYTES, spawn_framed_child


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose children are read frame by frame.
FRAMED_STREAM_PACKAGES = (
    "src/puffo_agent/agent/harness",
    "src/puffo_agent/agent/adapters",
)

# The rule is *not* "never spawn a subprocess". Callers that read to EOF with
# communicate() never reach the reader limit, and docker build/inspect need
# the ambient docker client environment the framed-child helper deliberately
# refuses to merge. Exemptions are named here so adding one is a decision.
DIRECT_SPAWN_EXEMPTIONS = {
    "src/puffo_agent/agent/harness/docker_support.py":
        "short docker commands read to EOF via communicate_with_timeout",
}


class _CapturedSpawn:
    def __init__(self) -> None:
        self.args: tuple = ()
        self.kwargs: dict = {}

    async def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return object()


@pytest.fixture
def spawned(monkeypatch) -> _CapturedSpawn:
    captured = _CapturedSpawn()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", captured)
    return captured


def test_limit_constant_is_well_above_the_asyncio_default():
    assert STREAM_READER_LIMIT_BYTES == 16 * 1024 * 1024
    assert STREAM_READER_LIMIT_BYTES > 64 * 1024


@pytest.mark.asyncio
async def test_framed_child_gets_the_shared_limit(spawned):
    await spawn_framed_child(["agent", "--rpc"], env={})

    assert spawned.kwargs["limit"] == STREAM_READER_LIMIT_BYTES


@pytest.mark.asyncio
async def test_framed_child_environment_replaces_rather_than_merges(
    spawned, monkeypatch
):
    """Sanitizing by omission only works if nothing merges the ambient env back."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-value")

    await spawn_framed_child(["agent"], env={"PATH": "/usr/bin"})

    assert spawned.kwargs["env"] == {"PATH": "/usr/bin"}


@pytest.mark.asyncio
async def test_framed_child_environment_is_copied_not_aliased(spawned):
    """The caller's dict must not stay live after the spawn decided the env."""
    env = {"PATH": "/usr/bin"}

    await spawn_framed_child(["agent"], env=env)
    env["ANTHROPIC_API_KEY"] = "added-later"

    assert spawned.kwargs["env"] == {"PATH": "/usr/bin"}


@pytest.mark.asyncio
async def test_framed_child_is_spawned_without_a_console(spawned, monkeypatch):
    monkeypatch.setattr(_proc, "no_window_kwargs", lambda: {"creationflags": 8})

    await spawn_framed_child(["agent"], env={})

    assert spawned.kwargs["creationflags"] == 8


@pytest.mark.asyncio
async def test_framed_child_pipes_both_output_streams(spawned):
    await spawn_framed_child(["agent"], env={})

    assert spawned.kwargs["stdout"] == asyncio.subprocess.PIPE
    assert spawned.kwargs["stderr"] == asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_stdin_is_a_pipe_unless_the_caller_says_otherwise(spawned):
    await spawn_framed_child(["agent"], env={})
    assert spawned.kwargs["stdin"] == asyncio.subprocess.PIPE

    await spawn_framed_child(
        ["agent"], env={}, stdin=asyncio.subprocess.DEVNULL
    )
    assert spawned.kwargs["stdin"] == asyncio.subprocess.DEVNULL


@pytest.mark.skipif(os.name != "posix", reason="pass_fds is POSIX-only")
@pytest.mark.asyncio
async def test_framed_child_inherits_only_explicit_control_fds(spawned):
    """A Driver authority socket must reach the child without opening ambient fds."""

    await spawn_framed_child(["agent"], env={}, pass_fds=(17,))

    assert spawned.kwargs["pass_fds"] == (17,)


def _direct_spawns(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in {"create_subprocess_exec", "create_subprocess_shell"}:
            lines.append(node.lineno)
    return lines


def _framed_stream_modules() -> list[str]:
    found = []
    for package in FRAMED_STREAM_PACKAGES:
        for path in sorted((_REPO_ROOT / package).rglob("*.py")):
            found.append(str(path.relative_to(_REPO_ROOT)))
    return found


def test_the_scan_actually_walks_the_driver_modules():
    """A directory walk that matches nothing would pass in silence."""
    modules = _framed_stream_modules()

    assert "src/puffo_agent/agent/harness/pi_driver.py" in modules
    assert "src/puffo_agent/agent/adapters/cli_session.py" in modules
    assert len(modules) > 5


@pytest.mark.parametrize("relpath", _framed_stream_modules())
def test_no_driver_spawns_a_framed_child_directly(relpath):
    if relpath in DIRECT_SPAWN_EXEMPTIONS:
        pytest.skip(DIRECT_SPAWN_EXEMPTIONS[relpath])

    lines = _direct_spawns(_REPO_ROOT / relpath)

    assert not lines, (
        f"{relpath} spawns a child directly at line(s) {lines}; use "
        "puffo_agent._proc.spawn_framed_child so the reader limit, the "
        "whole-environment replacement and the Windows console flags come "
        "from one place instead of being retyped correctly six times."
    )


@pytest.mark.parametrize("relpath", sorted(DIRECT_SPAWN_EXEMPTIONS))
def test_every_exemption_still_spawns_something(relpath):
    """An exemption for a module that no longer spawns is stale, not safe."""
    assert _direct_spawns(_REPO_ROOT / relpath), (
        f"{relpath} is exempted from the framed-child rule but no longer "
        "spawns a subprocess; drop the exemption."
    )
