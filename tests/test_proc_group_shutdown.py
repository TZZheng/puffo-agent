"""Process-group shutdown for re-spawning ACP children.

gemini's CLI replaces itself with a ``--max-old-space-size`` grandchild at
startup. Killing only the direct child pid orphans that grandchild: it
inherits the stdio pipes, the stream readers never reach EOF, and close()
hangs on a wait that even SIGKILL against the direct child cannot unstick.
The fix is granularity — spawn the child as a process-group leader
(``start_new_session=True``) and signal the whole group at shutdown
(``signal_process_group``). These tests observe both halves: the orphan the
old single-pid kill leaves behind, and the group signal actually reaping it.
"""

import asyncio
import os
import signal
import sys

import pytest

from puffo_agent._proc import signal_process_group, spawn_framed_child
from puffo_agent.agent.harness.acp_driver import AcpDriver
from puffo_agent.agent.harness.driver import RuntimeSpec

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="process groups are POSIX; Windows keeps the single-pid path"
)

# Mimics gemini's re-spawn: the direct child starts a long-lived grandchild
# that inherits our stdout pipe, prints the grandchild's pid, then lingers.
_RESPAWNER = (
    "import os, subprocess, sys, time\n"
    "grand = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "print(grand.pid, flush=True)\n"
    "time.sleep(120)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _spawn_respawner():
    proc = await spawn_framed_child(
        [sys.executable, "-c", _RESPAWNER],
        env={"PATH": os.environ.get("PATH", "")},
        start_new_session=True,
    )
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    return proc, int(line)


@pytest.mark.asyncio
async def test_start_new_session_makes_the_child_a_group_leader():
    proc = await spawn_framed_child(
        [sys.executable, "-c", "import os; print(int(os.getpgid(0) == os.getpid()))"],
        env={},
        start_new_session=True,
    )
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    await proc.wait()
    assert line.strip() == b"1"


@pytest.mark.asyncio
async def test_single_pid_kill_orphans_the_grandchild_and_wedges_the_pipe():
    """The failure mode the group kill exists for, kept observable.

    Killing only the direct child leaves the grandchild alive holding the
    stdout write end. asyncio's subprocess transport completes ``wait()``
    waiters only once every pipe transport has disconnected, so ``wait()``
    hangs even though the child it names is already dead — the exact shape
    of the original close() hang, which no SIGKILL to the direct pid can
    unstick.
    """
    proc, grand_pid = await _spawn_respawner()
    try:
        proc.kill()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1)
        assert _alive(grand_pid)
    finally:
        if _alive(grand_pid):
            os.kill(grand_pid, signal.SIGKILL)
        await asyncio.wait_for(proc.wait(), timeout=10)


@pytest.mark.asyncio
async def test_group_signal_reaps_the_grandchild_and_releases_the_pipe():
    proc, grand_pid = await _spawn_respawner()
    try:
        assert signal_process_group(proc, signal.SIGKILL)
        await asyncio.wait_for(proc.wait(), timeout=10)
        # EOF proves no surviving group member holds the pipe open.
        assert await asyncio.wait_for(proc.stdout.read(), timeout=10) == b""
        deadline = asyncio.get_running_loop().time() + 10
        while _alive(grand_pid):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.05)
    finally:
        if _alive(grand_pid):
            os.kill(grand_pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_group_signal_refuses_non_leaders_and_fakes():
    """Every guard fails toward the single-pid fallback.

    The non-leader refusal is the safety-critical one: a child spawned into
    our own group must never be group-signalled, or killpg would take down
    this process with it.
    """
    assert not signal_process_group(object(), signal.SIGTERM)
    assert not signal_process_group(
        type("P", (), {"pid": -1})(), signal.SIGTERM
    )
    proc = await spawn_framed_child(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={},
    )
    try:
        assert not signal_process_group(proc, signal.SIGTERM)
    finally:
        proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_acp_driver_spawn_creates_a_group_leader(tmp_path):
    driver = AcpDriver()
    spec = RuntimeSpec(
        workspace_dir=str(tmp_path),
        executable=sys.executable,
        launch_args=(
            "-c",
            "import os; print(int(os.getpgid(0) == os.getpid()))",
        ),
        environment={"PATH": os.environ.get("PATH", "")},
    )
    proc = await driver._spawn(spec)
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    await proc.wait()
    assert line.strip() == b"1"
