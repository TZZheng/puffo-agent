"""Subprocess spawn helpers."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping, Sequence

# One provider frame can carry a large tool result. asyncio's StreamReader
# defaults to 64 KiB and does not truncate on overrun -- it raises, killing
# the read loop, so the session hangs with the cause buried in a
# LimitOverrunError nobody is watching. Every spawn that reads a framed
# child stream must pass this.
STREAM_READER_LIMIT_BYTES = 16 * 1024 * 1024


def no_window_kwargs() -> dict:
    """Windows: spawn child console apps (claude/codex/docker) without a
    console window so a DETACHED ``start --background`` daemon — which has
    no console of its own — doesn't pop a window per subprocess. No-op on
    other platforms; stdio is piped either way."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


async def spawn_framed_child(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str | None = None,
    stdin: int = asyncio.subprocess.PIPE,
    start_new_session: bool = False,
) -> asyncio.subprocess.Process:
    """Spawn a child whose stdout is consumed frame by frame.

    Four things every such spawn has to get right, held here once instead of
    repeated at each call site:

    * ``limit`` -- asyncio's StreamReader defaults to 64 KiB and raises
      rather than truncates on overrun, which kills the read loop and hangs
      the session. One large tool result is enough.
    * ``env`` -- becomes the child's whole environment. Nothing is merged in
      here, which is what makes sanitizing by omission work: an omitted key
      carries no instruction to delete, so any merge would quietly restore
      everything the caller left out. What belongs in it stays the caller's
      decision -- an agent spawn passes a built allowlist, the container
      runtime passes the host docker client's own environment.
    * ``no_window_kwargs()`` -- a detached Windows daemon has no console of
      its own and must not pop one per child.
    * stdout and stderr both piped, so the caller owns backpressure on both.

    ``stdin`` is the one parameter because a one-shot child that is never
    written to takes DEVNULL; every long-lived one takes a pipe.

    ``start_new_session`` makes the child the leader of its own process
    group (POSIX ``setsid``; accepted and inert on Windows). Callers whose
    children re-spawn themselves — gemini's CLI replaces itself with a
    ``--max-old-space-size`` grandchild — need this so shutdown can signal
    the whole group; killing only the direct child pid leaves a grandchild
    that inherits the stdio pipes, which keeps the stream readers from
    reaching EOF and turns ``close()`` into a hang. Pair it with
    ``signal_process_group`` at shutdown.

    This does not spawn short-lived commands read to EOF with
    ``communicate()`` -- those never reach the reader limit and have their
    own environment policy.
    """
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=dict(env),
        limit=STREAM_READER_LIMIT_BYTES,
        start_new_session=start_new_session,
        **no_window_kwargs(),
    )


def signal_process_group(proc: object, sig: int) -> bool:
    """Signal the child's whole process group; report whether that happened.

    Returns True only when the group signal was actually sent. The caller
    falls back to single-pid ``terminate()``/``kill()`` on False, so every
    guard here fails toward the old behavior:

    * Windows has no ``killpg``.
    * A factory-injected process may have no real pid (test fakes).
    * The group signal is sent only when the child leads its own group
      (``pgid == pid``) — exactly what ``start_new_session=True`` produces.
      A child left in the parent's group must never be group-signalled:
      ``killpg`` would hit this process too.
    """
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is None or getpgid is None:
        return False
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if getpgid(pid) != pid:
            return False
        killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True
