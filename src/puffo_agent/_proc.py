"""Subprocess spawn helpers."""

from __future__ import annotations

import os
import subprocess

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
