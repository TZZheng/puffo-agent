"""Every framed-stream spawn must raise asyncio's default StreamReader limit.

asyncio's StreamReader defaults to 64 KiB. On overrun ``readline()`` does not
truncate -- it raises, which kills the read loop. The symptom is a session that
hangs, with the cause in a LimitOverrunError nobody is watching. A single large
tool result is enough.

Scope is deliberate. The rule is *not* "every create_subprocess_exec that pipes
stdout": callers using ``communicate()`` read to EOF and never hit the limit, so
requiring it there would be a false positive. What matters is spawns whose
stdout is consumed frame-by-frame, which is exactly the harness Driver modules
-- and exactly where new drivers (pi / opencode / acp) get added.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from puffo_agent._proc import STREAM_READER_LIMIT_BYTES

# Modules that spawn a child whose stdout is read as framed JSON lines.
# A new harness Driver that spawns a child belongs on this list.
FRAMED_STREAM_MODULES = (
    "src/puffo_agent/agent/harness/codex_driver.py",
    "src/puffo_agent/agent/harness/claude_code_driver.py",
    "src/puffo_agent/agent/harness/docker_runtime.py",
    "src/puffo_agent/agent/harness/opencode_driver.py",
    "src/puffo_agent/agent/harness/acp_driver.py",
    "src/puffo_agent/agent/harness/pi_driver.py",
    "src/puffo_agent/agent/adapters/cli_session.py",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _subprocess_spawns(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name == "create_subprocess_exec":
            yield node


def test_limit_constant_is_well_above_the_asyncio_default():
    assert STREAM_READER_LIMIT_BYTES == 16 * 1024 * 1024
    assert STREAM_READER_LIMIT_BYTES > 64 * 1024


@pytest.mark.parametrize("relpath", FRAMED_STREAM_MODULES)
def test_framed_stream_spawns_pass_an_explicit_limit(relpath):
    path = _REPO_ROOT / relpath
    spawns = list(_subprocess_spawns(path))
    assert spawns, f"{relpath}: expected at least one spawn; did it move?"

    for call in spawns:
        keywords = {kw.arg for kw in call.keywords}
        assert "limit" in keywords, (
            f"{relpath}:{call.lineno} spawns a framed-stream child without "
            "limit=; it will inherit asyncio's 64 KiB default and the read "
            "loop will die on the first oversized frame."
        )


@pytest.mark.parametrize("relpath", FRAMED_STREAM_MODULES)
def test_framed_stream_spawns_use_the_shared_constant(relpath):
    """No bare literals: four copies is how these drifted in the first place."""
    path = _REPO_ROOT / relpath
    for call in _subprocess_spawns(path):
        for kw in call.keywords:
            if kw.arg != "limit":
                continue
            assert isinstance(kw.value, ast.Name), (
                f"{relpath}:{call.lineno} passes a literal limit; import "
                "STREAM_READER_LIMIT_BYTES from puffo_agent._proc instead."
            )
            assert kw.value.id == "STREAM_READER_LIMIT_BYTES", (
                f"{relpath}:{call.lineno} uses {kw.value.id!r}"
            )
