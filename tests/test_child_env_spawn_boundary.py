"""The spawn boundary, not the spec builder, is where sanitising must hold.

Recorded as xfail because it currently fails. build_child_environment()
produces a clean RuntimeSpec.environment, but every Driver then does

    env = os.environ.copy()
    env.update(spec.environment)

and ``update`` only overwrites keys the spec *contains*. An allowlist
sanitises by removal, so the stripped key is absent from the spec, so nothing
overwrites the ambient value and it survives into the child.

Absence carries no instruction to delete. Removal-based sanitising only holds
when the consumer *replaces* the environment; merging it into ambient undoes
it completely.

Flips to pass when the Drivers treat RuntimeSpec.environment as the complete
child environment (T7). Do not "fix" this by adding the key back with an empty
value -- that hides the shape of the bug and would let the next merge-into-
ambient consumer reintroduce it.
"""

from __future__ import annotations

import os

import pytest

from puffo_agent.agent.harness.child_env import build_child_environment


def _driver_merge(spec_environment: dict[str, str]) -> dict[str, str]:
    """Verbatim shape of codex_driver._start_process / claude_code_driver."""
    env = os.environ.copy()
    env.update(spec_environment)
    return env


@pytest.mark.xfail(
    reason="T7: Drivers must use RuntimeSpec.environment as the complete "
           "child env instead of merging it into os.environ",
    strict=True,
)
def test_ambient_provider_key_does_not_survive_the_spawn_merge(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")

    spec_environment = build_child_environment(
        overrides={},
        controlled={"CODEX_HOME": "/agents/a/.codex"},
        extra_allowed=("CODEX_HOME",),
    )
    assert "OPENAI_API_KEY" not in spec_environment, (
        "precondition: the allowlist itself works"
    )

    child_env = _driver_merge(spec_environment)
    assert "OPENAI_API_KEY" not in child_env


def test_replacing_rather_than_merging_does_hold(monkeypatch):
    """The same inputs under the T7 shape, to show the fix is the merge."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")

    spec_environment = build_child_environment(
        overrides={},
        controlled={"CODEX_HOME": "/agents/a/.codex"},
        extra_allowed=("CODEX_HOME",),
    )
    child_env = dict(spec_environment)  # replace, do not merge
    assert "OPENAI_API_KEY" not in child_env
    assert child_env["CODEX_HOME"] == "/agents/a/.codex"
