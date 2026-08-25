"""Harness child environments are built from an allowlist, not a deny-list."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from puffo_agent.agent.harness.child_env import (
    PROVIDER_CREDENTIAL_ENV_NAMES,
    build_child_environment,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_AMBIENT = {
    "PATH": "/usr/bin",
    "HOME": "/home/op",
    "LC_ALL": "C.UTF-8",
    "XDG_CONFIG_HOME": "/home/op/.config",
    "HTTPS_PROXY": "http://proxy:8080",
    "NODE_EXTRA_CA_CERTS": "/etc/ca.pem",
    "OPENAI_API_KEY": "ambient-openai",
    "ANTHROPIC_API_KEY": "ambient-anthropic",
    "AWS_SECRET_ACCESS_KEY": "ambient-aws",
    "SOME_INTERNAL_TOKEN": "not-on-any-list",
}


@pytest.mark.parametrize("name", sorted(PROVIDER_CREDENTIAL_ENV_NAMES))
def test_ambient_provider_credentials_are_never_inherited(name):
    env = build_child_environment(source={**_AMBIENT, name: "ambient"})
    assert name not in env


@pytest.mark.parametrize("name", sorted(PROVIDER_CREDENTIAL_ENV_NAMES))
def test_an_override_cannot_reintroduce_a_provider_credential(name):
    """The ordering the Claude path already had, now enforced for everyone.

    Stripping only before merging overrides would let operator config smuggle
    an ambient key back in. The strip therefore runs after the merge.
    """
    env = build_child_environment(source=_AMBIENT, overrides={name: "smuggled"})
    assert name not in env


def test_controlled_injection_is_the_one_permitted_path():
    env = build_child_environment(
        source=_AMBIENT,
        overrides={"OPENAI_API_KEY": "smuggled"},
        controlled={"OPENAI_API_KEY": "controlled"},
    )
    assert env["OPENAI_API_KEY"] == "controlled"


def test_unlisted_ambient_variables_are_dropped():
    """The allowlist property: an unnamed secret does not pass by default.

    This is what a deny-list cannot give. SOME_INTERNAL_TOKEN is on no list
    and must still not reach the child.
    """
    env = build_child_environment(source=_AMBIENT)
    assert "SOME_INTERNAL_TOKEN" not in env


def test_operational_variables_survive():
    """Guards the other failure mode: an allowlist so tight it breaks agents.

    Dropping proxy or CA settings strands every agent behind a corporate
    proxy, and presents as "the harness is broken".
    """
    env = build_child_environment(source=_AMBIENT)
    for name in ("PATH", "HOME", "HTTPS_PROXY", "NODE_EXTRA_CA_CERTS"):
        assert env[name] == _AMBIENT[name], name


def test_open_ended_prefixes_survive():
    env = build_child_environment(source=_AMBIENT)
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["XDG_CONFIG_HOME"] == "/home/op/.config"


def test_extra_allowed_admits_a_runtime_specific_name():
    env = build_child_environment(
        source={**_AMBIENT, "CODEX_HOME": "/agents/a/.codex"},
        extra_allowed=("CODEX_HOME",),
    )
    assert env["CODEX_HOME"] == "/agents/a/.codex"


def test_spec_builders_do_not_touch_os_environ_directly():
    """Tripwire: a new runtime spec must go through the builder.

    Both drift cases started as a local ``dict(os.environ)``. This fails the
    moment one reappears in the module that builds RuntimeSpec environments.
    """
    path = _REPO_ROOT / "src/puffo_agent/agent/harness/local_runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    offenders = []
    for node in ast.walk(tree):
        # os.environ.copy() / dict(os.environ) / {**os.environ, ...}
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            value = node.value
            if isinstance(value, ast.Name) and value.id == "os":
                offenders.append(node.lineno)

    assert not offenders, (
        f"local_runtime.py reads os.environ directly at line(s) {offenders}; "
        "build the child environment with child_env.build_child_environment "
        "so ambient provider credentials cannot reach the harness."
    )
