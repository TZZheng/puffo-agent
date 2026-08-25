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


def _ambient_env_reads(path) -> list[int]:
    """Line numbers where a module rebuilds the ambient child environment.

    Catches os.environ.copy() / dict(os.environ) / {**os.environ, ...} and
    SDK-owned default_environment(), whose contents can drift independently
    of Puffo's credential contract.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            value = node.value
            if isinstance(value, ast.Name) and value.id == "os":
                offenders.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "default_environment"
        ):
            offenders.append(node.lineno)
    return offenders


# Modules allowed to read the ambient environment, and why. This is the whole
# harness package rather than a list of drivers on purpose: a hand-written
# inclusion list makes every module nobody thought of exempt by default, which
# is the same shape as the deny-list this allowlist replaced. docker_runtime
# was in exactly that position -- it reads os.environ and was never on the old
# six-module list.
AMBIENT_ENV_EXEMPTIONS = {
    "src/puffo_agent/agent/harness/child_env.py":
        "the allowlist boundary itself; it is what reads ambient",
    "src/puffo_agent/agent/harness/docker_runtime.py":
        "os.environ there is the host docker *client* env, not the agent "
        "child's: the container's environment is set by `docker exec -e` "
        "flags, and API keys travel by name so they stay out of argv",
}


def _harness_modules() -> list[str]:
    return [
        str(path.relative_to(_REPO_ROOT))
        for path in sorted((_REPO_ROOT / "src/puffo_agent/agent/harness").rglob("*.py"))
    ]


def test_the_ambient_scan_actually_walks_the_harness_package():
    """A directory walk that matches nothing would pass in silence."""
    modules = _harness_modules()

    assert "src/puffo_agent/agent/harness/local_runtime.py" in modules
    assert "src/puffo_agent/agent/harness/pi_driver.py" in modules
    assert len(modules) > 10


@pytest.mark.parametrize("relpath", sorted(AMBIENT_ENV_EXEMPTIONS))
def test_every_ambient_exemption_still_reads_ambient(relpath):
    """An exemption for a module that stopped reading ambient is stale."""
    assert _ambient_env_reads(_REPO_ROOT / relpath), (
        f"{relpath} is exempted from the ambient-environment rule but no "
        "longer reads os.environ; drop the exemption."
    )


@pytest.mark.parametrize("relpath", _harness_modules())
def test_harness_child_environment_boundary_never_rereads_ambient(relpath):
    """Spec construction and real spawn must share one allowlist boundary.

    Sanitizing a RuntimeSpec is ineffective if a Driver merges ``os.environ``
    back at spawn. SDK-owned default allowlists are also forbidden here: their
    contents can drift independently of Puffo's credential contract.
    """
    if relpath in AMBIENT_ENV_EXEMPTIONS:
        pytest.skip(AMBIENT_ENV_EXEMPTIONS[relpath])
    offenders = _ambient_env_reads(_REPO_ROOT / relpath)

    assert not offenders, (
        f"{relpath} rebuilds ambient child env at line(s) {offenders}; "
        "build the child environment with child_env.build_child_environment "
        "once, then pass RuntimeSpec.environment through unchanged."
    )
