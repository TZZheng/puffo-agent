"""Live acceptance checks against an installed OpenCode binary.

The default check is network-free.  A separately gated check runs a real free
model turn, proving that OpenCode exposes the installed skill to the model and
that the model can load and follow it.  Both use isolated OpenCode state so a
passing test cannot be explained by unrelated host-level skills.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from puffo_agent.agent.adapters.desired_install import install_desired


class _SkillTemplateHttp:
    async def get(self, path: str):
        if path == "/v2/skill-templates/puffo-e2e":
            return {
                "body": (
                    "---\n"
                    "name: puffo-e2e\n"
                    "description: Use only for the Puffo OpenCode E2E sentinel.\n"
                    "---\n\n"
                    "# Puffo E2E\n\n"
                    "After loading this skill, reply with exactly "
                    "`SENTINEL-OPENCODE-SKILL`.\n"
                )
            }
        raise AssertionError(path)


def _isolated_opencode_environment(tmp_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        }
    )
    return environment


@pytest.mark.skipif(shutil.which("opencode") is None, reason="OpenCode absent")
def test_desired_skill_is_discoverable_by_real_opencode_cli(tmp_path: Path):
    """User-selected skill survives the full Puffo install -> CLI discovery path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    asyncio.run(
        install_desired(
            http=_SkillTemplateHttp(),
            agent_home=tmp_path / "agent-home",
            workspace_dir=workspace,
            agent_id="opencode-live-e2e",
            harness_name="opencode",
            desired_skills=["puffo-e2e"],
            desired_mcps=[],
        )
    )

    environment = _isolated_opencode_environment(tmp_path)
    # OpenCode consults PWD while resolving project-local skills.  Keep it
    # consistent with cwd so the test cannot accidentally discover skills
    # from the repository that launched pytest.
    environment["PWD"] = str(workspace)
    result = subprocess.run(
        [shutil.which("opencode") or "opencode", "debug", "skill"],
        cwd=workspace,
        env=_isolated_opencode_environment(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    skills = json.loads(result.stdout)
    sentinel = next(item for item in skills if item["name"] == "puffo-e2e")

    assert sentinel["location"] == str(
        workspace / ".agents" / "skills" / "puffo-e2e" / "SKILL.md"
    )
    assert "SENTINEL-OPENCODE-SKILL" in sentinel["content"]

    disabled_environment = _isolated_opencode_environment(tmp_path)
    disabled_environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "1"
    disabled = subprocess.run(
        [shutil.which("opencode") or "opencode", "debug", "skill"],
        cwd=workspace,
        env=disabled_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    disabled_skills = json.loads(disabled.stdout)
    assert not any(item["name"] == "puffo-e2e" for item in disabled_skills)


@pytest.mark.skipif(
    os.environ.get("PUFFO_RUN_LIVE_OPENCODE_E2E") != "1",
    reason="set PUFFO_RUN_LIVE_OPENCODE_E2E=1 for a real model turn",
)
@pytest.mark.skipif(shutil.which("opencode") is None, reason="OpenCode absent")
def test_real_opencode_model_loads_and_follows_desired_skill(tmp_path: Path):
    """Opt-in user journey: install, model tool-call, and skill-directed reply."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    asyncio.run(
        install_desired(
            http=_SkillTemplateHttp(),
            agent_home=tmp_path / "agent-home",
            workspace_dir=workspace,
            agent_id="opencode-live-model-e2e",
            harness_name="opencode",
            desired_skills=["puffo-e2e"],
            desired_mcps=[],
        )
    )

    model = os.environ.get(
        "PUFFO_OPENCODE_E2E_MODEL", "opencode/mimo-v2.5-free"
    )
    environment = _isolated_opencode_environment(tmp_path)
    environment["PWD"] = str(workspace)
    result = subprocess.run(
        [
            shutil.which("opencode") or "opencode",
            "run",
            "--format",
            "json",
            "--model",
            model,
            (
                "Load the puffo-e2e skill, follow it, and return only its "
                "required sentinel."
            ),
        ],
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    frames = [json.loads(line) for line in result.stdout.splitlines() if line]

    assert any(
        frame.get("type") == "tool_use"
        and frame.get("part", {}).get("tool") == "skill"
        and frame.get("part", {}).get("state", {}).get("input", {}).get("name")
        == "puffo-e2e"
        for frame in frames
    ), result.stdout
    assert any(
        frame.get("type") == "text"
        and frame.get("part", {}).get("text") == "SENTINEL-OPENCODE-SKILL"
        for frame in frames
    ), result.stdout
