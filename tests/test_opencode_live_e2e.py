"""Opt-in live acceptance checks against an installed OpenCode binary.

These tests do not call a model or require provider credentials.  They exercise
the real OpenCode CLI in an isolated home/config/data/cache environment so a
passing unit test cannot hide a projection path that OpenCode itself ignores.
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
                    "description: E2E sentinel installed by Puffo.\n"
                    "---\n\n"
                    "# Puffo E2E\n\n"
                    "SENTINEL-OPENCODE-SKILL\n"
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
