"""Compatibility checks for agent.yml files written by older releases."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    return tmp_path


def _write_agent(agent_id: str, body: dict) -> None:
    from puffo_agent.portal.state import agent_yml_path

    path = agent_yml_path(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def test_legacy_agent_config_load_save_preserves_existing_values(home):
    """Missing new fields get defaults without losing legacy settings."""
    from puffo_agent.portal.state import AgentConfig, agent_yml_path

    agent_id = "legacy-agent"
    _write_agent(agent_id, {
        "id": agent_id,
        "state": "paused",
        "display_name": "Legacy Agent",
        "avatar_url": "https://cdn.example/avatar.png",
        "role": "Keeps the old workspace running",
        "role_short": "Keeper",
        "created_at": 123456789,
        "puffo_core": {
            "server_url": "https://api.example",
            "slug": "legacy-1234",
            "device_id": "device-old",
            "space_id": "space-old",
            "operator_slug": "owner-5678",
        },
        "runtime": {
            "kind": "chat-only",
            "provider": "anthropic",
            "model": "legacy-model",
            "api_key": "fake-old-key",
            "allowed_tools": ["Read", "Bash(git status)"],
            "permission_mode": "bypassPermissions",
            "harness": "claude-code",
            "max_turns": 23,
        },
        "profile": "identity/profile.md",
        "memory_dir": "memory-old",
        "workspace_dir": "workspace-old",
        "triggers": {"on_mention": False, "on_dm": True},
    })

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.kind == "cli-local"
    assert loaded.runtime.task_timeout_seconds == 600.0
    loaded.save()

    saved = yaml.safe_load(agent_yml_path(agent_id).read_text(encoding="utf-8"))
    assert saved["state"] == "paused"
    assert saved["display_name"] == "Legacy Agent"
    assert saved["avatar_url"] == "https://cdn.example/avatar.png"
    assert saved["role"] == "Keeps the old workspace running"
    assert saved["role_short"] == "Keeper"
    assert saved["created_at"] == 123456789
    assert saved["puffo_core"]["server_url"] == "https://api.example"
    assert saved["puffo_core"]["slug"] == "legacy-1234"
    assert saved["puffo_core"]["device_id"] == "device-old"
    assert saved["puffo_core"]["space_id"] == "space-old"
    assert saved["puffo_core"]["operator_slug"] == "owner-5678"
    assert saved["runtime"]["kind"] == "cli-local"
    assert saved["runtime"]["provider"] == "anthropic"
    assert saved["runtime"]["model"] == "legacy-model"
    assert saved["runtime"]["api_key"] == "fake-old-key"
    assert saved["runtime"]["allowed_tools"] == ["Read", "Bash(git status)"]
    assert saved["runtime"]["max_turns"] == 23
    assert saved["profile"] == "identity/profile.md"
    assert saved["memory_dir"] == "memory-old"
    assert saved["workspace_dir"] == "workspace-old"
    assert saved["triggers"] == {"on_mention": False, "on_dm": True}


@pytest.mark.parametrize(
    ("provider", "expected_harness"),
    [("", "claude-code"), ("anthropic", "claude-code"), ("openai", "codex")],
)
def test_stale_hermes_harness_migrates_to_driver_harness(
    home, provider, expected_harness,
):
    """``cli-local`` + ``hermes`` was valid before the Driver runtime landed.

    Such configs must migrate to the provider-resolved harness rather than
    leave the agent unloadable.
    """
    from puffo_agent.portal.state import AgentConfig

    agent_id = f"stale-hermes-{provider or 'default'}"
    runtime = {"kind": "cli-local", "harness": "hermes"}
    if provider:
        runtime["provider"] = provider
    _write_agent(agent_id, {"id": agent_id, "runtime": runtime})

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.harness == expected_harness
    assert loaded.runtime.provider == provider


def test_stale_hermes_migration_clears_unsupported_inference_level(home):
    """``minimal`` is a codex level; migrating to claude-code must drop it."""
    from puffo_agent.portal.state import AgentConfig

    agent_id = "stale-hermes-level"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {
            "kind": "cli-local",
            "provider": "anthropic",
            "harness": "hermes",
            "inference_level": "minimal",
        },
    })

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.harness == "claude-code"
    assert loaded.runtime.inference_level == ""


def test_stale_hermes_migration_keeps_supported_inference_level(home):
    from puffo_agent.portal.state import AgentConfig

    agent_id = "stale-hermes-keeps-level"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {
            "kind": "cli-local",
            "provider": "openai",
            "harness": "hermes",
            "inference_level": "medium",
        },
    })

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.harness == "codex"
    assert loaded.runtime.inference_level == "medium"


def test_local_gemini_cli_config_still_fails_to_load(home):
    """No Driver harness serves google, so this stays an explicit error."""
    from puffo_agent.portal.state import AgentConfig

    agent_id = "stale-gemini"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {
            "kind": "cli-local",
            "provider": "google",
            "harness": "gemini-cli",
        },
    })

    with pytest.raises(RuntimeError, match="not implemented by the Driver runtime"):
        AgentConfig.load(agent_id)


def test_docker_hermes_config_is_left_alone(home):
    """Migration is scoped to ``cli-local``; hermes is valid on cli-docker."""
    from puffo_agent.portal.state import AgentConfig

    agent_id = "docker-hermes"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {
            "kind": "cli-docker",
            "provider": "openai",
            "harness": "hermes",
        },
    })

    assert AgentConfig.load(agent_id).runtime.harness == "hermes"


def test_legacy_kind_with_openai_provider_resolves_codex(home):
    """A legacy kind carrying the default claude-code harness still migrates."""
    from puffo_agent.portal.state import AgentConfig

    agent_id = "legacy-openai"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {"kind": "sdk-local", "provider": "openai"},
    })

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.kind == "cli-local"
    assert loaded.runtime.harness == "codex"


def test_task_timeout_seconds_round_trips(home):
    from puffo_agent.portal.state import AgentConfig

    agent_id = "custom-timeout"
    _write_agent(agent_id, {
        "id": agent_id,
        "runtime": {
            "kind": "cli-local",
            "provider": "openai",
            "harness": "codex",
            "task_timeout_seconds": 123.5,
        },
    })

    loaded = AgentConfig.load(agent_id)
    assert loaded.runtime.task_timeout_seconds == 123.5
    loaded.save()
    assert AgentConfig.load(agent_id).runtime.task_timeout_seconds == 123.5
