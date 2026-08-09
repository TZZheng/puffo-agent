from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from _portal_support import isolated_home, write_test_agent
from puffo_agent.agent.adapters.docker_cli import DockerCLIAdapter
from puffo_agent.agent.adapters.local_cli import LocalCLIAdapter
from puffo_agent.portal import cli, state
from puffo_agent.portal.daemon import Daemon
from puffo_agent.portal.state import AgentConfig, DaemonConfig
from puffo_agent.portal.worker import build_adapter


def _local_adapter(tmp_path: Path, *, api_key: str = "") -> LocalCLIAdapter:
    return LocalCLIAdapter(
        agent_id="local-key-policy",
        model="claude-sonnet-5",
        workspace_dir=str(tmp_path / "workspace"),
        claude_dir=str(tmp_path / "workspace" / ".claude"),
        session_file=str(tmp_path / "session.json"),
        mcp_config_file=str(tmp_path / "mcp.json"),
        agent_home_dir=str(tmp_path / "agent-home"),
        claude_api_key=api_key,
    )


def _docker_adapter(tmp_path: Path, *, api_key: str = "") -> DockerCLIAdapter:
    return DockerCLIAdapter(
        agent_id="docker-key-policy",
        model="claude-sonnet-5",
        image="puffo/agent-runtime:test",
        workspace_dir=str(tmp_path / "workspace"),
        claude_dir=str(tmp_path / "workspace" / ".claude"),
        session_file=str(tmp_path / "session.json"),
        agent_home_dir=str(tmp_path / "agent-home"),
        shared_fs_dir=str(tmp_path / "shared"),
        claude_api_key=api_key,
    )


def test_daemon_anthropic_cli_api_key_opt_in_round_trips(tmp_path, monkeypatch):
    config_path = tmp_path / "daemon.yml"
    monkeypatch.setattr(state, "daemon_yml_path", lambda: config_path)
    cfg = DaemonConfig()
    assert cfg.anthropic.cli_use_api_key is False
    cfg.anthropic.api_key = "daemon-key"
    cfg.anthropic.cli_use_api_key = True

    cfg.save()

    loaded = DaemonConfig.load()
    assert loaded.anthropic.api_key == "daemon-key"
    assert loaded.anthropic.cli_use_api_key is True
    assert "cli_use_api_key" not in loaded.openai.__dict__


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, False), (False, False), ("true", False), (True, True)],
)
def test_daemon_cli_api_key_requires_yaml_boolean_true(
    tmp_path, monkeypatch, raw, expected,
):
    config_path = tmp_path / "daemon.yml"
    monkeypatch.setattr(state, "daemon_yml_path", lambda: config_path)
    anthropic = {"api_key": "daemon-key"}
    if raw is not None:
        anthropic["cli_use_api_key"] = raw
    config_path.write_text(
        "anthropic:\n" + "".join(f"  {key}: {json.dumps(value)}\n" for key, value in anthropic.items()),
        encoding="utf-8",
    )

    assert DaemonConfig.load().anthropic.cli_use_api_key is expected


def test_config_command_preserves_cli_api_key_opt_in(monkeypatch):
    isolated_home()
    cfg = DaemonConfig()
    cfg.anthropic.api_key = "daemon-key"
    cfg.anthropic.cli_use_api_key = True
    cfg.save()
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert cli.main(["config"]) == 0

    loaded = DaemonConfig.load()
    assert loaded.anthropic.api_key == "daemon-key"
    assert loaded.anthropic.cli_use_api_key is True


def test_config_command_ignores_ambient_anthropic_api_key(monkeypatch):
    isolated_home()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert cli.main(["config"]) == 0

    assert DaemonConfig.load().anthropic.api_key == ""


def test_settings_scrubber_handles_non_object_and_empty_env(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[]", encoding="utf-8")
    assert state.strip_claude_api_key_from_settings(path) is False
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "stale-key"}}),
        encoding="utf-8",
    )

    assert state.strip_claude_api_key_from_settings(path) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_settings_scrubber_reports_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "stale-key"}}),
        encoding="utf-8",
    )

    def fail_write(_path, _data):
        raise OSError("read-only")

    monkeypatch.setattr(state, "_atomic_write_json", fail_write)

    assert state.strip_claude_api_key_from_settings(path) is False


def test_local_ignores_ambient_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    adapter = _local_adapter(tmp_path)
    adapter._prepare_mcp_args = lambda: []

    session = adapter._ensure_session()

    assert "ANTHROPIC_API_KEY" not in session.env


def test_local_injects_only_configured_daemon_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    adapter = _local_adapter(tmp_path, api_key="daemon-key")
    adapter._prepare_mcp_args = lambda: []

    session = adapter._ensure_session()

    assert session.env["ANTHROPIC_API_KEY"] == "daemon-key"


def test_local_removes_persisted_api_key_settings(tmp_path):
    adapter = _local_adapter(tmp_path)
    adapter._prepare_mcp_args = lambda: []
    paths = (
        adapter.agent_home_dir / ".claude" / "settings.json",
        Path(adapter.claude_dir) / "settings.json",
        Path(adapter.claude_dir) / "settings.local.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "env": {"ANTHROPIC_API_KEY": "stale-key", "KEEP": "value"},
            }),
            encoding="utf-8",
        )

    adapter._ensure_session()

    for path in paths:
        assert json.loads(path.read_text(encoding="utf-8"))["env"] == {
            "KEEP": "value",
        }


def test_docker_ignores_ambient_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    adapter = _docker_adapter(tmp_path)
    adapter._prepare_mcp_args = lambda: []

    session = adapter._ensure_session()
    command = session.build_command([], session.env_overrides)

    assert "ANTHROPIC_API_KEY" not in session.env
    assert "ANTHROPIC_API_KEY=" in command


def test_docker_injects_configured_key_without_argv_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    adapter = _docker_adapter(tmp_path, api_key="daemon-key")
    adapter._prepare_mcp_args = lambda: []

    session = adapter._ensure_session()
    command = session.build_command([], session.env_overrides)

    assert session.env["ANTHROPIC_API_KEY"] == "daemon-key"
    key_index = command.index("ANTHROPIC_API_KEY")
    assert command[key_index - 1] == "-e"
    assert "daemon-key" not in command
    assert "ambient-key" not in command


@pytest.mark.parametrize("kind", ["cli-local", "cli-docker"])
@pytest.mark.parametrize(
    ("api_key", "enabled", "expected"),
    [("daemon-key", False, ""), ("", True, ""), ("daemon-key", True, "daemon-key")],
)
def test_worker_applies_daemon_key_policy(kind, api_key, enabled, expected):
    home = isolated_home()
    write_test_agent(home, "key-policy")
    cfg = AgentConfig.load("key-policy")
    cfg.runtime.kind = kind
    cfg.runtime.provider = "anthropic"
    cfg.runtime.harness = "claude-code"
    daemon = DaemonConfig()
    daemon.anthropic.api_key = api_key
    daemon.anthropic.cli_use_api_key = enabled

    adapter = build_adapter(daemon, cfg)

    assert adapter.claude_api_key == expected


@pytest.mark.parametrize("kind", ["cli-local", "cli-docker"])
def test_codex_never_receives_anthropic_daemon_key(kind):
    home = isolated_home()
    write_test_agent(home, "codex-key-policy")
    cfg = AgentConfig.load("codex-key-policy")
    cfg.runtime.kind = kind
    cfg.runtime.provider = "openai"
    cfg.runtime.harness = "codex"
    daemon = DaemonConfig()
    daemon.anthropic.api_key = "daemon-key"
    daemon.anthropic.cli_use_api_key = True

    adapter = build_adapter(daemon, cfg)

    assert adapter.claude_api_key == ""


@pytest.mark.parametrize(
    ("api_key", "enabled", "uses_api_key"),
    [("", True, False), ("daemon-key", False, False), ("daemon-key", True, True)],
)
def test_daemon_oauth_gate_follows_api_key_policy(api_key, enabled, uses_api_key):
    home = isolated_home()
    write_test_agent(home, "refresh-policy")
    cfg = AgentConfig.load("refresh-policy")
    cfg.runtime.kind = "cli-local"
    cfg.runtime.harness = "claude-code"
    daemon_cfg = DaemonConfig()
    daemon_cfg.anthropic.api_key = api_key
    daemon_cfg.anthropic.cli_use_api_key = enabled
    daemon = Daemon(daemon_cfg)

    if uses_api_key:
        assert daemon._notify_refresh_for(cfg) is None
        assert daemon._ensure_fresh_for(cfg) is None
        worker = SimpleNamespace()
        daemon._register_with_refresher(cfg, worker)
        assert not hasattr(worker, "_refresh_success_callback")
    else:
        assert daemon._notify_refresh_for(cfg) is not None
        assert daemon._ensure_fresh_for(cfg) is not None
