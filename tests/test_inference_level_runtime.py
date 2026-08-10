"""Inference-level persistence across CLI and daemon refresh paths."""

from __future__ import annotations

import pytest

from puffo_agent.mcp.host_tools import _write_refresh_model_flag
from puffo_agent.portal import daemon as daemon_module
from puffo_agent.portal.cli import build_parser
from puffo_agent.portal.state import AgentConfig, RuntimeConfig


def _save_codex_agent(tmp_path, monkeypatch, *, level: str = "minimal") -> AgentConfig:
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    cfg = AgentConfig(
        id="agent-inference",
        runtime=RuntimeConfig(
            kind="cli-local",
            provider="openai",
            harness="codex",
            model="gpt-5.5",
            inference_level=level,
        ),
    )
    cfg.save()
    return cfg


def test_daemon_refresh_persists_standalone_inference_level(tmp_path, monkeypatch):
    cfg = _save_codex_agent(tmp_path, monkeypatch, level="low")
    _write_refresh_model_flag(
        cfg.resolve_workspace_dir(),
        harness="",
        model="",
        inference_level="high",
    )

    daemon_module._process_daemon_refresh_flags(cfg.id)

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "codex"
    assert loaded.runtime.model == "gpt-5.5"
    assert loaded.runtime.inference_level == "high"


def test_daemon_harness_swap_clears_incompatible_inference_level(
    tmp_path, monkeypatch,
):
    cfg = _save_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daemon_module, "_validate_daemon_refresh_model", lambda harness, model: None,
    )
    _write_refresh_model_flag(
        cfg.resolve_workspace_dir(),
        harness="claude-code",
        model="claude-sonnet-4-6",
    )

    daemon_module._process_daemon_refresh_flags(cfg.id)

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.provider == "anthropic"
    assert loaded.runtime.harness == "claude-code"
    assert loaded.runtime.model == "claude-sonnet-4-6"
    assert loaded.runtime.inference_level == ""


def test_cli_harness_swap_clears_incompatible_inference_level(
    tmp_path, monkeypatch, capsys,
):
    cfg = _save_codex_agent(tmp_path, monkeypatch)
    args = build_parser().parse_args([
        "agent", "runtime", cfg.id,
        "--provider", "anthropic",
        "--harness", "claude-code",
    ])

    assert args.func(args) == 0

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "claude-code"
    assert loaded.runtime.inference_level == ""
    assert "incompatible prior value cleared" in capsys.readouterr().out


def test_load_rejects_inference_level_incompatible_with_harness(
    tmp_path, monkeypatch,
):
    cfg = _save_codex_agent(tmp_path, monkeypatch)
    cfg.runtime.harness = "claude-code"
    cfg.runtime.provider = "anthropic"
    cfg.save()

    with pytest.raises(RuntimeError, match="inference_level='minimal'.*claude-code"):
        AgentConfig.load(cfg.id)


def _save_claude_agent(tmp_path, monkeypatch, *, level: str = "xhigh") -> AgentConfig:
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    cfg = AgentConfig(
        id="agent-inference",
        runtime=RuntimeConfig(
            kind="cli-local",
            provider="anthropic",
            harness="claude-code",
            model="claude-sonnet-4-6",
            inference_level=level,
        ),
    )
    cfg.save()
    return cfg


def test_bridge_patch_harness_swap_saves_a_loadable_config(tmp_path, monkeypatch):
    """A PATCH that swaps harness must not persist the old harness's level."""
    from puffo_agent.portal.api.runtime_patch import apply_runtime_patch

    cfg = _save_claude_agent(tmp_path, monkeypatch)

    assert apply_runtime_patch(
        cfg.runtime, {"harness": "codex", "provider": "openai"},
    ) is None
    cfg.save()

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "codex"
    assert loaded.runtime.inference_level == ""


def test_bridge_patch_harness_swap_saves_a_loadable_config_reverse(
    tmp_path, monkeypatch,
):
    from puffo_agent.portal.api.runtime_patch import apply_runtime_patch

    cfg = _save_codex_agent(tmp_path, monkeypatch, level="minimal")

    assert apply_runtime_patch(
        cfg.runtime, {"harness": "claude-code", "provider": "anthropic"},
    ) is None
    cfg.save()

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "claude-code"
    assert loaded.runtime.inference_level == ""


def test_bridge_patch_keeps_a_still_supported_inference_level(tmp_path, monkeypatch):
    from puffo_agent.portal.api.runtime_patch import apply_runtime_patch

    cfg = _save_claude_agent(tmp_path, monkeypatch, level="high")

    assert apply_runtime_patch(
        cfg.runtime, {"harness": "codex", "provider": "openai"},
    ) is None
    cfg.save()

    assert AgentConfig.load(cfg.id).runtime.inference_level == "high"


def test_control_edit_harness_swap_saves_a_loadable_config(tmp_path, monkeypatch):
    """The cloud-control ``edit`` writer follows the same rule as the CLI."""
    from puffo_agent.portal.control.client import _apply_edit_runtime

    cfg = _save_claude_agent(tmp_path, monkeypatch)

    changed, error = _apply_edit_runtime(
        cfg, {"runtime": {"harness": "codex", "provider": "openai"}},
    )
    assert (changed, error) == (True, None)
    cfg.save()

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "codex"
    assert loaded.runtime.inference_level == ""


def test_cli_switch_to_harness_without_inference_support_clears_level(
    tmp_path, monkeypatch,
):
    cfg = _save_codex_agent(tmp_path, monkeypatch, level="high")
    args = build_parser().parse_args([
        "agent", "runtime", cfg.id,
        "--kind", "cli-docker",
        "--harness", "hermes",
    ])

    assert args.func(args) == 0

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "hermes"
    assert loaded.runtime.inference_level == ""
