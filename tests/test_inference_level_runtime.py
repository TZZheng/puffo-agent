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


def test_cli_switch_to_harness_without_inference_support_clears_level(
    tmp_path, monkeypatch,
):
    cfg = _save_codex_agent(tmp_path, monkeypatch, level="high")
    args = build_parser().parse_args([
        "agent", "runtime", cfg.id,
        "--harness", "hermes",
    ])

    assert args.func(args) == 0

    loaded = AgentConfig.load(cfg.id)
    assert loaded.runtime.harness == "hermes"
    assert loaded.runtime.inference_level == ""
