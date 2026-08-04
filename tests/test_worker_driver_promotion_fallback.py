from __future__ import annotations

from types import SimpleNamespace

import pytest

from puffo_agent.agent.harness.runtime_manager import RuntimeManagerAdapter
from puffo_agent.portal.state import AgentConfig, DaemonConfig, RuntimeConfig
from puffo_agent.portal import worker as worker_module


@pytest.mark.asyncio
async def test_failed_warm_skips_driver_promotion(monkeypatch):
    adapter = object()
    called = False

    async def must_not_promote(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("promotion must not run after failed warm-up")

    monkeypatch.setattr(
        worker_module, "_promote_supported_local_driver", must_not_promote
    )
    result = await worker_module._promote_local_driver_after_warm(
        adapter,
        AgentConfig(id="warm-failed"),
        warm_ok=False,
        system_prompt="system",
        outbox=object(),
        logical_session_ref="session",
    )

    assert result is adapter
    assert called is False


@pytest.mark.asyncio
async def test_driver_promotion_failure_keeps_legacy_adapter(monkeypatch):
    adapter = object()

    async def fail_promotion(*args, **kwargs):
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(
        worker_module, "_promote_supported_local_driver", fail_promotion
    )
    result = await worker_module._promote_local_driver_after_warm(
        adapter,
        AgentConfig(id="promotion-failed"),
        warm_ok=True,
        system_prompt="system",
        outbox=object(),
        logical_session_ref="session",
    )

    assert result is adapter


@pytest.mark.asyncio
async def test_promotion_warms_driver_before_closing_legacy_adapter(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    config = AgentConfig(
        id="promotion-order",
        runtime=RuntimeConfig(kind="cli-local", harness="codex"),
    )
    adapter = worker_module.build_adapter(DaemonConfig(), config)
    adapter.workspace_dir = str(tmp_path)
    adapter._verify = lambda: None

    async def no_install():
        return None

    adapter._install_desired = no_install
    adapter._ensure_codex_session = lambda: SimpleNamespace(
        argv=["codex", "app-server"],
        env={},
        get_provider_session_id=lambda: None,
    )
    legacy_closed = False
    promoted_closed = False

    async def close_legacy():
        nonlocal legacy_closed
        legacy_closed = True

    async def fail_warm(self, system_prompt):
        raise RuntimeError("driver unavailable")

    async def close_promoted(self):
        nonlocal promoted_closed
        promoted_closed = True

    adapter.aclose = close_legacy
    monkeypatch.setattr(
        "puffo_agent.agent.harness.build_driver", lambda _name: object()
    )
    monkeypatch.setattr(RuntimeManagerAdapter, "warm", fail_warm)
    monkeypatch.setattr(RuntimeManagerAdapter, "aclose", close_promoted)
    outbox = SimpleNamespace(state=lambda: {})

    with pytest.raises(RuntimeError, match="driver unavailable"):
        await worker_module._promote_supported_local_driver(
            adapter,
            config,
            system_prompt="system",
            outbox=outbox,
            logical_session_ref="logical-session",
        )

    assert legacy_closed is False
    assert promoted_closed is True
