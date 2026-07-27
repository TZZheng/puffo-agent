from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from puffo_agent.agent.adapters.base import Adapter, TurnContext, TurnResult
from puffo_agent.agent.adapters.chat_only import ChatOnlyAdapter
from puffo_agent.agent.adapters.docker_cli import DockerCLIAdapter
from puffo_agent.agent.adapters.local_cli import LocalCLIAdapter
from puffo_agent.agent.adapters.sdk import SDKAdapter
from puffo_agent.agent.context_controller import (
    AdmissionCandidate,
    CompactionResult,
    ContextCapabilities,
    ContextController,
    ContextDecision,
    ContextSnapshot,
    DecisionOutcome,
    FALLBACK_CONTEXT_WINDOW,
    ProviderAdmissionEvent,
    RolloverResult,
    normalize_context_snapshot,
)


NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def snapshot(used=60_000, window=200_000, source="provider"):
    return ContextSnapshot(used, window, source, NOW)


def candidate(
    batch=30_000, wrapper=5_000, reserve=5_000, minimum=0, cycle="cycle-1",
):
    return AdmissionCandidate(
        planning_cycle_key=cycle,
        formatted_batch_tokens=batch,
        wrapper_overhead_tokens=wrapper,
        output_tool_reserve_tokens=reserve,
        minimum_formatted_batch_tokens=minimum,
        payload={"opaque": True},
    )


class FakeProvider:
    def __init__(
        self,
        snapshots,
        *,
        compact_results=(),
        native_compaction=False,
        rollover=False,
    ):
        self.snapshots = list(snapshots)
        self.compact_results = list(compact_results)
        self.capabilities = ContextCapabilities(
            native_compaction=native_compaction,
            rollover=rollover,
            native_measurement=True,
        )
        self.calls = []

    async def get_context_snapshot(self):
        self.calls.append("snapshot")
        return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]

    def get_context_capabilities(self):
        return self.capabilities

    async def compact_context(self):
        self.calls.append("compact")
        return self.compact_results.pop(0)

    async def rollover_context(self):
        self.calls.append("rollover")
        return RolloverResult(True, "old", None)

    def get_provider_session_id(self):
        return "provider-session"


def test_projection_soft_target_exact_boundary_and_one_token_pressure():
    provider = FakeProvider([snapshot()])
    controller = ContextController(provider)
    admitted = asyncio.run(controller.decide(candidate()))
    assert admitted.outcome is DecisionOutcome.ADMIT
    assert admitted.projected_tokens == 100_000

    pressured = asyncio.run(
        ContextController(FakeProvider([snapshot()])).decide(candidate(batch=30_001))
    )
    assert pressured.outcome is not DecisionOutcome.ADMIT
    assert ContextController.projection(snapshot(), candidate(batch=30_001)) == 100_001


def test_projection_includes_all_four_terms():
    base = snapshot(used=11)
    cand = candidate(batch=13, wrapper=17, reserve=19)
    assert ContextController.projection(base, cand) == 60


def test_fallback_precedence_and_inspectable_source():
    provider = normalize_context_snapshot(
        used_tokens=1,
        provider_context_window=123,
        verified_model_context_window=456,
        measured_at=NOW,
    )
    model = normalize_context_snapshot(
        used_tokens=1,
        verified_model_context_window=456,
        measured_at=NOW,
    )
    fallback = normalize_context_snapshot(used_tokens=1, measured_at=NOW)
    assert (provider.context_window, provider.source) == (123, "provider")
    assert (model.context_window, model.source) == (456, "verified_model")
    assert fallback.context_window == FALLBACK_CONTEXT_WINDOW == 200_000
    assert "fallback" in fallback.source


def test_contract_records_are_frozen_and_decisions_are_closed_set():
    records = [
        snapshot(),
        ContextCapabilities(),
        CompactionResult(False),
        RolloverResult(False),
        candidate(),
        ContextDecision(DecisionOutcome.ADMIT, candidate(), snapshot(), 1),
        ProviderAdmissionEvent("k", None, NOW),
    ]
    for record in records:
        with pytest.raises(FrozenInstanceError):
            setattr(record, next(iter(record.__dataclass_fields__)), "changed")
    assert {outcome.value for outcome in DecisionOutcome} == {
        "admit", "replan", "shrink", "rollover", "degraded",
    }


def test_replan_order_and_arrival_after_compaction():
    provider = FakeProvider(
        [snapshot(100_000), snapshot(10_000)],
        compact_results=[CompactionResult(True)],
        native_compaction=True,
    )

    async def replan(old):
        provider.calls.append("replan")
        return AdmissionCandidate(
            "cycle-2", 7, 5, 5, payload={"arrival": "new-message"},
        )

    decision = asyncio.run(ContextController(provider, replan).decide(candidate()))
    assert decision.outcome is DecisionOutcome.REPLAN
    assert decision.candidate.payload == {"arrival": "new-message"}
    assert provider.calls == ["snapshot", "compact", "snapshot", "replan"]


def test_successful_compaction_is_bounded_per_lifecycle_cycle():
    provider = FakeProvider(
        [snapshot(100_000), snapshot(100_000)],
        compact_results=[CompactionResult(True)],
        native_compaction=True,
    )
    controller = ContextController(provider, lambda old: asyncio.sleep(0, result=old))
    assert asyncio.run(controller.decide(candidate())).outcome is DecisionOutcome.REPLAN
    assert asyncio.run(controller.decide(candidate())).outcome in {
        DecisionOutcome.SHRINK, DecisionOutcome.DEGRADED,
    }
    assert provider.calls.count("compact") == 1


def test_failed_compaction_bounded_to_two_then_shrink():
    provider = FakeProvider(
        [snapshot(60_000)],
        compact_results=[CompactionResult(False), CompactionResult(False)],
        native_compaction=True,
    )
    controller = ContextController(provider)
    for _ in range(3):
        decision = asyncio.run(controller.decide(candidate(batch=30_001, minimum=1)))
    assert provider.calls.count("compact") == 2
    assert decision.outcome is DecisionOutcome.SHRINK


def test_failed_compaction_bound_is_independent_per_planning_cycle():
    provider = FakeProvider(
        [snapshot(60_000)],
        compact_results=[CompactionResult(False) for _ in range(4)],
        native_compaction=True,
    )
    controller = ContextController(provider)
    for cycle in ("cycle-a", "cycle-b"):
        for _ in range(3):
            decision = asyncio.run(
                controller.decide(
                    candidate(batch=30_001, minimum=1, cycle=cycle),
                ),
            )
        assert decision.outcome is DecisionOutcome.SHRINK
    assert provider.calls.count("compact") == 4


def test_rollover_is_bounded_when_fixed_occupancy_cannot_fit():
    provider = FakeProvider([snapshot(100_000), snapshot(0)], rollover=True)
    controller = ContextController(provider)
    first = asyncio.run(controller.decide(candidate(minimum=30_000)))
    second = asyncio.run(controller.decide(candidate(minimum=30_000)))
    assert first.outcome is DecisionOutcome.ROLLOVER
    assert second.outcome is DecisionOutcome.ADMIT
    assert provider.calls.count("rollover") == 1


def test_degraded_when_control_unsupported_and_nothing_can_fit():
    result = asyncio.run(
        ContextController(FakeProvider([snapshot(100_000)])).decide(
            candidate(minimum=30_000),
        )
    )
    assert result.outcome is DecisionOutcome.DEGRADED


class MinimalAdapter(Adapter):
    async def run_turn(self, ctx: TurnContext) -> TurnResult:
        return TurnResult("ok")


def test_default_adapter_compatibility_and_one_shot_admission_callback():
    adapter = MinimalAdapter()
    events = []

    async def callback(event):
        events.append(event)
        raise RuntimeError("consumer error")

    adapter.register_admission_callback(callback, "cycle")
    event = ProviderAdmissionEvent("cycle", None, NOW)
    with pytest.raises(RuntimeError):
        asyncio.run(adapter._fire_admission_callback(event))
    asyncio.run(adapter._fire_admission_callback(event))
    assert events == [event]
    assert adapter.get_provider_session_id() is None
    snap = asyncio.run(adapter.get_context_snapshot())
    assert snap.context_window == 200_000
    assert "estimated" in snap.source


def test_provider_protocol_is_fakeable_without_adapter():
    provider = FakeProvider([snapshot()])
    result = asyncio.run(ContextController(provider).decide(candidate()))
    assert result.outcome is DecisionOutcome.ADMIT


def test_chat_stateless_admission_after_valid_completion():
    order = []

    class Provider:
        def complete(self, system, messages):
            order.append("complete")
            return "reply", 2, 3

    adapter = ChatOnlyAdapter(Provider())

    async def admitted(event):
        order.append("admitted")
        assert event.provider_session_id is None

    adapter.register_admission_callback(admitted, "chat-cycle")
    result = asyncio.run(adapter.run_turn(TurnContext("system", [])))
    assert result.reply == "reply"
    assert order == ["complete", "admitted"]
    assert adapter.get_provider_session_id() is None
    assert "stateless" in asyncio.run(adapter.get_context_snapshot()).source


def test_local_and_docker_wrapper_delegate_context_contract():
    class Harness:
        def name(self):
            return "claude-code"

    class Session(FakeProvider):
        def __init__(self):
            super().__init__([snapshot(12)])
            self.registered = None

        def register_admission_callback(self, callback, planning_cycle_key=""):
            self.registered = (callback, planning_cycle_key)

    callback = lambda event: asyncio.sleep(0)
    for wrapper_type in (LocalCLIAdapter, DockerCLIAdapter):
        wrapper = wrapper_type.__new__(wrapper_type)
        wrapper.harness = Harness()
        wrapper._session = Session()
        wrapper._one_shot_provider_session_id = None
        if wrapper_type is LocalCLIAdapter:
            wrapper._codex_session = None
        assert asyncio.run(wrapper.get_context_snapshot()).used_tokens == 12
        wrapper.register_admission_callback(callback, "wrapped-cycle")
        assert wrapper._session.registered == (callback, "wrapped-cycle")
        assert wrapper.get_provider_session_id() == "provider-session"


def test_local_codex_and_one_shot_hermes_gemini_compatibility(
    tmp_path, monkeypatch,
):
    import puffo_agent.agent.adapters.docker_cli as docker_module
    import puffo_agent.agent.adapters.local_cli as local_module

    class Harness:
        def __init__(self, value):
            self.value = value

        def name(self):
            return self.value

    codex = FakeProvider([snapshot(44)])
    local = LocalCLIAdapter.__new__(LocalCLIAdapter)
    local.harness = Harness("codex")
    local._codex_session = codex
    local._session = None
    local._one_shot_provider_session_id = None
    assert asyncio.run(local.get_context_snapshot()).used_tokens == 44
    assert local.get_context_capabilities() is codex.capabilities
    assert local.get_provider_session_id() == "provider-session"

    hermes = LocalCLIAdapter.__new__(LocalCLIAdapter)
    hermes.harness = Harness("hermes")
    hermes._codex_session = None
    hermes._session = None
    hermes._one_shot_provider_session_id = "hermes-real-session"
    hermes.session_file = tmp_path / "hermes-session.json"
    hermes_snapshot = asyncio.run(hermes.get_context_snapshot())
    assert "hermes_unsupported" in hermes_snapshot.source
    assert "unsupported" in hermes.get_context_capabilities().diagnostic
    assert hermes.get_provider_session_id() == "hermes-real-session"

    gemini = DockerCLIAdapter.__new__(DockerCLIAdapter)
    gemini.harness = Harness("gemini-cli")
    gemini._session = None
    gemini._one_shot_provider_session_id = "gemini-real-session"
    gemini.session_file = tmp_path / "gemini-session.json"
    gemini_snapshot = asyncio.run(gemini.get_context_snapshot())
    assert "gemini-cli_unsupported" in gemini_snapshot.source
    assert "gemini-cli" in gemini.get_context_capabilities().diagnostic
    assert gemini.get_provider_session_id() == "gemini-real-session"

    admission_events = []

    async def admitted(event):
        admission_events.append(event)

    async def fake_hermes_run(*args, **kwargs):
        return 0, b"session_id: hermes-observed\nreply", b""

    hermes._hermes_bin = "hermes"
    hermes._hermes_home = tmp_path
    hermes._hermes_mcp_registered = True
    hermes._hermes_audit = None
    hermes.puffo_core_mcp_env = None
    hermes.agent_id = "hermes-test"
    hermes.model = ""
    hermes.workspace_dir = str(tmp_path)
    monkeypatch.setattr(local_module, "hermes_run_cmd", fake_hermes_run)
    hermes.register_admission_callback(admitted, "hermes-boundary")
    asyncio.run(hermes._run_hermes_turn("hello", "system"))

    async def fake_gemini_run(*args, **kwargs):
        return (
            0,
            b'{"response":"reply","session_id":"gemini-observed"}',
            b"",
        )

    gemini.google_api_key = "test-key"
    gemini.container_name = "test-container"
    gemini.agent_id = "gemini-test"
    gemini.model = ""
    monkeypatch.setattr(docker_module, "_run_cmd", fake_gemini_run)
    gemini.register_admission_callback(admitted, "gemini-boundary")
    asyncio.run(gemini._run_gemini_chat("hello", "system"))
    assert [
        (event.planning_cycle_key, event.provider_session_id)
        for event in admission_events
    ] == [
        ("hermes-boundary", "hermes-observed"),
        ("gemini-boundary", "gemini-observed"),
    ]


def test_sdk_stateless_admits_on_first_yielded_provider_message():
    class Result:
        usage = {"input_tokens": 2, "output_tokens": 3}

    class Never:
        pass

    async def query(**kwargs):
        yield object()
        yield Result()

    adapter = SDKAdapter.__new__(SDKAdapter)
    adapter._query = query
    adapter._Options = lambda **kwargs: kwargs
    adapter._AssistantMessage = Never
    adapter._TextBlock = Never
    adapter._ToolUseBlock = Never
    adapter._ResultMessage = Result
    adapter.api_key = ""
    adapter.model = ""
    adapter.patterns = []
    adapter.permission_mode = None
    adapter.workspace_dir = ""
    adapter.max_turns = 1
    adapter.mcp_servers_override = None
    order = []

    async def admitted(event):
        order.append("admitted")
        assert event.provider_session_id is None

    adapter.register_admission_callback(admitted, "sdk-cycle")
    result = asyncio.run(adapter.run_turn(TurnContext("system", [])))
    assert order == ["admitted"]
    assert (result.input_tokens, result.output_tokens) == (2, 3)
    assert adapter.get_provider_session_id() is None
    assert "stateless" in asyncio.run(adapter.get_context_snapshot()).source
