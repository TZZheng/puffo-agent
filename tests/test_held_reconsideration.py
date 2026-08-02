from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from puffo_agent.agent.send_coordinator import (
    CHANNEL_SEND_PATH,
    KEYLESS_CHANNEL_SEND_PATH,
    SemanticSendRequest,
    SendCoordinator,
    _HeldEvidence,
)


@pytest.mark.asyncio
async def test_bridge_releases_held_wait_only_after_exact_durable_pair(tmp_path):
    from puffo_agent.agent.global_inbox_runtime import (
        ActiveBoundaryAdapter,
        GlobalInboxRuntime,
    )
    from .test_cloud_bridge_msgflow import FakeBridge, _bridge_client
    from .test_global_inbox_runtime import ToolReturnAdapter

    bridge = FakeBridge()
    client = _bridge_client(tmp_path, bridge, db="held-bridge.db")
    base = {
        "type": "message",
        "sender_slug": "alice-0001",
        "space_id": "sp_1",
        "channel_id": "ch_1",
        "plaintext": "context",
    }
    await client._dispatch_bridge_frame({
        **base, "seq": 2, "envelope_id": "already-admitted",
    })
    await client.store.admit_messages(
        ["already-admitted"],
        turn_id="turn-a",
        provider_session_id="session-a",
    )
    adapter = ToolReturnAdapter()
    adapter.session = "session-a"
    runtime = GlobalInboxRuntime(
        store=client.store,
        adapter=adapter,
        run_turn=lambda _planned: None,
        workspace=tmp_path,
    )
    runtime.active.turn_id = "turn-a"
    runtime.active.message_ids[:] = ["already-admitted"]
    runtime.active.provider_session_id = "session-a"
    client.global_runtime = runtime

    cfg, http, _unused_store = _setup_keyless()
    freshness = Freshness(baseline=2)
    boundary = ActiveBoundaryAdapter(client.store, runtime.active)
    coordinator = SendCoordinator(
        slug=cfg.slug,
        keystore=cfg.keystore,
        http_client=http,
        data_client=client.store,
        baseline_source=freshness,
        active_turn_source=boundary,
        held_recovery_source=runtime.held_recovery_source,
        provider_session_id="session-a",
    )
    calls = []

    async def post_unsigned(path, body=None):
        calls.append((path, body))
        if len(calls) == 1:
            return {
                "state": "held",
                "seen_seq": body["freshness"]["seen_seq"],
                "context_baseline_seq": (
                    body["freshness"]["context_baseline_seq"]
                ),
                "latest_seq": 5,
                "latest_envelope_id": "held-exact",
            }
        return {
            "state": "sent",
            "envelope_id": "explicit-retry",
            "seq": 7,
            "replay": False,
            "missing_devices": [],
            "freshness": {
                **body["freshness"],
                "latest_seq_before_send": body["freshness"]["seen_seq"],
            },
        }

    http.post_unsigned = post_unsigned
    held_task = asyncio.create_task(coordinator.send(
        SemanticSendRequest(destination="ch_1", text="held draft")
    ))
    while not calls:
        await asyncio.sleep(0)
    await client._dispatch_bridge_frame({
        **base, "seq": 4, "envelope_id": "unrelated",
    })
    await client._dispatch_bridge_frame({
        **base, "envelope_id": "legacy",
    })
    await client._dispatch_bridge_frame({
        **base, "seq": 6, "envelope_id": "wrong-envelope",
    })
    await asyncio.sleep(0)
    assert not held_task.done()
    assert len(calls) == 1
    assert await client.store.get_message_by_envelope("held-exact") is None
    await client._dispatch_bridge_frame({
        **base, "seq": 5, "envelope_id": "held-exact",
    })
    held = await held_task
    assert held["state"] == "held"
    assert held["synchronized"] is True
    assert len(calls) == 1
    exact = await client.store.get_message_by_envelope("held-exact")
    assert exact is not None and exact.server_seq == 5

    staged = await coordinator.send(SemanticSendRequest(
        destination="ch_1", text="too early", send_anyway=True,
    ))
    assert staged["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    read = await runtime.stage_model_visible_read(
        space_id="sp_1",
        channel_id="ch_1",
        through_seq=5,
        through_envelope_id="held-exact",
        tool_name="get_channel_history",
        tool_arguments={"channel": "ch_1"},
    )
    assert read["state"] == "admitted"
    # The selective history read cannot leapfrog the locally-known pending
    # seq=4 receipt; the safe channel prefix remains the admitted seq=2.
    assert await boundary.get_active_turn_through_seq("sp_1", "ch_1") == 2
    sent = await coordinator.send(SemanticSendRequest(
        destination="ch_1", text="explicit retry", send_anyway=True,
    ))
    # The pending seq=4 receipt also keeps reconsideration ineligible.
    assert sent["state"] == "failed"
    assert len(calls) == 1
    assert calls[-1][1]["freshness"] == {
        "context_baseline_seq": 2,
        "seen_seq": 2,
        "mode": "require_current",
    }
    await _unused_store.close()
    await client.store.close()

from .test_puffo_core_tools import _setup_keyless
from .test_send_coordinator import Freshness, coordinator_fixture


class Recovery:
    def __init__(self, rows=None, available=True):
        self.rows = rows or []
        self.available = available
        self.waits = []
        self.queries = []

    async def wait_for_held_delivery(self, space_id, channel_id, seq, envelope_id):
        self.waits.append((space_id, channel_id, seq, envelope_id))
        return self.available

    async def query_held_messages(
        self, space_id, channel_id, seq, envelope_id, provider_session_id,
    ):
        self.queries.append((space_id, channel_id, seq, envelope_id, provider_session_id))
        return self.rows


def exact_row(
    *, seq=5, envelope_id="msg_latest", session_id="session-a",
):
    return {
        "envelope_id": envelope_id,
        "server_seq": seq,
        "latest_seq": seq,
        "latest_envelope_id": envelope_id,
        "provider_session_id": session_id,
    }


def bind_turn(coordinator, freshness, *, boundary=2):
    coordinator.provider_session_id = "session-a"
    freshness.active = SimpleNamespace(
        turn_id="turn-a", provider_session_id="session-a"
    )
    coordinator.held_recovery_source = Recovery(rows=[exact_row()])

    async def active_boundary(_space_id, _channel_id):
        return active_boundary.value

    active_boundary.value = boundary
    freshness.get_active_turn_through_seq = active_boundary
    return active_boundary


def held_response(body, *, latest_seq=5, latest_envelope_id="msg_latest"):
    return {
        "state": "held",
        "envelope_id": body["envelope"]["envelope_id"],
        "context_baseline_seq": body["freshness"]["context_baseline_seq"],
        "seen_seq": body["freshness"]["seen_seq"],
        "latest_seq": latest_seq,
        "latest_envelope_id": latest_envelope_id,
    }


@pytest.mark.asyncio
async def test_held_recovery_keeps_channel_wide_terminal_pair_for_thread_send():
    """A thread draft still needs the channel's actual latest pair."""
    coordinator, freshness, _http = await coordinator_fixture()
    coordinator.provider_session_id = "session-a"
    freshness.active = SimpleNamespace(
        turn_id="turn-a", provider_session_id="session-a"
    )
    recovery = Recovery(rows=[
        {
            **exact_row(envelope_id="other-thread-latest"),
            "thread_root_id": "other-root", "content": "other thread",
        },
        {
            "envelope_id": "requested-root", "server_seq": 4,
            "latest_seq": 5, "latest_envelope_id": "other-thread-latest",
            "provider_session_id": "session-a", "thread_root_id": "",
            "content": "requested thread",
        },
    ])
    coordinator.held_recovery_source = recovery
    key = ("session-a", "turn-a", "sp_1", "ch_a")
    coordinator._held_evidence[key] = _HeldEvidence(
        latest_seq=5,
        latest_envelope_id="other-thread-latest",
        thread_root_id="requested-root",
    )

    assert await coordinator._recover_held(
        "sp_1", "ch_a", 5, "other-thread-latest"
    )
    assert [row["envelope_id"] for row in coordinator._held_evidence[key].recovered_messages] == [
        "other-thread-latest", "requested-root",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "keyless"])
async def test_complete_exact_held_identity_and_one_shot_contract(transport):
    if transport == "native":
        coordinator, freshness, http = await coordinator_fixture(
            baseline=2, active=2
        )
        boundary = bind_turn(coordinator, freshness)
        calls = []
        paths = []

        async def post(path, body):
            paths.append(path)
            calls.append(body)
            if len(calls) == 1:
                return held_response(body)
            return {
                "state": "sent",
                "envelope_id": body["envelope"]["envelope_id"],
                "seq": 6,
                "replay": False,
                "missing_devices": [],
                "freshness": {
                    "mode": "send_anyway",
                    "context_baseline_seq": 2,
                    "seen_seq": 5,
                    "latest_seq_before_send": 5,
                },
            }

        http.post = post
        expected_path = CHANNEL_SEND_PATH
    else:
        cfg, http, store = _setup_keyless()
        await store.mark_channel_space("ch_abc", "sp_test")
        freshness = Freshness(baseline=2)
        freshness.active = SimpleNamespace(
            turn_id="turn-a", provider_session_id="session-a"
        )

        async def active_boundary(_space_id, _channel_id):
            return active_boundary.value

        active_boundary.value = 2
        freshness.get_active_turn_through_seq = active_boundary
        boundary = active_boundary
        coordinator = SendCoordinator(
            slug=cfg.slug,
            keystore=cfg.keystore,
            http_client=http,
            data_client=store,
            baseline_source=freshness,
            active_turn_source=freshness,
            held_recovery_source=Recovery(rows=[exact_row()]),
            provider_session_id="session-a",
        )
        calls = []
        paths = []

        async def post_unsigned(path, body=None):
            paths.append(path)
            calls.append(body)
            if len(calls) == 1:
                return {
                    "state": "held",
                    "seen_seq": body["freshness"]["seen_seq"],
                    "context_baseline_seq": (
                        body["freshness"]["context_baseline_seq"]
                    ),
                    "latest_seq": 5,
                    "latest_envelope_id": "msg_latest",
                }
            return {
                "state": "sent",
                "envelope_id": "keyless-sent",
                "seq": 6,
                "replay": False,
                "missing_devices": [],
                "freshness": {
                    "mode": "send_anyway",
                    "context_baseline_seq": 2,
                    "seen_seq": 5,
                    "latest_seq_before_send": 5,
                },
            }

        http.post_unsigned = post_unsigned
        expected_path = KEYLESS_CHANNEL_SEND_PATH

    destination = "ch_a" if transport == "native" else "ch_abc"
    held = await coordinator.send(SemanticSendRequest(
        destination=destination, text="draft"
    ))
    assert held["state"] == "held"
    assert held["latest_seq"] == 5
    assert held["latest_envelope_id"] == "msg_latest"
    assert held["synchronized"] is True

    staged_only = await coordinator.send(SemanticSendRequest(
        destination=destination, text="too soon", send_anyway=True
    ))
    assert staged_only["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1

    boundary.value = 5
    freshness.active = SimpleNamespace(
        turn_id="wrong-turn", provider_session_id="session-a"
    )
    wrong_turn = await coordinator.send(SemanticSendRequest(
        destination=destination, text="wrong turn", send_anyway=True
    ))
    assert wrong_turn["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    freshness.active = SimpleNamespace(
        turn_id="turn-a", provider_session_id="session-a"
    )

    assert len(coordinator._held_evidence) == 1
    held_evidence = next(iter(coordinator._held_evidence.values()))
    held_snapshot = (
        held_evidence.latest_seq,
        held_evidence.latest_envelope_id,
        held_evidence.synchronized,
    )
    held_evidence.latest_envelope_id = "wrong-envelope"
    held_evidence.synchronized = False
    wrong_pair = await coordinator.send(SemanticSendRequest(
        destination=destination,
        text="wrong held pair",
        send_anyway=True,
    ))
    assert wrong_pair["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    (
        held_evidence.latest_seq,
        held_evidence.latest_envelope_id,
        held_evidence.synchronized,
    ) = held_snapshot

    identity_mismatches = [
        (
            "wrong coordinator session",
            lambda: setattr(
                coordinator, "provider_session_id", "session-b"
            ),
        ),
        (
            "wrong active session",
            lambda: setattr(
                freshness,
                "active",
                SimpleNamespace(
                    turn_id="turn-a", provider_session_id="session-b"
                ),
            ),
        ),
        (
            "wrong turn",
            lambda: setattr(
                freshness,
                "active",
                SimpleNamespace(
                    turn_id="wrong-turn",
                    provider_session_id="session-a",
                ),
            ),
        ),
    ]
    for label, mutate in identity_mismatches:
        original_session = coordinator.provider_session_id
        original_active = freshness.active
        mutate()
        blocked = await coordinator.send(SemanticSendRequest(
            destination=destination,
            text=label,
            send_anyway=True,
        ))
        assert blocked["error_kind"] == "reconsideration_ineligible"
        assert len(calls) == 1
        coordinator.provider_session_id = original_session
        freshness.active = original_active

    sent = await coordinator.send(SemanticSendRequest(
        destination=destination, text="approved", send_anyway=True
    ))
    assert sent["state"] == "sent"
    assert len(calls) == 2
    assert calls[-1]["freshness"] == {
        "context_baseline_seq": 2,
        "seen_seq": 5,
        "mode": "send_anyway",
    }
    reused = await coordinator.send(SemanticSendRequest(
        destination=destination, text="reuse", send_anyway=True
    ))
    assert reused["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 2
    assert paths == [expected_path, expected_path]
    if transport == "keyless":
        await store.close()


@pytest.mark.asyncio
async def test_held_attempted_no_automatic_resend_and_preserves_watermarks():
    coordinator, freshness, http = await coordinator_fixture(baseline=2)

    async def post(path, body):
        http.calls.append(("POST", path, body))
        return held_response(body)

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    assert result["state"] == "held"
    assert result["synchronized"] is False
    reconsideration = result["reconsideration"]
    assert reconsideration["context_ready"] is False
    assert reconsideration["draft"] == "draft"
    assert reconsideration["based_on_through_seq"] == 2
    assert reconsideration["target"] == {
        "space_id": "sp_1", "channel_id": "ch_a", "thread_root_id": "",
    }
    assert len([c for c in http.calls if c[0] == "POST"]) == 1
    assert freshness.advances == []


@pytest.mark.asyncio
async def test_durable_recovery_same_provider_session_only():
    coordinator, _, http = await coordinator_fixture(baseline=2)
    recovery = Recovery(rows=[
        {
            "envelope_id": "msg_latest", "latest_seq": 5,
            "latest_envelope_id": "msg_latest", "provider_session_id": "session-a",
        },
        {
            "envelope_id": "wrong-session", "latest_seq": 5,
            "latest_envelope_id": "msg_latest", "provider_session_id": "session-b",
        },
    ])
    coordinator.held_recovery_source = recovery
    coordinator.provider_session_id = "session-a"

    async def post(path, body):
        return held_response(body)

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    assert "recovered_messages" not in result
    assert result["synchronized"] is False


@pytest.mark.asyncio
async def test_recovery_unavailable_exposes_nothing_and_does_not_advance():
    coordinator, freshness, http = await coordinator_fixture(baseline=2)
    coordinator.held_recovery_source = Recovery(
        rows=[{"provider_session_id": "session-a"}], available=False,
    )

    async def post(path, body):
        return held_response(body)

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    assert "recovered_messages" not in result
    assert freshness.advances == []


@pytest.mark.asyncio
async def test_unsynchronized_held_stays_blocked_after_admitted_boundary_advance():
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    boundary = bind_turn(coordinator, freshness)
    coordinator.held_recovery_source = Recovery(available=False)
    calls = []

    async def post(path, body):
        calls.append(body)
        if len(calls) == 1:
            return held_response(body)
        return {
            "state": "sent",
            "envelope_id": body["envelope"]["envelope_id"],
            "seq": 6,
            "replay": False,
            "missing_devices": [],
            "freshness": {
                "mode": "send_anyway",
                "context_baseline_seq": 2,
                "seen_seq": 2,
                "latest_seq_before_send": 5,
            },
        }

    http.post = post
    first = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    boundary.value = 5
    second = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="new draft", send_anyway=True,
    ))
    assert first["state"] == "held"
    assert second["state"] == "failed"
    assert second["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    assert freshness.advances == []
    assert all(
        path == CHANNEL_SEND_PATH
        for method, path, _body in http.calls
        if method == "POST"
    )


@pytest.mark.asyncio
async def test_same_session_turn_admitted_boundary_makes_override_eligible():
    coordinator, freshness, http = await coordinator_fixture(baseline=2, active=2)
    boundary = bind_turn(coordinator, freshness)
    coordinator.held_recovery_source = Recovery(rows=[exact_row()])
    calls = []

    async def post(path, body):
        calls.append(body)
        if len(calls) == 1:
            return held_response(body)
        return {
            "state": "sent",
            "envelope_id": body["envelope"]["envelope_id"],
            "seq": 6,
            "replay": False,
            "missing_devices": [],
            "freshness": {
                "mode": "send_anyway",
                "context_baseline_seq": 2,
                "seen_seq": 5,
                "latest_seq_before_send": 5,
            },
        }

    http.post = post
    first = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="draft")
    )
    assert first["state"] == "held"
    assert first["synchronized"] is True
    blocked = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="too soon", send_anyway=True
    ))
    assert blocked["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    boundary.value = 5
    result = await coordinator.send(
        SemanticSendRequest(
            destination="ch_a", text="reconsidered", send_anyway=True
        )
    )
    assert result["state"] == "sent"
    assert len(calls) == 2
    assert calls[-1]["freshness"]["mode"] == "send_anyway"
    reused = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="reuse", send_anyway=True
    ))
    assert reused["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_change", ["target", "session", "active_session", "turn"]
)
async def test_override_wrong_target_session_or_turn_is_blocked_without_request(
    identity_change,
):
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    coordinator.provider_session_id = "session-a"
    freshness.active = SimpleNamespace(
        turn_id="turn-a", provider_session_id="session-a"
    )
    coordinator.held_recovery_source = Recovery(rows=[exact_row()])

    async def boundary(_space_id, _channel_id):
        return boundary.value

    boundary.value = 2
    freshness.get_active_turn_through_seq = boundary
    calls = []

    async def post(_path, body):
        calls.append(body)
        return held_response(body)

    http.post = post
    assert (await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="draft")
    ))["synchronized"] is True
    boundary.value = 5
    destination = "ch_a"
    if identity_change == "target":
        destination = "ch_b"
    elif identity_change == "session":
        coordinator.provider_session_id = "session-b"
    elif identity_change == "active_session":
        freshness.active = SimpleNamespace(
            turn_id="turn-a", provider_session_id="session-b"
        )
    else:
        freshness.active = SimpleNamespace(
            turn_id="turn-b", provider_session_id="session-a"
        )
    result = await coordinator.send(SemanticSendRequest(
        destination=destination,
        text="override",
        send_anyway=True,
    ))
    assert result["state"] == "failed"
    assert result["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_staged_but_unadmitted_held_context_cannot_authorize_override():
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    coordinator.provider_session_id = "session-a"
    freshness.active = SimpleNamespace(
        turn_id="turn-a", provider_session_id="session-a"
    )
    coordinator.held_recovery_source = Recovery(rows=[exact_row()])

    async def boundary(_space_id, _channel_id):
        return 2

    freshness.get_active_turn_through_seq = boundary
    calls = []

    async def post(_path, body):
        calls.append(body)
        return held_response(body)

    http.post = post
    assert (await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="draft")
    ))["synchronized"] is True
    # Catch-up may have made the row durable/readable, but the admitted
    # active boundary intentionally remains below the held watermark.
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="override", send_anyway=True
    ))
    assert result["state"] == "failed"
    assert result["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recovery",
    [
        Recovery(available=False),
        Recovery(rows=[]),
        Recovery(rows=[{
            "latest_seq": 5,
            "latest_envelope_id": "msg_latest",
            "provider_session_id": "session-a",
        }]),
        Recovery(rows=[exact_row(seq=4)]),
        Recovery(rows=[exact_row(envelope_id="wrong")]),
        Recovery(rows=[exact_row(session_id="session-b")]),
    ],
)
async def test_incomplete_or_mismatched_recovery_fails_closed(recovery):
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    boundary = bind_turn(coordinator, freshness)
    coordinator.held_recovery_source = recovery
    calls = []

    async def post(_path, body):
        calls.append(body)
        return held_response(body)

    http.post = post
    held = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="draft")
    )
    assert held["state"] == "held"
    assert held["synchronized"] is False
    boundary.value = 5
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="override", send_anyway=True
    ))
    assert result["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("late"), RuntimeError("boom")])
async def test_recovery_timeout_or_exception_fails_closed(failure):
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    boundary = bind_turn(coordinator, freshness)

    class FailingRecovery:
        async def wait_for_held_delivery(self, *_args):
            raise failure

    coordinator.held_recovery_source = FailingRecovery()
    calls = []

    async def post(_path, body):
        calls.append(body)
        return held_response(body)

    http.post = post
    await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="draft"
    ))
    boundary.value = 5
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="override", send_anyway=True
    ))
    assert result["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_later_exact_recovery_can_unlock_only_current_held_record():
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    boundary = bind_turn(coordinator, freshness)
    recovery = Recovery(available=False)
    coordinator.held_recovery_source = recovery
    calls = []

    async def post(_path, body):
        calls.append(body)
        if len(calls) == 1:
            return held_response(body)
        return {
            "state": "sent",
            "envelope_id": body["envelope"]["envelope_id"],
            "seq": 6,
            "replay": False,
            "missing_devices": [],
            "freshness": {
                "mode": "send_anyway",
                "context_baseline_seq": 2,
                "seen_seq": 5,
                "latest_seq_before_send": 5,
            },
        }

    http.post = post
    first = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="draft"
    ))
    assert first["synchronized"] is False
    boundary.value = 5
    blocked = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="still blocked", send_anyway=True
    ))
    assert blocked["error_kind"] == "reconsideration_ineligible"
    assert len(calls) == 1
    recovery.available = True
    recovery.rows = [exact_row()]
    sent = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="now eligible", send_anyway=True
    ))
    assert sent["state"] == "sent"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda response, body: {
            **response,
            "context_baseline_seq": body["freshness"]["context_baseline_seq"] + 1,
        },
        lambda response, body: {
            **response,
            "latest_seq": body["freshness"]["seen_seq"],
        },
    ],
)
async def test_invalid_held_baseline_or_nonadvancing_watermark_fails_closed(
    mutate,
):
    coordinator, _, http = await coordinator_fixture(baseline=2)
    calls = []

    async def post(_path, body):
        calls.append(body)
        return mutate(held_response(body), body)

    http.post = post
    result = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="draft")
    )
    assert result["state"] == "failed"
    assert result["error_kind"] == "protocol"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_held_attachment_is_uploaded_once_and_not_resent(tmp_path):
    coordinator, _, http = await coordinator_fixture(baseline=2)
    coordinator.workspace = str(tmp_path)
    (tmp_path / "report.txt").write_text("report", encoding="utf-8")
    uploads = []
    posts = []

    async def upload(path, body):
        uploads.append((path, body))
        return {"blob_id": "blob_held"}

    async def post(path, body):
        posts.append((path, body))
        return held_response(body)

    http.post_bytes = upload
    http.post = post
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a",
        attachment_paths=("report.txt",),
        caption="attached",
    ))
    assert result["state"] == "held"
    assert len(uploads) == 1
    assert len(posts) == 1
    assert not any(path in ("/messages", "/blobs/delete") for path, _ in posts)


@pytest.mark.asyncio
async def test_stale_recovery_completion_cannot_synchronize_superseded_pair():
    coordinator, freshness, http = await coordinator_fixture(
        baseline=2, active=2
    )
    boundary = bind_turn(coordinator, freshness)
    started = asyncio.Event()
    release = asyncio.Event()

    class CoalescingRecovery:
        def __init__(self):
            self.waits = []
            self.queries = []

        async def wait_for_held_delivery(self, space_id, channel_id, seq, envelope_id):
            self.waits.append((seq, envelope_id))
            if seq == 5:
                started.set()
                await release.wait()
                return True
            return False

        async def query_held_messages(
            self, space_id, channel_id, seq, envelope_id, provider_session_id,
        ):
            self.queries.append((seq, envelope_id))
            return [exact_row(
                seq=seq, envelope_id=envelope_id,
                session_id=provider_session_id,
            )]

    recovery = CoalescingRecovery()
    coordinator.held_recovery_source = recovery
    attempts = 0

    async def post(path, body):
        nonlocal attempts
        attempts += 1
        return held_response(
            body, latest_seq=4 + attempts,
            latest_envelope_id=f"msg_latest_{attempts}",
        )

    http.post = post
    first_task = asyncio.create_task(coordinator.send(
        SemanticSendRequest(destination="ch_a", text="one")
    ))
    await started.wait()
    second = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="two")
    )
    release.set()
    first = await first_task
    assert recovery.queries == []
    assert first["synchronized"] is False
    assert second["synchronized"] is False
    boundary.value = 6
    blocked = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="override", send_anyway=True
    ))
    assert blocked["error_kind"] == "reconsideration_ineligible"
    assert attempts == 2
