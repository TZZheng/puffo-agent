from __future__ import annotations

import asyncio

import pytest

from puffo_agent.agent.send_coordinator import CHANNEL_SEND_PATH, SemanticSendRequest

from .test_send_coordinator import coordinator_fixture


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


def held_response(body, *, latest_seq=5, latest_envelope_id="msg_latest"):
    return {
        "state": "held",
        "envelope_id": body["envelope"]["envelope_id"],
        "seen_seq": body["freshness"]["seen_seq"],
        "latest_seq": latest_seq,
        "latest_envelope_id": latest_envelope_id,
    }


@pytest.mark.asyncio
async def test_held_attempted_no_automatic_resend_and_preserves_watermarks():
    coordinator, freshness, http = await coordinator_fixture(baseline=2)

    async def post(path, body):
        http.calls.append(("POST", path, body))
        return held_response(body)

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    assert result == {
        "state": "held",
        "attempted": True,
        "envelope_id": result["envelope_id"],
        "context_baseline_seq": 2,
        "seen_seq": 2,
        "latest_seq": 5,
        "latest_envelope_id": "msg_latest",
        "note": result["note"],
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
            "content": "new",
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
    assert [row["envelope_id"] for row in result["recovered_messages"]] == ["msg_latest"]


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
async def test_later_explicit_send_anyway_is_new_checked_request():
    coordinator, freshness, http = await coordinator_fixture(baseline=2)
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
                "seen_seq": 2,
                "latest_seq_before_send": 5,
            },
        }

    http.post = post
    first = await coordinator.send(SemanticSendRequest(destination="ch_a", text="draft"))
    second = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="new draft", send_anyway=True,
    ))
    assert first["state"] == "held"
    assert second["state"] == "sent"
    assert calls[0]["envelope"]["envelope_id"] != calls[1]["envelope"]["envelope_id"]
    assert calls[1]["freshness"]["mode"] == "send_anyway"
    assert freshness.advances == []
    assert all(
        path == CHANNEL_SEND_PATH
        for method, path, _body in http.calls
        if method == "POST"
    )


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
async def test_concurrent_held_watermarks_coalesce_to_newest_pair():
    coordinator, _, http = await coordinator_fixture(baseline=2)
    coordinator.provider_session_id = "session-a"
    release = asyncio.Event()

    class CoalescingRecovery:
        def __init__(self):
            self.waits = []
            self.queries = []

        async def wait_for_held_delivery(self, space_id, channel_id, seq, envelope_id):
            self.waits.append((seq, envelope_id))
            if len(self.waits) == 2:
                release.set()
            await release.wait()
            return True

        async def query_held_messages(
            self, space_id, channel_id, seq, envelope_id, provider_session_id,
        ):
            self.queries.append((seq, envelope_id))
            return [{
                "latest_seq": seq,
                "latest_envelope_id": envelope_id,
                "provider_session_id": provider_session_id,
                "envelope_id": envelope_id,
            }]

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
    first, second = await asyncio.gather(
        coordinator.send(SemanticSendRequest(destination="ch_a", text="one")),
        coordinator.send(SemanticSendRequest(destination="ch_a", text="two")),
    )
    assert recovery.queries == [(6, "msg_latest_2"), (6, "msg_latest_2")]
    assert first["recovered_messages"][0]["envelope_id"] == "msg_latest_2"
    assert second["recovered_messages"][0]["envelope_id"] == "msg_latest_2"
