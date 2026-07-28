from __future__ import annotations

import copy

import pytest

from puffo_agent.agent.send_coordinator import (
    CHANNEL_SEND_PATH,
    SemanticSendRequest,
    SendCoordinator,
)
from puffo_agent.crypto.encoding import base64url_encode
from puffo_agent.crypto.http_client import HttpError
from puffo_agent.crypto.primitives import KemKeyPair
from puffo_agent.mcp.data_client import DataNotFound

from .test_puffo_core_tools import _setup


class Freshness:
    def __init__(self, baseline=0, active=None):
        self.baseline = baseline
        self.active = active
        self.lookups = []
        self.advances = []

    async def get_context_baseline_seq(self, space_id, channel_id):
        self.lookups.append(("baseline", space_id, channel_id))
        return self.baseline

    async def get_active_turn_through_seq(self, space_id, channel_id):
        self.lookups.append(("active", space_id, channel_id))
        return self.active

    async def advance_active_turn_through_seq(self, space_id, channel_id, seq):
        self.advances.append((space_id, channel_id, seq))
        self.active = seq


async def coordinator_fixture(*, baseline=0, active=None):
    cfg, http, _store = _setup()

    class Data:
        async def lookup_channel_space(self, channel_id):
            return {"ch_a": "sp_1", "ch_b": "sp_1"}.get(channel_id)

        async def get_message_by_envelope(self, _envelope_id):
            raise DataNotFound("not found")

        async def get_send_encryption(self, _slug, _root):
            return True

    data = Data()
    device = KemKeyPair.generate()
    for channel in ("ch_a", "ch_b"):
        http.responses[f"/spaces/sp_1/channels/{channel}/members"] = {
            "members": [{"slug": "alice-1"}],
        }
    http.responses["/certs/sync?slugs=alice-1"] = {
        "entries": [{
            "seq": 1,
            "kind": "device_cert",
            "cert": {
                "device_id": "dev_a",
                "kem_public_key": base64url_encode(device.public_key_bytes()),
            },
        }],
        "has_more": False,
    }
    freshness = Freshness(baseline, active)
    coordinator = SendCoordinator(
        slug=cfg.slug,
        keystore=cfg.keystore,
        http_client=http,
        data_client=data,
        baseline_source=freshness,
        active_turn_source=freshness,
    )
    return coordinator, freshness, http


@pytest.mark.asyncio
async def test_exact_rust_request_shape_and_send_anyway():
    coordinator, _, http = await coordinator_fixture(baseline=4, active=6)
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="hello", visibility_level="human",
        send_anyway=True,
    ))
    assert result["state"] == "sent"
    assert result["attempted"] is True
    path, body = [(path, body) for method, path, body in http.calls if method == "POST"][-1]
    assert path == CHANNEL_SEND_PATH
    assert set(body) == {"envelope", "freshness"}
    assert body["freshness"] == {
        "context_baseline_seq": 4,
        "seen_seq": 6,
        "mode": "send_anyway",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("baseline", [-1, "2", True])
async def test_baseline_invalid_fails_without_send(baseline):
    coordinator, _, http = await coordinator_fixture(baseline=baseline)
    result = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="hello"),
    )
    assert result["state"] == "failed"
    assert result["attempted"] is True
    assert not [call for call in http.calls if call[0].startswith("POST")]


@pytest.mark.asyncio
async def test_missing_baseline_defaults_to_zero_and_active_turn_advances_seen():
    coordinator, _, http = await coordinator_fixture(baseline=None, active=7)
    result = await coordinator.send(
        SemanticSendRequest(destination="ch_a", text="hello"),
    )
    assert result["state"] == "sent"
    body = [
        body for method, path, body in http.calls
        if method == "POST" and path == CHANNEL_SEND_PATH
    ][-1]
    assert body["freshness"] == {
        "context_baseline_seq": 0,
        "seen_seq": 7,
        "mode": "require_current",
    }


@pytest.mark.asyncio
async def test_boundary_multiple_shared_channel_and_independent_channels():
    coordinator, freshness, http = await coordinator_fixture(baseline=0)
    first = await coordinator.send(SemanticSendRequest(destination="ch_a", text="one"))
    second = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="two", root_id="unknown",
    ))
    assert first["state"] == second["state"] == "sent"
    bodies = [
        body for method, path, body in http.calls
        if method == "POST" and path == CHANNEL_SEND_PATH
    ]
    assert bodies[0]["freshness"]["seen_seq"] == 0
    assert bodies[1]["freshness"]["seen_seq"] == first["seq"]
    freshness.active = None
    await coordinator.send(SemanticSendRequest(destination="ch_b", text="other"))
    channel_bodies = [
        body for method, path, body in http.calls
        if method == "POST" and path == CHANNEL_SEND_PATH
    ]
    assert channel_bodies[-1]["freshness"]["seen_seq"] == 0
    assert ("sp_1", "ch_a", first["seq"]) in freshness.advances


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {},
    {"state": "wat"},
    {"state": "sent", "envelope_id": "wrong", "seq": 1, "replay": False},
    {"state": "sent", "seq": "1", "replay": False},
    {"state": "held", "seen_seq": 0},
])
async def test_response_validation_failed_attempted(bad):
    coordinator, _, http = await coordinator_fixture()
    http.responses[CHANNEL_SEND_PATH] = bad
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="x"))
    assert result["state"] == "failed"
    assert result["attempted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 413, 429, 500, 503])
async def test_status_matrix_and_deployment_error(status):
    coordinator, _, http = await coordinator_fixture()

    async def post(path, body):
        http.calls.append(("POST", path, body))
        raise HttpError(status, '{"error":"NOPE","message":"not accepted"}')

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="x"))
    assert result["state"] == "failed"
    assert result["status"] == status
    assert result["error_kind"] == ("deployment" if status in (404, 405) else "http")
    assert all(call[1] != "/messages" for call in http.calls if call[0] == "POST")


@pytest.mark.asyncio
async def test_lost_response_replay_reuses_exact_body():
    coordinator, _, http = await coordinator_fixture(baseline=8)
    bodies = []

    async def post(path, body):
        bodies.append(copy.deepcopy(body))
        if len(bodies) == 1:
            raise TimeoutError("response lost")
        return {
            "state": "sent",
            "envelope_id": body["envelope"]["envelope_id"],
            "seq": 11,
            "replay": True,
            "missing_devices": [],
            "freshness": {
                "mode": "require_current",
                "seen_seq": 8,
                "latest_seq_before_send": 8,
            },
        }

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="x"))
    assert result["state"] == "sent" and result["replay"] is True
    assert bodies[0] == bodies[1]


@pytest.mark.asyncio
async def test_dm_route_has_no_freshness():
    coordinator, _, http = await coordinator_fixture()
    device = KemKeyPair.generate()
    http.responses["/certs/sync?slugs=agent-0001,alice-1"] = {
        "entries": [{
            "seq": 1, "kind": "device_cert",
            "cert": {
                "device_id": "dev_dm",
                "kem_public_key": base64url_encode(device.public_key_bytes()),
            },
        }],
        "has_more": False,
    }
    result = await coordinator.send(SemanticSendRequest(
        destination="@alice-1", text="hi", visibility_level="human",
    ))
    assert result["state"] == "sent"
    path, body = [(p, b) for m, p, b in http.calls if m == "POST"][-1]
    assert path == "/messages"
    assert "freshness" not in body


@pytest.mark.asyncio
async def test_plaintext_dm_route_has_no_freshness():
    coordinator, _, http = await coordinator_fixture()

    async def plaintext(_slug, _root):
        return False

    coordinator.data_client.get_send_encryption = plaintext
    result = await coordinator.send(SemanticSendRequest(
        destination="@alice-1", text="hi", visibility_level="human",
    ))
    assert result["state"] == "sent"
    path, body = [(p, b) for m, p, b in http.calls if m == "POST"][-1]
    assert path == "/v2/messages/plaintext"
    assert "freshness" not in body


@pytest.mark.asyncio
async def test_plaintext_channel_no_downgrade():
    coordinator, _, http = await coordinator_fixture()

    async def plaintext(_slug, _root):
        return False

    coordinator.data_client.get_send_encryption = plaintext
    result = await coordinator.send(SemanticSendRequest(
        destination="ch_a", text="must not downgrade",
    ))
    assert result["state"] == "failed"
    assert result["error_kind"] == "encryption_required"
    assert not [call for call in http.calls if call[0].startswith("POST")]


@pytest.mark.asyncio
async def test_non_json_nominal_success_is_protocol_failure():
    coordinator, _, http = await coordinator_fixture()

    async def post(_path, _body):
        raise HttpError(200, "non-JSON body on 200 response: <html>")

    http.post = post
    result = await coordinator.send(SemanticSendRequest(destination="ch_a", text="x"))
    assert result["state"] == "failed"
    assert result["error_kind"] == "protocol"
