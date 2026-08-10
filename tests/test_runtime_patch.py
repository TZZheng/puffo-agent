"""PATCH /v1/agents/{id}/runtime — field contract and writer normalization.

``allowed_tools`` / ``max_turns`` are part of the endpoint's contract; a
payload carrying them must apply and echo them rather than return 200 with
the fields silently dropped.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from _bridge_support import (
    isolated_home, make_user, pair_request_body, signed_headers,
    write_test_agent,
)
from puffo_agent.crypto.encoding import base64url_encode
from puffo_agent.portal.api.runtime_patch import apply_runtime_patch, runtime_response
from puffo_agent.portal.api.server import build_app
from puffo_agent.portal.state import AgentConfig, DaemonConfig, RuntimeConfig

pytestmark = pytest.mark.asyncio

_HOST = {"Host": "127.0.0.1:63387"}


@pytest_asyncio.fixture
async def owner_client():
    """Paired client owning ``owned-bot``, so runtime PATCH is authorized."""
    user = make_user()
    home = isolated_home()
    write_test_agent(
        home, "owned-bot",
        owner_root_pubkey=base64url_encode(user.root_key.public_key_bytes()),
    )
    app = build_app(DaemonConfig().bridge)
    async with TestClient(TestServer(app)) as c:
        body = pair_request_body(user)
        h = signed_headers(user, "POST", "/v1/pair", body); h.update(_HOST)
        assert (await c.post("/v1/pair", data=body, headers=h)).status == 200
        yield c, user


async def _patch(client, user, payload: dict):
    path = "/v1/agents/owned-bot/runtime"
    body = json.dumps(payload).encode("utf-8")
    h = signed_headers(user, "PATCH", path, body); h.update(_HOST)
    return await client.patch(path, data=body, headers=h)


async def test_patch_applies_and_echoes_allowed_tools_and_max_turns(owner_client):
    client, user = owner_client
    r = await _patch(client, user, {"allowed_tools": ["Read"], "max_turns": 5})

    assert r.status == 200, await r.text()
    runtime = (await r.json())["runtime"]
    assert runtime["allowed_tools"] == ["Read"]
    assert runtime["max_turns"] == 5
    saved = AgentConfig.load("owned-bot").runtime
    assert saved.allowed_tools == ["Read"]
    assert saved.max_turns == 5


async def test_patch_rejects_non_list_allowed_tools(owner_client):
    client, user = owner_client
    r = await _patch(client, user, {"allowed_tools": "Read"})

    assert r.status == 400
    assert "allowed_tools must be a list of strings" in await r.text()
    assert AgentConfig.load("owned-bot").runtime.allowed_tools != "Read"


async def test_patch_rejects_non_integer_max_turns(owner_client):
    client, user = owner_client
    before = AgentConfig.load("owned-bot").runtime.max_turns
    r = await _patch(client, user, {"max_turns": "x"})

    assert r.status == 400
    assert "max_turns must be an integer" in await r.text()
    assert AgentConfig.load("owned-bot").runtime.max_turns == before


def _codex_runtime(level: str = "minimal") -> RuntimeConfig:
    return RuntimeConfig(
        kind="cli-local", provider="openai", harness="codex",
        model="gpt-5.5", inference_level=level,
    )


async def test_apply_patch_coerces_tool_names_to_strings():
    runtime = _codex_runtime()
    assert apply_runtime_patch(runtime, {"allowed_tools": ["Read", 7]}) is None
    assert runtime.allowed_tools == ["Read", "7"]


async def test_runtime_response_echoes_worker_limits():
    runtime = _codex_runtime()
    runtime.allowed_tools = ["Bash(git status)"]
    runtime.max_turns = 23
    body = runtime_response(runtime)
    assert body["allowed_tools"] == ["Bash(git status)"]
    assert body["max_turns"] == 23
