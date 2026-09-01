"""``monid_spend`` forwards to the server spend gateway and formats its
result. The tool holds no key and does no budgeting itself — the server
enforces the cap (PR-1 reserve); the tool's checks are UX pre-guards and
its errors are the server's own mapped messages.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from puffo_agent.crypto.http_client import HttpError
from puffo_agent.mcp.core_monid_tools import register_monid_tools


class _FakeHttp:
    def __init__(self, *, response=None, post_error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.keyless = False
        self._response = response if response is not None else {"ok": True}
        self._post_error = post_error

    async def post(self, path, body=None):
        self.calls.append(("POST", path, body))
        if self._post_error is not None:
            raise self._post_error
        return self._response


def _tools(http):
    cfg = SimpleNamespace(http_client=http, keyless=http.keyless)
    mcp = FastMCP("test")
    register_monid_tools(mcp, cfg)
    return mcp


async def _call(mcp, args):
    result = await mcp.call_tool("monid_spend", args)
    if isinstance(result, tuple):
        result = result[0]
    return "".join(getattr(item, "text", str(item)) for item in result)


@pytest.mark.asyncio
async def test_forwards_and_formats_result():
    http = _FakeHttp(
        response={
            "ledger_id": "led_1",
            "provider": "amazon",
            "endpoint": "product_search",
            "cost_micro": 3000,
            "provider_http_status": 200,
            "output": {"items": ["a", "b"]},
        }
    )
    mcp = _tools(http)
    text = await _call(
        mcp,
        {"query": "find widgets", "input": {"q": "widgets"}, "max_cost_micro": 10000},
    )
    assert "cost 3000 micro-dollars" in text
    assert "provider status 200" in text
    assert "items" in text
    assert http.calls[-1][0] == "POST"
    assert http.calls[-1][1] == "/v2/monid/spend"
    body = http.calls[-1][2]
    assert body["query"] == "find widgets"
    assert body["input"] == {"q": "widgets"}
    assert body["max_cost_micro"] == 10000


@pytest.mark.asyncio
async def test_reports_pending_reconcile():
    http = _FakeHttp(
        response={
            "error": "PENDING_RECONCILE",
            "message": "still resolving",
            "ledger_id": "led_pending",
        }
    )
    mcp = _tools(http)
    text = await _call(
        mcp, {"query": "x", "input": {"q": "x"}, "max_cost_micro": 5000}
    )
    assert "still resolving upstream" in text
    assert "led_pending" in text


@pytest.mark.asyncio
async def test_rejects_bad_input_without_calling_server():
    http = _FakeHttp()
    mcp = _tools(http)
    with pytest.raises(Exception):
        await _call(mcp, {"query": "x", "input": {"q": "x"}, "max_cost_micro": 0})
    with pytest.raises(Exception):
        await _call(mcp, {"query": "   ", "input": {"q": "x"}, "max_cost_micro": 5000})
    assert not [c for c in http.calls if c[1] == "/v2/monid/spend"]


@pytest.mark.asyncio
async def test_surfaces_server_error_message():
    http = _FakeHttp(
        post_error=HttpError(
            403,
            json.dumps(
                {"error": "FORBIDDEN", "message": "monid is not enabled for this agent"}
            ),
        )
    )
    mcp = _tools(http)
    with pytest.raises(Exception) as excinfo:
        await _call(mcp, {"query": "x", "input": {"q": "x"}, "max_cost_micro": 5000})
    assert "not enabled for this agent" in str(excinfo.value)


@pytest.mark.asyncio
async def test_not_registered_for_keyless_agents():
    # A keyless bridge agent cannot reach the subkey-gated spend route, so the
    # tool is simply not exposed for it — not registered, no error path.
    http = _FakeHttp()
    http.keyless = True
    mcp = _tools(http)
    tool_names = {t.name for t in await mcp.list_tools()}
    assert "monid_spend" not in tool_names

    # Native agents DO get it.
    native = _tools(_FakeHttp())
    native_names = {t.name for t in await native.list_tools()}
    assert "monid_spend" in native_names
