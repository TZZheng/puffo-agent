"""``monid_prepare`` and ``monid_spend`` forward to the server gateway and
format its result. The tools hold no key and do no budgeting themselves — the
server enforces the cap (PR-1 reserve) and picks the capability; the tools'
checks are UX pre-guards and their errors are the server's own mapped messages.
``monid_prepare`` is free (discover + inspect); only ``monid_spend`` charges.
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


async def _call(mcp, name, args):
    result = await mcp.call_tool(name, args)
    if isinstance(result, tuple):
        result = result[0]
    return "".join(getattr(item, "text", str(item)) for item in result)


# ------------------------------ monid_prepare ------------------------------


@pytest.mark.asyncio
async def test_prepare_forwards_query_and_returns_descriptor():
    http = _FakeHttp(
        response={
            "provider": "indeed",
            "endpoint": "/get_company_profile",
            "category": "company_profile",
            "price": {"price_type": "PER_CALL", "unit_price_micro": 10000},
            "input": {
                "queryParams": {
                    "type": "object",
                    "required": ["company"],
                    "properties": {"company": {"type": "string"}},
                }
            },
            "description": "Company profile.",
        }
    )
    mcp = _tools(http)
    text = await _call(mcp, "monid_prepare", {"query": "company profile", "limit": 3})
    # The whole descriptor is handed back as JSON so the model can build input.
    assert '"queryParams"' in text
    assert '"/get_company_profile"' in text
    assert http.calls[-1][1] == "/v2/monid/prepare"
    body = http.calls[-1][2]
    assert body["query"] == "company profile"
    assert body["limit"] == 3


@pytest.mark.asyncio
async def test_prepare_rejects_empty_query_without_calling_server():
    http = _FakeHttp()
    mcp = _tools(http)
    with pytest.raises(Exception):
        await _call(mcp, "monid_prepare", {"query": "   "})
    assert not [c for c in http.calls if c[1] == "/v2/monid/prepare"]


@pytest.mark.asyncio
async def test_prepare_surfaces_server_error_message():
    http = _FakeHttp(
        post_error=HttpError(
            400,
            json.dumps(
                {
                    "error": "BAD_REQUEST",
                    "message": "no allowlisted read-only capability matched the query",
                }
            ),
        )
    )
    mcp = _tools(http)
    with pytest.raises(Exception) as excinfo:
        await _call(mcp, "monid_prepare", {"query": "buy a car"})
    assert "no allowlisted read-only capability" in str(excinfo.value)


# ------------------------------- monid_spend -------------------------------


@pytest.mark.asyncio
async def test_spend_forwards_and_formats_result():
    http = _FakeHttp(
        response={
            "ledger_id": "led_1",
            "provider": "indeed",
            "endpoint": "/get_company_profile",
            "cost_micro": 3000,
            "provider_http_status": 200,
            "output": {"items": ["a", "b"]},
        }
    )
    mcp = _tools(http)
    text = await _call(
        mcp,
        "monid_spend",
        {
            "provider": "indeed",
            "endpoint": "/get_company_profile",
            "input": {"queryParams": {"company": "Google"}},
            "max_cost_micro": 10000,
        },
    )
    assert "cost 3000 micro-dollars" in text
    assert "provider status 200" in text
    assert "items" in text
    assert http.calls[-1][0] == "POST"
    assert http.calls[-1][1] == "/v2/monid/spend"
    body = http.calls[-1][2]
    assert body["provider"] == "indeed"
    assert body["endpoint"] == "/get_company_profile"
    assert body["input"] == {"queryParams": {"company": "Google"}}
    assert body["max_cost_micro"] == 10000
    assert "query" not in body  # reshaped: no free-text query on the paid path


@pytest.mark.asyncio
async def test_spend_reports_pending_reconcile():
    http = _FakeHttp(
        response={
            "error": "PENDING_RECONCILE",
            "message": "still resolving",
            "ledger_id": "led_pending",
        }
    )
    mcp = _tools(http)
    text = await _call(
        mcp,
        "monid_spend",
        {
            "provider": "indeed",
            "endpoint": "/get_company_profile",
            "input": {"queryParams": {"company": "x"}},
            "max_cost_micro": 5000,
        },
    )
    assert "still resolving upstream" in text
    assert "led_pending" in text


@pytest.mark.asyncio
async def test_spend_rejects_bad_input_without_calling_server():
    http = _FakeHttp()
    mcp = _tools(http)
    # Non-positive ceiling.
    with pytest.raises(Exception):
        await _call(
            mcp,
            "monid_spend",
            {
                "provider": "indeed",
                "endpoint": "/get_company_profile",
                "input": {"queryParams": {"company": "x"}},
                "max_cost_micro": 0,
            },
        )
    # Missing provider/endpoint.
    with pytest.raises(Exception):
        await _call(
            mcp,
            "monid_spend",
            {
                "provider": "  ",
                "endpoint": "",
                "input": {"queryParams": {"company": "x"}},
                "max_cost_micro": 5000,
            },
        )
    assert not [c for c in http.calls if c[1] == "/v2/monid/spend"]


@pytest.mark.asyncio
async def test_spend_surfaces_server_error_message():
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
        await _call(
            mcp,
            "monid_spend",
            {
                "provider": "indeed",
                "endpoint": "/get_company_profile",
                "input": {"queryParams": {"company": "x"}},
                "max_cost_micro": 5000,
            },
        )
    assert "not enabled for this agent" in str(excinfo.value)


@pytest.mark.asyncio
async def test_spend_surfaces_input_schema_so_the_model_can_retry():
    # When the server rejects the input before spending, it returns the
    # capability's own schema (the whole run-input descriptor); the tool hands it
    # back so the model can rebuild `input` and retry — including a queryParams
    # capability whose schema is NOT under `body`.
    http = _FakeHttp(
        post_error=HttpError(
            400,
            json.dumps(
                {
                    "error": "INVALID_INPUT",
                    "message": (
                        "input.queryParams is required: Monid wraps a run's input "
                        'as {"queryParams": { … }}'
                    ),
                    "input_schema": {
                        "queryParams": {
                            "type": "object",
                            "required": ["company"],
                            "properties": {
                                "company": {
                                    "type": "string",
                                    "description": "company slug e.g. Google",
                                }
                            },
                        },
                    },
                }
            ),
        )
    )
    mcp = _tools(http)
    with pytest.raises(Exception) as excinfo:
        await _call(
            mcp,
            "monid_spend",
            {
                "provider": "indeed",
                "endpoint": "/get_company_profile",
                "input": {"company": "https://indeed.com/cmp/Google"},
                "max_cost_micro": 5000,
            },
        )
    msg = str(excinfo.value)
    assert "input.queryParams is required" in msg
    assert "Rebuild `input`" in msg  # the schema is included for a retry
    assert '"required"' in msg


@pytest.mark.asyncio
async def test_tools_not_registered_for_keyless_agents():
    # A keyless bridge agent cannot reach the subkey-gated routes, so neither
    # tool is exposed for it — not registered, no error path.
    http = _FakeHttp()
    http.keyless = True
    mcp = _tools(http)
    tool_names = {t.name for t in await mcp.list_tools()}
    assert "monid_spend" not in tool_names
    assert "monid_prepare" not in tool_names

    # Native agents DO get both.
    native = _tools(_FakeHttp())
    native_names = {t.name for t in await native.list_tools()}
    assert "monid_spend" in native_names
    assert "monid_prepare" in native_names
