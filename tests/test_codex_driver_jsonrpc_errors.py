"""Focused recovery classification for Codex app-server JSON-RPC errors."""

from __future__ import annotations

import asyncio
import json

import pytest

from puffo_agent.agent.core import AgentAPIError
from puffo_agent.agent.harness.codex_driver import (
    CodexAppServerDriver,
    _classify_jsonrpc_error,
)


@pytest.mark.parametrize(
    "error",
    [
        {"code": 401, "message": "access token has expired"},
        {"code": -32000, "message": "token_invalidated"},
    ],
)
def test_jsonrpc_auth_errors_request_operator_reauthentication(error):
    exc = _classify_jsonrpc_error(error)

    assert isinstance(exc, AgentAPIError)
    assert exc.is_auth is True
    assert "code=" in str(exc)


@pytest.mark.parametrize(
    "error",
    [
        {"code": 429, "message": "rate limit exceeded"},
        {"code": -32000, "data": {"status": 503}, "message": "unavailable"},
    ],
)
def test_jsonrpc_retryable_provider_errors_requeue(error):
    exc = _classify_jsonrpc_error(error)

    assert isinstance(exc, AgentAPIError)
    assert exc.is_auth is False
    assert "code=" in str(exc)


def test_jsonrpc_protocol_error_keeps_safe_diagnostic_without_retry():
    exc = _classify_jsonrpc_error(
        {
            "code": -32602,
            "message": "invalid params authorization=Bearer sk_secret_token_123456789",
        }
    )

    assert type(exc) is RuntimeError
    assert "code=-32602" in str(exc)
    assert "invalid params" in str(exc)
    assert "sk_secret_token_123456789" not in str(exc)
    assert "[REDACTED]" in str(exc)


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ('invalid params {"api_key":"unmasked-secret-value"}',
         "unmasked-secret-value"),
        ('invalid params {"apiKey":"camel-secret-value"}',
         "camel-secret-value"),
        ("invalid params accessToken='quoted access secret'",
         "quoted access secret"),
        ("invalid params Authorization: Bearer plain-bearer-secret",
         "plain-bearer-secret"),
    ],
)
def test_jsonrpc_diagnostics_redact_structured_credentials(message, secret):
    exc = _classify_jsonrpc_error({"code": -32602, "message": message})

    assert type(exc) is RuntimeError
    assert secret not in str(exc)
    assert "[REDACTED]" in str(exc)


@pytest.mark.asyncio
async def test_reader_loop_delivers_classified_jsonrpc_error_to_requester():
    driver = CodexAppServerDriver()
    driver._closed = True
    driver._proc = type("Process", (), {"stdout": asyncio.StreamReader()})()
    future = asyncio.get_running_loop().create_future()
    driver._pending[7] = future
    driver._proc.stdout.feed_data(
        json.dumps(
            {
                "id": 7,
                "error": {"code": 429, "message": "rate limit exceeded"},
            }
        ).encode()
        + b"\n"
    )
    driver._proc.stdout.feed_eof()

    await driver._read_loop()

    with pytest.raises(AgentAPIError) as raised:
        future.result()
    assert raised.value.is_auth is False
