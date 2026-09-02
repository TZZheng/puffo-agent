"""Transport self-heal + honest health (8/30 App Nap incident).

Two key behaviors:
- a dead default executor (RuntimeError: cannot schedule new futures
  after shutdown) heals in-place: fresh executor + rebuilt session +
  one retry, instead of failing every request forever;
- WS reconnect-failure streaks flip runtime.json health to
  "server_unreachable" and back, instead of reporting "ok" while the
  server is unreachable for hours.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from puffo_agent.crypto.http_client import PuffoCoreHttpClient


class _FakeResponse:
    status = 200

    async def text(self) -> str:
        return json.dumps({"healed": True})


class _RequestCtx:
    def __init__(self, fail: bool):
        self._fail = fail

    async def __aenter__(self):
        if self._fail:
            raise RuntimeError("cannot schedule new futures after shutdown")
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, fail: bool):
        self._fail = fail
        self.closed = False

    def request(self, method, url, **kwargs):
        return _RequestCtx(self._fail)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_dead_default_executor_heals_and_request_succeeds(monkeypatch):
    client = PuffoCoreHttpClient.__new__(PuffoCoreHttpClient)
    client.server_url = "https://x"
    sessions = [_FakeSession(fail=True), _FakeSession(fail=False)]

    async def fake_get_session():
        return sessions[0] if not sessions[0].closed else sessions[1]

    monkeypatch.setattr(client, "_get_session", fake_get_session)
    client._session = sessions[0]
    monkeypatch.setattr(client, "_egress_headers", lambda base=None: dict(base or {}))

    loop = asyncio.get_running_loop()
    executor_before = loop._default_executor

    result = await client.get_unsigned("/ping")

    assert result == {"healed": True}
    # The broken session was replaced, and the loop-global executor was
    # renewed — that is what also revives WS reconnects.
    assert sessions[0].closed
    assert loop._default_executor is not executor_before
    assert loop._default_executor is not None


def test_ws_streaks_flip_health_to_server_unreachable_and_back(tmp_path, monkeypatch):
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    from puffo_agent.portal.state import RuntimeState, agent_dir
    from puffo_agent.portal.worker import _WS_DEGRADE_THRESHOLD, Worker

    agent_dir("a1").mkdir(parents=True)
    worker = Worker.__new__(Worker)
    worker.runtime = RuntimeState(status="running", health="ok")
    note = worker._transport_state_listener("a1")

    for streak in range(1, _WS_DEGRADE_THRESHOLD):
        note(False, streak)
    assert worker.runtime.health == "ok"

    note(False, _WS_DEGRADE_THRESHOLD)
    assert worker.runtime.health == "server_unreachable"
    assert "consecutive WS reconnect failures" in worker.runtime.error
    on_disk = json.loads((agent_dir("a1") / "runtime.json").read_text())
    assert on_disk["health"] == "server_unreachable"

    # Recovery clears only the transport-set state...
    note(True, 0)
    assert worker.runtime.health == "ok"
    assert worker.runtime.error == ""

    # ...and stronger signals are never clobbered in either direction.
    worker.runtime.health = "auth_failed"
    note(False, _WS_DEGRADE_THRESHOLD + 1)
    assert worker.runtime.health == "auth_failed"
    note(True, 0)
    assert worker.runtime.health == "auth_failed"
