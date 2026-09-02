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


@pytest.mark.asyncio
async def test_ws_dns_dead_executor_heals_without_http(monkeypatch):
    """Valid subkey: connect_once() makes no HTTP request before
    websockets.connect hits loop.getaddrinfo, so the reconnect loop
    itself must heal the dead executor for the next attempt to work."""
    from puffo_agent.crypto import ws_client as ws_module

    client = ws_module.PuffoCoreWsClient.__new__(ws_module.PuffoCoreWsClient)
    client.slug = "s"
    client.ws_url = "ws://x/subscribe"
    client._ws = None
    client.session_id = None
    client.on_transport_state = None
    client._reconnect_failures = 0

    class _ValidSubkeyHttp:
        async def _ensure_subkey(self):
            return None

    client.http_client = _ValidSubkeyHttp()

    calls = {"n": 0}

    def fake_connect(url, ssl=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cannot schedule new futures after shutdown")
        client._running = False
        raise ConnectionError("stop the loop")

    monkeypatch.setattr(ws_module.websockets, "connect", fake_connect)
    monkeypatch.setattr(ws_module, "INITIAL_BACKOFF", 0)

    loop = asyncio.get_running_loop()
    executor_before = loop._default_executor
    await client.run()

    assert calls["n"] == 2
    assert client._reconnect_failures >= 1
    assert loop._default_executor is not None
    assert loop._default_executor is not executor_before


def test_ws_streaks_flip_health_to_server_unreachable_and_back(tmp_path, monkeypatch):
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    from types import SimpleNamespace

    from puffo_agent.portal import worker as worker_module
    from puffo_agent.portal.state import RuntimeState, agent_dir
    from puffo_agent.portal.worker import _WS_DEGRADE_THRESHOLD, Worker

    agent_dir("a1").mkdir(parents=True)
    worker = Worker.__new__(Worker)
    worker.runtime = RuntimeState(status="running", health="ok")
    worker.agent_cfg = SimpleNamespace(id="a1")
    worker.daemon_cfg = None
    monkeypatch.setattr(
        worker_module,
        "_build_puffo_core_client",
        lambda cfg, agent_id, daemon_cfg=None: SimpleNamespace(),
    )
    # Through the wiring seam both runtime paths use — a client built
    # here must arrive with the listener attached, or streaks reach
    # nobody (the standard daemon path shipped exactly that hole once).
    note = worker._build_wired_client().transport_state_listener
    assert note is not None

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
