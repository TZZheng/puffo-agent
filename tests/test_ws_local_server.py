from __future__ import annotations

import logging
import socket

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from puffo_agent.portal import state
from puffo_agent.portal.state import DaemonConfig, WsLocalServiceConfig
from puffo_agent.portal.ws_local.server import (
    build_app,
    start_ws_local_server,
    stop_ws_local_server,
)
from puffo_agent.portal.ws_local.route import WS_LOCAL_HUB_KEY


def test_app_exposes_only_ws_local_route():
    app = build_app(object())
    routes = {(route.method, route.resource.canonical) for route in app.router.routes()}
    assert routes == {("GET", "/v1/ws-local"), ("HEAD", "/v1/ws-local")}
    assert app[WS_LOCAL_HUB_KEY] is not None


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/v1/info"),
        ("get", "/v1/providers"),
        ("post", "/v1/pair"),
        ("get", "/v1/agents"),
        ("post", "/v1/agents"),
        ("get", "/v1/agents/example/log"),
        ("post", "/v1/agents/export"),
    ],
)
@pytest.mark.asyncio
async def test_removed_http_apis_return_not_found(method, path):
    async with TestClient(TestServer(build_app())) as client:
        response = await getattr(client, method)(path)
    assert response.status == 404


@pytest.mark.asyncio
async def test_ws_route_rejects_when_hub_is_unavailable():
    async with TestClient(TestServer(build_app())) as client:
        websocket = await client.ws_connect("/v1/ws-local")
        message = await websocket.receive_json()
    assert message == {"type": "error", "reason": "ws-local is not enabled on this daemon"}


@pytest.mark.asyncio
async def test_disabled_service_does_not_start(caplog):
    with caplog.at_level(logging.INFO):
        runner = await start_ws_local_server(WsLocalServiceConfig(enabled=False))
    assert runner is None
    assert "disabled in daemon.yml" in caplog.text


@pytest.mark.asyncio
async def test_service_starts_and_stops_on_ephemeral_port():
    runner = await start_ws_local_server(WsLocalServiceConfig(port=0), object())
    assert runner is not None
    await stop_ws_local_server(runner)


@pytest.mark.asyncio
async def test_bind_failure_is_nonfatal(caplog):
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = occupied.getsockname()[1]
    try:
        with caplog.at_level(logging.WARNING):
            runner = await start_ws_local_server(WsLocalServiceConfig(port=port))
        assert runner is None
        assert "failed to bind" in caplog.text
    finally:
        occupied.close()


@pytest.mark.asyncio
async def test_cleanup_failure_is_nonfatal(caplog):
    class FailingRunner:
        async def cleanup(self):
            raise RuntimeError("cleanup boom")

    with caplog.at_level(logging.WARNING):
        await stop_ws_local_server(FailingRunner())
    assert "cleanup failed" in caplog.text


@pytest.mark.asyncio
async def test_stopping_absent_runner_is_noop():
    await stop_ws_local_server(None)


def test_legacy_bridge_config_is_ignored_and_dropped(tmp_path, monkeypatch):
    config_path = tmp_path / "daemon.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "bridge": {
                    "enabled": False,
                    "bind_host": "0.0.0.0",
                    "port": 61234,
                    "allowed_origins": ["https://example.test"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(state, "daemon_yml_path", lambda: config_path)

    config = DaemonConfig.load()
    assert config.ws_local_service == WsLocalServiceConfig()
    config.save()
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "bridge" not in saved
    assert saved["ws_local_service"] == {
        "enabled": True,
        "bind_host": "127.0.0.1",
        "port": 63387,
    }


def test_ws_local_service_config_round_trips(tmp_path, monkeypatch):
    config_path = tmp_path / "daemon.yml"
    monkeypatch.setattr(state, "daemon_yml_path", lambda: config_path)
    config = DaemonConfig(
        ws_local_service=WsLocalServiceConfig(
            enabled=False, bind_host="::1", port=62000,
        )
    )
    config.save()
    assert DaemonConfig.load().ws_local_service == config.ws_local_service
