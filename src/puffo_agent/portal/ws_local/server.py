"""Loopback HTTP server for the ws-local WebSocket endpoint."""

from __future__ import annotations

import logging

from aiohttp import web

from ..state import WsLocalServiceConfig
from .route import WS_LOCAL_HUB_KEY, WS_LOCAL_PATH, handle_ws_local

logger = logging.getLogger(__name__)


def build_app(ws_local_hub=None) -> web.Application:
    app = web.Application()
    app[WS_LOCAL_HUB_KEY] = ws_local_hub
    app.router.add_get(WS_LOCAL_PATH, handle_ws_local)
    return app


async def start_ws_local_server(
    cfg: WsLocalServiceConfig, ws_local_hub=None,
) -> web.AppRunner | None:
    if not cfg.enabled:
        logger.info("ws-local: disabled in daemon.yml; not starting")
        return None
    runner = web.AppRunner(
        build_app(ws_local_hub),
        access_log=logging.getLogger("puffo_agent.portal.ws_local.access"),
        access_log_format="%r -> %s (%Tf s)",
    )
    await runner.setup()
    site = web.TCPSite(runner, host=cfg.bind_host, port=cfg.port)
    try:
        await site.start()
    except OSError as exc:
        logger.warning(
            "ws-local: failed to bind %s:%d (%s); continuing without it",
            cfg.bind_host,
            cfg.port,
            exc,
        )
        await runner.cleanup()
        return None
    logger.info("ws-local: listening on ws://%s:%d%s", cfg.bind_host, cfg.port, WS_LOCAL_PATH)
    return runner


async def stop_ws_local_server(runner: web.AppRunner | None) -> None:
    if runner is None:
        return
    try:
        await runner.cleanup()
    except Exception as exc:
        logger.warning("ws-local: cleanup failed: %s", exc)
