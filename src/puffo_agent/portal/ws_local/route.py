"""aiohttp WS route for ws-local tools: ``GET /v1/ws-local``.

Loopback-only (the bridge binds loopback). Auth is the handshake's own
``.puffoagent`` decryption — this path is exempt from the bridge's HTTP
signature middleware. The handler wires the hub's per-agent attach point
into ``serve_connection``: the session relays replies + judges liveness,
the consumer (``client.listen``) feeds batches and advances the cursor
on ack.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

from .aiohttp_transport import AiohttpTransport
from .auth import authenticate_bundle
from .bundles import BundleQueue
from .endpoint import serve_connection
from .hub import AttachPoint, WsLocalHub
from .in_process_data_client import InProcessDataClient
from .protocol import Error, encode
from .session import Transport, WsLocalSession
from .tool_dispatch import build_dispatch as _build_dispatch

logger = logging.getLogger(__name__)


class _WsLocalContextAdapter:
    """Provider context/admission seam for one authenticated attachment."""

    def __init__(self) -> None:
        self._callback = None
        self._planning_cycle_key = ""
        self._provider_session_id: str | None = None

    async def get_context_snapshot(self):
        from ...agent.context_controller import ContextSnapshot

        return ContextSnapshot(0, 200_000, "ws-local", datetime.now(timezone.utc))

    def get_context_capabilities(self):
        from ...agent.context_controller import ContextCapabilities

        return ContextCapabilities()

    async def compact_context(self):
        raise RuntimeError("ws-local context compaction is unavailable")

    async def rollover_context(self):
        raise RuntimeError("ws-local context rollover is unavailable")

    def get_provider_session_id(self) -> str | None:
        return self._provider_session_id

    def register_admission_callback(self, callback, planning_cycle_key="") -> None:
        self._callback = callback
        self._planning_cycle_key = planning_cycle_key

    async def emit_admission(self, *, turn_id: str, correlation_key: str) -> None:
        from ...agent.context_controller import ProviderAdmissionEvent

        if self._callback is None or correlation_key != self._planning_cycle_key:
            raise RuntimeError("ws-local admission correlation failed")
        callback, self._callback = self._callback, None
        self._provider_session_id = f"ws-local:{turn_id}"
        await callback(ProviderAdmissionEvent(
            planning_cycle_key=correlation_key,
            provider_session_id=self._provider_session_id,
            provider_turn_id=turn_id,
            admitted_at=datetime.now(timezone.utc),
        ))


def _build_tool_dispatch(point: AttachPoint):
    from ...mcp.puffo_core_tools import PuffoCoreToolsConfig
    client = point.client
    send_coordinator = getattr(client, "send_delegate", None)
    cfg = PuffoCoreToolsConfig(
        slug=client.slug,
        agent_id=point.agent_id,
        device_id=client.device_id,
        keystore=client.keystore,
        http_client=client.http,
        data_client=InProcessDataClient(client.store, client),
        space_id=getattr(client, "space_id", None),
        workspace=getattr(client, "workspace", None),
        message_client=client,
        send_coordinator=send_coordinator,
        # T23: the daemon owns the single per-agent bridge WS, so only
        # the in-process ws-local tools can drive it. None on native
        # agents → send_message keeps the signed-crypto path. The
        # subprocess/RPC MCP site can't own this WS, so it stays None.
        bridge_client=getattr(client, "_bridge", None),
    )
    return _build_dispatch(cfg)

WS_LOCAL_PATH = "/v1/ws-local"


async def handle_ws_local(request: web.Request) -> web.WebSocketResponse:
    hub: WsLocalHub | None = request.app.get("ws_local_hub")
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    transport = AiohttpTransport(ws)
    if hub is None:
        await transport.send(encode(Error("ws-local is not enabled on this daemon")))
        await transport.close()
        return ws
    await serve_attached(transport, hub)
    return ws


async def serve_attached(transport: Transport, hub: WsLocalHub) -> None:
    """Wire the hub into ``serve_connection``. Split out from the aiohttp
    boilerplate so it's exercisable over any transport."""

    async def agent_context(slug: str) -> dict:
        point = hub.get(slug)
        if point is None:
            return {}
        cfg = point.agent_cfg
        try:
            profile_md = Path(cfg.resolve_profile_path()).read_text(encoding="utf-8")
        except OSError:
            profile_md = ""
        return {
            "slug": slug,
            "display_name": getattr(cfg, "display_name", ""),
            "profile_md": profile_md,
        }

    def make_session(authed, session_id, t, bridge, capabilities) -> WsLocalSession:
        point = hub.get(authed.slug)
        client = point.client
        runtime = getattr(client, "global_runtime", None)
        v2_capable = {
            "multi-target-v2", "explicit-admission-v2",
        }.issubset(capabilities)
        owned_runtime = runtime is None and v2_capable
        session = None
        if owned_runtime:
            from ...agent.global_inbox_runtime import (
                ActiveBoundaryAdapter,
                BaselineAdapter,
                GlobalInboxRuntime,
                TrackingSendDelegate,
            )
            from ...agent.send_coordinator import SendCoordinator

            adapter = _WsLocalContextAdapter()

            async def run_turn(planned):
                await bridge.dispatch_planned(session, planned)

            runtime = GlobalInboxRuntime(
                store=client.store,
                adapter=adapter,
                run_turn=run_turn,
                workspace=client.workspace or point.agent_cfg.resolve_workspace_dir(),
                send_mode_keys=(point.agent_id, client.slug),
                agent_id=point.agent_id,
            )
            coordinator = SendCoordinator(
                slug=client.slug,
                keystore=client.keystore,
                http_client=client.http,
                data_client=InProcessDataClient(client.store, client),
                workspace=client.workspace or point.agent_cfg.resolve_workspace_dir(),
                baseline_source=BaselineAdapter(client.store),
                active_turn_source=ActiveBoundaryAdapter(client.store, runtime.active),
                held_recovery_source=runtime.held_recovery_source,
            )
            runtime.coordinator = coordinator
            runtime.send_delegate = TrackingSendDelegate(
                coordinator, runtime.attempts, runtime,
            )
            client.global_runtime = runtime
            client.send_coordinator = coordinator
            client.send_delegate = runtime.send_delegate
            client._ws_local_owned_runtime = runtime
        bridge._runtime = runtime
        session = WsLocalSession(
            slug=authed.slug,
            session_id=session_id,
            transport=t,
            queue=BundleQueue(),
            reporter=point.reporter,
            tool_dispatch=_build_tool_dispatch(point),
            on_acked=bridge.on_acked,
            on_admitted=bridge.on_admitted,
            on_dead=bridge.on_dead,
            capabilities=capabilities,
            now=time.monotonic,
            ack_timeout_s=point.ack_timeout_s,
            ping_interval_s=point.ping_interval_s,
        )
        return session

    async def start_consumer(authed, on_message):
        from ...agent.global_inbox_runtime import await_listener_with_runtime

        point = hub.get(authed.slug)
        client = point.client
        owned_runtime = getattr(client, "_ws_local_owned_runtime", None)
        # Attaching is what brings the agent online: run the heartbeat
        # for the lifetime of the consumer.
        hb = asyncio.ensure_future(point.reporter.run_heartbeat_loop())
        runtime_task = (
            asyncio.ensure_future(owned_runtime.run())
            if owned_runtime is not None else None
        )
        try:
            if runtime_task is None:
                await client.listen(on_message)
            else:
                await await_listener_with_runtime(
                    client.listen(on_message),
                    runtime_task,
                    label=f"ws-local {authed.slug} global inbox runtime",
                )
        finally:
            point.reporter.stop()
            hb.cancel()
            if runtime_task is not None:
                owned_runtime.stop()
                runtime_task.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
            if runtime_task is not None:
                try:
                    await runtime_task
                except (asyncio.CancelledError, Exception):
                    pass
                if getattr(client, "_ws_local_owned_runtime", None) is owned_runtime:
                    client.global_runtime = None
                    client.send_coordinator = None
                    client.send_delegate = None
                    del client._ws_local_owned_runtime

    await serve_connection(
        transport,
        authenticate=authenticate_bundle,
        is_servable=hub.is_servable,
        agent_context=agent_context,
        registry=hub.registry,
        make_session=make_session,
        start_consumer=start_consumer,
        new_session_id=lambda: f"wsl_{uuid.uuid4().hex}",
        base64_decode=base64.b64decode,
    )
