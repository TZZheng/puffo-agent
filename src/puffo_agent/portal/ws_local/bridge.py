"""Bridge the durable global inbox runtime to a connected local tool.

``GlobalInboxRuntime`` plans and admits one turn at a time.
``WsLocalBridge.dispatch_planned`` sends that frozen plan and blocks until
the tool acknowledges it; disconnects requeue the active message ids.

``dispatch`` is retained as the frozen v1 single-root wire adapter. Current
daemon scheduling uses only ``dispatch_planned``; removing v1 requires a
separate protocol deprecation.

The bridge owns no transport or crypto — the ``WsLocalSession`` runs the
frame loop, relays replies, and judges liveness; the bridge only couples
"this batch" to "its ack".
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping

from .bundles import Bundle


class BridgeClosed(Exception):
    """Raised from ``dispatch`` when the session died before the batch was
    acked — signals the consumer to preserve the cursor for redelivery."""


class WsLocalBridge:
    def __init__(self, runtime=None) -> None:
        self._waiter: asyncio.Future[None] | None = None
        self._dead_reason: str | None = None
        self._runtime = runtime
        self._admitted = False

    # ── session hooks (passed to WsLocalSession) ─────────────────────────────

    async def on_acked(self, bundle: Bundle) -> None:
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)

    async def on_tool_result(
        self, tool_name: str, tool_arguments: Mapping[str, object], result: object,
    ) -> None:
        """Keep actual tool evidence private while linking it to its receipt."""
        if self._runtime is None or not isinstance(result, Mapping):
            return
        marker = result.get("tool_result_admission")
        if not isinstance(marker, str):
            return
        match = re.search(r"\[puffo:model-visible-read:([^\]]+)\]", marker)
        observe = getattr(self._runtime.adapter, "observe_tool_result", None)
        if match is not None and callable(observe):
            observe(match.group(1), tool_name, tool_arguments)

    async def on_admitted(self, bundle: Bundle, frame=None) -> None:
        """Apply the exact model-visible transition; ``ack`` never does."""
        if self._runtime is None or not bundle.turn_id:
            return
        planned = getattr(self._runtime, "_ws_local_planned", None)
        if planned is None or planned.turn_id != bundle.turn_id:
            raise BridgeClosed("ws-local admission correlation failed")
        correlation_key = getattr(frame, "correlation_key", "") or (
            planned.planning_cycle_key if not self._admitted else ""
        )
        emit = getattr(self._runtime.adapter, "emit_admission", None)
        if emit is None:
            raise BridgeClosed("ws-local runtime has no admission seam")
        await emit(
            turn_id=bundle.turn_id,
            correlation_key=correlation_key,
        )
        if correlation_key == planned.planning_cycle_key:
            self._admitted = True

    async def on_dead(self, reason: str) -> None:
        self._dead_reason = reason
        if self._runtime is not None and self._runtime.active.turn_id:
            await self._runtime.store.requeue_messages(
                tuple(self._runtime.active.message_ids),
                turn_id=self._runtime.active.turn_id,
            )
            self._runtime.active.clear()
            self._runtime.notify()
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_exception(BridgeClosed(reason))

    async def dispatch_planned(self, session, planned) -> None:
        if self._dead_reason is not None:
            raise BridgeClosed(self._dead_reason)
        self._runtime._ws_local_planned = planned
        self._admitted = False
        loop = asyncio.get_event_loop()
        self._waiter = loop.create_future()
        await session.deliver_planned(planned)
        try:
            await self._waiter
        finally:
            self._waiter = None

    # ── consumer callback ────────────────────────────────────────────────────

    async def dispatch(self, session, root_id: str, batch: list[dict], channel_meta: dict) -> None:
        """Send ``batch`` to the tool and block until it acks. Raises
        ``BridgeClosed`` if the session is (or becomes) dead."""
        if self._dead_reason is not None:
            raise BridgeClosed(self._dead_reason)
        loop = asyncio.get_event_loop()
        self._waiter = loop.create_future()
        # Set the waiter before delivering so an ack processed on the
        # session's frame loop can't race ahead of us.
        await session.deliver_batch(root_id, batch, channel_meta)
        try:
            await self._waiter
        finally:
            self._waiter = None
