"""Inbox and durable reminder MCP tool registration."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ..agent.message_projection import format_inbox_read_result
from ..agent.message_store_models import parse_inbox_target
from .core_history_read_tools import (
    read_history_messages,
    read_messages_tool_arguments,
)


def _normalize_inbox_read_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize the pre-window RPC shape during rolling local upgrades."""
    return {
        "context_version": int(result.get("context_version", 1)),
        "messages": result["messages"],
        "prior_context": result.get("prior_context", ()),
        "prior_context_has_more": bool(result.get("prior_context_has_more", False)),
        "next_cursor": result["next_cursor"],
        "has_more": result["has_more"],
        "remaining_count": result["remaining_count"],
        "snapshot_generation": result["snapshot_generation"],
    }


def register_message_read_tool(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def read_messages(
        view: str = "pending",
        target: str = "",
        cursor: str = "",
        limit: int = 50,
        since_message_id: str = "",
        after_seq: int | None = None,
        before_seq: int | None = None,
        after_timestamp_ms: int | None = None,
        before_timestamp_ms: int | None = None,
    ) -> str:
        """Read one semantic conversation window.

        ``view='pending'`` drains pending Inbox work; target is optional and
        cursor continues the returned snapshot. ``view='history'`` reads a
        canonical DM/channel/thread target without changing pending intent and
        accepts message, sequence, or timestamp bounds. ``limit`` bounds the
        page size. Results use the same context/message grammar and explicit
        older/newer window boundaries. See the managed ``read-messages`` skill
        for paging semantics.
        """
        if isinstance(limit, bool):
            raise RuntimeError("limit must be an integer")
        arguments = read_messages_tool_arguments(
            view=view,
            target=target,
            cursor=cursor,
            limit=limit,
            since_message_id=since_message_id,
            after_seq=after_seq,
            before_seq=before_seq,
            after_timestamp_ms=after_timestamp_ms,
            before_timestamp_ms=before_timestamp_ms,
        )
        if view == "history":
            if cursor:
                raise RuntimeError("cursor is only valid when view='pending'")
            return await read_history_messages(
                cfg,
                target=target,
                limit=limit,
                since_message_id=since_message_id,
                after_seq=after_seq,
                before_seq=before_seq,
                after_timestamp_ms=after_timestamp_ms,
                before_timestamp_ms=before_timestamp_ms,
                tool_arguments=arguments,
            )
        if view != "pending":
            raise RuntimeError("view must be 'pending' or 'history'")
        if any(
            value not in (None, "")
            for value in (
                since_message_id,
                after_seq,
                before_seq,
                after_timestamp_ms,
                before_timestamp_ms,
            )
        ):
            raise RuntimeError(
                "history bounds are only valid when view='history'"
            )
        if target:
            try:
                parse_inbox_target(target)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from None
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(
                getattr(cfg, "message_client", None), "global_runtime", None
            )
        if runtime is not None:
            result = await runtime.read_inbox(
                target=target,
                cursor=cursor,
                limit=limit,
                tool_arguments=arguments,
            )
        elif cfg.rpc_client is not None:
            result = await cfg.rpc_client.read_inbox(
                target=target, cursor=cursor, limit=limit
            )
        else:
            raise RuntimeError("global Inbox runtime is unavailable")
        return format_inbox_read_result(_normalize_inbox_read_result(result))


def register_reminder_tools(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def create_reminder(
        content: str,
        target: str,
        intended_at: str,
    ) -> dict[str, Any]:
        """Create one durable local reminder for a canonical Inbox target.

        ``intended_at`` is an explicit-offset RFC3339 timestamp. The result
        is a provider-neutral reminder object. A due occurrence enters the
        ordinary durable Inbox and leaves any action decision to the model.
        """
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(
                getattr(cfg, "message_client", None),
                "global_runtime",
                None,
            )
        if runtime is not None:
            return await runtime.create_reminder(
                content=content,
                target=target,
                intended_at=intended_at,
            )
        if cfg.rpc_client is not None:
            return await cfg.rpc_client.create_reminder(
                content=content,
                target=target,
                intended_at=intended_at,
            )
        raise RuntimeError("global Inbox runtime is unavailable")

    @mcp.tool()
    async def list_reminders(
        state: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """List durable reminders, including target, content, time, and state."""
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(
                getattr(cfg, "message_client", None),
                "global_runtime",
                None,
            )
        if runtime is not None:
            return await runtime.list_reminders(state=state, limit=limit)
        if cfg.rpc_client is not None:
            return await cfg.rpc_client.list_reminders(state=state, limit=limit)
        raise RuntimeError("global Inbox runtime is unavailable")

    @mcp.tool()
    async def cancel_reminder(reminder_id: str) -> dict[str, Any]:
        """Idempotently cancel a reminder whose delivery has not started."""
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(
                getattr(cfg, "message_client", None),
                "global_runtime",
                None,
            )
        if runtime is not None:
            return await runtime.cancel_reminder(reminder_id=reminder_id)
        if cfg.rpc_client is not None:
            return await cfg.rpc_client.cancel_reminder(reminder_id=reminder_id)
        raise RuntimeError("global Inbox runtime is unavailable")


def register_reminder_replace_tool(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def replace_reminder(
        reminder_id: str,
        content: str = "",
        target: str = "",
        intended_at: str = "",
    ) -> dict[str, Any]:
        """Atomically replace one scheduled reminder.

        Supply one or more changed fields; empty fields inherit the existing
        value. ``intended_at`` uses explicit-offset RFC3339. The result contains
        the cancelled reminder and its scheduled replacement. Delivery that
        already started is never revoked.
        """
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(
                getattr(cfg, "message_client", None),
                "global_runtime",
                None,
            )
        if runtime is not None:
            return await runtime.replace_reminder(
                reminder_id=reminder_id,
                content=content,
                target=target,
                intended_at=intended_at,
            )
        if cfg.rpc_client is not None:
            return await cfg.rpc_client.replace_reminder(
                reminder_id=reminder_id,
                content=content,
                target=target,
                intended_at=intended_at,
            )
        raise RuntimeError("global Inbox runtime is unavailable")


def register_inbox_tools(mcp: FastMCP, cfg: Any) -> None:
    """Register Inbox and reminder tools in their established order."""
    register_message_read_tool(mcp, cfg)
    register_reminder_tools(mcp, cfg)
    register_reminder_replace_tool(mcp, cfg)
