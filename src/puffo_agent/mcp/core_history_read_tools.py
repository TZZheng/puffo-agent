"""MCP registration for history read tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic.json_schema import SkipJsonSchema

from ..agent.message_projection import (
    format_message_group,
)
from .data_client import DataNotFound
from .puffo_core_tools import (
    _resolve_channel_space,
    _stage_model_visible_messages,
)


def _resolve_text_alias(
    explicit_name: str,
    explicit_value: str,
    legacy_name: str,
    legacy_value: str,
) -> str:
    if explicit_value and legacy_value and explicit_value != legacy_value:
        raise RuntimeError(
            f"conflicting values for {explicit_name} and deprecated {legacy_name}"
        )
    return explicit_value or legacy_value


def _resolve_int_alias(
    explicit_name: str,
    explicit_value: int | None,
    legacy_name: str,
    legacy_value: int,
) -> int | None:
    explicit = int(explicit_value) if explicit_value is not None else None
    legacy = int(legacy_value) if legacy_value else None
    if explicit is not None and legacy is not None and explicit != legacy:
        raise RuntimeError(
            f"conflicting values for {explicit_name} and deprecated {legacy_name}"
        )
    value = explicit if explicit is not None else legacy
    if value is not None and value < 0:
        raise RuntimeError(f"{explicit_name} must be non-negative")
    return value


def _optional_non_negative(name: str, value: int | None) -> int | None:
    parsed = int(value) if value is not None else None
    if parsed is not None and parsed < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return parsed


def _register_channel_history_tool(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def get_channel_history(
        channel: str,
        limit: int = 20,
        since_envelope_id: str = "",
        after_seq: int | None = None,
        before_seq: int | None = None,
        after_timestamp_ms: int | None = None,
        before_timestamp_ms: int | None = None,
        since: SkipJsonSchema[str] = "",
        after: SkipJsonSchema[int] = 0,
        before: SkipJsonSchema[int] = 0,
    ) -> str:
        """List local channel root posts.

        ``channel`` is ``ch_<uuid>``. Prefer the explicitly named
        ``since_envelope_id``, ``after_seq`` / ``before_seq``, and
        ``after_timestamp_ms`` / ``before_timestamp_ms`` filters. Hidden legacy
        aliases remain accepted for compatibility but are not advertised to the
        model. Results are semantic target/row projection groups with root reply
        counts; use ``get_thread_history`` for replies.
        """
        limit = max(1, min(int(limit), 200))
        resolved_since = _resolve_text_alias(
            "since_envelope_id", since_envelope_id, "since", since
        )
        resolved_after_ts = _resolve_int_alias(
            "after_timestamp_ms", after_timestamp_ms, "after", after
        )
        resolved_before_ts = _resolve_int_alias(
            "before_timestamp_ms", before_timestamp_ms, "before", before
        )
        channel_ref = channel.strip()
        if channel_ref.startswith("#"):
            raise RuntimeError(
                "'#<name>' channel addressing isn't supported; pass the "
                "channel id directly."
            )
        # Local-store read would return empty on a slug — route non-
        # ``ch_`` refs through the resolver purely for its hint error.
        if not channel_ref.startswith("ch_"):
            await _resolve_channel_space(cfg, channel_ref)
        channel_id = channel_ref

        try:
            roots = await cfg.data_client.get_channel_roots(
                channel_id,
                limit=limit,
                since_envelope_id=resolved_since or None,
                before_ts=resolved_before_ts,
                after_ts=resolved_after_ts,
                before_seq=_optional_non_negative("before_seq", before_seq),
                after_seq=_optional_non_negative("after_seq", after_seq),
            )
        except DataNotFound:
            return f"(no such channel: {channel_id})"
        if not roots:
            return "(no root posts in the requested window)"
        tool_arguments: dict[str, object] = {"channel": channel}
        if limit != 20:
            tool_arguments["limit"] = limit
        if since_envelope_id:
            tool_arguments["since_envelope_id"] = since_envelope_id
        if after_seq is not None:
            tool_arguments["after_seq"] = after_seq
        if before_seq is not None:
            tool_arguments["before_seq"] = before_seq
        if after_timestamp_ms is not None:
            tool_arguments["after_timestamp_ms"] = after_timestamp_ms
        if before_timestamp_ms is not None:
            tool_arguments["before_timestamp_ms"] = before_timestamp_ms
        if since:
            tool_arguments["since"] = since
        if after:
            tool_arguments["after"] = after
        if before:
            tool_arguments["before"] = before
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            [entry.message for entry in roots],
            tool_name="get_channel_history",
            tool_arguments=tool_arguments,
        )
        result = format_message_group(
            [entry.message for entry in roots],
            current_agent_aliases=(cfg.slug,),
            reply_counts={
                entry.message.envelope_id: entry.reply_count for entry in roots
            },
        )
        return f"{result}\n{receipt_marker}" if receipt_marker else result


def _register_dm_history_tool(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def get_dm_history(
        peer: str,
        limit: int = 20,
        before: int = 0,
    ) -> str:
        """List local DMs with ``peer``, oldest first.

        ``before`` is an exclusive ms-epoch paging bound. Results use the
        semantic target/row projection. See the managed ``channel-history``
        skill for supplementary-context guidance.
        """
        limit = max(1, min(int(limit), 200))
        peer_slug = peer.strip().lstrip("@")
        if not peer_slug:
            raise RuntimeError("pass the peer's slug to read DM history.")
        msgs = await cfg.data_client.get_dm_history(
            peer_slug,
            limit=limit,
            before=int(before) if before else None,
        )
        if not msgs:
            return "(no direct messages with that peer in the requested window)"
        return format_message_group(msgs, current_agent_aliases=(cfg.slug,))


def register_channel_and_dm_history_tools(mcp: FastMCP, cfg: Any) -> None:
    """Register channel and DM readers in their established order."""
    _register_channel_history_tool(mcp, cfg)
    _register_dm_history_tool(mcp, cfg)


def register_thread_history_tools(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def get_thread_history(
        root_id: str,
        limit: int = 50,
        since_envelope_id: str = "",
        after_seq: int | None = None,
        before_seq: int | None = None,
        after_timestamp_ms: int | None = None,
        before_timestamp_ms: int | None = None,
        since: SkipJsonSchema[str] = "",
        after: SkipJsonSchema[int] = 0,
        before: SkipJsonSchema[int] = 0,
    ) -> str:
        """List a local thread's root and replies, oldest first.

        ``root_id`` is the root envelope id. Prefer the explicitly named
        envelope, sequence, or ms-epoch timestamp filters. Hidden legacy aliases
        remain accepted for compatibility but are not advertised to the model.
        Results use the semantic target/row projection.
        """
        if not root_id.strip():
            raise RuntimeError("root_id required")
        limit = max(1, min(int(limit), 200))
        resolved_since = _resolve_text_alias(
            "since_envelope_id", since_envelope_id, "since", since
        )
        resolved_after_ts = _resolve_int_alias(
            "after_timestamp_ms", after_timestamp_ms, "after", after
        )
        resolved_before_ts = _resolve_int_alias(
            "before_timestamp_ms", before_timestamp_ms, "before", before
        )
        try:
            msgs = await cfg.data_client.get_thread_messages(
                root_id.strip(),
                limit=limit,
                since_envelope_id=resolved_since or None,
                before_ts=resolved_before_ts,
                after_ts=resolved_after_ts,
                before_seq=_optional_non_negative("before_seq", before_seq),
                after_seq=_optional_non_negative("after_seq", after_seq),
            )
        except DataNotFound:
            return f"(no such thread: {root_id.strip()})"
        if not msgs:
            return "(no messages in this thread for the requested window)"
        tool_arguments = {"root_id": root_id}
        if limit != 50:
            tool_arguments["limit"] = limit
        if since_envelope_id:
            tool_arguments["since_envelope_id"] = since_envelope_id
        if after_seq is not None:
            tool_arguments["after_seq"] = after_seq
        if before_seq is not None:
            tool_arguments["before_seq"] = before_seq
        if after_timestamp_ms is not None:
            tool_arguments["after_timestamp_ms"] = after_timestamp_ms
        if before_timestamp_ms is not None:
            tool_arguments["before_timestamp_ms"] = before_timestamp_ms
        if since:
            tool_arguments["since"] = since
        if after:
            tool_arguments["after"] = after
        if before:
            tool_arguments["before"] = before
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            msgs,
            tool_name="get_thread_history",
            tool_arguments=tool_arguments,
        )
        result = format_message_group(
            msgs,
            current_agent_aliases=(cfg.slug,),
            thread_root_id=root_id.strip(),
        )
        return f"{result}\n{receipt_marker}" if receipt_marker else result


def register_history_read_tools(mcp: FastMCP, cfg: Any) -> None:
    """Register conversation history tools in their established order."""
    register_channel_and_dm_history_tools(mcp, cfg)
    register_thread_history_tools(mcp, cfg)
