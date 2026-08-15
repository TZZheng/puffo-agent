"""Internal history implementation for the unified message read tool."""

from __future__ import annotations

from typing import Any

from ..agent.message_projection import (
    format_history_read_result,
    format_message_group,
)
from ..agent.message_store_models import parse_inbox_target
from .data_client import DataNotFound
from .puffo_core_tools import _stage_model_visible_messages


def _optional_non_negative(name: str, value: int | None) -> int | None:
    parsed = int(value) if value is not None else None
    if parsed is not None and parsed < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return parsed


def _bounded_window(
    rows: list[Any],
    *,
    limit: int,
    reads_forward: bool,
) -> tuple[list[Any], bool, bool]:
    """Trim one look-ahead row and expose the active paging direction."""
    has_more = len(rows) > limit
    if reads_forward:
        return rows[:limit], False, has_more
    return rows[-limit:], has_more, False


def _reads_forward(
    *,
    since_message_id: str,
    after_seq: int | None,
    after_timestamp_ms: int | None,
) -> bool:
    return bool(
        since_message_id or after_seq is not None or after_timestamp_ms is not None
    )


def read_messages_tool_arguments(
    *,
    view: str,
    target: str,
    cursor: str,
    limit: int,
    since_message_id: str,
    after_seq: int | None,
    before_seq: int | None,
    after_timestamp_ms: int | None,
    before_timestamp_ms: int | None,
) -> dict[str, object]:
    """Mirror the public call while omitting only public default values."""
    arguments: dict[str, object] = {}
    if view != "pending":
        arguments["view"] = view
    if target:
        arguments["target"] = target
    if cursor:
        arguments["cursor"] = cursor
    if limit != 50:
        arguments["limit"] = limit
    for key, value in (
        ("since_message_id", since_message_id),
        ("after_seq", after_seq),
        ("before_seq", before_seq),
        ("after_timestamp_ms", after_timestamp_ms),
        ("before_timestamp_ms", before_timestamp_ms),
    ):
        if value not in (None, ""):
            arguments[key] = value
    return arguments


async def _read_channel_history(
    cfg: Any,
    *,
    target: str,
    channel_id: str,
    limit: int,
    since_message_id: str,
    after_seq: int | None,
    before_seq: int | None,
    after_timestamp_ms: int | None,
    before_timestamp_ms: int | None,
    tool_arguments: dict[str, object],
) -> str:
    try:
        roots = await cfg.data_client.get_channel_roots(
            channel_id,
            limit=limit + 1,
            since_envelope_id=since_message_id or None,
            before_ts=before_timestamp_ms,
            after_ts=after_timestamp_ms,
            before_seq=before_seq,
            after_seq=after_seq,
        )
    except DataNotFound:
        raise RuntimeError(f"no such channel target: {target}") from None
    roots, has_older, has_newer = _bounded_window(
        roots,
        limit=limit,
        reads_forward=_reads_forward(
            since_message_id=since_message_id,
            after_seq=after_seq,
            after_timestamp_ms=after_timestamp_ms,
        ),
    )
    messages = [entry.message for entry in roots]
    marker = await _stage_model_visible_messages(
        cfg,
        messages,
        tool_name="read_messages",
        tool_arguments=tool_arguments,
    )
    body = format_message_group(
        messages,
        current_agent_aliases=(cfg.slug,),
        reply_counts={
            entry.message.envelope_id: entry.reply_count for entry in roots
        },
    )
    result = format_history_read_result(
        messages,
        body=body,
        has_older=has_older,
        has_newer=has_newer,
        target_ref=target,
    )
    return f"{result}\n{marker}" if marker else result


async def _read_thread_history(
    cfg: Any,
    *,
    target: str,
    root_id: str,
    limit: int,
    since_message_id: str,
    after_seq: int | None,
    before_seq: int | None,
    after_timestamp_ms: int | None,
    before_timestamp_ms: int | None,
    tool_arguments: dict[str, object],
) -> str:
    try:
        messages = await cfg.data_client.get_thread_messages(
            root_id,
            limit=limit + 1,
            since_envelope_id=since_message_id or None,
            before_ts=before_timestamp_ms,
            after_ts=after_timestamp_ms,
            before_seq=before_seq,
            after_seq=after_seq,
        )
    except DataNotFound:
        raise RuntimeError(f"no such thread target: {target}") from None
    messages, has_older, has_newer = _bounded_window(
        messages,
        limit=limit,
        reads_forward=_reads_forward(
            since_message_id=since_message_id,
            after_seq=after_seq,
            after_timestamp_ms=after_timestamp_ms,
        ),
    )
    marker = await _stage_model_visible_messages(
        cfg,
        messages,
        tool_name="read_messages",
        tool_arguments=tool_arguments,
    )
    body = format_message_group(
        messages,
        current_agent_aliases=(cfg.slug,),
        thread_root_id=root_id,
    )
    result = format_history_read_result(
        messages,
        body=body,
        has_older=has_older,
        has_newer=has_newer,
        target_ref=target,
    )
    return f"{result}\n{marker}" if marker else result


async def _read_dm_history(
    cfg: Any,
    *,
    target: str,
    peer: str,
    limit: int,
    before_timestamp_ms: int | None,
) -> str:
    messages = await cfg.data_client.get_dm_history(
        peer,
        limit=limit + 1,
        before=before_timestamp_ms,
    )
    messages, has_older, has_newer = _bounded_window(
        messages,
        limit=limit,
        reads_forward=False,
    )
    body = format_message_group(messages, current_agent_aliases=(cfg.slug,))
    return format_history_read_result(
        messages,
        body=body,
        has_older=has_older,
        has_newer=has_newer,
        target_ref=target,
    )


async def read_history_messages(
    cfg: Any,
    *,
    target: str,
    limit: int,
    since_message_id: str,
    after_seq: int | None,
    before_seq: int | None,
    after_timestamp_ms: int | None,
    before_timestamp_ms: int | None,
    tool_arguments: dict[str, object],
) -> str:
    """Read one explicit history window for a canonical Inbox target."""
    if not target:
        raise RuntimeError("target is required when view='history'")
    try:
        kind, _space_id, channel_id, tail = parse_inbox_target(target)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None

    limit = max(1, min(int(limit), 200))
    after_seq = _optional_non_negative("after_seq", after_seq)
    before_seq = _optional_non_negative("before_seq", before_seq)
    after_timestamp_ms = _optional_non_negative(
        "after_timestamp_ms", after_timestamp_ms
    )
    before_timestamp_ms = _optional_non_negative(
        "before_timestamp_ms", before_timestamp_ms
    )

    if kind == "dm":
        unsupported = (
            since_message_id
            or after_seq is not None
            or before_seq is not None
            or after_timestamp_ms is not None
        )
        if unsupported:
            raise RuntimeError(
                "DM history supports only limit and before_timestamp_ms bounds"
            )
        return await _read_dm_history(
            cfg,
            target=target,
            peer=tail.lstrip("@"),
            limit=limit,
            before_timestamp_ms=before_timestamp_ms,
        )
    if tail:
        return await _read_thread_history(
            cfg,
            target=target,
            root_id=tail,
            limit=limit,
            since_message_id=since_message_id,
            after_seq=after_seq,
            before_seq=before_seq,
            after_timestamp_ms=after_timestamp_ms,
            before_timestamp_ms=before_timestamp_ms,
            tool_arguments=tool_arguments,
        )
    return await _read_channel_history(
        cfg,
        target=target,
        channel_id=channel_id,
        limit=limit,
        since_message_id=since_message_id,
        after_seq=after_seq,
        before_seq=before_seq,
        after_timestamp_ms=after_timestamp_ms,
        before_timestamp_ms=before_timestamp_ms,
        tool_arguments=tool_arguments,
    )
