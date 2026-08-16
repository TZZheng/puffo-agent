"""Outbound message MCP tool registration."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ..agent.send_coordinator import SemanticSendRequest
from .puffo_core_tools import _dispatch_semantic_send
from .tool_result_projection import (
    ToolResultSurface,
    project_send_result,
)


def register_message_tools(
    mcp: FastMCP,
    cfg: Any,
    *,
    result_surface: ToolResultSurface = "stdio_mcp",
) -> None:
    @mcp.tool(structured_output=False)
    async def send_message(
        channel: str,
        text: str,
        root_id: str = "",
        visibility_level: str = "default",
        send_anyway: bool = False,
    ) -> Any:
        """Post text to a channel ``ch_<uuid>`` or DM ``@<slug>``.

        ``root_id`` is an optional thread root; ``visibility_level`` is
        ``human``, ``default`` (default), or ``agent_only``; ``send_anyway``
        is an explicit held-send flag. Results are ``sent``, ``held``, or an
        error. See the managed ``send-message`` skill for routing, visibility,
        thread-root validation, and held-send guidance.
        """
        return project_send_result(
            await _dispatch_semantic_send(
                cfg,
                SemanticSendRequest(
                    destination=channel,
                    text=text,
                    root_id=root_id,
                    visibility_level=visibility_level,
                    send_anyway=send_anyway,
                ),
            ),
            surface=result_surface,
        )

    @mcp.tool(structured_output=False)
    async def send_message_with_attachments(
        paths: list[str],
        channel: str,
        caption: str = "",
        root_id: str = "",
        visibility_level: str = "default",
        send_anyway: bool = False,
    ) -> Any:
        """Send workspace files and an optional caption to a channel or DM.

        ``paths`` are workspace-relative; ``channel``, ``root_id``,
        ``visibility_level``, and ``send_anyway`` match ``send_message``.
        Results are ``sent``, ``held``, or an error. See the managed
        ``send-message-with-attachments`` skill and its common
        ``send-message`` held-send procedure.
        """
        return project_send_result(
            await _dispatch_semantic_send(
                cfg,
                SemanticSendRequest(
                    destination=channel,
                    attachment_paths=(tuple(paths) if isinstance(paths, list) else ()),
                    caption=caption,
                    root_id=root_id,
                    visibility_level=visibility_level,
                    send_anyway=send_anyway,
                ),
                tool_name="send_message_with_attachments",
            ),
            surface=result_surface,
        )
