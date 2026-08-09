"""MCP registration for dm preference tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ..agent.message_projection import CONTEXT_VERSION
from .puffo_core_tools import _note_contact


def register_dm_preference_tools(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def get_dm_allowlists() -> dict[str, Any]:
        """List your DM allowlist (peers whose DMs skip the approval
        gate). Per-agent — every identity keeps its own list; this
        reads yours only."""
        data = await cfg.http_client.get("/allowlists")
        slugs = sorted(
            e.get("peer_slug", "")
            for e in (data.get("entries") or [])
            if e.get("peer_slug")
        )
        return {
            "context_version": CONTEXT_VERSION,
            "count": len(slugs),
            "identities": [f"@{slug.lstrip('@')}" for slug in slugs],
        }

    @mcp.tool()
    async def get_dm_blocklists() -> dict[str, Any]:
        """List your DM blocklist (senders whose messages are silently
        dropped). Per-agent — every identity keeps its own list; this
        reads yours only."""
        data = await cfg.http_client.get("/blocklists")
        slugs = sorted(
            b.get("id", "")
            for b in (data.get("blocks") or [])
            if b.get("target") == "user" and b.get("id")
        )
        return {
            "context_version": CONTEXT_VERSION,
            "count": len(slugs),
            "identities": [f"@{slug.lstrip('@')}" for slug in slugs],
        }

    @mcp.tool()
    async def add_dm_allowlist(slug: str) -> str:
        """Allow a user to DM you. Future DMs from this sender skip
        the ``auto_accept_dm`` approval prompt and deliver directly.
        Idempotent: re-allowlisting an entry is a no-op. Per-agent —
        only your own allowlist changes; other agents are unaffected.

        slug: peer to allow (e.g. ``alice-1234``).
        """
        target = (slug or "").strip()
        if not target:
            raise RuntimeError("slug is required")
        await cfg.http_client.post("/allowlists", {"slugs": [target]})
        _note_contact(cfg, target, allowed=True)
        return f"allowlisted {target}"

    @mcp.tool()
    async def update_dm_blocklist(slug: str, on: bool) -> str:
        """Block (``on=True``) or unblock (``on=False``) a sender.
        Server-enforced — blocked senders' messages are silently
        dropped at the server, so you never see them. Per-agent —
        only your own blocklist changes; other agents are unaffected.

        slug: peer to (un)block (e.g. ``alice-1234``).
        on: ``True`` adds to blocklist, ``False`` removes.
        """
        target = (slug or "").strip()
        if not target:
            raise RuntimeError("slug is required")
        if on:
            await cfg.http_client.post(
                "/blocklists",
                {"target": "user", "id": target},
            )
            _note_contact(cfg, target, blocked=True)
            return f"blocked {target}"
        await cfg.http_client.delete(
            "/blocklists",
            body={"id": target},
        )
        _note_contact(cfg, target, blocked=False)
        return f"unblocked {target}"
