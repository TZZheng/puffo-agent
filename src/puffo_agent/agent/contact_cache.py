"""The agent's own DM contact cache (allowlist + blocklist), hydrated
from puffo-server. Per-agent - the server scopes both lists to the
authenticated identity. Single read/write point for every allow/block
decision - never hit /allowlists + /blocklists ad hoc.
"""

from __future__ import annotations

import time
from typing import Any


class ContactCache:
    def __init__(
        self,
        http_client: Any,
        log: Any,
        *,
        ttl: float = 300.0,
        miss_refresh_interval: float = 15.0,
    ):
        self._http = http_client
        self._log = log
        self._ttl = ttl
        self._miss_refresh_interval = miss_refresh_interval
        self._allow: set[str] = set()
        self._block: set[str] = set()
        self._fetched_at: float = 0.0

    async def refresh(self) -> None:
        """Replace both sets, preserving stale data when refresh fails."""
        try:
            allow = await self._http.get("/allowlists")
            block = await self._http.get("/blocklists")
        except Exception as exc:  # noqa: BLE001
            self._log.warning("contact_cache: refresh failed: %s", exc)
            return
        self._allow = {
            entry.get("peer_slug", "")
            for entry in (allow.get("entries") or [])
        } - {""}
        self._block = {
            entry.get("id", "")
            for entry in (block.get("blocks") or [])
            if entry.get("target") == "user"
        } - {""}
        self._fetched_at = time.monotonic()

    def _age(self) -> float:
        if not self._fetched_at:
            return float("inf")
        return time.monotonic() - self._fetched_at

    async def _maybe_refresh(self, *, on_miss: bool) -> None:
        age = self._age()
        if age >= self._ttl:
            await self.refresh()
        elif on_miss and age >= self._miss_refresh_interval:
            await self.refresh()

    async def is_allowed(self, slug: str) -> bool:
        if not slug:
            return False
        await self._maybe_refresh(on_miss=slug not in self._allow)
        return slug in self._allow

    async def is_blocked(self, slug: str) -> bool:
        if not slug:
            return False
        await self._maybe_refresh(on_miss=False)
        return slug in self._block

    def note_allowed(self, slug: str) -> None:
        if slug:
            self._allow.add(slug)

    def note_blocked(self, slug: str, blocked: bool) -> None:
        if not slug:
            return
        if blocked:
            self._block.add(slug)
        else:
            self._block.discard(slug)
