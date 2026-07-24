"""MCP-side client for the daemon's ``rpc_service``. Host writes go through
the daemon for single-writer semantics; cli-docker reaches the daemon via
``host.docker.internal``."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class PuffoRpcClient:
    """Async client for the daemon's loopback RPC service.
    Transport failures + non-2xx responses raise ``RuntimeError``."""

    def __init__(self, base_url: str, agent_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            # Match the bare-address repr aiohttp gc-emits on a leak.
            logger.info(
                "aiohttp ClientSession created (class=PuffoRpcClient "
                "base_url=%s agent_id=%s)",
                self.base_url, self.agent_id,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _post(self, route: str, body: dict[str, Any]) -> str:
        """POST + return the ``message`` field. Raises on transport or non-2xx."""
        path = (
            f"/v1/rpc/{urllib.parse.quote(self.agent_id, safe='')}/"
            f"{route.lstrip('/')}"
        )
        url = f"{self.base_url}{path}"
        session = await self._get_session()
        try:
            async with session.post(url, json=body) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    text = await resp.text()
                    raise RuntimeError(
                        f"rpc {route} returned non-JSON body "
                        f"(status {resp.status}): {text[:500]}"
                    )
                if resp.status >= 400:
                    err = (
                        data.get("error")
                        if isinstance(data, dict) else None
                    )
                    raise RuntimeError(
                        err or f"rpc {route} failed with status {resp.status}"
                    )
                msg = (
                    data.get("message") if isinstance(data, dict) else None
                )
                if not isinstance(msg, str):
                    raise RuntimeError(
                        f"rpc {route} returned a JSON object without a "
                        f"`message` string field"
                    )
                return msg
        except aiohttp.ClientError as exc:
            raise RuntimeError(
                f"rpc {route} transport error: {exc}"
            ) from exc

    async def install_mcp(
        self,
        *,
        name: str,
        template_id: str = "",
        spec: Optional[dict[str, Any]] = None,
    ) -> str:
        return await self._post(
            "install-mcp",
            {"name": name, "template_id": template_id, "spec": spec},
        )

    async def sync_mcp(self, *, template_id: str) -> str:
        return await self._post(
            "sync-mcp", {"template_id": template_id},
        )

    async def request_leave(
        self,
        *,
        kind: str,
        space_id: str,
        channel_id: str = "",
        reason: str = "",
    ) -> str:
        return await self._post(
            "leave-request",
            {
                "kind": kind,
                "space_id": space_id,
                "channel_id": channel_id,
                "reason": reason,
            },
        )

    # ── PUF-394: persistent scheduler ────────────────────────────────

    async def cron_create(
        self, *, name: str, cron_expr: str, prompt: str, channel_id: str = "",
    ) -> str:
        return await self._post(
            "cron-create",
            {"name": name, "cron_expr": cron_expr, "prompt": prompt,
             "channel_id": channel_id},
        )

    async def cron_list(self) -> str:
        return await self._post("cron-list", {})

    async def cron_update(
        self,
        *,
        job_id: str,
        name: Optional[str] = None,
        cron_expr: Optional[str] = None,
        prompt: Optional[str] = None,
        channel_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> str:
        return await self._post(
            "cron-update",
            {"job_id": job_id, "name": name, "cron_expr": cron_expr,
             "prompt": prompt, "channel_id": channel_id, "enabled": enabled},
        )

    async def cron_delete(self, *, job_id: str) -> str:
        return await self._post("cron-delete", {"job_id": job_id})
