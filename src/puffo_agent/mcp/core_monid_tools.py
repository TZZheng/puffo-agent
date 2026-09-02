"""Monid spend MCP tool registration.

One generic tool, ``monid_spend``, lets an agent fetch paid, read-only
external data through Monid with the server as the spend-control middle
layer. The agent never holds the Monid key or money: it forwards to
``POST /v2/monid/spend`` via the native signed client, and the server
discovers a capability, checks the budget, pays Monid, and returns the
result. Step one targets native (key-holding) agents; the keyless bridge
transport is out of scope here (that path is unsigned and could not reach
the subkey-gated spend route).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..crypto.http_client import HttpError

logger = logging.getLogger(__name__)


def _monid_error_message(exc: HttpError) -> str:
    """Pull the server's human-readable ``message`` out of a failed monid
    response body (JSON ``{error, message, input_schema?}``), falling back to a
    terse ``HTTP <status>``. Keeps the tool's error clean instead of a raw blob.

    When the server rejects the ``input`` before spending it returns the
    capability's own ``input_schema``; that is appended so the model can rebuild
    ``input`` to match and retry.
    """
    try:
        parsed = json.loads(exc.body)
        if isinstance(parsed, dict) and parsed.get("message"):
            message = str(parsed["message"])
            schema = parsed.get("input_schema")
            if schema is not None:
                return (
                    f"{message}\nRebuild `input` to match this schema and call "
                    f"again:\n{json.dumps(schema, ensure_ascii=False)}"
                )
            return message
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return f"HTTP {exc.status}"


def register_monid_tools(mcp: FastMCP, cfg: Any) -> None:
    # Native-only tool. A keyless (T23 bridge) agent authenticates via the
    # unsigned `/v2/cloud-agents/*` proxy and holds no subkey, so it cannot
    # reach the subkey-gated `/v2/monid/spend`. Rather than expose a tool that
    # would only ever error there (and to avoid opening any second auth path),
    # it is simply not registered for keyless agents — the same conditional-
    # registration pattern `register_core_tools` uses for the bridge lifecycle
    # tools. Cloud/keyless Monid is out of scope for step one.
    if getattr(cfg, "keyless", False):
        return

    @mcp.tool()
    async def monid_spend(
        query: str,
        input: Optional[dict[str, Any]] = None,
        max_cost_micro: int = 0,
        limit: int = 5,
        idempotency_key: str = "",
    ) -> str:
        """Fetch paid, read-only external data through Monid.

        Puffo is the middle layer: it holds the Monid key, checks your
        spend budget, pays Monid, and returns the result. You never see the
        key and never hold money. The spend comes out of the shared Monid
        balance, and your operator must have enabled Monid for you and set a
        cap first (otherwise this returns a "not enabled" error).

        This tool is the ONLY way to reach Monid. Do NOT install or run a
        Monid CLI, and never ask anyone for — or try to hold — your own Monid
        API key: you neither have nor need one, the server holds it. If a call
        fails, fix your `input` or ask the operator to raise the cap; do not
        route around this tool.

        Args:
            query: What data you want, in natural language. Puffo discovers a
                matching read-only capability for it.
            input: The parameters for the chosen capability, in Monid's run
                envelope: wrap them under `body` (Puffo forwards `input`
                verbatim), i.e. `{"body": { ... }}`. For example, an Amazon
                search takes an `input` array of query objects:
                `{"body": {"input": [{"keyword": "Men Shoes", "domainCode":
                "com", "sortBy": "recent", "maxPages": 1, "category":
                "aps"}]}}`. If the shape does not match the capability's schema,
                the error returns that schema — rebuild `input` to match it and
                call again.
            max_cost_micro: Your hard ceiling for THIS one call, in
                micro-dollars (1_000_000 = $1). Must be positive. If the
                quoted price is above it, the call is rejected before any
                money is spent.
            limit: How many candidate capabilities to consider (1-25).
            idempotency_key: Optional. Pass a stable value to make a retried
                call safe — it reconciles the earlier attempt instead of
                paying a second time.

        Returns the provider's result and what the call cost.
        """
        if not query.strip():
            raise RuntimeError("query is required")
        if max_cost_micro <= 0:
            raise RuntimeError(
                "max_cost_micro must be a positive integer in micro-dollars "
                "(1_000_000 = $1)"
            )

        body: dict[str, Any] = {
            "query": query,
            "input": input if input is not None else {},
            "max_cost_micro": max_cost_micro,
            "limit": limit,
        }
        if idempotency_key:
            body["idempotency_key"] = idempotency_key

        try:
            data = await cfg.http_client.post("/v2/monid/spend", body)
        except HttpError as exc:
            raise RuntimeError(
                f"monid spend failed: {_monid_error_message(exc)}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected monid response: {data!r}")

        # A 202 is not an HTTP error to the client, but it means the spend is
        # still resolving upstream and an owner will reconcile it — there is no
        # result yet, so report that rather than a settled cost.
        if data.get("error") == "PENDING_RECONCILE" or (
            "cost_micro" not in data and data.get("ledger_id")
        ):
            return (
                "monid spend is still resolving upstream; your operator will "
                f"reconcile it (ledger {data.get('ledger_id', '?')}). No result yet."
            )

        cost = data.get("cost_micro")
        status = data.get("provider_http_status")
        output = data.get("output")
        header = f"monid spend settled: cost {cost} micro-dollars"
        if status is not None:
            header += f", provider status {status}"
        return header + "\nresult:\n" + json.dumps(output, indent=2, ensure_ascii=False)
