"""Monid spend MCP tool registration.

Two generic tools let an agent fetch paid, read-only external data through
Monid with the server as the spend-control middle layer:

* ``monid_prepare`` — FREE. Find a capability for what you want and get its
  input schema, an example, and the price. No charge.
* ``monid_spend`` — PAID. Run the capability you prepared, with the ``input``
  you built from its schema.

The agent never holds the Monid key or money: it forwards to the server via the
native signed client, and the server discovers/inspects a capability, checks the
budget, pays Monid, and returns the result. This two-step flow is what lets one
generic tool reach any of Monid's endpoints — the agent reads each capability's
own schema instead of us hardcoding a template per endpoint. Step one targets
native (key-holding) agents; the keyless bridge transport is out of scope here
(that path is unsigned and could not reach the subkey-gated routes).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..crypto.http_client import HttpError

logger = logging.getLogger(__name__)

# Provenance is mandatory. A successful `monid_spend` result is stamped as
# Monid-sourced (paid, real) so the model can attribute it; but only the model
# writes the final answer, so on any failure we tell it that if it falls back to
# its own knowledge or the web it MUST label that as non-Monid — never pass it
# off as Monid data. This is prompt-level steering, not a hard block.
_LABEL_NON_MONID = (
    "If you answer from your own knowledge or the web instead, you MUST clearly "
    "label it as NOT a Monid result — never present non-Monid data as Monid data."
)


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
    # Native-only tools. A keyless (T23 bridge) agent authenticates via the
    # unsigned `/v2/cloud-agents/*` proxy and holds no subkey, so it cannot
    # reach the subkey-gated `/v2/monid/*` routes. Rather than expose tools that
    # would only ever error there (and to avoid opening any second auth path),
    # they are simply not registered for keyless agents — the same conditional-
    # registration pattern `register_core_tools` uses for the bridge lifecycle
    # tools. Cloud/keyless Monid is out of scope for step one.
    if getattr(cfg, "keyless", False):
        return
    _register_monid_prepare(mcp, cfg)
    _register_monid_spend(mcp, cfg)


def _register_monid_prepare(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def monid_prepare(query: str, limit: int = 5) -> str:
        """Find a Monid capability for the data you want, and see how to call it.

        FREE — this only looks things up, it does not fetch data or spend any
        money. Always call this BEFORE `monid_spend`: it tells you which
        capability to run and exactly how to shape its `input`.

        Say what you want in `query` (natural language). The result gives you:

        - `provider` and `endpoint` — pass these straight to `monid_spend`.
        - `price` — the price model and quoted amount (in micro-dollars).
        - `input` — the run-input schema. Monid wraps a run's input in one or
          more named envelopes: `body`, `queryParams`, and/or `pathParams`.
          Whichever the capability declares is here, each a JSON schema with a
          `description` per field (and sometimes a filled-in example). Build
          your `input` by filling those exact envelope(s) — e.g. if the schema
          is under `queryParams`, send `{"queryParams": { ...your values... }}`.
        - `description` — extra guidance on what a field expects.

        Args:
            query: What data you want, in natural language.
            limit: How many candidate capabilities to consider (1-25).

        Returns the prepared capability as JSON. If nothing matches, this
        errors — the data is not available through Monid (the capability may
        not exist, or is not one Puffo allows). You may still answer the user
        from your own knowledge or the web, but you MUST clearly label that as
        NOT a Monid result; never imply non-Monid data came from Monid.
        """
        if not query.strip():
            raise RuntimeError("query is required")

        try:
            data = await cfg.http_client.post(
                "/v2/monid/prepare", {"query": query, "limit": limit}
            )
        except HttpError as exc:
            # A prepare failure (no capability matched, or a transient upstream
            # error) means the data was not reached through Monid, so it must not
            # be passed off as a Monid result. Answering from elsewhere is allowed
            # as long as it is labeled non-Monid.
            raise RuntimeError(
                f"monid prepare failed: {_monid_error_message(exc)}\n"
                f"Couldn't retrieve this via Monid. {_LABEL_NON_MONID}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected monid response: {data!r}")
        return json.dumps(data, indent=2, ensure_ascii=False)


def _register_monid_spend(mcp: FastMCP, cfg: Any) -> None:
    @mcp.tool()
    async def monid_spend(
        provider: str,
        endpoint: str,
        input: dict[str, Any],
        max_cost_micro: int,
        idempotency_key: str = "",
    ) -> str:
        """Run a Monid capability you prepared, and pay for the data — PAID.

        Puffo is the middle layer: it holds the Monid key, checks your spend
        budget, pays Monid, and returns the result. You never see the key and
        never hold money. The spend comes out of the shared Monid balance, and
        your operator must have enabled Monid for you and set a cap first.

        This tool is the ONLY way to reach Monid: never install or run a Monid
        CLI, and never ask for or hold your own Monid key (the server holds it).

        Call `monid_prepare` FIRST to get `provider`, `endpoint`, and the input
        schema. Then:

        Args:
            provider: From `monid_prepare` — the capability's provider.
            endpoint: From `monid_prepare` — the capability's endpoint.
            input: The run payload you built from the prepared schema, in Monid's
                envelope shape: fill the envelope(s) the schema declared, i.e.
                `{"body": {...}}` and/or `{"queryParams": {...}}` and/or
                `{"pathParams": {...}}`. If the shape does not match, the error
                returns that schema — rebuild `input` to match it and call again.
            max_cost_micro: Your hard ceiling for THIS one call, in
                micro-dollars (1_000_000 = $1). Must be positive. If the quoted
                price is above it, the call is rejected before any money is spent.
            idempotency_key: Optional. Pass a stable value to make a retried call
                safe — it reconciles the earlier attempt instead of paying twice.

        Returns the provider's result and what the call cost, stamped
        `via Monid · <provider>/<endpoint> · <cost>` — mark data you got this
        way as Monid-sourced. Anything you instead answer from your own
        knowledge or the web MUST be labeled as NOT a Monid result; never
        present non-Monid data as a Monid result.
        """
        if not provider.strip() or not endpoint.strip():
            raise RuntimeError(
                "provider and endpoint are required (from monid_prepare)"
            )
        if max_cost_micro <= 0:
            raise RuntimeError(
                "max_cost_micro must be a positive integer in micro-dollars "
                "(1_000_000 = $1)"
            )

        body: dict[str, Any] = {
            "provider": provider,
            "endpoint": endpoint,
            "input": input if input is not None else {},
            "max_cost_micro": max_cost_micro,
        }
        if idempotency_key:
            body["idempotency_key"] = idempotency_key

        try:
            data = await cfg.http_client.post("/v2/monid/spend", body)
        except HttpError as exc:
            # A spend failure is usually a retryable input/schema mismatch — the
            # error carries the schema to rebuild `input` and retry, so try that
            # first. The label rule is the fallback: if you give up and answer
            # from elsewhere, it must be marked non-Monid.
            raise RuntimeError(
                f"monid spend failed: {_monid_error_message(exc)}\n{_LABEL_NON_MONID}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected monid response: {data!r}")
        return _format_spend_result(data)


def _format_spend_result(data: dict[str, Any]) -> str:
    """Render a `/v2/monid/spend` response for the model.

    A 202 is not an HTTP error to the client, but it means the spend is still
    resolving upstream and an owner will reconcile it — there is no result yet,
    so report that rather than a settled cost.
    """
    if data.get("error") == "PENDING_RECONCILE" or (
        "cost_micro" not in data and data.get("ledger_id")
    ):
        return (
            "monid spend is still resolving upstream; your operator will "
            f"reconcile it (ledger {data.get('ledger_id', '?')}). No result yet."
        )

    provider = data.get("provider", "?")
    endpoint = data.get("endpoint", "?")
    cost = data.get("cost_micro")
    status = data.get("provider_http_status")
    output = data.get("output")
    # Provenance stamp: this is paid, real Monid data — so the model can attribute
    # its source to the user and never conflate it with its own/web answers.
    header = f"via Monid · {provider}{endpoint} · cost {cost} micro-dollars"
    if status is not None:
        header += f", provider status {status}"
    return header + "\nresult:\n" + json.dumps(output, indent=2, ensure_ascii=False)
