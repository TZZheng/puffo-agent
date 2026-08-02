"""MCP tools for puffo-core: signed API + E2E encrypted messages.

Wire calls follow puffo-cli's conventions: ``/certs/sync`` for
device certs, ``/spaces/<sp>/channels/<ch>/members`` for channel
members, event-stream replay for channel discovery. Host-side /
local tools live in ``host_tools.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..crypto.attachments import (
    ATTACHMENT_CONTENT_TYPE,
    AttachmentMeta,
    encrypt_attachment,
)
from ..crypto.encoding import base64url_decode, base64url_encode
from ..crypto.http_client import PuffoCoreHttpClient
from ..crypto.keystore import KeyStore, decode_secret
from ..crypto.message import (
    build_plaintext_message,
    EncryptInput,
    RecipientDevice,
    build_supplementation_envelope,
    encrypt_message,
    encrypt_message_with_content_key,
)
from ..crypto.primitives import Ed25519KeyPair
from ..limits import MESSAGE_SEGMENT_CHARS
from ..agent.context_controller import MODEL_VISIBLE_READ_RECEIPT_PREFIX
from ..agent.message_projection import format_message_group, target_label
from ..agent._logging import log_runtime_event
from ..agent.send_coordinator import (
    SemanticSendRequest,
    SendCoordinator,
    failed_result,
)
from .data_client import DataClient, DataNotFound
from ._host_mcp import PuffoRpcClient

logger = logging.getLogger(__name__)


def _history_text(content: Any) -> str:
    """Render the human text/caption portion of stored structured content."""
    if isinstance(content, dict):
        value = content.get("text")
        if isinstance(value, str) and value:
            return value
        caption = content.get("caption")
        return caption if isinstance(caption, str) else ""
    return content if isinstance(content, str) else ""


async def _send_encryption_required(cfg, resolved_root):
    """Daemon-level send-mode decision. Data-client shims without the
    method (older harnesses) fail safe to E2EE."""
    getter = getattr(cfg.data_client, "get_send_encryption", None)
    if getter is None:
        return True
    return await getter(cfg.slug, resolved_root or None)


async def _resolve_channel_space(cfg: Any, channel_id: str) -> str:
    """Resolve ``channel_id`` → ``space_id`` from the local cache.

    The cache is filled by ``puffo_core_client._handle_event`` for
    every membership event the agent receives that carries both ids
    (``invite_to_channel`` where we're the invitee,
    ``accept_channel_invite`` where we're the signer, and
    ``create_channel`` — server only fans those to space members).
    ``mark_channel_space`` is also written synchronously inside
    ``_accept_invite`` to close the WS-echo race.

    Raises ``RuntimeError`` (which propagates to the LLM as an MCP
    tool error) on miss — that's the signal "agent has no way to
    reach this channel; you may not be a member, or the id is
    wrong." Earlier code walked ``GET /spaces`` as a fallback
    resolver, but with events feeding the cache that fallback is
    redundant and silently misleading (a hit there only proved
    access to the space, not membership in the channel).
    """
    space_id = await cfg.data_client.lookup_channel_space(channel_id)
    if not space_id:
        # Non-``ch_`` miss is almost always a bare user slug — hint at
        # the DM path instead of the membership-flavoured error below.
        if not channel_id.startswith("ch_"):
            raise RuntimeError(
                f"'{channel_id}' is not a channel id (channel ids "
                f"start with 'ch_'). If it's a user slug, prepend "
                f"'@' to DM them: send_message(channel='@{channel_id}', "
                f"...); to read a DM conversation use "
                f"get_dm_history(peer='{channel_id}'). To find a "
                f"channel id, call list_channels_in_all_spaces."
            )
        raise RuntimeError(
            f"agent has no record of channel {channel_id} — either it "
            f"isn't a channel the agent belongs to, the id is wrong, or "
            f"you were just added and the membership hasn't propagated to "
            f"this agent yet (retrying shortly resolves that last case). "
            f"Call list_channels_in_all_spaces to see the channels the "
            f"agent can currently reach."
        )
    return space_id


def _ts_to_iso(ms: int) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _enc_tag(m: Any) -> str:
    """`[encrypted]`/`[plaintext]` tag; legacy rows default to encrypted."""
    return "[encrypted]" if getattr(m, "is_encrypted", True) else "[plaintext]"


async def _stage_model_visible_messages(
    cfg: "PuffoCoreToolsConfig",
    messages: list[Any],
    *,
    tool_name: str,
    tool_arguments: dict[str, object],
) -> str:
    """Stage the highest channel watermark returned to the provider."""
    rpc = getattr(cfg, "rpc_client", None)
    if rpc is None:
        log_runtime_event(
            logger,
            "history.read_staged",
            level=logging.DEBUG,
            agent_id=cfg.agent_id,
            agent_slug=cfg.slug,
            state="unsupported_adapter",
        )
        return ""
    candidates = [
        message
        for message in messages
        if getattr(message, "envelope_kind", "") != "dm"
        and getattr(message, "space_id", None)
        and getattr(message, "channel_id", None)
        and isinstance(getattr(message, "server_seq", None), int)
        and not isinstance(getattr(message, "server_seq", None), bool)
    ]
    if not candidates:
        state = (
            "dm_unsupported"
            if any(getattr(message, "envelope_kind", "") == "dm" for message in messages)
            else "unsequenced"
            if any(getattr(message, "server_seq", None) is None for message in messages)
            else "unsupported_history"
        )
        log_runtime_event(
            logger,
            "history.read_staged",
            level=logging.DEBUG,
            agent_id=cfg.agent_id,
            agent_slug=cfg.slug,
            state=state,
        )
        return ""
    watermark = max(candidates, key=lambda message: message.server_seq)
    if any(
        message.space_id != watermark.space_id
        or message.channel_id != watermark.channel_id
        for message in candidates
    ):
        # One continuation has one channel watermark; do not claim partial
        # visibility for a mixed-channel presentation.
        return ""
    # Only stage the exact local rows that are represented in this response.
    visible = [message.envelope_id for message in candidates]
    staged = await rpc.stage_model_visible_read(
        space_id=watermark.space_id,
        channel_id=watermark.channel_id,
        through_seq=watermark.server_seq,
        through_envelope_id=watermark.envelope_id,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        visible_message_ids=visible,
    )
    receipt = staged.get("correlation_receipt")
    if not isinstance(receipt, str) or not receipt:
        raise RuntimeError("model-visible read staging returned no receipt")
    return f"[{MODEL_VISIBLE_READ_RECEIPT_PREFIX}{receipt}]"


# ── transport seam ─────────────────────────────────────────────────
#
# One helper per wire read/write. Each branches on ``cfg.keyless``: the
# keyless (T23 bridge) transport hits the unsigned, token-authed
# ``/v2/cloud-agents/*`` routes (the E2B egress proxy injects
# ``x-sandbox-token``); the native transport keeps the signed keystore
# path byte-for-byte. Kept as module-level ``_read_*``/``_send_*``
# functions to match the existing ``_resolve_channel_space`` /
# ``_fetch_device_keys`` idiom.


async def _read_spaces(cfg: Any) -> Any:
    if cfg.keyless:
        return await cfg.http_client.get_unsigned("/v2/cloud-agents/spaces")
    return await cfg.http_client.get("/spaces")


async def _read_space_channels(cfg: Any, space_id: str) -> Any:
    if cfg.keyless:
        return await cfg.http_client.get_unsigned(
            f"/v2/cloud-agents/spaces/{space_id}/channels"
        )
    return await cfg.http_client.get(f"/spaces/{space_id}/channels")


async def _read_channel_members(
    cfg: Any, space_id: str, channel_id: str,
) -> Any:
    """Read the exact channel roster on both transports."""
    if cfg.keyless:
        return await cfg.http_client.get_unsigned(
            f"/v2/cloud-agents/spaces/{space_id}/channels/{channel_id}/members"
        )
    return await cfg.http_client.get(
        f"/spaces/{space_id}/channels/{channel_id}/members"
    )


async def _read_profiles(cfg: Any, slugs_csv: str) -> Any:
    quoted = urllib.parse.quote(slugs_csv, safe=",")
    if cfg.keyless:
        return await cfg.http_client.get_unsigned(
            f"/v2/cloud-agents/identities/profiles?slugs={quoted}"
        )
    return await cfg.http_client.get(
        f"/identities/profiles?slugs={quoted}"
    )


async def _send_keyless(cfg: Any, body: dict) -> dict:
    return await cfg.http_client.post_unsigned(
        "/v2/cloud-agents/messages", body,
    ) or {}


async def _upload_blob_keyless(cfg: Any, data: bytes) -> dict:
    return await cfg.http_client.post_bytes_unsigned(
        "/v2/cloud-agents/blobs/upload", data,
    ) or {}


@dataclass
class PuffoCoreToolsConfig:
    slug: str
    device_id: str
    keystore: KeyStore
    http_client: PuffoCoreHttpClient
    data_client: DataClient
    agent_id: str = ""
    space_id: Optional[str] = None
    # Workspace root used by ``send_message_with_attachments`` to
    # safety-resolve LLM-supplied relative paths (no ``..`` escape,
    # no absolutes).
    workspace: Optional[str] = None
    # None when PUFFO_RPC_URL isn't set; install/sync tools surface
    # a clear error rather than touching operator files in-process.
    rpc_client: Optional[PuffoRpcClient] = None
    # Set only on the in-process (ws-local) path, where tools run inside
    # the daemon and drive the message client directly instead of via RPC.
    message_client: Any = None
    # Package 4 wires the worker's one persistent instance. Optional keeps
    # older constructors source-compatible; sends fail closed while absent.
    send_coordinator: Any = None
    # T23 keyless bridge transport (``CloudBridgeClient``). Populated only
    # at the in-process ws-local site (from ``client._bridge``); the
    # subprocess/RPC MCP path leaves it None.
    bridge_client: Any = None
    # Live Inbox runtime for in-process tools. Subprocess tools use rpc_client.
    inbox_runtime: Any = None

    @property
    def keyless(self) -> bool:
        """Whether reads and sends use the keyless cloud-agent routes."""
        return getattr(self.http_client, "keyless", False)


async def _dispatch_semantic_send(
    cfg: "PuffoCoreToolsConfig", request: SemanticSendRequest,
    *, tool_name: str = "send_message",
) -> dict[str, Any]:
    coordinator = getattr(cfg, "send_coordinator", None)
    if coordinator is None:
        coordinator = getattr(
            getattr(cfg, "message_client", None), "send_delegate", None
        )
    if coordinator is not None:
        try:
            if hasattr(coordinator, "workspace"):
                coordinator.workspace = cfg.workspace
            result = await coordinator.send(request)
        except Exception as exc:
            return failed_result(
                f"persistent send coordinator failed: {exc}",
                kind="coordinator",
            )
        if isinstance(result, dict):
            result.setdefault("attempted", True)
            runtime = getattr(cfg, "inbox_runtime", None)
            if result.get("state") == "held" and runtime is not None:
                result = await runtime.stage_held_send_result(
                    result,
                    tool_name=tool_name,
                    # Match the public tool's stable semantic core. Optional
                    # defaults are provider-normalized differently, so adding
                    # them here would make a genuine original result ambiguous.
                    tool_arguments=(
                        _held_send_tool_arguments(request, attachments=False)
                        if not request.attachment_paths else _held_send_tool_arguments(
                            request, attachments=True,
                        )
                    ),
                )
            return result
        return failed_result(
            "persistent send coordinator returned a malformed result",
            kind="protocol",
        )


    rpc = getattr(cfg, "rpc_client", None)
    if rpc is not None:
        try:
            result = await rpc.send_message(**request.to_rpc_dict())
        except Exception as exc:
            return failed_result(f"send RPC unavailable: {exc}", kind="rpc_unavailable")
        if isinstance(result, dict):
            result.setdefault("attempted", True)
            return result
        return failed_result("send RPC returned a malformed result", kind="protocol")
    if cfg.keyless:
        coordinator = SendCoordinator(
            slug=cfg.slug,
            keystore=cfg.keystore,
            http_client=cfg.http_client,
            data_client=cfg.data_client,
            workspace=cfg.workspace,
        )
        return await coordinator.send(request)
    return failed_result(
        "persistent send coordinator is unavailable",
        kind="coordinator_unavailable",
    )


def _held_send_tool_arguments(
    request: SemanticSendRequest, *, attachments: bool,
) -> dict[str, Any]:
    """Mirror the actual public call without turning omitted defaults into requirements."""
    arguments: dict[str, Any] = {
        "channel": request.destination,
        **({"caption": request.caption, "paths": list(request.attachment_paths)}
           if attachments else {"text": request.text}),
    }
    if request.root_id:
        arguments["root_id"] = request.root_id
    if request.visibility_level != "default":
        arguments["visibility_level"] = request.visibility_level
    if request.send_anyway:
        arguments["send_anyway"] = True
    return arguments


def _note_contact(
    cfg: "PuffoCoreToolsConfig", slug: str, *,
    allowed: bool = False, blocked: Optional[bool] = None,
) -> None:
    """Reflect an allowlist/blocklist write into the in-process contact
    cache when the tool runs inside the daemon (ws-local). Out-of-process
    runtimes (cli-local) pick it up via the cache's TTL / miss refresh."""
    mc = getattr(cfg, "message_client", None)
    contacts = getattr(mc, "_contacts", None) if mc is not None else None
    if contacts is None:
        return
    if allowed:
        contacts.note_allowed(slug)
    if blocked is not None:
        contacts.note_blocked(slug, blocked)


async def _fetch_device_keys(
    http_client: PuffoCoreHttpClient,
    slugs: list[str],
) -> list[RecipientDevice]:
    """Paginate ``/certs/sync?slugs=...`` and collect
    ``(device_id, kem_pk)`` for every returned device_cert.
    """
    if not slugs:
        return []
    slugs_param = ",".join(slugs)
    devices: list[RecipientDevice] = []
    seen_ids: set[str] = set()
    since = 0
    while True:
        data = await http_client.get(
            f"/certs/sync?slugs={slugs_param}&since={since}"
        )
        for entry in data.get("entries", []):
            if entry.get("kind") == "device_cert":
                cert = entry.get("cert", {})
                dev_id = cert.get("device_id", "")
                # v2 nests under ``keys.encryption.public_key``; fall
                # back to the v1 flat field for legacy entries.
                keys_block = cert.get("keys") or {}
                enc_block = keys_block.get("encryption") or {}
                kem_b64 = enc_block.get("public_key") or cert.get("kem_public_key", "")
                if dev_id and kem_b64 and dev_id not in seen_ids:
                    try:
                        devices.append(RecipientDevice(
                            device_id=dev_id,
                            kem_public_key=base64url_decode(kem_b64),
                        ))
                        seen_ids.add(dev_id)
                    except Exception:
                        # Skip malformed entry; don't abort the fetch.
                        pass
            since = entry.get("seq", since)
        if not data.get("has_more"):
            break
    return devices


async def _supplement_missing_devices(
    http_client: PuffoCoreHttpClient,
    envelope: dict,
    content_key: bytes,
    recipient_slugs: list[str],
    missing_device_ids: list[str],
) -> None:
    """Best-effort: re-fetch certs, build a same-``envelope_id``
    envelope for the missing device_ids, POST. Logs + swallows on
    failure (original send is already durable)."""
    envelope_id = envelope.get("envelope_id", "?")
    try:
        fresh = await _fetch_device_keys(http_client, recipient_slugs)
        wanted = set(missing_device_ids)
        supp_devices = [d for d in fresh if d.device_id in wanted]
        if not supp_devices:
            logger.warning(
                "supplementation: server reported %d missing device(s) "
                "for %s but fresh /certs/sync returned none of them; "
                "those devices won't receive this message",
                len(missing_device_ids), envelope_id,
            )
            return
        supp_env = build_supplementation_envelope(
            envelope, content_key, supp_devices,
        )
        await http_client.post("/messages", supp_env)
        logger.debug(
            "supplementation: re-posted %s to %d device(s)",
            envelope_id, len(supp_devices),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "supplementation: failed for %s: %s", envelope_id, exc,
        )


from ..agent._visibility import resolve_visibility as _resolve_visibility


_RESOLVE_ROOT_MAX_DEPTH = 8


async def _resolve_outgoing_root(
    root_id: str,
    data_client: Any,
    *,
    self_slug: str,
    channel_id: Optional[str],
    space_id: Optional[str],
    dm_peer: Optional[str],
) -> tuple[Optional[str], str]:
    """Resolve the agent-supplied ``root_id`` to a server-valid thread
    root for an outbound send. Returns ``(root_or_None, note)``:

      - daemon-local system envelope (sender ``system`` / self-referencing
        root) -> ``None``: the server has no such row, so the message goes
        out as a new top-level root;
      - reference from another channel/DM -> raises ``RuntimeError`` so
        the agent can correct itself;
      - same-scope reply -> walks up and returns the true root;
      - not in the local store / lookup failure -> ``None`` + note (agents
        may only thread under roots they hold locally);
      - cycle / over-deep chain (corrupt data) -> original id + warning.
    """
    if not root_id.strip():
        return None, ""

    current = root_id
    seen: set[str] = set()
    walked = False
    cycle = False
    for _ in range(_RESOLVE_ROOT_MAX_DEPTH):
        if current in seen:
            cycle = True
            break
        seen.add(current)
        try:
            msg = await data_client.get_message_by_envelope(current)
        except DataNotFound:
            msg = None
        except Exception as exc:
            logger.warning(
                "resolve_outgoing_root: wiped %s — lookup transport error: %s",
                root_id, exc,
            )
            return None, (
                f"\nnote: thread_root_id {root_id} could not be verified "
                "(local cache lookup failed); sent as top-level."
            )
        if msg is None:
            logger.info(
                "resolve_outgoing_root: wiped %s — %s not in local cache",
                root_id, current,
            )
            return None, (
                f"\nnote: thread_root_id {root_id} not in local cache; "
                "sent as top-level. Agents can only reply in threads "
                "whose root is in their own local message store."
            )
        # Cross-scope first: rejecting before the system/self-ref wipe
        # keeps misdirected content out of the wrong conversation.
        if dm_peer is not None:
            kind = getattr(msg, "envelope_kind", None)
            peer = (
                getattr(msg, "recipient_slug", None)
                if getattr(msg, "sender_slug", None) == self_slug
                else getattr(msg, "sender_slug", None)
            )
            if kind != "dm" or peer != dm_peer:
                logger.info(
                    "resolve_outgoing_root: rejected %s — not part of the "
                    "DM with %s (kind=%r peer=%r)",
                    root_id, dm_peer, kind, peer,
                )
                raise RuntimeError(
                    f"thread_root_id {root_id} does not belong to this DM "
                    f"with @{dm_peer}; pass a root from this conversation "
                    "or omit root_id to start a new thread."
                )
        else:
            msg_space = getattr(msg, "space_id", None)
            if msg.channel_id != channel_id or (
                space_id and msg_space and msg_space != space_id
            ):
                logger.info(
                    "resolve_outgoing_root: rejected %s — belongs to "
                    "channel %r, outbound is %r",
                    root_id, msg.channel_id, channel_id,
                )
                raise RuntimeError(
                    f"thread_root_id {root_id} belongs to channel "
                    f"{msg.channel_id!r}, not this send's channel "
                    f"{channel_id!r}; pass a root from the current channel "
                    "or omit root_id to start a new thread."
                )
        parent_root = getattr(msg, "thread_root_id", None)
        if getattr(msg, "sender_slug", None) == "system" or parent_root == current:
            # Daemon-minted system envelopes have no server row — threading
            # under them would dangle. Send as a new root instead.
            logger.info(
                "resolve_outgoing_root: wiped %s — daemon-local system thread",
                root_id,
            )
            return None, (
                f"\nnote: thread_root_id {root_id} refers to a local "
                "system message that doesn't exist on the server; sent "
                "as a new top-level message."
            )
        if parent_root is None:
            if walked:
                return current, (
                    f"\nnote: root_id {root_id} was a reply, not a root — "
                    f"auto-corrected to {current}. Pass the metadata "
                    "block's thread_root_id, not post_id."
                )
            return current, ""
        walked = True
        current = parent_root

    reason = (
        "cycle detected in thread chain"
        if cycle
        else f"chain deeper than {_RESOLVE_ROOT_MAX_DEPTH} levels"
    )
    return root_id, (
        f"\nnote: could not resolve root_id {root_id} to a true thread "
        f"root ({reason}); sent as-is. The relay's thread chain looks "
        "corrupt — please flag this to the operator and pass the "
        "metadata block's thread_root_id directly."
    )


def register_core_tools(mcp: FastMCP, cfg: PuffoCoreToolsConfig) -> None:

    @mcp.tool()
    async def read_inbox(
        target: str = "",
        cursor: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Read one stable pending Inbox page.

        target: optional canonical ``channel:<space>:<channel>[:thread:<root>]``
            or ``dm:<peer>`` projection.
        cursor: opaque cursor returned by the preceding page.
        limit: page size from 1 through 50. Continue with ``next_cursor``;
            there is no total read-depth cap.
        result: content-bearing results include the exact pending ``messages``
            page plus bounded, read-only ``prior_context`` blocks from the
            same conversation route(s), strictly earlier than that page;
            prior rows are not admitted or acknowledged. Each message block's
            ``is_self`` metadata is true only when its durable ``sender_slug``
            matches this Agent's current runtime identity; a true prior row is
            evidence of an earlier contribution. If that contribution already
            completed the same originating assignment and the new page adds no
            unresolved action for this Agent, choose ``[SILENT]``. Send again
            for a follow-up, correction, direct mention, newly exposed
            dependency, or otherwise changed assignment.
            Ordinary peer progress on an unchanged originating intent does not
            by itself reopen an ``is_self: true`` contribution that already
            completed this Agent's part. Peer-exposed work or an evolving
            assignment permits another response only when newly observed
            content creates or changes an unresolved obligation belonging to
            this Agent. A genuine follow-up, correction, direct mention,
            newly exposed dependency, otherwise changed work, including
            genuine new peer-exposed work (new work exposed by peer progress),
            can create real work and permit another reply. Treat a changed
            objective, changed scope, changed constraint, or changed deliverable
            as genuine evolution.
            Multi-target turns must address destinations explicitly or stay
            silent. Keep the final send-or-silence decision model-owned: choose
            a send or ``[SILENT]`` from current evidence.
        """
        arguments: dict[str, Any] = {}
        if target:
            arguments["target"] = target
        if cursor:
            arguments["cursor"] = cursor
        if limit != 50:
            arguments["limit"] = limit
        runtime = getattr(cfg, "inbox_runtime", None)
        if runtime is None:
            runtime = getattr(getattr(cfg, "message_client", None), "global_runtime", None)
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
        receipt = result.pop("correlation_receipt", "")
        if receipt:
            result["admission_receipt"] = (
                f"[{MODEL_VISIBLE_READ_RECEIPT_PREFIX}{receipt}]"
            )
        return result

    @mcp.tool()
    async def whoami() -> str:
        """Return your own identity: display name, slug, device_id, and
        subkey info."""
        if cfg.keyless:
            # T23 keyless bridge transport: the sandbox holds no local
            # keystore, so build identity from the config instead of
            # ``load_identity``/``load_session``. display_name resolves
            # over the unsigned profiles route; the signing subkey is
            # managed server-side.
            lines = []
            try:
                data = await _read_profiles(cfg, cfg.slug)
                profiles = (
                    data.get("profiles", []) if isinstance(data, dict) else []
                )
                display_name = (
                    (profiles[0].get("display_name") or "").strip()
                    if profiles else ""
                )
                if display_name:
                    lines.append(f"display_name: {display_name}")
            except Exception as exc:
                logger.warning(
                    "whoami: failed to fetch own display_name: %s", exc,
                )
            lines += [
                f"slug:      {cfg.slug}",
                f"device_id: {cfg.device_id}",
                f"server:    {cfg.http_client.server_url}",
                "subkey:    (managed server-side; keyless transport)",
            ]
            return "\n".join(lines)

        identity = cfg.keystore.load_identity(cfg.slug)
        lines = []
        # display_name lives on the server (the local keystore only has
        # the slug); fetch best-effort so whoami still works if offline.
        try:
            data = await cfg.http_client.get(
                "/identities/profiles?slugs="
                f"{urllib.parse.quote(cfg.slug, safe='')}"
            )
            profiles = data.get("profiles", []) if isinstance(data, dict) else []
            display_name = (
                (profiles[0].get("display_name") or "").strip()
                if profiles else ""
            )
            if display_name:
                lines.append(f"display_name: {display_name}")
        except Exception as exc:
            logger.warning("whoami: failed to fetch own display_name: %s", exc)
        lines += [
            f"slug:      {identity.slug}",
            f"device_id: {identity.device_id}",
            f"server:    {identity.server_url}",
        ]
        try:
            sess = cfg.keystore.load_session(cfg.slug)
            lines.append(f"subkey_id: {sess.subkey_id}")
            lines.append(f"expires:   {_ts_to_iso(sess.expires_at)}")
        except FileNotFoundError:
            lines.append("subkey:    (no active session)")
        return "\n".join(lines)

    @mcp.tool()
    async def send_message(
        channel: str,
        text: str,
        root_id: str = "",
        visibility_level: str = "default",
        send_anyway: bool = False,
    ) -> dict[str, Any]:
        """Post a message to a Puffo.ai channel or DM a user.

        channel: '@<slug>' for a DM (e.g. '@alice-1234'), or a raw
            channel id (e.g. 'ch_<uuid>'). Use
            ``list_channels_in_all_spaces`` (or ``list_spaces`` +
            ``list_channels_in_space``) to discover ids — '#name'
            shortcuts are not supported.
        text: message body. Markdown preserved verbatim.
        root_id: optional — reply inside a thread; pass the
            envelope_id of the message you're replying to. Non-root
            ids auto-correct to their thread root; roots from other
            channels/DMs are rejected; replies to daemon-local system
            messages go out as new top-level posts.
        visibility_level: one of ``"human"`` | ``"default"`` |
            ``"agent_only"`` (default: ``"default"``).
            - ``"human"`` — anything a person should read (replies,
              status updates, operator pings). Sent visible.
            - ``"default"`` — agent-to-agent chatter human clients
              fold away. Sent hidden BUT with safety-net floors: DMs
              and messages whose text @-mentions a human are forced
              visible with a note explaining why. Root-level (non-
              threaded) posts are also forced visible because they
              can't fold in the UI.
            - ``"agent_only"`` — you're explicitly telling the daemon
              this is agent-to-agent traffic; the DM / @-mention
              safety net is skipped. Use only when you're confident
              no human is waiting for this reply. Root-level posts
              are still forced visible (can't fold either way).
        send_anyway: for a channel send, keep the chosen content even
            after a prior held result and a correlated same-Turn read has
            admitted context through the held watermark. Content revised or
            derived from conversation progress is context-dependent and must
            use normal freshness. After a held send of that content, reread
            the relevant route and make a fresh send-or-silence decision
            before sending. Switching targets does not make that content
            context-independent. For context-dependent content, actual newer
            message content must be successfully synchronized, returned and
            inspected before choosing ``send_anyway``. A newer watermark or
            sequence advance alone, an empty recovery/read result, or a
            failed history lookup is not semantic inspection. Do not infer
            unseen content or force a stale context-dependent draft. After
            the existing same-turn held/read technical eligibility checks,
            explicit ``send_anyway=True`` remains available as a model-owned
            choice only when the chosen content is genuinely
            context-independent; this is a deliberate judgment owned by the
            model, not an automatic retry and not a way to bypass normal
            freshness for context-dependent content.
        Assignment completion: ordinary peer progress on an unchanged
        originating intent does not by itself reopen an ``is_self: true``
        contribution that already completed this Agent's part. Peer-exposed
        work or an evolving assignment permits another response only when
        newly observed content creates or changes an unresolved obligation
        belonging to this Agent. A genuine follow-up, correction, direct
        mention, newly exposed dependency, otherwise changed work, including
        genuine new peer-exposed work (new work exposed by peer progress), can
        create real work and permit another reply. Treat a changed objective,
        changed scope, changed constraint, or changed deliverable as genuine
        evolution. Multi-target turns must address destinations explicitly or
        stay silent. Keep the final send-or-silence decision model-owned:
        choose a send or
        ``[SILENT]`` from current evidence.
        """
        return await _dispatch_semantic_send(
            cfg,
            SemanticSendRequest(
                destination=channel,
                text=text,
                root_id=root_id,
                visibility_level=visibility_level,
                send_anyway=send_anyway,
            ),
        )

        # Retained temporarily as unreachable reference code for the other
        # read-oriented helpers in this module; the semantic return above is
        # the only model-authored send path.
        channel_ref = channel.strip()
        if not channel_ref:
            raise RuntimeError("channel is required")
        if channel_ref.startswith("#"):
            raise RuntimeError(
                "'#<name>' channel addressing isn't supported; "
                "use the channel id (e.g. 'ch_<uuid>') or call "
                "list_channels_in_all_spaces to look one up."
            )

        if cfg.keyless:
            # T23 keyless bridge transport: POST plaintext to the unsigned
            # ``/v2/cloud-agents/messages`` route (the E2B egress proxy
            # injects ``x-sandbox-token``); the server holds all crypto and
            # fans out recipients. DM ('@slug') vs channel ('ch_') routing
            # only — no device-key fetch, no encrypt, no signed POST, no
            # bridge WS. Threaded replies carry the same snake_case
            # ``thread_root_id`` / ``reply_to_id`` field names a human/web
            # message uses: ``thread_root_id`` is the resolved+validated
            # true root (same resolvers the native branch runs, driven off
            # the local store — no network), ``reply_to_id`` is the raw
            # parent id the agent passed. Resolution is fail-soft — a miss
            # falls through with a note and the send still completes.
            if channel_ref.startswith("@"):
                keyless_recipient = channel_ref[1:]
                if not keyless_recipient:
                    raise RuntimeError("DM recipient slug is required after '@'")
                resolved_root, root_note = await _resolve_root_id(
                    root_id, cfg.data_client,
                )
                resolved_root, validate_note = await _validate_root_same_channel(
                    resolved_root, None, None, cfg.data_client,
                )
                body: dict[str, Any] = {
                    "plaintext": text,
                    "recipient_slug": keyless_recipient,
                }
            else:
                keyless_channel_id = channel_ref
                keyless_space_id = await _resolve_channel_space(
                    cfg, keyless_channel_id,
                )
                resolved_root, root_note = await _resolve_root_id(
                    root_id, cfg.data_client,
                )
                resolved_root, validate_note = await _validate_root_same_channel(
                    resolved_root, keyless_channel_id, keyless_space_id,
                    cfg.data_client,
                )
                body = {
                    "plaintext": text,
                    "space_id": keyless_space_id,
                    "channel_id": keyless_channel_id,
                }
            # Only truthy thread keys ride the body. F4: drop the raw parent
            # ref when root validation wiped the thread — no dangling
            # reply_to for a root the send no longer threads under.
            if resolved_root:
                body["thread_root_id"] = resolved_root
                if root_id:
                    body["reply_to_id"] = root_id
            ack = await _send_keyless(cfg, body)
            return (
                f"posted {(ack or {}).get('envelope_id', '?')} to {channel}"
                f"{root_note}"
                f"{validate_note}"
            )

        if channel_ref.startswith("@"):
            recipient_slug = channel_ref[1:]
            if not recipient_slug:
                raise RuntimeError("DM recipient slug is required after '@'")
            envelope_kind = "dm"
            channel_id: Optional[str] = None
            send_space_id: Optional[str] = None
            # Fan to the recipient AND our own other devices so any
            # other logged-in clients see the DM too.
            recipient_slugs = [cfg.slug, recipient_slug]
        else:
            channel_id = channel_ref
            envelope_kind = "channel"
            recipient_slug = None
            # The local cache (filled by membership events landing
            # over the WS — see puffo_core_client._handle_event /
            # _maybe_cache_channel_space) is authoritative for
            # channels the agent can reach. Miss → bail loud.
            send_space_id = await _resolve_channel_space(cfg, channel_id)
            members_resp = await cfg.http_client.get(
                f"/spaces/{send_space_id}/channels/{channel_id}/members"
            )
            recipient_slugs = [
                m.get("slug", "")
                for m in members_resp.get("members", [])
                if m.get("slug")
            ]
            if not recipient_slugs:
                raise RuntimeError(
                    f"channel {channel_id} has no resolvable members "
                    f"(searched space {send_space_id})"
                )

        sess = cfg.keystore.load_session(cfg.slug)
        signing_key = Ed25519KeyPair.from_secret_bytes(
            decode_secret(sess.subkey_secret_key)
        )

        resolved_root, root_note = await _resolve_outgoing_root(
            root_id, cfg.data_client,
            self_slug=cfg.slug,
            channel_id=channel_id,
            space_id=send_space_id,
            dm_peer=recipient_slug,
        )
        encrypt = await _send_encryption_required(cfg, resolved_root)
        devices: list = []
        if encrypt:
            devices = await _fetch_device_keys(cfg.http_client, recipient_slugs)
            if not devices:
                raise RuntimeError("no recipient devices found")
        # Visibility floors key off the RESOLVED root — a wiped root makes
        # this a root-level post, which can't fold in the UI.
        effective_visible, visibility_note = await _resolve_visibility(
            visibility_level, channel_ref, text, resolved_root or "", cfg.http_client,
        )
        inp = EncryptInput(
            envelope_kind=envelope_kind,
            sender_slug=cfg.slug,
            sender_subkey_id=sess.subkey_id,
            is_visible_to_human=effective_visible,
            space_id=send_space_id,
            channel_id=channel_id,
            recipient_slug=recipient_slug,
            thread_root_id=resolved_root,
            content_type="text/plain",
            content=text,
            recipients=devices,
        )
        if encrypt:
            envelope, content_key = encrypt_message_with_content_key(
                inp, signing_key,
            )
            # Server expects the envelope at the top level, not wrapped.
            resp = await cfg.http_client.post("/messages", envelope) or {}
            missing = resp.get("missing_devices") or []
            if missing:
                asyncio.create_task(_supplement_missing_devices(
                    cfg.http_client, envelope, content_key,
                    recipient_slugs, missing,
                ))
        else:
            envelope = build_plaintext_message(inp, signing_key)
            await cfg.http_client.post("/v2/messages/plaintext", envelope)
        return (
            f"posted {envelope.get('envelope_id', '?')} to {channel}"
            f"{visibility_note}"
            f"{root_note}"
        )

    @mcp.tool()
    async def get_channel_history(
        channel: str,
        limit: int = 20,
        since: str = "",
        before: int = 0,
        after: int = 0,
    ) -> str:
        """List recent **root posts** in a channel from local storage,
        with the reply count for each thread.

        Replies are NOT inlined — call ``get_thread_history`` if you
        want to drill into a specific thread. This keeps a single
        ``get_channel_history`` call from dragging hundreds of replies
        into your context just because one thread is active.

        Filters (optional, can be combined):
        - ``since`` — an envelope_id (``msg_<uuid>``). Results have
          ``sent_at >`` that envelope's ``sent_at``. Use this when
          you remember the latest root you already saw.
        - ``after`` — ms-epoch timestamp; exclusive lower bound.
        - ``before`` — ms-epoch timestamp; exclusive upper bound.

        Output uses semantic target headers and message rows; root rows include
        their current reply count. Oldest-first inside the returned window.
        Channel id is a raw
        ``ch_<uuid>`` (no ``#name`` shortcut)."""
        limit = max(1, min(int(limit), 200))
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
                since_envelope_id=since or None,
                before_ts=int(before) if before else None,
                after_ts=int(after) if after else None,
            )
        except DataNotFound:
            return f"(no such channel: {channel_id})"
        if not roots:
            return "(no root posts in the requested window)"
        tool_arguments: dict[str, object] = {"channel": channel}
        if limit != 20:
            tool_arguments["limit"] = limit
        if since:
            tool_arguments["since"] = since
        if before:
            tool_arguments["before"] = before
        if after:
            tool_arguments["after"] = after
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            [entry.message for entry in roots],
            tool_name="get_channel_history",
            tool_arguments=tool_arguments,
        )
        result = format_message_group(
            [entry.message for entry in roots],
            current_agent_aliases=(cfg.slug,),
            reply_counts={entry.message.envelope_id: entry.reply_count for entry in roots},
        )
        return f"{result}\n{receipt_marker}" if receipt_marker else result

    @mcp.tool()
    async def get_dm_history(
        peer: str,
        limit: int = 20,
        before: int = 0,
    ) -> str:
        """List recent **direct messages** between you and ``peer``,
        oldest-first, from local storage.

        ``peer`` is the other party's slug (e.g. ``alice-1a2b``) — the
        same slug you'd DM with ``send_message``. ``before`` is an
        optional ms-epoch upper bound (exclusive) for paging back.

        Output uses semantic target headers and message rows, oldest-first."""
        limit = max(1, min(int(limit), 200))
        peer_slug = peer.strip().lstrip("@")
        if not peer_slug:
            raise RuntimeError("pass the peer's slug to read DM history.")
        msgs = await cfg.data_client.get_dm_history(
            peer_slug, limit=limit, before=int(before) if before else None,
        )
        if not msgs:
            return "(no direct messages with that peer in the requested window)"
        return format_message_group(msgs, current_agent_aliases=(cfg.slug,))

    @mcp.tool()
    async def get_thread_history(
        root_id: str,
        limit: int = 50,
        since: str = "",
        before: int = 0,
        after: int = 0,
    ) -> str:
        """List messages in one thread (the root post + every reply
        that points at it) from local storage.

        Used after ``get_channel_history`` shows a thread you want
        to read into. Same filter semantics as
        ``get_channel_history``: ``since`` is an envelope_id whose
        ``sent_at`` becomes the exclusive lower bound; ``after`` /
        ``before`` are ms-epoch bounds. All filters optional.

        ``root_id`` is the thread root envelope_id (``msg_<uuid>``).
        For a top-level post that has no replies, this returns just
        that post.

        Output uses semantic target headers and message rows, oldest-first."""
        if not root_id.strip():
            raise RuntimeError("root_id required")
        limit = max(1, min(int(limit), 200))
        try:
            msgs = await cfg.data_client.get_thread_messages(
                root_id.strip(),
                limit=limit,
                since_envelope_id=since or None,
                before_ts=int(before) if before else None,
                after_ts=int(after) if after else None,
            )
        except DataNotFound:
            return f"(no such thread: {root_id.strip()})"
        if not msgs:
            return "(no messages in this thread for the requested window)"
        tool_arguments = {"root_id": root_id}
        if limit != 50:
            tool_arguments["limit"] = limit
        if since:
            tool_arguments["since"] = since
        if before:
            tool_arguments["before"] = before
        if after:
            tool_arguments["after"] = after
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            msgs,
            tool_name="get_thread_history",
            tool_arguments=tool_arguments,
        )
        result = format_message_group(msgs, current_agent_aliases=(cfg.slug,))
        return f"{result}\n{receipt_marker}" if receipt_marker else result

    @mcp.tool()
    async def list_spaces() -> str:
        """List spaces this agent is a member of (id + name).

        ``GET /spaces`` is server-filtered to memberships the
        agent actually has, so the result reflects authoritative
        permissions — channels can be enumerated for any space
        listed here via ``list_channels_in_space``."""
        data = await _read_spaces(cfg)
        spaces_entries = (data or {}).get("spaces", []) or []
        if not spaces_entries:
            return "(not a member of any space)"
        lines: list[str] = []
        for sp in spaces_entries:
            sid = sp.get("space_id", "")
            name = sp.get("name", "") or sid
            if sid:
                lines.append(f"- {sid}  {name}")
        return "\n".join(lines) if lines else "(not a member of any space)"

    @mcp.tool()
    async def list_channels_in_space(space_id: str) -> str:
        """List channels in a single space the agent is a member of.

        ``GET /spaces/<space_id>/channels`` is server-filtered to the
        agent's actual channel memberships; this tool just formats the
        result. The legacy ``cfg.space_id`` is not consulted — pass
        the explicit ``space_id`` to scope the query. Use
        ``list_spaces`` to enumerate valid ``space_id``s first.

        Tight-race note: just after AcceptSpaceInvite the endpoint can
        briefly return the SPA-route HTML stub (decoded as ``str``)
        while the materialiser commits. Treat that as "no channels yet"
        rather than crashing the tool.
        """
        sid = (space_id or "").strip()
        if not sid:
            raise RuntimeError("space_id is required")
        data = await _read_space_channels(cfg, sid)
        channels = (
            data.get("channels", []) if isinstance(data, dict) else []
        ) or []
        if not channels:
            return "(no channels — agent may not be a member of this space yet)"
        lines: list[str] = []
        for ch in channels:
            cid = ch.get("channel_id", "")
            name = ch.get("name", "") or cid
            if cid:
                lines.append(f"- {cid}  {name}")
        return "\n".join(lines) if lines else "(no channels)"

    @mcp.tool()
    async def list_channels_in_all_spaces() -> str:
        """List channels in every space the agent is a member of.

        Output is grouped by space::

            Space sp_X (Team):
              - ch_a  general
              - ch_b  random
            Space sp_Y (Other):
              - ch_c  general

        Convenience over ``list_spaces`` + ``list_channels_in_space``
        for the case where the LLM wants the full membership picture
        in one tool call. Walks one ``GET /spaces`` plus one
        ``GET /spaces/<sp>/channels`` per space."""
        spaces_data = await _read_spaces(cfg)
        spaces_entries = (spaces_data or {}).get("spaces", []) or []
        if not spaces_entries:
            return "(not a member of any space)"
        lines: list[str] = []
        for sp in spaces_entries:
            space_id = sp.get("space_id", "")
            space_name = sp.get("name", "") or space_id
            if not space_id:
                continue
            ch_data = await _read_space_channels(cfg, space_id)
            channels = (
                ch_data.get("channels", []) if isinstance(ch_data, dict) else []
            )
            lines.append(f"Space {space_id} ({space_name}):")
            if not channels:
                lines.append("  (no channels)")
                continue
            for ch in channels:
                cid = ch.get("channel_id", "")
                name = ch.get("name", "") or cid
                if cid:
                    lines.append(f"  - {cid}  {name}")
        return "\n".join(lines) if lines else "(no channels)"

    @mcp.tool()
    async def list_channel_members(channel: str) -> str:
        """List the members of a channel as ``- <slug>  (<role>)``.
        Role is one of owner / admin / member.
        """
        channel_ref = channel.strip()
        if channel_ref.startswith("#"):
            raise RuntimeError(
                "'#<name>' channel addressing isn't supported; pass the "
                "channel id directly."
            )
        channel_id = channel_ref
        # Resolve from the local channel→space cache (populated by
        # membership events). Misses raise — the previous version
        # silently used ``cfg.space_id`` (home space), which broke
        # for any channel not in the agent's home space.
        space_id = await _resolve_channel_space(cfg, channel_id)

        # Keyless degrades to the space roster (no keyless channel-members
        # route exists); native reads the channel-scoped members. See
        # ``_read_channel_members``.
        data = await _read_channel_members(cfg, space_id, channel_id)
        rows = []
        for m in data.get("members", []):
            slug = m.get("slug", "?")
            role = m.get("role") or "member"
            rows.append(f"- {slug}  ({role})")
        return "\n".join(rows) or "(empty channel)"

    @mcp.tool()
    async def get_user_info(username: str) -> str:
        """Look up a user by slug or @-handle.
        Returns slug, display name, bio, and avatar URL when set.

        Always fetches fresh from puffo-server (bypasses the daemon's
        TTL'd profile cache) and writes the result back to that cache
        so the next inbound message renders with the new values.
        Use this when the operator says someone renamed themselves.
        """
        slug = (username or "").lstrip("@").strip()
        if not slug:
            raise RuntimeError("username is required")
        # ``/identities/profiles?slugs=`` accepts a comma-separated
        # list; we read back the first entry. Empty list means the
        # slug isn't registered.
        data = await _read_profiles(cfg, slug)
        profiles = data.get("profiles", []) if isinstance(data, dict) else []
        if not profiles:
            return f"(no profile for {slug})"
        p = profiles[0]
        # Server returns ``display_name`` (was previously read as
        # ``username`` here, which silently dropped the field for
        # every lookup — the line was never printed).
        display_name = (p.get("display_name") or "").strip()
        avatar_url = (p.get("avatar_url") or "").strip()
        bio = (p.get("bio") or "").strip()
        # Push the just-fetched values into the daemon's profile
        # cache for this agent's view so the next render of an
        # inbound message uses the fresh display_name + avatar
        # instead of waiting for the TTL to expire.
        try:
            await cfg.data_client.update_profile_cache(
                slug, display_name, avatar_url,
            )
        except Exception as exc:
            logger.warning(
                "get_user_info: failed to refresh daemon cache for %s: %s",
                slug, exc,
            )
        lines = [f"slug: {p.get('slug', slug)}"]
        if display_name:
            lines.append(f"display_name: {display_name}")
        if bio:
            lines.append(f"bio: {bio}")
        if avatar_url:
            lines.append(f"avatar_url: {avatar_url}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_post(post_ref: str) -> str:
        """Fetch one message by its envelope_id from local storage.

        post_ref: an envelope_id (e.g. 'env_...'). Returns sender,
        timestamp, and message text.
        """
        envelope_id = (post_ref or "").strip()
        if not envelope_id:
            raise RuntimeError("post_ref (envelope_id) is required")

        msg = await cfg.data_client.get_message_by_envelope(envelope_id)
        if msg is None:
            return f"message {envelope_id} not found in local storage"
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            [msg],
            tool_name="get_post",
            tool_arguments={"post_ref": post_ref},
        )

        result = format_message_group([msg], current_agent_aliases=(cfg.slug,))
        return f"{result}\n{receipt_marker}" if receipt_marker else result

    @mcp.tool()
    async def get_post_segment(
        envelope_id: str,
        segment: int,
        segment_size: int = MESSAGE_SEGMENT_CHARS,
    ) -> str:
        """Page a long message body back in chunks.

        When the daemon redacts an oversize inbound message it
        replaces the in-prompt body with a ``[puffo-agent system
        message]`` placeholder citing this tool's name plus the
        envelope_id and total segment count. Call this tool with
        ``segment=N`` (zero-indexed) and the same ``segment_size``
        the placeholder reported to retrieve chunk ``N`` of the
        full body. Only fetch the segments you actually need —
        the placeholder preview usually tells you whether the
        content is worth paging through.

        Returns: ``segment <i>/<total> (chars <start>..<end> of <total>):
        \n<chunk body>``. Out-of-range segment numbers return
        ``segment out of range`` so the agent knows it overshot.

        Special cases:
          * unknown envelope_id → "message <id> not found in local storage"
          * empty content       → "message <id> has no text body"

        ``segment_size`` defaults to the daemon's default redaction
        page size; pass the value the placeholder cited if the operator
        has overridden it on their host.
        """
        envelope_id = (envelope_id or "").strip()
        if not envelope_id:
            raise RuntimeError("envelope_id is required")
        if segment < 0:
            raise RuntimeError("segment must be >= 0")
        if segment_size <= 0:
            raise RuntimeError("segment_size must be > 0")

        msg = await cfg.data_client.get_message_by_envelope(envelope_id)
        if msg is None:
            return f"message {envelope_id} not found in local storage"

        # ``content`` carries either a bare string (plain message)
        # or the ``puffo/message+attachments/v1`` dict shape; pull
        # the text out of the latter so segmenting works on the
        # human-readable portion in both cases.
        content = msg.content
        text = _history_text(content)

        if not text:
            return f"message {envelope_id} has no text body"

        total = len(text)
        # ceil(total / segment_size); at least 1 when total > 0.
        seg_count = (total + segment_size - 1) // segment_size
        if segment >= seg_count:
            return (
                f"segment {segment} out of range (envelope_id={envelope_id} "
                f"has {seg_count} segment(s) at segment_size={segment_size}, "
                "indexed 0..{0})".format(seg_count - 1)
            )
        start = segment * segment_size
        end = min(start + segment_size, total)
        chunk = text[start:end]
        tool_arguments = {
            "envelope_id": envelope_id,
            "segment": segment,
        }
        if segment_size != MESSAGE_SEGMENT_CHARS:
            tool_arguments["segment_size"] = segment_size
        receipt_marker = await _stage_model_visible_messages(
            cfg,
            [msg],
            tool_name="get_post_segment",
            tool_arguments=tool_arguments,
        )
        result = (
            f"source target={target_label(msg)} envelope_id={msg.envelope_id}\n"
            f"segment {segment}/{seg_count - 1} "
            f"(chars {start}..{end - 1} of {total}):\n{chunk}"
        )
        return f"{result}\n{receipt_marker}" if receipt_marker else result

    @mcp.tool()
    async def send_message_with_attachments(
        paths: list[str],
        channel: str,
        caption: str = "",
        root_id: str = "",
        visibility_level: str = "default",
        send_anyway: bool = False,
    ) -> dict[str, Any]:
        """Send a message carrying one or more workspace files to a
        channel or DM.

        All files ride in a single envelope — recipients see one
        message bubble with N attachments.

        paths: workspace-relative file paths. ``..`` and absolute
            paths are rejected.
        channel: same syntax as ``send_message`` (``@<slug>`` or a
            raw channel id).
        caption: optional text alongside the files.
        root_id: optional thread reply, same semantics as
            ``send_message``'s ``root_id``.
        visibility_level: same semantics as ``send_message`` —
            ``"human"`` | ``"default"`` | ``"agent_only"``, default
            ``"default"``. The @-mention floor keys off ``caption``.
        send_anyway: same channel-freshness override as
            ``send_message``. Content revised or derived from conversation
            progress is context-dependent and must use normal freshness.
            After a held send of that content, reread the relevant route and
            make a fresh send-or-silence decision before sending. Switching
            targets does not make that content context-independent. For
            context-dependent content, actual newer message content must be
            successfully synchronized, returned and inspected before
            choosing ``send_anyway``. A newer watermark or sequence advance
            alone, an empty recovery/read result, or a failed history lookup
            is not semantic inspection. Do not infer unseen content or force
            a stale context-dependent draft. After the existing same-turn
            held/read technical eligibility checks, explicit
            ``send_anyway=True`` remains available as a model-owned choice only
            when the chosen content is genuinely context-independent; this is
            a deliberate judgment owned by the model, not an automatic retry
            and not a way to bypass normal freshness for context-dependent
            content.
        Assignment completion: ordinary peer progress on an unchanged
        originating intent does not by itself reopen an ``is_self: true``
        contribution that already completed this Agent's part. Peer-exposed
        work or an evolving assignment permits another response only when
        newly observed content creates or changes an unresolved obligation
        belonging to this Agent. A genuine follow-up, correction, direct
        mention, newly exposed dependency, otherwise changed work, including
        genuine new peer-exposed work (new work exposed by peer progress), can
        create real work and permit another reply. Treat a changed objective,
        changed scope, changed constraint, or changed deliverable as genuine
        evolution. Multi-target turns must address destinations explicitly or
        stay silent. Keep the final send-or-silence decision model-owned:
        choose a send or
        ``[SILENT]`` from current evidence.
        """
        return await _dispatch_semantic_send(
            cfg,
            SemanticSendRequest(
                destination=channel,
                attachment_paths=tuple(paths) if isinstance(paths, list) else (),
                caption=caption,
                root_id=root_id,
                visibility_level=visibility_level,
                send_anyway=send_anyway,
            ),
            tool_name="send_message_with_attachments",
        )

        import mimetypes
        from pathlib import Path

        if not cfg.workspace:
            raise RuntimeError(
                "send_message_with_attachments: agent has no configured "
                "workspace dir"
            )
        if not paths or not isinstance(paths, list):
            raise RuntimeError(
                "send_message_with_attachments: paths is required "
                "(non-empty list)"
            )
        if len(paths) > 10:
            raise RuntimeError(
                f"send_message_with_attachments: too many files "
                f"({len(paths)} > 10 cap)"
            )
        workspace_dir = Path(cfg.workspace).resolve()

        # Validate all paths up front so a late failure doesn't
        # leave orphan blob uploads on the server.
        targets: list[Path] = []
        for raw in paths:
            rel = (raw or "").strip()
            if not rel:
                raise RuntimeError(
                    "send_message_with_attachments: paths contains empty entry"
                )
            rel_path = Path(rel)
            if rel_path.is_absolute():
                raise RuntimeError(
                    f"send_message_with_attachments: absolute paths not "
                    f"allowed ({rel!r})"
                )
            try:
                target = (workspace_dir / rel_path).resolve()
                target.relative_to(workspace_dir)
            except (OSError, ValueError):
                raise RuntimeError(
                    f"send_message_with_attachments: {rel!r} escapes the "
                    f"workspace"
                )
            if not target.is_file():
                raise RuntimeError(
                    f"send_message_with_attachments: {rel!r} is not a file"
                )
            targets.append(target)

        if cfg.keyless:
            # T23 keyless bridge transport: the server holds the at-rest
            # blob store, so we upload each file's PLAINTEXT bytes unsigned
            # (``post_bytes_unsigned`` → egress-injected ``x-sandbox-token``,
            # raw body) and pass the returned ``blob_id``s as top-level
            # ``AttachmentRef``s into the same plaintext ``/v2/cloud-agents/
            # messages`` POST ``send_message`` uses. No ``encrypt_attachment``,
            # no device-key fetch, no signed ``/messages`` POST — a keyless
            # agent has no signed-crypto seam. DM vs channel routing +
            # fail-soft thread resolution mirror ``send_message``'s keyless
            # branch.
            #
            # F5: run EVERY precondition — destination resolve/validate,
            # root resolve/validate, and all per-file size checks — BEFORE
            # the first upload, so a rejected route or an oversized Nth file
            # raises with no orphaned blobs left on the server.
            channel_ref = channel.strip()
            if channel_ref.startswith("#"):
                raise RuntimeError(
                    "'#<name>' channel addressing isn't supported; pass "
                    "the channel id directly."
                )

            # (1) Resolve + validate the destination first. A bare ``@`` or
            # a stale ``ch_`` (``_resolve_channel_space`` raises) fails here,
            # before any upload.
            is_dm = channel_ref.startswith("@")
            keyless_recipient = ""
            keyless_channel_id = ""
            keyless_space_id = None
            if is_dm:
                keyless_recipient = channel_ref[1:]
                if not keyless_recipient:
                    raise RuntimeError(
                        "DM recipient slug is required after '@'"
                    )
            else:
                keyless_channel_id = channel_ref
                keyless_space_id = await _resolve_channel_space(
                    cfg, keyless_channel_id,
                )

            # (2) Resolve + validate the thread root before uploads.
            resolved_root, root_note = await _resolve_root_id(
                root_id, cfg.data_client,
            )
            resolved_root, validate_note = await _validate_root_same_channel(
                resolved_root,
                None if is_dm else keyless_channel_id,
                None if is_dm else keyless_space_id,
                cfg.data_client,
            )

            # (3) Read every file + run the 8 MiB check for ALL files
            # before uploading any — a later oversized file must not orphan
            # earlier blobs.
            prepared: list[tuple[Any, bytes, str]] = []
            total_bytes = 0
            for target in targets:
                plaintext = target.read_bytes()
                if len(plaintext) > 8 * 1024 * 1024:
                    raise RuntimeError(
                        f"send_message_with_attachments: {target.name!r} is "
                        f"{len(plaintext)} bytes (server caps at 8 MiB)"
                    )
                mime_type, _ = mimetypes.guess_type(target.name)
                mime_type = mime_type or "application/octet-stream"
                prepared.append((target, plaintext, mime_type))
                total_bytes += len(plaintext)

            # (4) Only now upload each blob → build refs.
            refs: list[dict] = []
            for target, plaintext, mime_type in prepared:
                up = await _upload_blob_keyless(cfg, plaintext)
                blob_id = up.get("blob_id") if isinstance(up, dict) else None
                if not blob_id:
                    raise RuntimeError(
                        f"send_message_with_attachments: keyless upload "
                        f"returned no blob_id for {target.name!r} ({up!r})"
                    )
                refs.append({
                    "blob_id": blob_id,
                    "filename": target.name,
                    "mime_type": mime_type,
                    "size_bytes": len(plaintext),
                })

            # (5) One send using the pre-resolved route + the F4 reply_to
            # gate (drop the raw parent when root validation wiped it).
            if is_dm:
                body: dict[str, Any] = {
                    "plaintext": caption,
                    "recipient_slug": keyless_recipient,
                    "attachments": refs,
                }
            else:
                body = {
                    "plaintext": caption,
                    "space_id": keyless_space_id,
                    "channel_id": keyless_channel_id,
                    "attachments": refs,
                }
            if resolved_root:
                body["thread_root_id"] = resolved_root
                if root_id:
                    body["reply_to_id"] = root_id
            ack = await _send_keyless(cfg, body)
            names = ", ".join(t.name for t in targets)
            thread_note = f" in thread {resolved_root}" if resolved_root else ""
            return (
                f"uploaded {len(targets)} file(s) [{names}] ({total_bytes} "
                f"bytes total) to {channel}{thread_note} "
                f"(envelope_id {(ack or {}).get('envelope_id', '?')})"
                f"{root_note}"
                f"{validate_note}"
            )

        # Encrypt + upload each file. ``blob_id`` is patched in
        # after /blobs/upload returns — AAD doesn't depend on it.
        attachment_metas: list[AttachmentMeta] = []
        total_bytes = 0
        for target in targets:
            plaintext = target.read_bytes()
            if len(plaintext) > 8 * 1024 * 1024:
                raise RuntimeError(
                    f"send_message_with_attachments: {target.name!r} is {len(plaintext)} bytes "
                    "(server caps at 8 MiB)"
                )
            mime_type, _ = mimetypes.guess_type(target.name)
            mime_type = mime_type or "application/octet-stream"
            ciphertext, meta = encrypt_attachment(
                plaintext=plaintext,
                filename=target.name,
                mime_type=mime_type,
                blob_id="",
            )
            upload = await cfg.http_client.post_bytes(
                "/blobs/upload", ciphertext,
            )
            blob_id = upload.get("blob_id") if isinstance(upload, dict) else None
            if not blob_id:
                raise RuntimeError(
                    f"send_message_with_attachments: server returned no blob_id for "
                    f"{target.name!r} ({upload!r})"
                )
            meta.blob_id = blob_id
            attachment_metas.append(meta)
            total_bytes += len(plaintext)

        # Compose one envelope carrying all attachments, reusing
        # ``send_message``'s routing logic.
        channel_ref = channel.strip()
        if channel_ref.startswith("#"):
            raise RuntimeError(
                "'#<name>' channel addressing isn't supported; pass the "
                "channel id directly."
            )
        if channel_ref.startswith("@"):
            recipient_slug = channel_ref[1:]
            if not recipient_slug:
                raise RuntimeError("DM recipient slug is required after '@'")
            envelope_kind = "dm"
            channel_id: Optional[str] = None
            send_space_id: Optional[str] = None
            recipient_slugs = [cfg.slug, recipient_slug]
        else:
            channel_id = channel_ref
            envelope_kind = "channel"
            recipient_slug = None
            # Same cache-only resolution as ``send_message`` — no
            # silent fallback to ``cfg.space_id``, because targeting
            # the wrong space produced FB-76 mismaps.
            send_space_id = await _resolve_channel_space(cfg, channel_id)
            members_resp = await cfg.http_client.get(
                f"/spaces/{send_space_id}/channels/{channel_id}/members"
            )
            recipient_slugs = [
                m.get("slug", "")
                for m in members_resp.get("members", [])
                if m.get("slug")
            ]
            if not recipient_slugs:
                raise RuntimeError(
                    f"send_message_with_attachments: channel {channel_id} has no resolvable members"
                )

        sess = cfg.keystore.load_session(cfg.slug)
        signing_key = Ed25519KeyPair.from_secret_bytes(
            decode_secret(sess.subkey_secret_key)
        )
        body_content = {
            "text": caption,
            "attachments": [m.to_dict() for m in attachment_metas],
        }
        resolved_root, root_note = await _resolve_outgoing_root(
            root_id, cfg.data_client,
            self_slug=cfg.slug,
            channel_id=channel_id,
            space_id=send_space_id,
            dm_peer=recipient_slug,
        )
        encrypt = await _send_encryption_required(cfg, resolved_root)
        devices: list = []
        if encrypt:
            devices = await _fetch_device_keys(cfg.http_client, recipient_slugs)
            if not devices:
                raise RuntimeError(
                    "send_message_with_attachments: no recipient devices found"
                )
        effective_visible, visibility_note = await _resolve_visibility(
            visibility_level, channel_ref, caption, resolved_root or "", cfg.http_client,
        )
        inp = EncryptInput(
            envelope_kind=envelope_kind,
            sender_slug=cfg.slug,
            sender_subkey_id=sess.subkey_id,
            is_visible_to_human=effective_visible,
            space_id=send_space_id,
            channel_id=channel_id,
            recipient_slug=recipient_slug,
            thread_root_id=resolved_root,
            content_type=ATTACHMENT_CONTENT_TYPE,
            content=body_content,
            recipients=devices,
        )
        if encrypt:
            envelope, content_key = encrypt_message_with_content_key(
                inp, signing_key,
            )
            resp = await cfg.http_client.post("/messages", envelope) or {}
            missing = resp.get("missing_devices") or []
            if missing:
                asyncio.create_task(_supplement_missing_devices(
                    cfg.http_client, envelope, content_key,
                    recipient_slugs, missing,
                ))
        else:
            envelope = build_plaintext_message(inp, signing_key)
            await cfg.http_client.post("/v2/messages/plaintext", envelope)
        names = ", ".join(t.name for t in targets)
        thread_note = f" in thread {resolved_root}" if resolved_root else ""
        return (
            f"uploaded {len(targets)} file(s) [{names}] ({total_bytes} bytes "
            f"total) to {channel}{thread_note} "
            f"(envelope_id {envelope.get('envelope_id', '?')})"
            f"{visibility_note}"
            f"{root_note}"
        )

    @mcp.tool()
    async def install_host_mcp(
        name: str,
        spec: Optional[dict] = None,
        template_id: str = "",
    ) -> str:
        """Lay down an MCP server spec into the operator's host
        ``~/.claude.json`` so they can complete OAuth / paste API keys
        on their own claude session, then auto-DM them a one-line
        install confirmation. Pair with ``sync_host_mcp`` once they
        confirm. If you have setup-context to share (docs URL, env
        keys to populate, gotchas) send a separate follow-up message
        — the auto-DM is intentionally minimal.

        ``name``: the key the entry registers under
            (``mcpServers[<name>]`` on host).

        Pass exactly ONE of the two source forms:

        - ``template_id``: look up the spec from puffo-server's
          ``/v2/mcp-templates/<id>`` catalog. Use when the MCP is
          operator-curated and ``desired_mcp`` ships an empty-env
          placeholder you need credentials for.
        - ``spec``: pass an inline MCP server config dict transcribed
          from the MCP package's own README — useful when you find
          an MCP on the web (e.g. Coinbase CDP MCP) that isn't in
          puffo-server's catalog. Shape:
            ``{"type": "stdio", "command": "npx", "args": [...], "env": {...}}``
            ``{"type": "http"|"sse", "url": "https://...", "env": {...}}``
          Set ``env`` values to empty strings for placeholders the
          operator needs to populate.

        Behaviour:
          - host already has the entry → file untouched, no DM, tells
            you to skip to ``sync_host_mcp``.
          - catalog / spec validation / file write fails → tool errors,
            no side effects.
          - host write succeeds + DM succeeds → returns the DM's
            envelope_id; wait for the operator's ping.
          - host write succeeds + DM fails → returns the prebuilt body
            so you can retry via ``send_message`` yourself.
        """
        if cfg.rpc_client is None:
            raise RuntimeError(
                "install_host_mcp unavailable — PUFFO_RPC_URL not set "
                "on this MCP runtime, so the puffo-agent daemon's "
                "rpc_service isn't reachable."
            )
        return await cfg.rpc_client.install_mcp(
            name=name, template_id=template_id, spec=spec,
        )

    @mcp.tool()
    async def sync_host_mcp(template_id: str) -> str:
        """Copy the operator's ``~/.claude.json#mcpServers[<id>]``
        entry into your own ``<agent>/.claude.json``. Pair with
        ``install_host_mcp`` once the operator finishes OAuth on host,
        then call ``refresh()`` so claude respawns and picks up the
        new MCP.

        If the host config doesn't have the entry yet, returns an
        error asking you to call ``install_host_mcp`` first (and
        relay the result to the operator).
        """
        if cfg.rpc_client is None:
            raise RuntimeError(
                "sync_host_mcp unavailable — PUFFO_RPC_URL not set "
                "on this MCP runtime, so the puffo-agent daemon's "
                "rpc_service isn't reachable."
            )
        return await cfg.rpc_client.sync_mcp(template_id=template_id)

    @mcp.tool()
    async def leave_space(space_id: str, reason: str = "") -> str:
        """Request to leave a space. This does NOT leave immediately —
        it asks your operator to approve. The operator gets a DM and
        replies `y` (you leave) or `n` (you stay); you'll see their
        decision in that thread.

        space_id: the space to leave (e.g. 'sp_<uuid>'). Use
            ``list_spaces`` to find it.
        reason: optional — a short why, shown to the operator in the
            approval DM. Be honest and specific.

        Note: a space owner can't leave directly, and the request only
        goes through while you're a member.
        """
        sp = space_id.strip()
        if not sp:
            raise RuntimeError("space_id is required")
        # ws-local runs in-process → drive the client directly; harness
        # runtimes go through the daemon's rpc_service.
        if cfg.message_client is not None:
            return await cfg.message_client.request_leave_approval(
                kind="leave_space", space_id=sp, channel_id="", reason=reason,
            )
        if cfg.rpc_client is None:
            raise RuntimeError(
                "leave_space unavailable — PUFFO_RPC_URL not set on this "
                "MCP runtime, so the puffo-agent daemon isn't reachable."
            )
        return await cfg.rpc_client.request_leave(
            kind="leave_space", space_id=sp, channel_id="", reason=reason,
        )

    @mcp.tool()
    async def leave_channel(channel_id: str, reason: str = "") -> str:
        """Request to leave a channel. This does NOT leave immediately —
        it asks your operator to approve. The operator gets a DM and
        replies `y` (you leave) or `n` (you stay); you'll see their
        decision in that thread.

        channel_id: the channel to leave (e.g. 'ch_<uuid>'). Use
            ``list_channels_in_all_spaces`` to find it.
        reason: optional — a short why, shown to the operator in the
            approval DM.

        Note: public channels can't be left on their own — to leave one
        you'd leave the whole space (``leave_space``).
        """
        ch = channel_id.strip()
        if not ch:
            raise RuntimeError("channel_id is required")
        space_id = await _resolve_channel_space(cfg, ch)
        if cfg.message_client is not None:
            return await cfg.message_client.request_leave_approval(
                kind="leave_channel", space_id=space_id, channel_id=ch,
                reason=reason,
            )
        if cfg.rpc_client is None:
            raise RuntimeError(
                "leave_channel unavailable — PUFFO_RPC_URL not set on "
                "this MCP runtime, so the puffo-agent daemon isn't "
                "reachable."
            )
        return await cfg.rpc_client.request_leave(
            kind="leave_channel", space_id=space_id, channel_id=ch,
            reason=reason,
        )

    @mcp.tool()
    async def get_dm_allowlists() -> str:
        """List your DM allowlist (peers whose DMs skip the approval
        gate). Per-agent — every identity keeps its own list; this
        reads yours only."""
        data = await cfg.http_client.get("/allowlists")
        slugs = sorted(
            e.get("peer_slug", "")
            for e in (data.get("entries") or [])
            if e.get("peer_slug")
        )
        if not slugs:
            return "DM allowlist is empty."
        return "DM allowlist:\n" + "\n".join(f"- {s}" for s in slugs)

    @mcp.tool()
    async def get_dm_blocklists() -> str:
        """List your DM blocklist (senders whose messages are silently
        dropped). Per-agent — every identity keeps its own list; this
        reads yours only."""
        data = await cfg.http_client.get("/blocklists")
        slugs = sorted(
            b.get("id", "")
            for b in (data.get("blocks") or [])
            if b.get("target") == "user" and b.get("id")
        )
        if not slugs:
            return "DM blocklist is empty."
        return "DM blocklist:\n" + "\n".join(f"- {s}" for s in slugs)

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
                "/blocklists", {"target": "user", "id": target},
            )
            _note_contact(cfg, target, blocked=True)
            return f"blocked {target}"
        await cfg.http_client.delete(
            "/blocklists", body={"id": target},
        )
        _note_contact(cfg, target, blocked=False)
        return f"unblocked {target}"

    # Bridge-only sandbox-lifecycle tools (schedule_wake / cancel_wake /
    # get_scheduled_wake / get_runtime_status / keep_alive). Gated on
    # ``bridge_client`` — set ONLY at the in-process ws-local site — so a
    # native/subprocess agent never registers them (and never exposes the
    # keyless ``x-sandbox-token`` lifecycle surface). Imported lazily so
    # the native path doesn't pay for a module it will never use.
    if cfg.bridge_client is not None:
        from .lifecycle_tools import register_lifecycle_tools
        register_lifecycle_tools(mcp, cfg)
