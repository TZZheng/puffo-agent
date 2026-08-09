"""Inbound receipt processing behind ``PuffoCoreMessageClient.listen``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..crypto.message import MessagePayload, decrypt_message, read_plaintext_message
from ..crypto.primitives import KemKeyPair
from ..crypto.ws_client import ServerDelivery, TransportOutcome
from ._logging import log_runtime_event
from .message_context import MENTION_RE, maybe_redact_long_text
from .message_store import ReceiptDisposition, ReceiptWriteStatus

if TYPE_CHECKING:
    from .puffo_core_client import PuffoCoreMessageClient


BLOCKED_MESSAGE_PLACEHOLDER = "[message from a blocked account was dropped]"


@dataclass
class _ReceiptCommitter:
    client: PuffoCoreMessageClient
    payload: MessagePayload
    server_seq: int
    stored_payload: dict[str, Any]

    async def commit(
        self,
        disposition: ReceiptDisposition,
        reason: str,
        *,
        content: Any | None = None,
    ) -> TransportOutcome:
        self._log_received()
        row = dict(self.stored_payload)
        if content is not None:
            row["content"] = content
            row["content_type"] = "text/plain"

        result = None
        if disposition is ReceiptDisposition.ELIGIBLE:
            promoted = await self.client.store.promote_gated_receipt(
                self.payload.envelope_id,
                self.server_seq,
                reason=reason,
            )
            if promoted.status is not ReceiptWriteStatus.CONFLICT:
                result = promoted
        if result is None:
            result = await self.client.store.store_receipt(
                row,
                server_seq=self.server_seq,
                disposition=disposition,
                reason=reason,
            )
        if result.status is ReceiptWriteStatus.CONFLICT:
            return TransportOutcome.HOLD

        await self._log_commit(result)
        self._notify_runtime(result)
        if result.acknowledge:
            return TransportOutcome.ACK
        if (
            result.disposition is ReceiptDisposition.FOREIGN_DM_GATED
            and result.status
            in (ReceiptWriteStatus.COMMITTED, ReceiptWriteStatus.IDEMPOTENT)
        ):
            return TransportOutcome.DEFER
        return TransportOutcome.HOLD

    def _log_received(self) -> None:
        log_runtime_event(
            self.client._log,
            "inbox.received",
            agent_id=self.client.agent_id,
            agent_slug=self.client.slug,
            message_id=self.payload.envelope_id,
            server_seq=self.server_seq,
            space_id=self.payload.space_id,
            channel_id=self.payload.channel_id,
            outcome="received",
        )

    async def _log_commit(self, result) -> None:
        if result.status is not ReceiptWriteStatus.COMMITTED:
            return
        fields = dict(
            agent_id=self.client.agent_id,
            agent_slug=self.client.slug,
            message_id=self.payload.envelope_id,
            server_seq=self.server_seq,
            space_id=self.payload.space_id,
            channel_id=self.payload.channel_id,
        )
        log_runtime_event(
            self.client._log,
            "inbox.receipt_committed",
            level=logging.INFO,
            envelope_id=self.payload.envelope_id,
            seq=self.server_seq,
            mode="transport_receipt",
            state=result.disposition.value,
            **fields,
        )
        log_runtime_event(
            self.client._log,
            "inbox.persisted",
            outcome=result.disposition.value,
            **fields,
        )
        notice = await self.client.store.get_notice_state()
        if notice.delivery_pending:
            log_runtime_event(
                self.client._log,
                "notice.armed",
                agent_id=self.client.agent_id,
                agent_slug=self.client.slug,
                notice_generation=notice.generation,
                message_count=notice.pending_count,
                outcome="armed",
            )

    def _notify_runtime(self, result) -> None:
        runtime = getattr(self.client, "global_runtime", None)
        if runtime is None:
            return
        if self.payload.envelope_kind == "channel" and result.status in (
            ReceiptWriteStatus.COMMITTED,
            ReceiptWriteStatus.IDEMPOTENT,
        ):
            runtime.notify_delivery()
        if (
            result.disposition is ReceiptDisposition.ELIGIBLE
            and result.status is ReceiptWriteStatus.COMMITTED
        ):
            runtime.notify()


class InboundReceiptHandler:
    """Turn one durable server delivery into one local receipt decision."""

    def __init__(
        self,
        client: PuffoCoreMessageClient,
        kem_keypair: KemKeyPair,
        *,
        decrypt=decrypt_message,
        read_plaintext=read_plaintext_message,
    ) -> None:
        self.client = client
        self.kem_keypair = kem_keypair
        self._decrypt = decrypt
        self._read_plaintext = read_plaintext

    async def handle(self, delivery: ServerDelivery) -> TransportOutcome:
        opened = await self._open_payload(delivery)
        if opened is None:
            return TransportOutcome.HOLD
        payload, is_plaintext = opened
        committer = await self._build_committer(
            payload,
            delivery["seq"],
            is_plaintext=is_plaintext,
        )

        outcome = await self._blocked_outcome(committer)
        if outcome is not None:
            return outcome
        outcome = await self._self_echo_outcome(committer)
        if outcome is not None:
            return outcome
        outcome = await self._operator_control_outcome(committer)
        if outcome is not None:
            return outcome
        outcome = await self._stale_outcome(committer)
        if outcome is not None:
            return outcome

        raw_text, attachment_paths = await self._extract_content(payload)
        outcome = await self._foreign_dm_outcome(committer, raw_text)
        if outcome is not None:
            return outcome
        self._cache_channel_space(payload)
        committer.stored_payload["content"] = await self._prompt_content(
            payload,
            raw_text,
            attachment_paths,
        )
        return await committer.commit(
            ReceiptDisposition.ELIGIBLE,
            "eligible message",
        )

    async def _open_payload(
        self,
        delivery: ServerDelivery,
    ) -> tuple[MessagePayload, bool] | None:
        envelope = delivery["envelope"]
        is_plaintext = envelope.get("type") == "plaintext_message_envelope"
        sender_slug = self._sender_slug(envelope, is_plaintext)
        try:
            sender_keys = await self.client._key_cache.get_signing_keys(sender_slug)
        except Exception as exc:
            self.client._log.warning(
                "could not fetch signing keys for %s — skipping (%s)",
                sender_slug,
                exc,
            )
            return None

        payload = self._decode_with_keys(envelope, sender_keys, is_plaintext)
        if payload is None:
            self.client._key_cache.invalidate(sender_slug)
            try:
                sender_keys = await self.client._key_cache.get_signing_keys(sender_slug)
                payload = self._decode_with_keys(
                    envelope,
                    sender_keys,
                    is_plaintext,
                )
            except Exception:
                pass
        if payload is None:
            self.client._log.warning(
                "decryption failed for %s (%d sender keys tried) — skipping",
                envelope.get("envelope_id"),
                len(sender_keys),
            )
            return None
        return payload, is_plaintext

    @staticmethod
    def _sender_slug(envelope: dict[str, Any], is_plaintext: bool) -> str:
        if is_plaintext:
            return (
                envelope.get("signed_payload", {})
                .get("payload", {})
                .get("sender_slug", "")
            )
        return envelope.get("sender_slug", "")

    def _decode_with_keys(
        self,
        envelope: dict[str, Any],
        sender_keys: list[bytes],
        is_plaintext: bool,
    ) -> MessagePayload | None:
        for signing_key in sender_keys:
            try:
                if is_plaintext:
                    return self._read_plaintext(envelope, signing_key)
                return self._decrypt(
                    envelope,
                    self.client.device_id,
                    self.kem_keypair,
                    signing_key,
                )
            except Exception:
                continue
        return None

    async def _build_committer(
        self,
        payload: MessagePayload,
        server_seq: int,
        *,
        is_plaintext: bool,
    ) -> _ReceiptCommitter:
        thread_root_id = await self.client._resolve_incoming_thread_root(
            payload.thread_root_id,
            payload.channel_id,
            payload.space_id,
        )
        reply_to_id = await self.client._validate_incoming_parent_id(
            payload.reply_to_id,
            payload.channel_id,
            payload.space_id,
        )
        stored_payload = {
            "envelope_id": payload.envelope_id,
            "envelope_kind": payload.envelope_kind,
            "sender_slug": payload.sender_slug,
            "channel_id": payload.channel_id,
            "space_id": payload.space_id,
            "recipient_slug": payload.recipient_slug,
            "content_type": payload.content_type,
            "content": payload.content,
            "sent_at": payload.sent_at,
            "thread_root_id": thread_root_id,
            "reply_to_id": reply_to_id,
            "is_encrypted": not is_plaintext,
        }
        return _ReceiptCommitter(
            client=self.client,
            payload=payload,
            server_seq=server_seq,
            stored_payload=stored_payload,
        )

    async def _blocked_outcome(
        self,
        committer: _ReceiptCommitter,
    ) -> TransportOutcome | None:
        payload = committer.payload
        if payload.sender_slug == self.client.slug:
            return None
        if not await self.client._contacts.is_blocked(payload.sender_slug):
            return None

        if payload.envelope_kind == "dm":
            self.client._log.info(
                "dm_gate: dropped DM from blocked %s",
                payload.sender_slug,
            )
            reason = "blocked dm tombstone"
        else:
            self.client._log.info(
                "contact: dropped channel message from blocked %s "
                "(redacted placeholder stored)",
                payload.sender_slug,
            )
            reason = "blocked channel tombstone"
        return await committer.commit(
            ReceiptDisposition.TERMINAL,
            reason,
            content=BLOCKED_MESSAGE_PLACEHOLDER,
        )

    async def _self_echo_outcome(
        self,
        committer: _ReceiptCommitter,
    ) -> TransportOutcome | None:
        payload = committer.payload
        if payload.sender_slug != self.client.slug:
            return None
        if payload.envelope_kind == "dm":
            await self.client._maybe_allowlist_outbound_dm(payload.recipient_slug)
        return await committer.commit(ReceiptDisposition.TERMINAL, "self echo")

    async def _operator_control_outcome(
        self,
        committer: _ReceiptCommitter,
    ) -> TransportOutcome | None:
        payload = committer.payload
        if not (
            payload.envelope_kind == "dm"
            and payload.sender_slug == self.client.operator_slug
        ):
            return None

        text = str(payload.content) if payload.content else ""
        root_id = committer.stored_payload["thread_root_id"] or ""
        targets, is_direct = self.client._resolve_invite_targets(root_id, text)
        handled_labels = (
            await self.client._apply_invite_replies(targets, text) if targets else []
        )
        if handled_labels:
            if is_direct:
                await self.client._send_invite_bulk_summary(
                    handled_labels,
                    text,
                    root_id,
                )
            return await committer.commit(
                ReceiptDisposition.TERMINAL,
                "handled invite reply",
            )

        controls = (
            (self.client._maybe_handle_leave_reply, "handled leave reply"),
            (
                self.client._maybe_handle_permission_reply,
                "handled permission reply",
            ),
            (
                self.client._maybe_handle_dm_approval_reply,
                "handled dm approval reply",
            ),
        )
        for handler, reason in controls:
            if await handler(thread_root_id=root_id, text=text):
                return await committer.commit(ReceiptDisposition.TERMINAL, reason)
        return None

    async def _stale_outcome(
        self,
        committer: _ReceiptCommitter,
    ) -> TransportOutcome | None:
        payload = committer.payload
        if not self.client._is_stale_for_catchup(payload.sent_at):
            return None
        root_id = committer.stored_payload["thread_root_id"]
        self.client._log.info(
            "handle_envelope: staleness-gate-skipped envelope=%s "
            "(sent_at=%d, threshold_ms=%d, root=%s) — stored, no LLM",
            payload.envelope_id,
            payload.sent_at,
            self.client._catchup_stale_ms,
            root_id or payload.envelope_id,
        )
        self.client._report_stale_processed(payload.envelope_id)
        return await committer.commit(
            ReceiptDisposition.TERMINAL,
            "stale catch-up",
        )

    async def _extract_content(
        self,
        payload: MessagePayload,
    ) -> tuple[str, list[str]]:
        if payload.content_type == "puffo/message+attachments/v1" and isinstance(
            payload.content, dict
        ):
            raw_text = str(payload.content.get("text") or "")
            raw_attachments = payload.content.get("attachments") or []
            if isinstance(raw_attachments, list):
                paths = await self.client._save_inbound_attachments(
                    envelope_id=payload.envelope_id,
                    metas_raw=raw_attachments,
                )
                return raw_text, paths
            return raw_text, []
        return str(payload.content) if payload.content else "", []

    async def _foreign_dm_outcome(
        self,
        committer: _ReceiptCommitter,
        raw_text: str,
    ) -> TransportOutcome | None:
        payload = committer.payload
        if payload.envelope_kind != "dm":
            return None
        if not await self.client._is_foreign_dm_sender(payload.sender_slug):
            await self.client._ensure_trusted_contact(payload.sender_slug)
            return None

        await self.client._maybe_send_dm_notice(payload.sender_slug)
        trusted = (
            self.client.auto_accept_dm
            or await self.client._contacts.is_allowed(payload.sender_slug)
            or await self.client._shares_space_with(payload.sender_slug)
        )
        if trusted:
            return None
        if not await self.client._maybe_gate_foreign_dm(
            sender_slug=payload.sender_slug,
            text=raw_text,
        ):
            return None
        return await committer.commit(
            ReceiptDisposition.FOREIGN_DM_GATED,
            "foreign dm awaiting approval",
        )

    def _cache_channel_space(self, payload: MessagePayload) -> None:
        if payload.envelope_kind != "dm" and payload.channel_id and payload.space_id:
            self.client._channel_space[payload.channel_id] = payload.space_id

    async def _prompt_content(
        self,
        payload: MessagePayload,
        raw_text: str,
        attachment_paths: list[str],
    ) -> dict[str, Any]:
        mentions, is_mention = await self._mentions(payload, raw_text)
        clean_text = (
            raw_text.replace(
                f"@{self.client.slug}",
                f"@you({self.client.slug})",
            ).strip()
            if is_mention
            else raw_text
        )
        names = await self._display_context(payload)
        llm_text = maybe_redact_long_text(
            clean_text,
            envelope_id=payload.envelope_id,
            sender_slug=payload.sender_slug,
            sender_display_name=names["sender_display_name"],
            max_inline_chars=self.client._max_inline_chars,
            segment_chars=self.client._segment_chars,
            agent_slug=self.client.slug,
        )
        return {
            "text": llm_text,
            "attachment_paths": attachment_paths,
            "mentions": mentions,
            "sender_display_name": names["sender_display_name"],
            "sender_owner_slug": names["sender_owner_slug"],
            "is_from_operator": names["is_from_operator"],
            "sender_is_agent": names["sender_is_agent"],
            "is_visible_to_human": payload.is_visible_to_human,
            "channel_name": names["channel_name"],
            "space_name": names["space_name"],
        }

    async def _mentions(
        self,
        payload: MessagePayload,
        raw_text: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        self_slug = self.client.slug.lower()
        parsed: list[str] = []
        seen: set[str] = set()
        for match in MENTION_RE.finditer(raw_text):
            slug = match.group(1).lower()
            if slug not in seen:
                seen.add(slug)
                parsed.append(slug)

        space_members = (
            await self.client._get_space_members(payload.space_id)
            if payload.space_id
            else {}
        )
        mentions: list[dict[str, Any]] = []
        for slug in parsed:
            if slug == self_slug:
                mentions.append(
                    {"username": self.client.slug, "is_agent": True, "is_self": True}
                )
            elif not space_members or slug in space_members:
                mentions.append(
                    {
                        "username": slug,
                        "is_agent": space_members.get(slug) == "agent",
                        "is_self": False,
                    }
                )
        return mentions, self_slug in seen

    async def _display_context(self, payload: MessagePayload) -> dict[str, Any]:
        space_id = payload.space_id or ""
        space_name = await self.client._resolve_space_name(space_id) if space_id else ""
        if payload.envelope_kind == "dm":
            channel_name = "Direct message"
        elif payload.channel_id:
            channel_name = await self.client._resolve_channel_name(
                space_id,
                payload.channel_id,
            )
        else:
            channel_name = payload.channel_id or ""

        sender_display_name = await self.client._fetch_display_name(payload.sender_slug)
        sender_owner_slug = await self.client._fetch_owner_slug(payload.sender_slug)
        return {
            "channel_name": channel_name,
            "space_name": space_name,
            "sender_display_name": sender_display_name,
            "sender_owner_slug": sender_owner_slug,
            "is_from_operator": bool(
                self.client.operator_slug
                and payload.sender_slug == self.client.operator_slug
            ),
            "sender_is_agent": bool(sender_owner_slug),
        }
