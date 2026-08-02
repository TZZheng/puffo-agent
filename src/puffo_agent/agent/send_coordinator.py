"""Daemon-owned semantic message send coordination.

The coordinator is deliberately independent of provider/model state.  Its
freshness inputs are injected so the daemon can keep one coordinator alive
while Package 4 owns the sources that implement those protocols.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..crypto.attachments import ATTACHMENT_CONTENT_TYPE, AttachmentMeta, encrypt_attachment
from ..crypto.http_client import HttpError
from ..crypto.keystore import decode_secret
from ..crypto.message import (
    EncryptInput,
    build_plaintext_message,
    build_supplementation_envelope,
    encrypt_message_with_content_key,
)
from ..crypto.primitives import Ed25519KeyPair

logger = logging.getLogger(__name__)

CHANNEL_SEND_PATH = "/v2/agent-runtime/messages:send"
KEYLESS_CHANNEL_SEND_PATH = "/v2/cloud-agents/agent-runtime/messages:send"
_KNOWN_ERROR_STATUSES = {400, 401, 403, 404, 405, 409, 413, 429, 500, 503}


@dataclass(frozen=True)
class SemanticSendRequest:
    """The complete model-facing send contract (and nothing freshness-related)."""

    destination: str
    text: str = ""
    attachment_paths: tuple[str, ...] = ()
    caption: str = ""
    root_id: str = ""
    visibility_level: str = "default"
    send_anyway: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticSendRequest":
        allowed = {
            "destination", "channel", "text", "attachment_paths", "paths",
            "caption", "root_id", "visibility_level", "send_anyway",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown send field(s): {', '.join(sorted(unknown))}")
        destination = value.get("destination", value.get("channel", ""))
        paths = value.get("attachment_paths", value.get("paths", ())) or ()
        if not isinstance(paths, (list, tuple)) or not all(
            isinstance(item, str) for item in paths
        ):
            raise ValueError("attachment paths must be a list of strings")
        return cls(
            destination=str(destination or ""),
            text=str(value.get("text") or ""),
            attachment_paths=tuple(paths),
            caption=str(value.get("caption") or ""),
            root_id=str(value.get("root_id") or ""),
            visibility_level=str(value.get("visibility_level") or "default"),
            send_anyway=value.get("send_anyway", False) is True,
        )

    def to_rpc_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "channel": self.destination,
            "root_id": self.root_id,
            "visibility_level": self.visibility_level,
            "send_anyway": self.send_anyway,
        }
        if self.attachment_paths:
            body["paths"] = list(self.attachment_paths)
            body["caption"] = self.caption
        else:
            body["text"] = self.text
        return body


@dataclass
class SendResult:
    state: str
    attempted: bool = True
    envelope_id: Optional[str] = None
    seq: Optional[int] = None
    replay: Optional[bool] = None
    devices_queued: Optional[int] = None
    context_baseline_seq: Optional[int] = None
    seen_seq: Optional[int] = None
    latest_seq: Optional[int] = None
    latest_envelope_id: Optional[str] = None
    latest_seq_before_send: Optional[int] = None
    mode: Optional[str] = None
    missing_devices: list[str] = field(default_factory=list)
    recovered_messages: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[str] = None
    status: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [], "")}


@dataclass
class _HeldEvidence:
    latest_seq: int
    latest_envelope_id: str
    synchronized: bool = False
    draft: str = ""
    based_on_through_seq: int | None = None
    thread_root_id: str = ""
    recovered_messages: list[dict[str, Any]] = field(default_factory=list)
    visible_draft_basis: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: str = ""


@dataclass(frozen=True)
class _ReconsiderationDecision:
    eligible: bool
    reason: str
    provider_session_id: str
    turn_id: str
    latest_seq: int | None = None
    latest_envelope_id: str | None = None
    admitted_seq: int | None = None

    def audit_fields(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "decision_reason": self.reason,
            "provider_session_id": self.provider_session_id,
            "turn_id": self.turn_id,
            "latest_seq": self.latest_seq,
            "latest_envelope_id": self.latest_envelope_id,
            "admitted_seq": self.admitted_seq,
        }


@runtime_checkable
class ContextBaselineSource(Protocol):
    async def get_context_baseline_seq(self, space_id: str, channel_id: str) -> Optional[int]: ...


@runtime_checkable
class ActiveTurnBoundarySource(Protocol):
    async def get_active_turn_through_seq(self, space_id: str, channel_id: str) -> Optional[int]: ...
    async def advance_active_turn_through_seq(
        self, space_id: str, channel_id: str, seq: int
    ) -> None: ...


@runtime_checkable
class HeldRecoverySource(Protocol):
    async def wait_for_held_delivery(
        self, space_id: str, channel_id: str, latest_seq: int, latest_envelope_id: str
    ) -> Any: ...
    async def query_held_messages(
        self, space_id: str, channel_id: str, latest_seq: int,
        latest_envelope_id: str, provider_session_id: Optional[str],
    ) -> Sequence[Mapping[str, Any]]: ...


def failed_result(message: str, *, kind: str = "unavailable", status: int | None = None) -> dict[str, Any]:
    return SendResult(
        state="failed", error=message, error_kind=kind, status=status,
    ).to_dict()


async def _call_first(obj: Any, names: Sequence[str], *args: Any) -> Any:
    if obj is None:
        return None
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return await fn(*args)
    return None


class SendCoordinator:
    """Persistent per-worker coordinator for all model-authored sends."""

    def __init__(
        self,
        *,
        slug: str,
        keystore: Any,
        http_client: Any,
        data_client: Any,
        workspace: str | None = None,
        baseline_source: ContextBaselineSource | Any | None = None,
        active_turn_source: ActiveTurnBoundarySource | Any | None = None,
        held_recovery_source: HeldRecoverySource | Any | None = None,
        provider_session_id: str | None = None,
    ) -> None:
        self.slug = slug
        self.keystore = keystore
        self.http_client = http_client
        self.data_client = data_client
        self.workspace = workspace
        self.baseline_source = baseline_source
        self.active_turn_source = active_turn_source
        self.held_recovery_source = held_recovery_source
        self.provider_session_id = provider_session_id
        self._channel_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._held_lock = asyncio.Lock()
        self._held_evidence: dict[
            tuple[str, str, str, str], _HeldEvidence
        ] = {}

    def _turn_identity(self) -> tuple[str, str]:
        active = getattr(self.active_turn_source, "active", None)
        turn_id = str(getattr(active, "turn_id", "") or "")
        configured_session_id = str(self.provider_session_id or "")
        active_session_id = str(
            getattr(active, "provider_session_id", "") or ""
        )
        if (
            configured_session_id
            and active_session_id
            and configured_session_id != active_session_id
        ):
            return "", turn_id
        session_id = configured_session_id or active_session_id
        return session_id, turn_id

    async def send(
        self, request: SemanticSendRequest | Mapping[str, Any] | None = None, **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            if request is None:
                request = SemanticSendRequest.from_mapping(kwargs)
            elif isinstance(request, Mapping):
                request = SemanticSendRequest.from_mapping(request)
            elif kwargs:
                raise ValueError("pass either a request or keyword fields, not both")
            if not isinstance(request, SemanticSendRequest):
                raise ValueError("invalid semantic send request")
            return await self._send_request(request)
        except Exception as exc:  # semantic facade never leaks tool exceptions
            logger.exception("semantic send failed before transport")
            return failed_result(str(exc), kind="validation")

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        return await self.send(kwargs)

    async def _send_request(self, request: SemanticSendRequest) -> dict[str, Any]:
        destination = request.destination.strip()
        if not destination:
            return failed_result("channel is required", kind="validation")
        try:
            self._validate_attachment_targets(request)
        except Exception as exc:
            return failed_result(str(exc), kind="validation")
        if destination.startswith("#"):
            return failed_result(
                "'#<name>' channel addressing isn't supported; use a channel id",
                kind="validation",
            )
        if getattr(self.http_client, "keyless", False):
            return await self._send_keyless(request, destination)
        if destination.startswith("@"):
            return await self._send_dm(request, destination[1:])

        from ..mcp.puffo_core_tools import _resolve_channel_space

        try:
            space_id = await _resolve_channel_space(_CoordinatorConfig(self), destination)
        except Exception as exc:
            return failed_result(str(exc), kind="routing")
        key = (space_id, destination)
        lock = self._channel_locks.setdefault(key, asyncio.Lock())
        async with lock:
            result = await self._send_channel(request, space_id, destination)
        if result.get("state") == "held":
            synchronized = await self._recover_held(
                space_id, destination,
                result.get("latest_seq"), result.get("latest_envelope_id"),
            )
            result["synchronized"] = synchronized
            session_id, turn_id = self._turn_identity()
            result.update(await self._held_context_output(
                (session_id, turn_id, space_id, destination), space_id, destination,
            ))
        return result

    async def _send_keyless(
        self, request: SemanticSendRequest, destination: str,
    ) -> dict[str, Any]:
        """Use coordinated Runtime-v2 channels and preserve legacy keyless DMs."""
        from ..mcp.puffo_core_tools import (
            _resolve_channel_space,
            _resolve_outgoing_root,
        )
        from ._visibility import resolve_visibility

        if not destination.startswith("@"):
            try:
                space_id = await _resolve_channel_space(
                    _CoordinatorConfig(self), destination,
                )
            except Exception as exc:
                return failed_result(str(exc), kind="routing")
            key = (space_id, destination)
            lock = self._channel_locks.setdefault(key, asyncio.Lock())
            async with lock:
                return await self._send_keyless_channel(
                    request, space_id, destination
                )

        dm_peer: str | None
        channel_id: str | None
        space_id: str | None
        if destination.startswith("@"):
            dm_peer = destination[1:]
            if not dm_peer:
                return failed_result(
                    "DM recipient slug is required after '@'", kind="validation",
                )
            channel_id = None
            space_id = None
        else:
            dm_peer = None
            channel_id = destination
            try:
                space_id = await _resolve_channel_space(
                    _CoordinatorConfig(self), destination,
                )
            except Exception as exc:
                return failed_result(str(exc), kind="routing")

        try:
            root_id, root_note = await _resolve_outgoing_root(
                request.root_id,
                self.data_client,
                self_slug=self.slug,
                channel_id=channel_id,
                space_id=space_id,
                dm_peer=dm_peer,
            )
            visible, visibility_note = await resolve_visibility(
                request.visibility_level,
                destination,
                request.caption if request.attachment_paths else request.text,
                root_id or "",
                self.http_client,
            )
            body: dict[str, Any] = {
                "plaintext": (
                    request.caption if request.attachment_paths else request.text
                ),
                # Keyless envelopes have no signed payload from which the
                # server can recover this policy; always send it explicitly.
                "is_visible_to_human": visible,
            }
            if dm_peer is not None:
                body["recipient_slug"] = dm_peer
            else:
                body["space_id"] = space_id
                body["channel_id"] = channel_id
            if root_id:
                body["thread_root_id"] = root_id
                if request.root_id:
                    body["reply_to_id"] = request.root_id

            targets = self._validate_attachment_targets(request)
            total_bytes = 0
            refs: list[dict[str, Any]] = []
            prepared: list[tuple[Path, bytes, str]] = []
            for target in targets:
                plaintext = target.read_bytes()
                mime_type = (
                    mimetypes.guess_type(target.name)[0]
                    or "application/octet-stream"
                )
                prepared.append((target, plaintext, mime_type))
                total_bytes += len(plaintext)
            for target, plaintext, mime_type in prepared:
                upload = await self.http_client.post_bytes_unsigned(
                    "/v2/cloud-agents/blobs/upload", plaintext,
                )
                blob_id = (
                    upload.get("blob_id")
                    if isinstance(upload, Mapping) else None
                )
                if not blob_id:
                    raise RuntimeError(
                        f"keyless upload returned no blob_id for {target.name!r}"
                    )
                refs.append({
                    "blob_id": blob_id,
                    "filename": target.name,
                    "mime_type": mime_type,
                    "size_bytes": len(plaintext),
                })
            if refs:
                body["attachments"] = refs

            raw = await self.http_client.post_unsigned(
                "/v2/cloud-agents/messages", body,
            ) or {}
            envelope_id = (
                raw.get("envelope_id") if isinstance(raw, Mapping) else None
            )
            if not envelope_id:
                return failed_result(
                    "keyless message send returned no envelope_id",
                    kind="protocol",
                )
            def optional_int(name: str) -> int | None:
                value = raw.get(name) if isinstance(raw, Mapping) else None
                if value is None:
                    return None
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"keyless response has invalid {name}")
                return value
            seq = optional_int("seq")
            devices_queued = optional_int("devices_queued")
            missing_devices = raw.get("missing_devices", []) if isinstance(raw, Mapping) else []
            if not isinstance(missing_devices, list) or not all(isinstance(v, str) for v in missing_devices):
                raise ValueError("keyless response has invalid missing_devices")
            replay = raw.get("replay") if isinstance(raw, Mapping) else None
            if replay is not None and not isinstance(replay, bool):
                raise ValueError("keyless response has invalid replay")
            attachment_note = (
                f"\nuploaded {len(refs)} file(s) ({total_bytes} bytes total)"
                if refs else ""
            )
            return SendResult(
                state="sent",
                envelope_id=str(envelope_id),
                seq=seq,
                replay=replay,
                devices_queued=devices_queued,
                missing_devices=list(missing_devices),
                note=(
                    f"{'uploaded' if refs else 'posted'} {envelope_id} "
                    f"to {destination}{root_note}{visibility_note}{attachment_note}"
                ),
            ).to_dict()
        except HttpError as exc:
            return SendResult(
                state="failed",
                error=_http_error_detail(exc.body),
                error_kind="http",
                status=exc.status,
            ).to_dict()
        except Exception as exc:
            return failed_result(str(exc), kind="validation")

    async def _send_keyless_channel(
        self,
        request: SemanticSendRequest,
        space_id: str,
        channel_id: str,
    ) -> dict[str, Any]:
        baseline = await self._baseline(space_id, channel_id)
        if baseline is None:
            baseline = 0
        active = await self._active_boundary(space_id, channel_id)
        if (
            isinstance(baseline, bool)
            or not isinstance(baseline, int)
            or baseline < 0
            or isinstance(active, bool)
            or (active is not None and (not isinstance(active, int) or active < 0))
        ):
            return failed_result(
                "channel freshness is unavailable",
                kind="freshness_unavailable",
            )
        reconsideration: _ReconsiderationDecision | None = None
        if request.send_anyway:
            reconsideration = await self._reconsideration_decision(
                space_id, channel_id
            )
            if not reconsideration.eligible:
                result = failed_result(
                    "send_anyway requires exact held catch-up and an admitted "
                    "same-Turn read through that boundary",
                    kind="reconsideration_ineligible",
                )
                result["_reconsideration_audit"] = (
                    reconsideration.audit_fields()
                )
                return result
            active = reconsideration.admitted_seq
        seen_seq = max(baseline, active if active is not None else baseline)
        session_id, turn_id = self._turn_identity()
        held_key = (session_id, turn_id, space_id, channel_id)
        from ..mcp.puffo_core_tools import _resolve_outgoing_root
        try:
            root_id, root_note = await _resolve_outgoing_root(
                request.root_id,
                self.data_client,
                self_slug=self.slug,
                channel_id=channel_id,
                space_id=space_id,
                dm_peer=None,
            )
            from ._visibility import resolve_visibility
            visible, _ = await resolve_visibility(
                request.visibility_level, channel_id,
                request.caption if request.attachment_paths else request.text,
                root_id or "", self.http_client,
            )
            body: dict[str, Any] = {
                # One logical model send owns one stable idempotency reference.
                # _post_keyless_exact retries this exact body, while a later
                # tool call constructs a fresh reference here.
                "client_ref": f"send_{uuid.uuid4().hex}",
                "space_id": space_id,
                "channel_id": channel_id,
                "plaintext": (
                    request.caption if request.attachment_paths else request.text
                ),
                "is_visible_to_human": visible,
                "freshness": {
                    "context_baseline_seq": baseline,
                    "seen_seq": seen_seq,
                    "mode": (
                        "send_anyway"
                        if request.send_anyway
                        else "require_current"
                    ),
                },
            }
            if root_id:
                body["thread_root_id"] = root_id
                if request.root_id:
                    body["reply_to_id"] = request.root_id
            refs: list[dict[str, Any]] = []
            for target in self._validate_attachment_targets(request):
                plaintext = target.read_bytes()
                upload = await self.http_client.post_bytes_unsigned(
                    "/v2/cloud-agents/blobs/upload", plaintext
                )
                blob_id = upload.get("blob_id") if isinstance(upload, Mapping) else None
                if not blob_id:
                    return failed_result(
                        "keyless attachment upload returned no blob_id",
                        kind="protocol",
                    )
                refs.append(
                    {
                        "blob_id": blob_id,
                        "filename": target.name,
                        "mime_type": (
                            mimetypes.guess_type(target.name)[0]
                            or "application/octet-stream"
                        ),
                        "size_bytes": len(plaintext),
                    }
                )
            if refs:
                body["attachments"] = refs
            visible_draft_basis = await self._visible_draft_basis(
                space_id, channel_id, root_id or "",
            )
            raw = await self._post_keyless_exact(body)
            result = self._validate_keyless_response(raw, body)
            if result.state == "held":
                if result.latest_seq is not None and result.latest_envelope_id:
                    await self._record_held(
                        held_key, result.latest_seq, result.latest_envelope_id,
                        draft=request.caption if request.attachment_paths else request.text,
                        based_on_through_seq=seen_seq, thread_root_id=root_id or "",
                        visible_draft_basis=visible_draft_basis,
                    )
                result.note = "No channel message was committed; inspect newer Inbox context."
                synchronized = await self._recover_held(
                    space_id,
                    channel_id,
                    result.latest_seq,
                    result.latest_envelope_id,
                )
            elif result.state == "sent":
                await self._consume_held(held_key)
                result.note = f"posted {result.envelope_id} to {channel_id}{root_note}"
                if result.latest_seq_before_send == seen_seq and result.seq is not None:
                    await self._advance(space_id, channel_id, result.seq)
            output = result.to_dict()
            if result.state == "held":
                output["synchronized"] = synchronized
                output.update(await self._held_context_output(held_key, space_id, channel_id))
            if reconsideration is not None:
                output["_reconsideration_audit"] = (
                    reconsideration.audit_fields()
                )
            return output
        except HttpError as exc:
            return failed_result(
                _http_error_detail(exc.body),
                kind="freshness_unavailable" if exc.status in (404, 405, 503) else "http",
                status=exc.status,
            )
        except Exception as exc:
            return failed_result(str(exc), kind="validation")

    async def _post_keyless_exact(self, body: dict[str, Any]) -> Any:
        for attempt in range(2):
            try:
                return await self.http_client.post_unsigned(
                    KEYLESS_CHANNEL_SEND_PATH, body
                )
            except HttpError:
                raise
            except (TimeoutError, ConnectionError, OSError):
                if attempt == 0:
                    continue
                return SendResult(
                    state="failed",
                    error="coordinated keyless send outcome is unknown",
                    error_kind="transport_unknown",
                )
        raise AssertionError("unreachable")

    def _validate_keyless_response(
        self, raw: Any, request_body: Mapping[str, Any]
    ) -> SendResult:
        if isinstance(raw, SendResult):
            return raw
        if not isinstance(raw, Mapping):
            return SendResult(
                state="failed", error="malformed coordinated keyless response",
                error_kind="protocol",
            )
        state = raw.get("state")
        freshness = request_body["freshness"]
        if state == "sent":
            envelope_id = raw.get("envelope_id")
            seq = raw.get("seq")
            replay = raw.get("replay")
            missing_devices = raw.get("missing_devices")
            devices_queued = raw.get("devices_queued")
            echoed = raw.get("freshness")
            if (
                not isinstance(envelope_id, str)
                or not envelope_id
                or isinstance(seq, bool)
                or not isinstance(seq, int)
                or seq <= 0
                or not isinstance(replay, bool)
                or not isinstance(missing_devices, list)
                or not all(isinstance(item, str) for item in missing_devices)
                or (devices_queued is not None and (isinstance(devices_queued, bool) or not isinstance(devices_queued, int) or devices_queued < 0))
                or not isinstance(echoed, Mapping)
                or set(echoed) != {
                    "mode",
                    "context_baseline_seq",
                    "seen_seq",
                    "latest_seq_before_send",
                }
                or echoed.get("mode") != freshness["mode"]
                or echoed.get("seen_seq") != freshness["seen_seq"]
                or echoed.get("context_baseline_seq")
                != freshness["context_baseline_seq"]
                or isinstance(echoed.get("latest_seq_before_send"), bool)
                or not isinstance(echoed.get("latest_seq_before_send"), int)
                or echoed["latest_seq_before_send"] < freshness["seen_seq"]
                or (
                    freshness["mode"] == "require_current"
                    and echoed["latest_seq_before_send"] != freshness["seen_seq"]
                )
            ):
                return SendResult(
                    state="failed", error="invalid coordinated keyless commit",
                    error_kind="protocol",
                )
            return SendResult(
                state="sent",
                envelope_id=envelope_id,
                seq=seq,
                replay=replay,
                devices_queued=devices_queued,
                missing_devices=list(missing_devices),
                context_baseline_seq=freshness["context_baseline_seq"],
                seen_seq=freshness["seen_seq"],
                latest_seq_before_send=echoed["latest_seq_before_send"],
                mode=echoed["mode"],
            )
        if state == "held":
            latest = raw.get("latest_seq")
            if (
                raw.get("seen_seq") != freshness["seen_seq"]
                or (
                    raw.get("context_baseline_seq") is not None
                    and raw.get("context_baseline_seq")
                    != freshness["context_baseline_seq"]
                )
                or isinstance(latest, bool)
                or not isinstance(latest, int)
                or latest <= freshness["seen_seq"]
                or not isinstance(raw.get("latest_envelope_id"), str)
                or not raw.get("latest_envelope_id")
            ):
                return SendResult(
                    state="failed", error="invalid coordinated keyless hold",
                    error_kind="protocol",
                )
            return SendResult(
                state="held",
                context_baseline_seq=freshness["context_baseline_seq"],
                seen_seq=freshness["seen_seq"],
                latest_seq=latest,
                latest_envelope_id=raw["latest_envelope_id"],
            )
        return SendResult(
            state="failed", error="unknown coordinated keyless state",
            error_kind="protocol",
        )

    async def _baseline(self, space_id: str, channel_id: str) -> Any:
        return await _call_first(
            self.baseline_source,
            ("get_context_baseline_seq", "context_baseline_seq", "get_baseline", "baseline_for"),
            space_id, channel_id,
        )

    async def _active_boundary(self, space_id: str, channel_id: str) -> Any:
        return await _call_first(
            self.active_turn_source,
            ("get_active_turn_through_seq", "active_turn_through_seq", "get_boundary", "boundary_for"),
            space_id, channel_id,
        )

    async def _advance(self, space_id: str, channel_id: str, seq: int) -> None:
        await _call_first(
            self.active_turn_source,
            ("advance_active_turn_through_seq", "advance_boundary", "advance"),
            space_id, channel_id, seq,
        )

    async def _send_channel(
        self, request: SemanticSendRequest, space_id: str, channel_id: str,
    ) -> dict[str, Any]:
        baseline = await self._baseline(space_id, channel_id)
        if baseline is None:
            # No trusted visibility floor is a valid conservative state. Zero
            # claims no inaccessible history as model-visible; the Server can
            # hold the attempt, and the model can independently reconsider or
            # choose send_anyway.
            baseline = 0
        if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
            return failed_result(
                "context baseline is invalid for channel send",
                kind="freshness_unavailable",
            )
        active = await self._active_boundary(space_id, channel_id)
        if isinstance(active, bool) or (active is not None and (
            not isinstance(active, int) or active < 0
        )):
            return failed_result("active-turn boundary is invalid", kind="freshness_unavailable")
        reconsideration: _ReconsiderationDecision | None = None
        if request.send_anyway:
            reconsideration = await self._reconsideration_decision(
                space_id, channel_id
            )
            if not reconsideration.eligible:
                result = failed_result(
                    "send_anyway requires exact held catch-up and an admitted "
                    "same-Turn read through that boundary",
                    kind="reconsideration_ineligible",
                )
                result["_reconsideration_audit"] = (
                    reconsideration.audit_fields()
                )
                return result
            active = reconsideration.admitted_seq
        seen_seq = max(baseline, active if active is not None else baseline)
        session_id, turn_id = self._turn_identity()
        held_key = (session_id, turn_id, space_id, channel_id)

        try:
            resolved = await self._resolve_route_and_content(
                request, space_id=space_id, channel_id=channel_id, dm_peer=None,
                require_encryption=True,
            )
        except Exception as exc:
            return failed_result(str(exc), kind="validation")
        if not resolved["encrypt"]:
            return failed_result(
                "plaintext channel sends are not supported",
                kind="encryption_required",
            )

        envelope, content_key = encrypt_message_with_content_key(
            resolved["input"], resolved["signing_key"],
        )
        freshness = {
            "context_baseline_seq": baseline,
            "seen_seq": seen_seq,
            "mode": "send_anyway" if request.send_anyway else "require_current",
        }
        body = {"envelope": envelope, "freshness": freshness}
        visible_draft_basis = await self._visible_draft_basis(
            space_id, channel_id,
            str(resolved.get("root_id") or request.root_id or ""),
        )
        response = await self._post_channel_exact(body)
        result = self._validate_channel_response(response, envelope, freshness)
        action = "uploaded" if request.attachment_paths else "posted"
        if result.state == "sent":
            await self._consume_held(held_key)
            result.note = (
                f"{action} {envelope.get('envelope_id', '?')} to "
                f"{request.destination}{resolved['note']}"
            )
            # A stale send_anyway may cross messages this turn has not seen.
            # The outbound itself is visible, but the boundary is contiguous,
            # so it can advance only when there was no intervening gap.
            if result.latest_seq_before_send == result.seen_seq:
                await self._advance(
                    space_id, channel_id, result.seq  # type: ignore[arg-type]
                )
            if result.missing_devices:
                asyncio.create_task(self._supplement_channel(
                    envelope, content_key, resolved["recipient_slugs"],
                    result.missing_devices, freshness,
                ))
        elif result.state == "held":
            if result.latest_seq is not None and result.latest_envelope_id:
                await self._record_held(
                    held_key, result.latest_seq, result.latest_envelope_id,
                    draft=request.caption if request.attachment_paths else request.text,
                    based_on_through_seq=seen_seq,
                    thread_root_id=str(resolved.get("root_id") or request.root_id or ""),
                    visible_draft_basis=visible_draft_basis,
                )
            result.note = (
                "No message was sent because the channel advanced beyond this "
                "turn's visible boundary. Newer context is returned when "
                "available. You can decide whether to send revised content, "
                "send the chosen content with send_anyway=true, or leave it "
                f"unsent.{resolved['note']}"
            )
        output = result.to_dict()
        if result.state == "held":
            # Native sends recover in ``_send_request``; return the immutable
            # draft/basis now and enrich it after recovery below.
            output.update(await self._held_context_output(held_key, space_id, channel_id))
        if reconsideration is not None:
            output["_reconsideration_audit"] = reconsideration.audit_fields()
        return output

    async def _post_channel_exact(self, body: dict[str, Any]) -> Any:
        # The same object is deliberately reused after an uncertain outcome; the
        # signed client serializes it deterministically with json.dumps.
        for attempt in range(2):
            try:
                return await self.http_client.post(CHANNEL_SEND_PATH, body)
            except HttpError as exc:
                kind = (
                    "deployment" if exc.status in (404, 405)
                    else "protocol" if exc.status < 400
                    else "http"
                )
                detail = _http_error_detail(exc.body)
                return SendResult(
                    state="failed", error=detail, error_kind=kind,
                    status=exc.status,
                )
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt == 0:
                    continue
                return SendResult(
                    state="failed", error=str(exc), error_kind="transport_unknown",
                )
            except Exception as exc:
                # aiohttp and test transports need not share a concrete base type.
                if attempt == 0 and exc.__class__.__name__ in {
                    "ClientError", "ServerDisconnectedError", "ClientConnectionError",
                }:
                    continue
                return SendResult(state="failed", error=str(exc), error_kind="transport")
        raise AssertionError("unreachable")

    def _validate_channel_response(
        self, raw: Any, envelope: Mapping[str, Any], freshness: Mapping[str, Any],
    ) -> SendResult:
        if isinstance(raw, SendResult):
            return raw
        if not isinstance(raw, Mapping):
            return SendResult(
                state="failed", error="channel send returned a malformed response",
                error_kind="protocol",
            )
        state = raw.get("state")
        envelope_id = envelope.get("envelope_id")
        if not isinstance(envelope_id, str) or not envelope_id.strip():
            return SendResult(
                state="failed", error="request envelope_id is invalid",
                error_kind="protocol",
            )
        if state not in ("sent", "held"):
            return SendResult(
                state="failed", error="unknown channel send state",
                error_kind="protocol",
            )
        if raw.get("envelope_id") != envelope_id:
            return SendResult(
                state="failed", error="response envelope_id mismatch",
                error_kind="protocol",
            )
        if state == "sent":
            seq = raw.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
                return SendResult(
                    state="failed", error="sent response has invalid seq",
                    error_kind="protocol",
                )
            if "replay" not in raw:
                return SendResult(
                    state="failed", error="sent response omitted replay",
                    error_kind="protocol",
                )
            replay = raw["replay"]
            if not isinstance(replay, bool):
                return SendResult(
                    state="failed", error="sent response has invalid replay",
                    error_kind="protocol",
                )
            has_freshness = "freshness" in raw
            response_freshness = raw.get("freshness")
            # The frozen Server contract identifies legacy-created replay by
            # replay=true plus the absence of stored v2 freshness metadata.
            legacy_replay = replay is True and not has_freshness
            if not has_freshness and not legacy_replay:
                return SendResult(
                    state="failed", error="sent response omitted freshness",
                    error_kind="protocol",
                )
            latest_before = None
            if has_freshness:
                if not isinstance(response_freshness, Mapping):
                    return SendResult(
                        state="failed", error="sent response freshness is malformed",
                        error_kind="protocol",
                    )
                if set(response_freshness) != {
                    "mode",
                    "context_baseline_seq",
                    "seen_seq",
                    "latest_seq_before_send",
                }:
                    return SendResult(
                        state="failed",
                        error="sent response freshness fields mismatch",
                        error_kind="protocol",
                    )
                baseline_echo = response_freshness["context_baseline_seq"]
                seen_echo = response_freshness["seen_seq"]
                latest_before = response_freshness.get("latest_seq_before_send")
                if (
                    response_freshness.get("mode") != freshness["mode"]
                    or isinstance(baseline_echo, bool)
                    or not isinstance(baseline_echo, int)
                    or baseline_echo < 0
                    or baseline_echo != freshness["context_baseline_seq"]
                    or isinstance(seen_echo, bool)
                    or not isinstance(seen_echo, int)
                    or seen_echo < 0
                    or seen_echo != freshness["seen_seq"]
                    or isinstance(latest_before, bool)
                    or not isinstance(latest_before, int)
                    or latest_before < 0
                    or latest_before < max(baseline_echo, seen_echo)
                    or (
                        freshness["mode"] == "require_current"
                        and latest_before != seen_echo
                    )
                ):
                    return SendResult(
                        state="failed", error="sent response freshness mismatch",
                        error_kind="protocol",
                    )
            if "missing_devices" not in raw:
                return SendResult(
                    state="failed",
                    error="sent response omitted missing_devices",
                    error_kind="protocol",
                )
            missing = raw["missing_devices"]
            if not isinstance(missing, list) or not all(isinstance(v, str) for v in missing):
                return SendResult(
                    state="failed", error="sent response has invalid missing_devices",
                    error_kind="protocol",
                )
            devices_queued = raw.get("devices_queued")
            if devices_queued is not None and (isinstance(devices_queued, bool) or not isinstance(devices_queued, int) or devices_queued < 0):
                return SendResult(
                    state="failed", error="sent response has invalid devices_queued",
                    error_kind="protocol",
                )
            return SendResult(
                state="sent", envelope_id=str(envelope_id), seq=seq, replay=replay,
                devices_queued=devices_queued,
                context_baseline_seq=freshness["context_baseline_seq"],
                seen_seq=freshness["seen_seq"],
                latest_seq_before_send=(
                    latest_before if has_freshness else None
                ),
                missing_devices=missing,
            )

        if set(raw) != {
            "state",
            "envelope_id",
            "context_baseline_seq",
            "seen_seq",
            "latest_seq",
            "latest_envelope_id",
        }:
            return SendResult(
                state="failed", error="held response fields mismatch",
                error_kind="protocol",
            )
        values = {
            "context_baseline_seq": raw["context_baseline_seq"],
            "seen_seq": raw["seen_seq"],
            "latest_seq": raw["latest_seq"],
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            return SendResult(
                state="failed", error="held response has incomplete boundary watermarks",
                error_kind="protocol",
            )
        if (
            values["context_baseline_seq"] != freshness["context_baseline_seq"]
            or values["seen_seq"] != freshness["seen_seq"]
            or values["latest_seq"] <= max(
                freshness["context_baseline_seq"], freshness["seen_seq"]
            )
            or not isinstance(raw.get("latest_envelope_id"), str)
            or not raw.get("latest_envelope_id").strip()
        ):
            return SendResult(
                state="failed", error="held response watermark mismatch",
                error_kind="protocol",
            )
        return SendResult(
            state="held", envelope_id=str(envelope_id),
            context_baseline_seq=freshness["context_baseline_seq"],
            seen_seq=values["seen_seq"], latest_seq=values["latest_seq"],
            latest_envelope_id=raw["latest_envelope_id"],
        )

    async def _send_dm(self, request: SemanticSendRequest, recipient_slug: str) -> dict[str, Any]:
        if not recipient_slug:
            return failed_result("DM recipient slug is required after '@'", kind="validation")
        try:
            resolved = await self._resolve_route_and_content(
                request, space_id=None, channel_id=None, dm_peer=recipient_slug,
                require_encryption=False,
            )
            inp = resolved["input"]
            if resolved["encrypt"]:
                envelope, content_key = encrypt_message_with_content_key(
                    inp, resolved["signing_key"],
                )
                raw = await self.http_client.post("/messages", envelope) or {}
                metadata = self._legacy_dm_metadata(raw)
                missing = metadata[3]
                if missing:
                    from ..mcp.puffo_core_tools import _supplement_missing_devices
                    asyncio.create_task(_supplement_missing_devices(
                        self.http_client, envelope, content_key,
                        resolved["recipient_slugs"], list(missing),
                    ))
            else:
                envelope = build_plaintext_message(inp, resolved["signing_key"])
                raw = await self.http_client.post("/v2/messages/plaintext", envelope) or {}
                metadata = self._legacy_dm_metadata(raw)
            return SendResult(
                state="sent", envelope_id=envelope.get("envelope_id"),
                seq=metadata[0], replay=metadata[1], devices_queued=metadata[2],
                missing_devices=metadata[3],
                note=(
                    f"{'uploaded' if request.attachment_paths else 'posted'} "
                    f"{envelope.get('envelope_id', '?')} to {request.destination}"
                    f"{resolved['note']}"
                ),
            ).to_dict()
        except HttpError as exc:
            return SendResult(
                state="failed", error=_http_error_detail(exc.body),
                error_kind="http", status=exc.status,
            ).to_dict()
        except Exception as exc:
            return failed_result(str(exc), kind="validation")

    @staticmethod
    def _legacy_dm_metadata(raw: Any) -> tuple[int | None, bool | None, int | None, list[str]]:
        """Validate optional legacy-DM commit metadata without requiring it."""
        if not isinstance(raw, Mapping):
            return None, None, None, []
        def optional_int(name: str) -> int | None:
            value = raw.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DM response has invalid {name}")
            return value
        replay = raw.get("replay")
        if replay is not None and not isinstance(replay, bool):
            raise ValueError("DM response has invalid replay")
        missing = raw.get("missing_devices", [])
        if not isinstance(missing, list) or not all(isinstance(value, str) for value in missing):
            raise ValueError("DM response has invalid missing_devices")
        return optional_int("seq"), replay, optional_int("devices_queued"), list(missing)

    async def _resolve_route_and_content(
        self, request: SemanticSendRequest, *, space_id: str | None,
        channel_id: str | None, dm_peer: str | None, require_encryption: bool,
    ) -> dict[str, Any]:
        from ..mcp.puffo_core_tools import (
            _fetch_device_keys, _resolve_outgoing_root, _send_encryption_required,
        )
        from ._visibility import resolve_visibility

        destination = request.destination.strip()
        if channel_id is not None:
            members = await self.http_client.get(
                f"/spaces/{space_id}/channels/{channel_id}/members"
            )
            recipient_slugs = [
                row.get("slug") for row in (members or {}).get("members", [])
                if isinstance(row, Mapping) and row.get("slug")
            ]
            if not recipient_slugs:
                raise RuntimeError(f"channel {channel_id} has no resolvable members")
            kind = "channel"
        else:
            recipient_slugs = [self.slug, dm_peer]
            kind = "dm"

        root, root_note = await _resolve_outgoing_root(
            request.root_id, self.data_client, self_slug=self.slug,
            channel_id=channel_id, space_id=space_id, dm_peer=dm_peer,
        )
        encrypt = await _send_encryption_required(_CoordinatorConfig(self), root)
        if require_encryption and not encrypt:
            return {
                "encrypt": False, "note": root_note,
                "recipient_slugs": recipient_slugs,
            }

        attachments, attachment_note = await self._prepare_attachments(request)
        content: Any
        content_type: str
        visible_text = request.caption if request.attachment_paths else request.text
        if request.attachment_paths:
            content = {"text": request.caption, "attachments": [m.to_dict() for m in attachments]}
            content_type = ATTACHMENT_CONTENT_TYPE
        else:
            content = request.text
            content_type = "text/plain"
        devices = await _fetch_device_keys(self.http_client, recipient_slugs) if encrypt else []
        if encrypt and not devices:
            raise RuntimeError("no recipient devices found")
        visible, visibility_note = await resolve_visibility(
            request.visibility_level, destination, visible_text, root or "", self.http_client,
        )
        sess = self.keystore.load_session(self.slug)
        signing_key = Ed25519KeyPair.from_secret_bytes(decode_secret(sess.subkey_secret_key))
        inp = EncryptInput(
            envelope_kind=kind, sender_slug=self.slug,
            sender_subkey_id=sess.subkey_id, is_visible_to_human=visible,
            space_id=space_id, channel_id=channel_id, recipient_slug=dm_peer,
            thread_root_id=root, content_type=content_type, content=content,
            recipients=devices,
        )
        return {
            "input": inp, "signing_key": signing_key, "encrypt": encrypt,
            "recipient_slugs": recipient_slugs,
            "note": f"{visibility_note}{root_note}{attachment_note}",
        }

    async def _prepare_attachments(
        self, request: SemanticSendRequest,
    ) -> tuple[list[AttachmentMeta], str]:
        if not request.attachment_paths:
            return [], ""
        targets = self._validate_attachment_targets(request)
        metas: list[AttachmentMeta] = []
        total = 0
        for target in targets:
            plaintext = target.read_bytes()
            mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            ciphertext, meta = encrypt_attachment(
                plaintext=plaintext, filename=target.name,
                mime_type=mime_type, blob_id="",
            )
            upload = await self.http_client.post_bytes("/blobs/upload", ciphertext)
            blob_id = upload.get("blob_id") if isinstance(upload, Mapping) else None
            if not blob_id:
                raise RuntimeError(f"server returned no blob_id for {target.name!r}")
            meta.blob_id = blob_id
            metas.append(meta)
            total += len(plaintext)
        names = ", ".join(path.name for path in targets)
        return metas, f"\nuploaded {len(targets)} file(s) [{names}] ({total} bytes total)"

    def _validate_attachment_targets(
        self, request: SemanticSendRequest,
    ) -> list[Path]:
        if not request.attachment_paths:
            return []
        if not self.workspace:
            raise RuntimeError("send_message_with_attachments: agent has no configured workspace dir")
        if len(request.attachment_paths) > 10:
            raise RuntimeError("send_message_with_attachments: too many files (> 10 cap)")
        workspace = Path(self.workspace).resolve()
        targets: list[Path] = []
        for raw in request.attachment_paths:
            rel = raw.strip()
            if not rel:
                raise RuntimeError("send_message_with_attachments: paths contains empty entry")
            rel_path = Path(rel)
            if rel_path.is_absolute():
                raise RuntimeError(f"absolute paths not allowed ({rel!r})")
            target = (workspace / rel_path).resolve()
            try:
                target.relative_to(workspace)
            except ValueError as exc:
                raise RuntimeError(f"{rel!r} escapes the workspace") from exc
            if not target.is_file():
                raise RuntimeError(f"{rel!r} is not a file")
            if target.stat().st_size > 8 * 1024 * 1024:
                raise RuntimeError(f"{target.name!r} exceeds the 8 MiB cap")
            targets.append(target)
        return targets

    async def _supplement_channel(
        self, envelope: dict[str, Any], content_key: bytes,
        recipient_slugs: list[str], missing_ids: list[str],
        freshness: dict[str, Any],
    ) -> None:
        try:
            from ..mcp.puffo_core_tools import _fetch_device_keys
            fresh = await _fetch_device_keys(self.http_client, recipient_slugs)
            wanted = set(missing_ids)
            devices = [device for device in fresh if device.device_id in wanted]
            if not devices:
                return
            supplement = build_supplementation_envelope(envelope, content_key, devices)
            response = await self.http_client.post(
                CHANNEL_SEND_PATH,
                {"envelope": supplement, "freshness": freshness},
            )
            checked = self._validate_channel_response(response, supplement, freshness)
            if checked.state == "failed":
                logger.warning("channel supplementation rejected: %s", checked.error)
        except Exception as exc:
            logger.warning("channel supplementation failed: %s", exc)

    async def _record_held(
        self,
        key: tuple[str, str, str, str],
        latest_seq: int,
        latest_envelope_id: str,
        *,
        draft: str = "",
        based_on_through_seq: int | None = None,
        thread_root_id: str = "",
        visible_draft_basis: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        async with self._held_lock:
            old = self._held_evidence.get(key)
            if (
                old is None
                or latest_seq > old.latest_seq
                or (
                    latest_seq == old.latest_seq
                    and latest_envelope_id != old.latest_envelope_id
                )
            ):
                self._held_evidence[key] = _HeldEvidence(
                    latest_seq=latest_seq,
                    latest_envelope_id=latest_envelope_id,
                    draft=draft,
                    based_on_through_seq=based_on_through_seq,
                    thread_root_id=thread_root_id,
                    visible_draft_basis=[dict(row) for row in visible_draft_basis],
                )

    async def _held_context_output(
        self, key: tuple[str, str, str, str], space_id: str, channel_id: str,
    ) -> dict[str, Any]:
        async with self._held_lock:
            held = self._held_evidence.get(key)
            if held is None:
                return {"context_ready": False}
            rows = list(held.recovered_messages)
            data: dict[str, Any] = {
                "reconsideration": {
                    "context_ready": bool(held.synchronized and rows),
                    "draft": held.draft,
                    "based_on_through_seq": held.based_on_through_seq,
                    "latest_seq": held.latest_seq,
                    "latest_envelope_id": held.latest_envelope_id,
                    "target": {
                        "space_id": space_id, "channel_id": channel_id,
                        "thread_root_id": held.thread_root_id,
                    },
                    "decision": (
                        "Inspect this context, then choose a revised normal-freshness "
                        "send, an unchanged send_anyway=True only if it is still clear "
                        "and appropriate, or send nothing."
                    ),
                },
            }
            if held.diagnostic:
                data["reconsideration"]["diagnostic"] = held.diagnostic
        if rows:
            from .message_projection import format_message_group
            aliases = (self.slug,)
            data["reconsideration"]["visible_draft_basis"] = format_message_group(
                held.visible_draft_basis,
                current_agent_aliases=aliases,
            )
            data["reconsideration"]["new_channel_context"] = format_message_group(
                rows, current_agent_aliases=aliases,
            )
        return data

    async def _consume_held(
        self, key: tuple[str, str, str, str]
    ) -> None:
        async with self._held_lock:
            self._held_evidence.pop(key, None)

    async def _visible_draft_basis(
        self, space_id: str, channel_id: str, thread_root_id: str,
    ) -> list[dict[str, Any]]:
        """Snapshot only rows visible before transport for this destination."""
        runtime = getattr(self.held_recovery_source, "runtime", None)
        active = getattr(runtime, "active", None)
        store = getattr(runtime, "store", None)
        if active is None or store is None:
            return []
        rows: list[dict[str, Any]] = []
        for envelope_id in tuple(getattr(active, "visible_message_ids", ())):
            row = await store.get_message_by_envelope(envelope_id)
            if row is None or row.space_id != space_id or row.channel_id != channel_id:
                continue
            if thread_root_id:
                if row.envelope_id != thread_root_id and row.thread_root_id != thread_root_id:
                    continue
            elif row.thread_root_id:
                continue
            rows.append({
                "space_id": row.space_id, "channel_id": row.channel_id,
                "thread_root_id": row.thread_root_id or "",
                "envelope_id": row.envelope_id, "server_seq": row.server_seq,
                "sender_slug": row.sender_slug, "envelope_kind": row.envelope_kind,
                "sent_at": row.sent_at, "is_encrypted": row.is_encrypted,
                "content": row.content,
            })
        return rows

    async def _reconsideration_decision(
        self, space_id: str, channel_id: str
    ) -> _ReconsiderationDecision:
        session_id, turn_id = self._turn_identity()
        if not session_id or not turn_id:
            return _ReconsiderationDecision(
                False, "missing_active_identity", session_id, turn_id
            )
        key = (session_id, turn_id, space_id, channel_id)
        async with self._held_lock:
            held = self._held_evidence.get(key)
            held_pair = (
                (held.latest_seq, held.latest_envelope_id)
                if held is not None
                else None
            )
            synchronized = bool(held and held.synchronized)
        if held_pair is None:
            return _ReconsiderationDecision(
                False, "missing_held_evidence", session_id, turn_id
            )
        if not synchronized:
            await self._recover_held(
                space_id, channel_id, held_pair[0], held_pair[1]
            )
        async with self._held_lock:
            current = self._held_evidence.get(key)
            if current is None:
                return _ReconsiderationDecision(
                    False, "missing_held_evidence", session_id, turn_id
                )
            latest_seq = current.latest_seq
            latest_envelope_id = current.latest_envelope_id
            synchronized = current.synchronized
        admitted = await self._active_boundary(space_id, channel_id)
        if isinstance(admitted, bool) or (
            admitted is not None
            and (not isinstance(admitted, int) or admitted < 0)
        ):
            admitted = None
        if not synchronized:
            reason = (
                "held_boundary_superseded"
                if (latest_seq, latest_envelope_id) != held_pair
                else "held_not_synchronized"
            )
            return _ReconsiderationDecision(
                False, reason, session_id, turn_id,
                latest_seq, latest_envelope_id, admitted,
            )
        if admitted is None:
            return _ReconsiderationDecision(
                False, "admission_unavailable", session_id, turn_id,
                latest_seq, latest_envelope_id,
            )
        if admitted < latest_seq:
            return _ReconsiderationDecision(
                False, "admission_before_held", session_id, turn_id,
                latest_seq, latest_envelope_id, admitted,
            )
        if self._turn_identity() != (session_id, turn_id):
            return _ReconsiderationDecision(
                False, "active_identity_changed", session_id, turn_id,
                latest_seq, latest_envelope_id, admitted,
            )
        return _ReconsiderationDecision(
            True, "synchronized_and_admitted", session_id, turn_id,
            latest_seq, latest_envelope_id, admitted,
        )

    async def _recover_held(
        self, space_id: str, channel_id: str,
        latest_seq: int | None, latest_envelope_id: str | None,
    ) -> bool:
        if latest_seq is None or not latest_envelope_id or self.held_recovery_source is None:
            return False
        session_id, turn_id = self._turn_identity()
        if not session_id or not turn_id:
            return False
        key = (session_id, turn_id, space_id, channel_id)
        async with self._held_lock:
            current = self._held_evidence.get(key)
            if (
                current is None
                or current.latest_seq != latest_seq
                or current.latest_envelope_id != latest_envelope_id
            ):
                return False
        try:
            waited = await _call_first(
                self.held_recovery_source,
                ("wait_for_held_delivery", "wait_for_delivery", "wait"),
                space_id, channel_id, latest_seq, latest_envelope_id,
            )
        except Exception:
            return False
        if waited is not True:
            return False
        async with self._held_lock:
            current = self._held_evidence.get(key)
            if (
                current is None
                or current.latest_seq != latest_seq
                or current.latest_envelope_id != latest_envelope_id
            ):
                return False
        if self.provider_session_id is None:
            return False
        try:
            rows = await _call_first(
                self.held_recovery_source,
                ("query_held_messages", "query_recovered_messages", "query"),
                space_id, channel_id, latest_seq, latest_envelope_id,
                self.provider_session_id,
            )
        except Exception:
            return False
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return False
        async with self._held_lock:
            current = self._held_evidence.get(key)
            thread_root_id = current.thread_root_id if current is not None else ""
        rows = [
            row for row in rows
            if isinstance(row, Mapping) and (
                (not thread_root_id and not row.get("thread_root_id"))
                or (bool(thread_root_id) and (
                    row.get("envelope_id") == thread_root_id
                    or row.get("thread_root_id") == thread_root_id
                ))
            )
        ]
        synchronized = any(
            isinstance(row, Mapping)
            and row.get("envelope_id") == latest_envelope_id
            and row.get("server_seq") == latest_seq
            and row.get("latest_seq") == latest_seq
            and row.get("latest_envelope_id") == latest_envelope_id
            and row.get("provider_session_id") == session_id
            for row in rows
        )
        if not synchronized or self._turn_identity() != (session_id, turn_id):
            return False
        # Exact-pair compare-and-set: a stale recovery completion cannot bless
        # a superseding held head or resurrect consumed evidence.
        async with self._held_lock:
            current = self._held_evidence.get(key)
            if (
                current is None
                or current.latest_seq != latest_seq
                or current.latest_envelope_id != latest_envelope_id
            ):
                return False
            current.synchronized = True
            # Only locally returned rows with a full plaintext projection are
            # model evidence. The metadata-only sentinel remains sufficient for
            # old callers' synchronization proof but never claims readiness.
            current.recovered_messages = [
                dict(row) for row in rows
                if isinstance(row, Mapping) and "content" in row
            ]
            if not current.recovered_messages:
                current.diagnostic = "local held context is unavailable or unreadable"
        return True


class _CoordinatorConfig:
    """Duck-shaped adapter for the established routing helpers."""

    def __init__(self, coordinator: SendCoordinator) -> None:
        self.slug = coordinator.slug
        self.keystore = coordinator.keystore
        self.http_client = coordinator.http_client
        self.data_client = coordinator.data_client
        self.workspace = coordinator.workspace


def _http_error_detail(body: str) -> str:
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return str(body)[:500] or "HTTP request failed"
    if isinstance(parsed, Mapping):
        return str(parsed.get("message") or parsed.get("error") or parsed)[:500]
    return str(parsed)[:500]
