"""Small, model-facing projection for locally decrypted conversation rows.

This is deliberately presentation-only: lifecycle and freshness continue to
belong to :mod:`message_store` and :mod:`global_inbox_runtime`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def _value(message: Any, name: str, default: Any = "") -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _content(message: Any) -> Mapping[str, Any]:
    value = _value(message, "content", {})
    return value if isinstance(value, Mapping) else {}


def message_text(message: Any) -> str:
    content = _value(message, "content", "")
    if isinstance(content, Mapping):
        return str(content.get("text") or content.get("caption") or "")
    return str(content or "")


def target_label(message: Any) -> str:
    """Deterministic, cross-space-safe target label."""
    channel = str(_value(message, "channel_id") or "")
    if not channel:
        peer = str(_value(message, "recipient_slug") or _value(message, "sender_slug") or "unknown")
        return f"dm:{peer.lstrip('@')}"
    space = str(_value(message, "space_id") or "unknown-space")
    label = f"space:{space}/#{channel}"
    root = str(_value(message, "thread_root_id") or "")
    return f"{label}:{root}" if root else label


def sender_type(message: Any) -> str:
    content = _content(message)
    explicit = content.get("sender_type") or _value(message, "sender_type", "")
    if explicit in {"human", "agent", "system"}:
        return str(explicit)
    if content.get("sender_is_agent") or _value(message, "sender_is_agent", False):
        return "agent"
    if str(_value(message, "envelope_kind", "")) in {"system", "runtime"}:
        return "system"
    return "human"


def sender_name(message: Any) -> str:
    content = _content(message)
    return str(
        content.get("sender_display_name")
        or content.get("sender_owner_slug")
        or _value(message, "sender_slug")
        or "unknown-sender"
    )


def iso_time(message: Any) -> str:
    raw = _value(message, "sent_at", 0)
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError):
        return "unknown-time"


def format_message_row(message: Any, *, reply_count: int | None = None) -> str:
    seq = _value(message, "server_seq", None)
    seq_text = str(seq) if isinstance(seq, int) else "unsequenced"
    annotations: list[str] = [f"envelope_id={_value(message, 'envelope_id', 'unknown')}"]
    encrypted = bool(_value(message, "is_encrypted", True))
    annotations.append("[encrypted]" if encrypted else "[plaintext]")
    annotations.append(f"is_encrypted={str(encrypted).lower()}")
    if not encrypted:
        annotations.append("is_encrypted: false")
    attachments = _content(message).get("attachment_paths", _content(message).get("attachments", ()))
    if attachments:
        annotations.append(f"attachments={len(attachments)}")
    if reply_count:
        annotations.append(f"replies={reply_count}")
    suffix = f" ({', '.join(annotations)})" if annotations else ""
    return f"[seq={seq_text} time={iso_time(message)} type={sender_type(message)}] @{sender_name(message)}: {message_text(message)}{suffix}"


def format_message_group(messages: Iterable[Any], *, reply_counts: Mapping[str, int] | None = None) -> str:
    grouped: list[str] = []
    previous: str | None = None
    for message in messages:
        target = target_label(message)
        if target != previous:
            grouped.append(f"## target={target}")
            previous = target
        grouped.append(format_message_row(message, reply_count=(reply_counts or {}).get(str(_value(message, "envelope_id", "")))))
    return "\n".join(grouped)
