"""Presentation-only projection for locally decrypted conversation rows."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


def _value(message: Any, name: str, default: Any = "") -> Any:
    return message.get(name, default) if isinstance(message, Mapping) else getattr(message, name, default)


def _content(message: Any) -> Mapping[str, Any]:
    content = _value(message, "content", {})
    return content if isinstance(content, Mapping) else {}


def _normalized_slug(value: Any) -> str:
    """Normalize the runtime's slug-shaped identities for comparison."""
    return str(value or "").strip().lstrip("@").lower()


def _quoted(value: Any) -> str:
    """Quote readable annotations without letting them alter the row grammar."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _annotation(label: str, value: Any) -> str:
    return f" {label}={_quoted(value)}" if value not in (None, "") else ""


def message_text(message: Any) -> str:
    content = _value(message, "content", "")
    if isinstance(content, Mapping):
        return str(content.get("text") or content.get("caption") or "")
    return str(content or "")


def _dm_peer(message: Any, current_agent_aliases: Sequence[str] = ()) -> str:
    sender = str(_value(message, "sender_slug") or "")
    recipient = str(_value(message, "recipient_slug") or "")
    aliases = {_normalized_slug(alias) for alias in current_agent_aliases if alias}
    if _normalized_slug(sender) in aliases and recipient:
        return recipient.lstrip("@")
    return (sender or recipient or "unknown").lstrip("@")


def target_label(
    message: Any,
    *,
    current_agent_aliases: Sequence[str] = (),
    thread_root_id: str = "",
) -> str:
    """Return the complete canonical target header, including durable IDs."""
    content = _content(message)
    channel_id = str(_value(message, "channel_id") or "")
    if not channel_id or str(_value(message, "envelope_kind", "")) == "dm":
        return f"target=dm peer_id={_dm_peer(message, current_agent_aliases)}"
    space_id = str(_value(message, "space_id") or "unknown-space")
    effective_thread_root_id = str(thread_root_id or _value(message, "thread_root_id", "") or "")
    target = "thread" if effective_thread_root_id else "channel"
    header = f"target={target} space_id={space_id}"
    header += _annotation("space", content.get("space_name"))
    header += f" channel_id={channel_id}"
    channel_name = content.get("channel_name")
    header += _annotation("channel", f"#{str(channel_name).lstrip('#')}") if channel_name else ""
    if target == "thread":
        header += f" thread_root_id={effective_thread_root_id}"
    return header


def sender_type(message: Any, *, current_agent_aliases: Sequence[str] = ()) -> str:
    content = _content(message)
    explicit = content.get("sender_type") or _value(message, "sender_type", "")
    if explicit in {"human", "agent", "system"}:
        return str(explicit)
    if content.get("sender_is_agent") or _value(message, "sender_is_agent", False):
        return "agent"
    if str(_value(message, "envelope_kind", "")) in {"system", "runtime"}:
        return "system"
    if content.get("is_from_operator") or content.get("sender_is_human") or _value(message, "sender_is_human", False):
        return "human"
    aliases = {_normalized_slug(alias) for alias in current_agent_aliases if alias}
    if _normalized_slug(_value(message, "sender_slug")) in aliases:
        return "agent"
    return "unknown"


def _author(message: Any) -> tuple[str, str]:
    content = _content(message)
    slug = str(_value(message, "sender_slug") or "unknown").lstrip("@")
    name = content.get("sender_display_name") or content.get("sender_owner_slug")
    return slug, str(name or "")


def iso_time(message: Any) -> str:
    raw = _value(message, "sent_at", None)
    try:
        return datetime.fromtimestamp(float(raw) / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (TypeError, ValueError, OSError, OverflowError):
        return "unknown-time"


def _attachment_count(message: Any) -> int | None:
    attachments = _content(message).get("attachment_paths", _content(message).get("attachments"))
    if isinstance(attachments, Sequence) and not isinstance(attachments, (str, bytes)):
        return len(attachments)
    return None


def _reminder_event(message: Any) -> Mapping[str, Any] | None:
    """Return a locally durable reminder event without reclassifying messages."""
    content = _content(message)
    return content if content.get("event_type") == "reminder" else None


def format_reminder_event(message: Any) -> str:
    """Project a reminder fact without sender prose or behavioral advice."""
    event = _reminder_event(message)
    if event is None:
        raise ValueError("message is not a reminder event")
    fields = [
        "event_type=reminder",
        f"reminder_id={_quoted(event.get('reminder_id', ''))}",
        f"occurrence_id={_quoted(event.get('occurrence_id', ''))}",
        f"target={_quoted(event.get('target', ''))}",
        f"intended_at={_quoted(event.get('intended_at', ''))}",
        f"actual_fire_at={_quoted(event.get('actual_fire_at', ''))}",
        f"id={_quoted(_value(message, 'envelope_id', 'unknown'))}",
    ]
    return f"[reminder {' '.join(fields)}]\n{str(event.get('content', ''))}"


def format_message_row(message: Any, *, current_agent_aliases: Sequence[str] = (), reply_count: int | None = None) -> str:
    if _reminder_event(message) is not None:
        return format_reminder_event(message)
    seq = _value(message, "server_seq", None)
    seq_text = str(seq) if isinstance(seq, int) else "unsequenced"
    aliases = {_normalized_slug(alias) for alias in current_agent_aliases if alias}
    author, display_name = _author(message)
    encrypted = bool(_value(message, "is_encrypted", True))
    fields = [
        f"seq={seq_text}", f"time={iso_time(message)}", f"type={sender_type(message, current_agent_aliases=current_agent_aliases)}",
        f"id={_value(message, 'envelope_id', 'unknown')}",
        f"self={'true' if _normalized_slug(author) in aliases else 'false'}", f"encrypted={'true' if encrypted else 'false'}",
    ]
    count = _attachment_count(message)
    if count is not None:
        fields.append(f"attachments={count}")
    if reply_count is not None:
        fields.append(f"replies={reply_count}")
    return f"[{' '.join(fields)}] @{author}" + (_annotation("name", display_name) if display_name else "") + f":\n{message_text(message)}"


def format_message_group(
    messages: Iterable[Any],
    *,
    current_agent_aliases: Sequence[str] = (),
    reply_counts: Mapping[str, int] | None = None,
    thread_root_id: str = "",
) -> str:
    """Group only adjacent rows with exactly the same canonical target header."""
    output: list[str] = []
    previous: str | None = None
    for message in messages:
        header = target_label(
            message,
            current_agent_aliases=current_agent_aliases,
            thread_root_id=thread_root_id,
        )
        if header != previous:
            output.append(f"## {header}")
            previous = header
        reply_count = (reply_counts or {}).get(str(_value(message, "envelope_id", "")))
        output.append(format_message_row(message, current_agent_aliases=current_agent_aliases, reply_count=reply_count))
    return "\n".join(output)
