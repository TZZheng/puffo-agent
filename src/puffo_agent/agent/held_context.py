"""Pure projection of held-send evidence into model-visible context."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .message_projection import CONTEXT_VERSION, format_message_group, sender_type


def _participation_snapshot(
    current_agent_slug: str,
    visible_draft_basis: Sequence[Mapping[str, Any]],
    recovered_messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    self_slug = current_agent_slug.lstrip("@")
    self_message_ids: list[str] = []
    peer_identities: set[str] = set()
    for row in (*visible_draft_basis, *recovered_messages):
        slug = str(row.get("sender_slug") or "").lstrip("@")
        if not slug:
            continue
        if slug == self_slug:
            envelope_id = str(row.get("envelope_id") or "")
            if envelope_id and envelope_id not in self_message_ids:
                self_message_ids.append(envelope_id)
            continue
        if sender_type(row, current_agent_aliases=(current_agent_slug,)) == "agent":
            peer_identities.add(f"@{slug}")
    return {
        "current_agent_identity": f"@{self_slug}",
        "current_agent_has_visible_message": bool(self_message_ids),
        "current_agent_visible_message_ids": self_message_ids,
        "other_visible_agent_count": len(peer_identities),
        "other_visible_agent_identities": sorted(peer_identities),
    }


def build_held_context_output(
    *,
    held: Any,
    current_agent_slug: str,
    space_id: str,
    channel_id: str,
    guidance: str,
) -> dict[str, Any]:
    rows = list(held.recovered_messages)
    target_type = "thread" if held.thread_root_id else "channel"
    target_ref = f"channel:{space_id}:{channel_id}"
    if held.thread_root_id:
        target_ref += f":thread:{held.thread_root_id}"
    reconsideration: dict[str, Any] = {
        "context_version": CONTEXT_VERSION,
        "context_ready": bool(held.synchronized and rows),
        "draft": held.draft,
        "based_on_through_seq": held.based_on_through_seq,
        "latest_seq": held.latest_seq,
        "latest_envelope_id": held.latest_envelope_id,
        "target": {
            "target_type": target_type,
            "target_ref": target_ref,
            "space_id": space_id,
            "channel_id": channel_id,
            "thread_root_id": held.thread_root_id,
        },
        "guidance": guidance,
        "participation_snapshot": _participation_snapshot(
            current_agent_slug,
            held.visible_draft_basis,
            rows,
        ),
    }
    if held.diagnostic:
        reconsideration["diagnostic"] = held.diagnostic
    if rows:
        aliases = (current_agent_slug,)
        reconsideration["visible_draft_basis"] = format_message_group(
            held.visible_draft_basis,
            current_agent_aliases=aliases,
            thread_root_id=held.thread_root_id,
            chronological=True,
        )
        reconsideration["new_channel_context"] = format_message_group(
            rows,
            current_agent_aliases=aliases,
            chronological=True,
        )
    return {"reconsideration": reconsideration}
