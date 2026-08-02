"""Canonical, presentation-only conversation projection matrix."""
from __future__ import annotations

import pytest

from puffo_agent.agent.message_projection import format_message_group


def message(**overrides):
    row = {
        "envelope_id": "msg_42",
        "envelope_kind": "channel",
        "server_seq": 42,
        "sent_at": 1785573072000,
        "sender_slug": "alice-1234",
        "space_id": "sp_1",
        "channel_id": "ch_1",
        "thread_root_id": "msg_root",
        "is_encrypted": True,
        "content": {
            "text": "message body",
            "sender_display_name": "Alice",
            "sender_type": "agent",
            "space_name": "Puffo AI",
            "channel_name": "general",
        },
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("name", "rows", "aliases", "reply_counts", "expected"),
    [
        (
            "canonical_thread",
            [message()], (), {},
            '## target=thread space_id=sp_1 space="Puffo AI" channel_id=ch_1 channel="#general" thread_root_id=msg_root\n'
            '[seq=42 time=2026-08-01T08:31:12.000Z type=agent id=msg_42 self=false encrypted=true] @alice-1234 name="Alice":\nmessage body',
        ),
        (
            "same_named_cross_space_and_gap_restarts_header",
            [
                message(envelope_id="one", server_seq=1, thread_root_id="", content={"text": "one", "space_name": "One", "channel_name": "general", "is_from_operator": True}),
                message(envelope_id="two", server_seq=3, thread_root_id="", content={"text": "two", "space_name": "One", "channel_name": "general", "sender_is_agent": True}),
                message(envelope_id="three", server_seq=4, thread_root_id="", space_id="sp_2", channel_id="ch_2", content={"text": "three", "space_name": "Two", "channel_name": "general"}),
            ], (), {},
            '## target=channel space_id=sp_1 space="One" channel_id=ch_1 channel="#general"\n[seq=1 time=2026-08-01T08:31:12.000Z type=human id=one self=false encrypted=true] @alice-1234:\none\n'
            '[seq=3 time=2026-08-01T08:31:12.000Z type=agent id=two self=false encrypted=true] @alice-1234:\ntwo\n'
            '## target=channel space_id=sp_2 space="Two" channel_id=ch_2 channel="#general"\n[seq=4 time=2026-08-01T08:31:12.000Z type=unknown id=three self=false encrypted=true] @alice-1234:\nthree',
        ),
        (
            "dm_self_multiline_attachments_plaintext_and_replies",
            [message(envelope_id="dm_1", envelope_kind="dm", channel_id="", space_id="", thread_root_id="", sender_slug="wire-agent", recipient_slug="bob-9", server_seq=None, is_encrypted=False, content={"text": "first line\nsecond line", "sender_display_name": "Wire", "attachments": ["a", "b"], "sender_type": "human"})],
            ("wire-agent",), {"dm_1": 2},
            '## target=dm peer_id=bob-9\n[seq=unsequenced time=2026-08-01T08:31:12.000Z type=human id=dm_1 self=true encrypted=false attachments=2 replies=2] @wire-agent name="Wire":\nfirst line\nsecond line',
        ),
        (
            "thread_sequence_gap_stays_in_one_target_group",
            [message(envelope_id="msg_42", server_seq=42), message(envelope_id="msg_44", server_seq=44, content={"text": "later", "sender_type": "agent", "space_name": "Puffo AI", "channel_name": "general"})], (), {},
            '## target=thread space_id=sp_1 space="Puffo AI" channel_id=ch_1 channel="#general" thread_root_id=msg_root\n[seq=42 time=2026-08-01T08:31:12.000Z type=agent id=msg_42 self=false encrypted=true] @alice-1234 name="Alice":\nmessage body\n[seq=44 time=2026-08-01T08:31:12.000Z type=agent id=msg_44 self=false encrypted=true] @alice-1234:\nlater',
        ),
        (
            "explicit_type_precedence_over_agent_evidence",
            [message(content={"text": "runtime", "sender_type": "human", "sender_is_agent": True}, envelope_kind="runtime")], (), {},
            '## target=thread space_id=sp_1 channel_id=ch_1 thread_root_id=msg_root\n[seq=42 time=2026-08-01T08:31:12.000Z type=human id=msg_42 self=false encrypted=true] @alice-1234:\nruntime',
        ),
        (
            "runtime_kind_is_system_after_agent_evidence_check",
            [message(envelope_kind="runtime", content={"text": "system event"})], (), {},
            '## target=thread space_id=sp_1 channel_id=ch_1 thread_root_id=msg_root\n[seq=42 time=2026-08-01T08:31:12.000Z type=system id=msg_42 self=false encrypted=true] @alice-1234:\nsystem event',
        ),
    ],
)
def test_canonical_message_projection_matrix(name, rows, aliases, reply_counts, expected):
    assert format_message_group(rows, current_agent_aliases=aliases, reply_counts=reply_counts) == expected
