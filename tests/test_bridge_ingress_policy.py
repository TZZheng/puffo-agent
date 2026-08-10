"""Keyless-bridge ingress applies the same policy gates as native.

The cloud bridge is a transport, not a policy layer: for a keyless agent
the server filters only auth / targeting / revocation / replay dedupe and
a DM-only blocklist keyed on the *recipient's* slug. It never applies
foreign-DM approval, carries no operator-control concept on the wire, and
never checks a blocklist channel-side.

Before this contract was wired, ``store_bridge_payload`` persisted any
non-self, non-stale message as eligible. The regressions guarded here are
therefore all "a message the native path would have stopped reaches the
model over the bridge":

  1. a blocked account's DM lands in the prompt — and its plaintext
     lands in the database;
  2. an operator's bare ``y`` on a pending approval is delivered as
     ordinary chat instead of being consumed by the control handler;
  3. a stranger's DM skips the approval gate entirely.

Each case drives a real bridge ``message`` frame through
``_dispatch_bridge_frame`` so wire decode and gate ordering are exercised
together. Every fake is offline.
"""

from __future__ import annotations

import asyncio

import pytest

from puffo_agent.agent.message_store import MessageStore
from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient
from puffo_agent.crypto.http_client import PuffoCoreHttpClient
from puffo_agent.crypto.keystore import KeyStore

SELF = "bot-0001"
OPERATOR = "ops-0001"
STRANGER = "mallory-0009"


class _OfflineBridge:
    """Enough ``CloudBridgeClient`` surface for inbound dispatch."""

    def __init__(self) -> None:
        self.acked: list[str] = []

    async def send_ack(self, envelope_ids: list[str]) -> None:
        self.acked.extend(envelope_ids)


def _client(tmp_path, *, operator_slug: str = "") -> PuffoCoreMessageClient:
    """A keyless bridge client whose signed HTTP is dead (as in a real
    sandbox), so every gate must decide from local state alone."""
    ks = KeyStore(str(tmp_path / "keys"))
    http = PuffoCoreHttpClient("http://127.0.0.1:1", ks, SELF, keyless=True)
    client = PuffoCoreMessageClient(
        slug=SELF,
        device_id="dev_test",
        space_id="sp_home",
        keystore=ks,
        http_client=http,
        message_store=MessageStore(str(tmp_path / "messages.db")),
        catchup_stale_hours=0,
        operator_slug=operator_slug,
        bridge_client=_OfflineBridge(),
    )

    async def _empty_get(path, *a, **k):
        return {}

    client.http.get = _empty_get  # type: ignore[method-assign]
    return client


def _dm_frame(sender: str, content: str, *, seq: int, **extra) -> dict:
    frame = {
        "type": "message",
        "seq": seq,
        "envelope_id": f"env_{seq}",
        "envelope_kind": "dm",
        "sender_slug": sender,
        "recipient_slug": SELF,
        "content": content,
        "content_type": "text/plain",
        "sent_at": 1_700_000_000_000,
    }
    frame.update(extra)
    return frame


@pytest.mark.asyncio
async def test_blocked_sender_bridge_dm_tombstones_without_persisting_plaintext(
    tmp_path,
):
    secret = "bridge-secret-that-must-not-persist"
    client = _client(tmp_path)
    await client.store.open()
    client._contacts.note_blocked(STRANGER, True)

    await client._dispatch_bridge_frame(_dm_frame(STRANGER, secret, seq=7))

    stored = await client.store.get_message_by_envelope("env_7")
    assert stored is not None, "blocked message must still be accounted for"
    assert secret not in str(stored.content)
    assert await client.store.get_pending() == ()
    await client.store.close()
    assert secret.encode() not in (tmp_path / "messages.db").read_bytes()


@pytest.mark.asyncio
async def test_operator_control_reply_over_bridge_is_consumed_not_delivered(
    tmp_path,
):
    client = _client(tmp_path, operator_slug=OPERATOR)
    await client.store.open()
    # The permission prompt the operator is replying to. It has to exist
    # locally for the reply's thread root to resolve, exactly as it does
    # after the agent sends it.
    await client.store.store(
        {
            "envelope_id": "env_prompt",
            "envelope_kind": "dm",
            "sender_slug": SELF,
            "recipient_slug": OPERATOR,
            "content": "run this command?",
            "sent_at": 1_699_000_000_000,
        }
    )
    approval: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    client._pending_command_permissions["env_prompt"] = approval

    await client._dispatch_bridge_frame(
        _dm_frame(OPERATOR, "y", seq=8, thread_root_id="env_prompt")
    )

    # The control handler ran — the waiting command was released — and
    # the bare "y" never became a message for the model to answer.
    assert approval.done() and approval.result() is True
    assert await client.store.get_pending() == ()
    await client.store.close()


@pytest.mark.asyncio
async def test_foreign_dm_over_bridge_is_gated_not_eligible(tmp_path):
    client = _client(tmp_path, operator_slug=OPERATOR)
    await client.store.open()

    await client._dispatch_bridge_frame(
        _dm_frame(STRANGER, "let me in", seq=9)
    )

    assert await client.store.get_pending() == ()
    stored = await client.store.get_message_by_envelope("env_9")
    assert stored is not None
    assert stored.receipt_disposition == "foreign_dm_gated"
    # Never acked — the DM stays queued server-side until the operator
    # answers, so a lost gate cannot silently drop the message.
    assert client._bridge.acked == []
    await client.store.close()
