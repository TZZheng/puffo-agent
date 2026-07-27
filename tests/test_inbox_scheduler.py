import asyncio
import os
import sys
from dataclasses import FrozenInstanceError

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from puffo_agent.agent.inbox_scheduler import InboxCoalescer, InboxPlanner
from puffo_agent.agent.message_store import ProcessingState, StoredMessage


def _message(
    envelope_id: str,
    *,
    kind: str = "channel",
    content: str = "x",
    channel: str = "ch",
    space: str = "sp",
    sender: str = "alice",
    recipient: str | None = None,
    thread: str | None = None,
) -> StoredMessage:
    return StoredMessage(
        envelope_id=envelope_id,
        envelope_kind=kind,
        sender_slug=sender,
        channel_id=channel if kind != "dm" else None,
        space_id=space if kind != "dm" else None,
        recipient_slug=recipient,
        content_type="text/plain",
        content=content,
        sent_at=1,
        received_at=1,
        thread_root_id=thread,
        processing_state=ProcessingState.PENDING,
    )


def test_count_cap_accepts_exactly_50_and_does_not_skip_51st_selection():
    items = [_message(f"m{i}") for i in range(51)]
    batch = InboxPlanner().plan(items, formatter=lambda item: item.content, estimator=lambda _: 1)
    assert len(batch.items) == 50
    assert batch.message_ids == tuple(f"m{i}" for i in range(50))
    assert batch.more_available


def test_token_cap_is_inclusive_and_one_over_stops_selection():
    items = [_message("exact"), _message("over"), _message("younger")]
    batch = InboxPlanner().plan(
        items,
        formatter=lambda item: item.envelope_id,
        estimator=lambda formatted: 32_000 if formatted == "exact" else 1,
    )
    assert batch.message_ids == ("exact",)
    assert batch.estimated_tokens == 32_000
    assert batch.more_available


def test_byte_cap_is_inclusive_and_independent_from_token_cap():
    items = [_message("exact", content="x" * 96_000), _message("over", content="y")]
    batch = InboxPlanner().plan(
        items,
        formatter=lambda item: item.content,
        estimator=lambda _: 1,
    )
    assert batch.message_ids == ("exact",)
    assert batch.formatted_bytes == 96_000
    assert batch.estimated_tokens == 1


def test_fallback_estimator_is_exact_utf8_bytes_with_minimum_one():
    items = [_message("empty", content=""), _message("unicode", content="é")]
    batch = InboxPlanner().plan(items, formatter=lambda item: item.content)
    assert batch.estimated_tokens == 3  # max(1, 0) + two UTF-8 bytes
    assert batch.formatted_bytes == 2


def test_planned_batch_is_immutable_tuple_backed_and_planning_is_read_only():
    item = _message("m")
    before = item.__dict__.copy()
    batch = InboxPlanner().plan([item], formatter=lambda _: "formatted", estimator=lambda _: 1)
    assert isinstance(batch.items, tuple)
    assert isinstance(batch.message_ids, tuple)
    assert isinstance(batch.formatted_messages, tuple)
    assert isinstance(batch.target_projections, tuple)
    with pytest.raises(FrozenInstanceError):
        batch.estimated_tokens = 99
    with pytest.raises(TypeError):
        batch.items[0] = item
    assert item.__dict__ == before


def test_order_local_metadata_neutral_targets_and_defensive_deduplication():
    ordered = [
        _message("s1", sender="a", content="@agent"),
        _message("local", sender="b", content="plain", thread="root"),
        _message("s2", sender="c", content="other"),
        _message("dm", kind="dm", sender="d", recipient="agent"),
    ]
    batch = InboxPlanner().plan(
        ordered + [ordered[1]],
        formatter=lambda item: str(item.content),
        estimator=lambda _: 1,
    )
    assert batch.message_ids == ("s1", "local", "s2", "dm")
    assert batch.target_projections == (
        ("channel", "sp", "ch"),
        ("thread", "sp", "ch", "root"),
        ("dm", "d", "agent"),
    )


@pytest.mark.asyncio
async def test_oversized_head_policy_and_guarded_quarantine_run_at_most_once():
    planner = InboxPlanner()
    batch = planner.plan(
        [_message("huge", content="x" * 96_001), _message("younger")],
        formatter=lambda item: item.content,
        estimator=lambda _: 1,
    )
    assert batch.items == ()
    assert batch.unfit_head_id == "huge"
    assert batch.more_available
    calls = []
    quarantined = set()

    async def policy(envelope_id, reason):
        calls.append(("policy", envelope_id, reason))
        return True

    async def quarantine(envelope_id, *, reason):
        calls.append(("quarantine", envelope_id, reason))
        if envelope_id in quarantined:
            return False
        quarantined.add(envelope_id)
        return True

    assert await planner.resolve_unfit_head(
        batch, policy=policy, quarantine=quarantine
    )
    assert not await planner.resolve_unfit_head(
        batch, policy=policy, quarantine=quarantine
    )
    assert [call[0] for call in calls] == [
        "policy", "quarantine", "policy", "quarantine",
    ]


@pytest.mark.asyncio
async def test_coalescer_uses_one_fixed_100ms_metadata_neutral_window():
    now = 10.0
    sleeps = []

    def monotonic():
        return now

    async def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        # A second wake inside the window cannot reset the deadline.
        coalescer.notify()
        now += delay

    coalescer = InboxCoalescer(sleep=sleep, monotonic=monotonic)
    coalescer.notify()
    await coalescer.wait_for_burst()
    assert sleeps == [pytest.approx(0.100)]
    with pytest.raises(TypeError):
        coalescer.notify({"sender": "metadata is forbidden"})


@pytest.mark.asyncio
async def test_coalescer_deadline_starts_at_notify_before_delayed_waiter():
    now = 20.0
    sleeps = []

    def monotonic():
        return now

    async def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay

    coalescer = InboxCoalescer(sleep=sleep, monotonic=monotonic)
    coalescer.notify()
    now += 0.080
    coalescer.notify()
    await coalescer.wait_for_burst()
    assert sleeps == [pytest.approx(0.020)]


@pytest.mark.asyncio
async def test_coalescer_preserves_notification_for_next_expired_window():
    now = 30.0
    sleeps = []

    def monotonic():
        return now

    async def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay
        if len(sleeps) == 1:
            # This wake arrives after the first deadline but before the waiter
            # consumes that completed window, so it must seed the next window.
            now += 0.001
            coalescer.notify()

    coalescer = InboxCoalescer(sleep=sleep, monotonic=monotonic)
    coalescer.notify()
    await coalescer.wait_for_burst()
    await coalescer.wait_for_burst()
    assert sleeps == [pytest.approx(0.100), pytest.approx(0.100)]
