"""T0: conformance contract for the Pi RPC harness, pinned at 0.84.3.

No live `pi` process is involved. These tests pin the protocol surface a Pi
Driver (T3) must satisfy, and lock the framing rule that a Python client is
most likely to get wrong.
"""

from __future__ import annotations

import json

import pytest

from tests.fixtures.pi_rpc_0_84_3 import (
    COMMANDS,
    DECLARED_CAPABILITIES,
    EVENTS,
    EXTENSION_UI_REQUESTS,
    PINNED_VERSION,
)


def iter_jsonl_frames(chunk: str):
    """Split Pi RPC output into frames. LF is the only record delimiter.

    Deliberately not ``str.splitlines()``. See the test below for why.
    """
    for line in chunk.split("\n"):
        line = line.rstrip("\r")
        if line:
            yield line


# --- framing ---------------------------------------------------------------


def test_python_splitlines_corrupts_valid_frames():
    """The single most likely way to break this client, pinned as a warning.

    docs/rpc.md warns Node users that ``readline`` is not protocol-compliant
    because it also splits on U+2028/U+2029, which are legal raw inside a JSON
    string. Python's ``str.splitlines()`` has the same defect.

    Only characters a JavaScript emitter leaves **raw** can reach us, so this
    covers exactly those: ``JSON.stringify`` escapes control characters below
    0x20 (so \x1c-\x1e can never arrive raw) but does not escape U+2028,
    U+2029 or U+0085. Hence three reachable hazards, not six.

    ``ensure_ascii=False`` is what makes this test faithful: it reproduces
    what Pi actually writes. With Python's default escaping the bug is
    invisible, which is precisely how it would ship unnoticed.
    """
    for separator in ("\u2028", "\u2029", "\x85"):
        frame = json.dumps(
            {"type": "prompt", "message": f"a{separator}b"},
            ensure_ascii=False,
        )
        assert separator in frame, "must be raw, as a JS emitter writes it"
        assert "\n" not in frame

        # One valid record that parses fine.
        assert json.loads(frame)["message"] == f"a{separator}b"

        # splitlines() tears it in two, leaving both halves unparseable...
        halves = frame.splitlines()
        assert len(halves) == 2, repr(separator)
        for half in halves:
            with pytest.raises(json.JSONDecodeError):
                json.loads(half)

        # ...while LF-only splitting keeps it whole.
        assert list(iter_jsonl_frames(frame + "\n")) == [frame], repr(separator)


def test_control_chars_below_0x20_cannot_arrive_raw():
    """Bounds the hazard above: a JS emitter escapes these, so they are safe."""
    for separator in ("\x1c", "\x1d", "\x1e"):
        frame = json.dumps({"m": f"a{separator}b"}, ensure_ascii=False)
        assert separator not in frame
        assert len(frame.splitlines()) == 1


def test_framing_strips_trailing_cr_and_skips_blank_records():
    chunk = '{"a":1}\r\n\n{"b":2}\n'
    assert list(iter_jsonl_frames(chunk)) == ['{"a":1}', '{"b":2}']


# --- pinned surface --------------------------------------------------------


def test_pinned_version_is_explicit():
    assert PINNED_VERSION == "0.84.3"


def test_command_and_event_surface_sizes():
    """Sizes are pinned so a protocol change fails loudly on version bump."""
    assert len(COMMANDS) == 32
    assert len(EVENTS) == 21
    assert len(EXTENSION_UI_REQUESTS) == 9


def test_turn_lifecycle_events_are_present():
    """The events a turn boundary is built from must all exist."""
    for name in ("agent_start", "agent_end", "agent_settled",
                 "turn_start", "turn_end"):
        assert name in EVENTS, name


def test_exhaustive_event_parsing_is_required():
    """A driver must branch on every pinned event, with no catch-all.

    This is the ``SESSION_UPDATED`` lesson in test form: an event routed to a
    fallback bucket that nobody consumes is indistinguishable from dropping
    it, while looking handled. T3 must replace this stub with its real
    dispatch table and keep the assertion.
    """
    handled: set[str] = set(EVENTS)
    assert handled == EVENTS, sorted(EVENTS - handled)


@pytest.mark.parametrize("field,expected", sorted(DECLARED_CAPABILITIES.items()))
def test_declared_capabilities_match_the_command_surface(field, expected):
    """Each capability claim is backed by a command that exists, or by its
    documented absence. Prevents T3 advertising something Pi cannot do."""
    backing = {
        "steer": "steer" in COMMANDS,
        "busy_delivery": "steer" in COMMANDS,
        "cancel": "abort" in COMMANDS,
        "context_status": "get_session_stats" in COMMANDS,
        "compact": "compact" in COMMANDS,
        "session_resume": "switch_session" in COMMANDS,
        # Pi ships no permission gate; the claim must stay False.
        "permission_bridge": expected is False,
        # No child-per-turn: one `pi --mode rpc` process spans turns.
        "lifecycle": expected == "persistent_child",
    }
    assert backing[field], f"{field}={expected!r} is not backed by 0.84.3"
