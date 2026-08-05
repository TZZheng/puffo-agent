"""PUF-417: a wrapped ``/note`` body kept only its first line.

The parser walks the body matching ``key: value`` per line and skips
anything that doesn't match, so every continuation line of a ``message:``
was dropped. The web client (parse-note-command.ts) had the same defect;
both sides must agree on the wire format or notes stop round-tripping
between an agent and a browser.
"""
from __future__ import annotations

import pytest

from puffo_agent.mcp.puffo_core_tools import _format_note, _parse_note


def note(message: str, *, mentions: list[str] | None = None) -> str:
    return _format_note("#db4cac", "Waiting", message, mentions or [])


def test_multi_line_body_keeps_every_line():
    parsed = _parse_note("/note \ncolor: #db4cac\nlabel: Waiting\nmessage: first\nsecond\nthird")
    assert parsed is not None
    assert parsed["message"] == "first\nsecond\nthird"


def test_body_stops_at_the_next_recognized_field():
    parsed = _parse_note("/note \nmessage: first\nsecond\nmentions: @alice-0001")
    assert parsed is not None
    assert parsed["message"] == "first\nsecond"
    assert parsed["mentions"] == ["alice-0001"]


def test_body_does_not_stop_on_a_colon_that_is_not_a_field():
    """A note is prose; a colon in it is ordinary. Stopping on any colon
    would truncate half the notes the fleet writes."""
    parsed = _parse_note(
        "/note \nmessage: check this\nTODO: fix the thing\nsee https://x.dev/a:b\ndone"
    )
    assert parsed is not None
    assert parsed["message"] == "check this\nTODO: fix the thing\nsee https://x.dev/a:b\ndone"


def test_blank_lines_inside_the_body_are_kept():
    parsed = _parse_note("/note \nmessage: para one\n\npara two")
    assert parsed is not None
    assert parsed["message"] == "para one\n\npara two"


def test_single_line_note_parses_as_before():
    parsed = _parse_note(
        "/note \ncolor: #c9f748\nlabel: Complete\nmessage: all done\nmentions: @bob-0001"
    )
    assert parsed == {"label": "Complete", "message": "all done", "mentions": ["bob-0001"]}


def test_note_without_a_body():
    parsed = _parse_note("/note \ncolor: #eee\nlabel: Note")
    assert parsed is not None
    assert parsed["message"] == ""


def test_non_note_content_still_rejected():
    assert _parse_note("/notebook something") is None
    assert _parse_note("hello") is None


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_round_trip_preserves_every_line(n: int):
    message = "\n".join(f"line {i + 1}" for i in range(n))
    parsed = _parse_note(note(message, mentions=["alice-0001"]))
    assert parsed is not None
    assert parsed["message"] == message
    assert parsed["mentions"] == ["alice-0001"]


def test_round_trip_with_blank_line_and_stray_colon():
    message = "intro\n\nNOTE: see below\ntail"
    parsed = _parse_note(note(message))
    assert parsed is not None
    assert parsed["message"] == message


def test_message_is_written_before_the_mentions_that_end_it():
    wire = note("a\nb", mentions=["x-0001"])
    assert wire.index("message:") < wire.index("mentions:")
