"""Phase 2 tests — CodexSession JSON-RPC plumbing.

The codex App Server is replaced by a fake subprocess we write
ourselves: a tiny Python script run via ``sys.executable`` that reads
JSON-RPC lines from stdin and emits scripted responses + notifications
on stdout. This is the same shape as the production binary from the
session's point of view, and it lets us assert end-to-end behaviour
without needing codex installed.

Coverage:

  * Conversation start: ``newConversation`` request → result with
    ``conversationId`` → persisted to ``codex_session.json``.
  * Single turn: ``sendUserTurn`` → ``item/agentMessage/delta`` deltas
    accumulated → ``turn/completed`` resolves the future, usage stats
    propagate to TurnResult.
  * Resume: second session with the same session file resumes the
    persisted conversation id.
  * Approval auto-bypass: server-initiated approval request is
    answered with ``{"decision": "approved"}`` without going through
    the agent loop.
  * Reload in-place: ``current_instructions`` is updated by
    ``reload``; next turn carries the new value.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from puffo_agent.agent.adapters.codex_session import CodexSession


# ─────────────────────────────────────────────────────────────────────────────
# Fake codex app-server
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario is a tiny Python script we hand to subprocess. The
# fake reads one JSON-RPC line at a time and writes scripted output.
# Kept inline (not a fixture file) so each test's expectations live
# next to the wire trace they care about.

FAKE_HEADER = textwrap.dedent('''\
    import json, sys

    def w(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    def r():
        line = sys.stdin.readline()
        return json.loads(line) if line else None

    def absorb_initialize():
        """Drain the JSON-RPC initialize handshake the session sends
        before any real method call. Tests don't care about the
        capability exchange — they assert against the method that
        follows."""
        msg = r()
        assert msg["method"] == "initialize", f"expected initialize, got {msg.get('method')!r}"
        w({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
''')


def _write_fake(tmp_path: Path, body: str) -> Path:
    """Write a fake codex app-server script to ``tmp_path`` and return
    its path. The script is invoked with python so we don't need a
    real binary."""
    path = tmp_path / "fake_codex.py"
    path.write_text(FAKE_HEADER + "\n" + body, encoding="utf-8")
    return path


def _argv_for(fake: Path) -> list[str]:
    return [sys.executable, str(fake)]


# ─────────────────────────────────────────────────────────────────────────────
# Single-turn happy path
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_TURN_SCRIPT = '''\
absorb_initialize()

# 1. Handle thread/start, return nested {thread: {id}} (real codex
#    shape per codex-rs/app-server). Verify the params don't carry
#    legacy ``instructions`` (codex doesn't accept that field).
msg = r()
assert msg["method"] == "thread/start"
assert "instructions" not in msg["params"]
w({"jsonrpc": "2.0", "id": msg["id"],
   "result": {"thread": {"id": "conv_42", "createdAt": "2026-05-15T00:00:00Z"}}})


# 2. Receive turn/start with structured ``input`` array
msg = r()
assert msg["method"] == "turn/start"
assert msg["params"]["threadId"] == "conv_42"
assert msg["params"]["input"] == [{"type": "text", "text": "hi there"}]
turn_id = msg["id"]

# 3. Stream two agentMessage deltas — codex's real shape puts the
#    text fragment at params.delta directly (NOT nested under
#    params.item.text — that was a wrong guess that lost most of the
#    streaming text in the first live run).
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "Hello, "}})
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "world!"}})

# 4. ACK the request (the App Server doesn't have to ACK before the
#    turn ends; we ACK here so the session's _send_raw_request future
#    resolves — the actual completion signal is turn/completed.)
w({"jsonrpc": "2.0", "id": turn_id, "result": None})

# 5. Final turn/completed with usage
w({"jsonrpc": "2.0", "method": "turn/completed",
   "params": {"usage": {"input_tokens": 12, "output_tokens": 4}}})

# 6. Wait for the session to close us (or a teardown)
while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_mcp_send_message_recorded_in_metadata(tmp_path):
    """When the agent invokes ``mcp__puffo__send_message`` (or its
    attachments sibling) and the call completes, the resulting
    TurnResult.metadata must carry a non-empty
    ``send_message_targets`` list. core.py reads that field to decide
    "agent already posted; skip the shell fallback" — without this
    detection the codex path silently triggers the
    is_visible_to_human=false fallback for every reply.
    """
    fake = _write_fake(tmp_path, '''\
absorb_initialize()
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

msg = r()  # turn/start
turn_id = msg["id"]

# Emit a completed mcpToolCall matching the real codex notification
# shape from the live debug log (server="puffo", tool="send_message",
# status="completed", arguments includes channel + root_id).
w({"jsonrpc": "2.0", "method": "item/completed",
   "params": {"item": {
       "type": "mcpToolCall",
       "id": "call_abc",
       "server": "puffo",
       "tool": "send_message",
       "status": "completed",
       "arguments": {
           "channel": "ch_test_123",
           "root_id": "msg_root_456",
           "text": "hi",
           "is_visible_to_human": True,
       },
   }}})

w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("system prompt")
        result = await cs.run_turn("hi", "system prompt")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    targets = result.metadata.get("send_message_targets") or []
    assert len(targets) == 1, targets
    assert targets[0]["channel"] == "ch_test_123"
    assert targets[0]["root_id"] == "msg_root_456"


def test_mcp_send_message_failed_status_does_not_record(tmp_path):
    """A failed/declined MCP tool call must NOT populate
    send_message_targets — otherwise the worker would skip its
    fallback and the agent's reply would silently disappear."""
    fake = _write_fake(tmp_path, '''\
absorb_initialize()
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

msg = r()
turn_id = msg["id"]

w({"jsonrpc": "2.0", "method": "item/completed",
   "params": {"item": {
       "type": "mcpToolCall",
       "server": "puffo", "tool": "send_message",
       "status": "failed",
       "arguments": {"channel": "ch_x", "root_id": "msg_x"},
   }}})
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m",
              "delta": "I tried but failed"}})

w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("system prompt")
        result = await cs.run_turn("hi", "system prompt")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    targets = result.metadata.get("send_message_targets") or []
    assert targets == [], targets


def test_single_turn_roundtrip(tmp_path):
    fake = _write_fake(tmp_path, SINGLE_TURN_SCRIPT)
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("system prompt v1")
        result = await cs.run_turn("hi there", "system prompt v1")
        await cs.aclose()
        return result

    result = asyncio.run(_run())

    assert result.reply == "Hello, world!"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.metadata["harness"] == "codex"
    assert result.metadata["conversation_id"] == "conv_42"

    # Persisted for the next process.
    persisted = json.loads(session_file.read_text(encoding="utf-8"))
    assert persisted["conversation_id"] == "conv_42"


def test_token_usage_from_thread_event(tmp_path):
    """Live codex reports per-turn tokens via ``thread/tokenUsage/updated``,
    not on ``turn/completed`` — and ``inputTokens`` bundles the re-sent cached
    history, so the recorded input must exclude ``cachedInputTokens``."""
    fake = _write_fake(tmp_path, '''\
absorb_initialize()
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

msg = r()  # turn/start
turn_id = msg["id"]

w({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
   "params": {"threadId": "c1", "turnId": "u1", "tokenUsage": {
       "last": {"inputTokens": 76544, "cachedInputTokens": 74624, "outputTokens": 21},
   }}})

w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {"turn": {"status": "completed"}}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("system prompt")
        result = await cs.run_turn("hi", "system prompt")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    assert result.input_tokens == 76544 - 74624  # cached history excluded
    assert result.output_tokens == 21


def test_context_snapshot_retained_without_active_turn(tmp_path):
    cs = CodexSession(
        "context-test", tmp_path / "session.json", ["true"],
    )

    async def drive():
        await cs._handle_notification("thread/tokenUsage/updated", {
            "threadId": "c1",
            "turnId": "t1",
            "tokenUsage": {
                "last": {
                    "totalTokens": 143262,
                    "inputTokens": 10,
                    "outputTokens": 2,
                },
                "total": {"inputTokens": 99, "outputTokens": 22},
                "modelContextWindow": 258400,
            },
        })
        return await cs.get_context_snapshot()

    snap = asyncio.run(drive())
    assert snap.used_tokens == 143262
    assert snap.context_window == 258400
    assert snap.source == "provider"
    assert cs._latest_usage_total == {"inputTokens": 99, "outputTokens": 22}


def test_compact_waits_for_matching_context_compaction_completion(tmp_path):
    cs = CodexSession("compact-test", tmp_path / "session.json", ["true"])
    cs._conversation_id = "thread-current"
    request_seen = asyncio.Event()

    async def fake_send(request_id, method, params):
        assert method == "thread/compact/start"
        assert params == {"threadId": "thread-current"}
        request_seen.set()
        return {}

    cs._send_raw_request = fake_send

    async def drive():
        task = asyncio.create_task(cs.compact_context())
        await request_seen.wait()
        await asyncio.sleep(0)
        assert not task.done()  # JSON-RPC acceptance is not completion.
        # Completion cannot establish its own identity.
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current",
            "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        assert not task.done()
        await cs._handle_notification("turn/started", {
            "threadId": "thread-current",
            "turn": {"id": "compact-turn"},
        })
        # Missing identity and wrong turn/item/type remain pending.
        for event in (
            {"threadId": "thread-current", "turnId": "", "item": {
                "id": "compact-item", "type": "contextCompaction",
            }},
            {"threadId": "thread-current", "turnId": "user-turn", "item": {
                "id": "compact-item", "type": "contextCompaction",
            }},
            {"threadId": "thread-current", "turnId": "compact-turn", "item": {
                "id": "wrong-type", "type": "agentMessage",
            }},
        ):
            await cs._handle_notification("item/started", event)
            assert not task.done()
        await cs._handle_notification("item/started", {
            "threadId": "thread-current",
            "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        await cs._handle_notification("item/completed", {
            "threadId": "wrong",
            "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        assert not task.done()
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current",
            "turnId": "compact-turn",
            "item": {"id": "wrong-item", "type": "contextCompaction"},
        })
        assert not task.done()
        await cs._handle_notification("thread/compacted", {
            "threadId": "thread-current",
            "turnId": "compact-turn",
        })
        assert not task.done()
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current",
            "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        return await task

    result = asyncio.run(drive())
    assert result.completed is True
    assert result.provider_session_id == "thread-current"
    assert result.before is not None and result.after is not None


def test_compact_timeout_cleans_state_without_rotation(tmp_path, monkeypatch):
    import puffo_agent.agent.adapters.codex_session as codex_module

    monkeypatch.setattr(codex_module, "COMPACT_TIMEOUT_SECONDS", 0.001)
    cs = CodexSession("compact-timeout", tmp_path / "session.json", ["true"])
    cs._conversation_id = "thread-still-valid"

    async def fake_send(request_id, method, params):
        return {}  # accepted, but no contextCompaction completion follows

    cs._send_raw_request = fake_send
    result = asyncio.run(cs.compact_context())
    assert result.completed is False
    assert "timed out" in result.diagnostic
    assert cs._pending_compact is None
    assert cs.get_provider_session_id() == "thread-still-valid"
    assert cs._consecutive_thread_failures == 0


def test_admission_fires_only_on_correlated_user_turn_started(tmp_path):
    from puffo_agent.agent.adapters.codex_session import _PendingTurn

    async def drive():
        cs = CodexSession("admission-test", tmp_path / "session.json", ["true"])
        cs._conversation_id = "thread-current"
        events = []

        async def admitted(event):
            events.append(event)

        cs.register_admission_callback(admitted, "cycle-c")
        cs._active_turn = _PendingTurn(1, time.time())
        assert events == []  # registration / request ACK does not admit.
        # A JSON-RPC ACK is handled outside notification dispatch and therefore
        # cannot consume provider admission.
        assert cs._admission_callback is admitted
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current",
            "turnId": "other",
            "item": {"id": "i", "type": "agentMessage"},
        })
        await cs._handle_notification("turn/completed", {
            "threadId": "wrong",
            "turn": {"id": "wrong-turn"},
        })
        assert events == []
        await cs._handle_notification("turn/started", {
            "threadId": "wrong",
            "turn": {"id": "wrong-turn"},
        })
        assert events == []
        await cs._handle_notification("turn/started", {
            "threadId": "thread-current",
            "turn": {"id": "user-turn"},
        })
        await cs._handle_notification("turn/started", {
            "threadId": "thread-current",
            "turn": {"id": "user-turn"},
        })
        assert len(events) == 1
        assert events[0].provider_turn_id == "user-turn"

    asyncio.run(drive())


def test_initial_admission_ignores_compact_and_continuation_is_distinct(tmp_path):
    from puffo_agent.agent.adapters.codex_session import _PendingCompact, _PendingTurn

    async def drive():
        cs = CodexSession("continuation-test", tmp_path / "session.json", ["true"])
        cs._conversation_id = "thread-current"
        events = []

        async def admitted(event):
            events.append(event)

        # Compact notifications with no active user turn do not consume the
        # independently registered initial callback.
        cs.register_admission_callback(admitted, "initial")
        cs._pending_compact = _PendingCompact(
            "thread-current",
            await cs.get_context_snapshot(),
            asyncio.get_running_loop().create_future(),
        )
        await cs._handle_notification("turn/started", {
            "threadId": "thread-current", "turn": {"id": "compact-turn"},
        })
        await cs._handle_notification("item/started", {
            "threadId": "thread-current", "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current", "turnId": "compact-turn",
            "item": {"id": "compact-item", "type": "contextCompaction"},
        })
        assert events == []
        cs._pending_compact = None

        user_turn = _PendingTurn(2, time.time())
        cs._active_turn = user_turn
        await cs._handle_notification("turn/started", {
            "threadId": "thread-current", "turn": {"id": "user-turn"},
        })
        assert [event.planning_cycle_key for event in events] == ["initial"]

        # Registration after turn start is correlated tool-result admission.
        cs.register_continuation_callback(
            admitted, "continuation", channel_id="ch-a",
        )
        for params in (
            {"threadId": "wrong", "turnId": "user-turn",
             "item": {"id": "m1", "type": "mcpToolCall", "server": "puffo",
                      "tool": "send_message", "status": "completed",
                      "arguments": {"channel": "ch-a"}}},
            {"threadId": "thread-current", "turnId": "wrong-turn",
             "item": {"id": "m2", "type": "mcpToolCall", "server": "puffo",
                      "tool": "send_message", "status": "completed",
                      "arguments": {"channel": "ch-a"}}},
            {"threadId": "thread-current", "turnId": "user-turn",
             "item": {"id": "c1", "type": "contextCompaction"}},
            {"threadId": "thread-current", "turnId": "user-turn",
             "item": {"id": "m3", "type": "mcpToolCall", "server": "puffo",
                      "tool": "send_message", "status": "completed",
                      "arguments": {"channel": "ch-other"}}},
        ):
            await cs._handle_notification("item/completed", params)
        assert len(events) == 1
        # Live Codex item/completed events may contain only params.item.
        # Missing correlation IDs inherit the current active turn; supplied
        # non-empty IDs above still have to match.
        accepted = {
            "item": {
                "id": "m4", "type": "mcpToolCall", "server": "puffo",
                "tool": "send_message", "status": "completed",
                "arguments": {"channel": "ch-a"},
            },
        }
        await cs._handle_notification("item/completed", accepted)
        await cs._handle_notification("item/completed", accepted)
        assert [event.planning_cycle_key for event in events] == [
            "initial", "continuation",
        ]
        assert events[-1].provider_turn_id == "user-turn"

        cs.register_continuation_callback(
            admitted, "continuation-a", channel_id="ch-a",
        )
        cs.register_continuation_callback(
            admitted, "continuation-b", channel_id="ch-b",
        )
        await cs._handle_notification("turn/completed", {
            "threadId": "thread-current", "turn": {"id": "user-turn"},
        })
        assert len(events) == 2
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current", "turnId": "user-turn",
            "item": {
                "id": "b", "type": "mcpToolCall", "server": "puffo",
                "tool": "send_message", "status": "completed",
                "arguments": {"channel": "ch-b"},
            },
        })
        await cs._handle_notification("item/completed", {
            "threadId": "thread-current", "turnId": "user-turn",
            "item": {
                "id": "a", "type": "mcpToolCall", "server": "puffo",
                "tool": "send_message", "status": "completed",
                "arguments": {"channel": "ch-a"},
            },
        })
        assert [event.planning_cycle_key for event in events] == [
            "initial", "continuation", "continuation-b", "continuation-a",
        ]

    asyncio.run(drive())


def test_token_usage_sums_multi_request_turn(tmp_path):
    """A turn with several model requests reports the whole turn's usage (the
    thread total's delta), not just the last request."""
    fake = _write_fake(tmp_path, '''\
absorb_initialize()
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})
msg = r()  # turn/start
turn_id = msg["id"]

w({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
   "params": {"tokenUsage": {
       "last": {"inputTokens": 1005, "cachedInputTokens": 1000, "outputTokens": 100},
       "total": {"inputTokens": 5005, "cachedInputTokens": 5000, "outputTokens": 100}}}})
w({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
   "params": {"tokenUsage": {
       "last": {"inputTokens": 1008, "cachedInputTokens": 1000, "outputTokens": 110},
       "total": {"inputTokens": 6013, "cachedInputTokens": 6000, "outputTokens": 210}}}})

w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {"turn": {"status": "completed"}}})
while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("system prompt")
        result = await cs.run_turn("hi", "system prompt")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    assert result.output_tokens == 210  # 100 + 110, not just the last request's 110
    assert result.input_tokens == 13    # cumulative non-cached, not just the last


# ─────────────────────────────────────────────────────────────────────────────
# Resume happy path — second instance picks up the persisted id
# ─────────────────────────────────────────────────────────────────────────────

RESUME_SCRIPT = '''\
absorb_initialize()

msg = r()
assert msg["method"] == "thread/resume", f"expected thread/resume, got {msg['method']}"
assert msg["params"]["threadId"] == "conv_42"
w({"jsonrpc": "2.0", "id": msg["id"],
   "result": {"thread": {"id": "conv_42"}}})

msg = r()
assert msg["method"] == "turn/start"
turn_id = msg["id"]
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "resumed"}})
w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_resume_existing_conversation(tmp_path):
    fake = _write_fake(tmp_path, RESUME_SCRIPT)
    session_file = tmp_path / "codex_session.json"
    session_file.write_text(json.dumps({"conversation_id": "conv_42"}))

    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        result = await cs.run_turn("next turn", "system prompt")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    assert result.reply == "resumed"


# ─────────────────────────────────────────────────────────────────────────────
# Approval auto-bypass
# ─────────────────────────────────────────────────────────────────────────────

APPROVAL_SCRIPT = '''\
import time
absorb_initialize()

# thread/start
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

# turn/start
msg = r()
turn_id = msg["id"]

# Server-initiated request: codex's new MCP elicitation contract
# (mcpServer/elicitation/request). Reply shape per app-server README:
#   accept  → {"action": "accept",  "content": {}}
#   decline → {"action": "decline", "content": null}
w({"jsonrpc": "2.0", "id": 9001, "method": "mcpServer/elicitation/request",
   "params": {
       "threadId": "c1",
       "serverName": "puffo",
       "meta": {"codex_approval_kind": "mcp_tool_call"},
   }})

reply = r()
assert reply["id"] == 9001
assert reply["result"]["action"] == "accept", reply
assert reply["result"]["content"] == {}, reply

# Also send the codex-canonical command-execution approval method
# (item/commandExecution/requestApproval). Different response shape
# from the MCP elicitation above: {decision: "accept" | ...}.
w({"jsonrpc": "2.0", "id": 9002,
   "method": "item/commandExecution/requestApproval",
   "params": {"command": ["rm", "-rf", "/"]}})

reply = r()
assert reply["id"] == 9002
assert reply["result"]["decision"] == "accept", reply

# And file-change approval — same {decision: ...} shape as exec.
w({"jsonrpc": "2.0", "id": 9003,
   "method": "item/fileChange/requestApproval",
   "params": {"path": "/some/file"}})

reply = r()
assert reply["id"] == 9003
assert reply["result"]["decision"] == "accept", reply

# Now complete the turn
w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "did the thing"}})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_approval_auto_bypass(tmp_path):
    fake = _write_fake(tmp_path, APPROVAL_SCRIPT)
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
        permission_mode="bypassPermissions",
    )

    async def _run():
        await cs.warm("sys")
        result = await cs.run_turn("do it", "sys")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    assert result.reply == "did the thing"


# ─────────────────────────────────────────────────────────────────────────────
# Turn failure surfaces as exception
# ─────────────────────────────────────────────────────────────────────────────

FAIL_SCRIPT = '''\
absorb_initialize()

msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

msg = r()
turn_id = msg["id"]
w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/failed",
   "params": {"error": {"message": "model overloaded"}}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_turn_failed_raises(tmp_path):
    fake = _write_fake(tmp_path, FAIL_SCRIPT)
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        try:
            await cs.run_turn("hi", "sys")
            await cs.aclose()
            return None
        except RuntimeError as exc:
            await cs.aclose()
            return str(exc)

    err = asyncio.run(_run())
    assert err is not None
    assert "model overloaded" in err


RECONNECT_SCRIPT = '''\
absorb_initialize()

msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

msg = r()
turn_id = msg["id"]
w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/failed",
   "params": {"error": {"message": "Reconnecting... 2/5"}}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_reconnect_turn_routes_to_retry_not_drop(tmp_path):
    # Non-auth AgentAPIError = the consumer's re-enqueue path; a raw
    # RuntimeError would be swallowed + dropped by the worker.
    from puffo_agent.agent.core import AgentAPIError

    fake = _write_fake(tmp_path, RECONNECT_SCRIPT)
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        try:
            await cs.run_turn("hi", "sys")
            return ("none", None)
        except AgentAPIError as exc:
            return ("apierror", exc)
        except RuntimeError as exc:
            return ("runtime", exc)
        finally:
            await cs.aclose()

    kind, exc = asyncio.run(_run())
    assert kind == "apierror", f"expected AgentAPIError retry route, got {kind}"
    assert exc.is_auth is False
    assert "Reconnecting" in str(exc)
    # Transient — not a wedged strike.
    assert cs._consecutive_thread_failures == 0


def test_looks_like_codex_reconnect():
    from puffo_agent.agent.adapters.codex_session import _looks_like_codex_reconnect
    assert _looks_like_codex_reconnect("codex turn failed: Reconnecting... 2/5")
    assert _looks_like_codex_reconnect("reconnecting to backend")
    assert not _looks_like_codex_reconnect("model overloaded")
    assert not _looks_like_codex_reconnect("agent thread limit reached")
    assert not _looks_like_codex_reconnect("")


RECONNECT_THEN_RECOVER_SCRIPT = '''\
absorb_initialize()

msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

# Turn 1 fails mid-reconnect.
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": None})
w({"jsonrpc": "2.0", "method": "turn/failed",
   "params": {"error": {"message": "Reconnecting... 2/5"}}})

# Turn 2 (the retry) succeeds on the same thread.
msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": None})
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "c1", "turnId": "u2", "itemId": "m", "delta": "recovered"}})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_reconnect_then_retry_recovers_on_same_session(tmp_path):
    # Mirrors the worker retry path: run_retry_turn re-enters run_turn on the
    # same session after the AgentAPIError.
    from puffo_agent.agent.core import AgentAPIError

    fake = _write_fake(tmp_path, RECONNECT_THEN_RECOVER_SCRIPT)
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        try:
            with pytest.raises(AgentAPIError):
                await cs.run_turn("hi", "sys")
            return await cs.run_turn("hi again", "sys")
        finally:
            await cs.aclose()

    result = asyncio.run(_run())
    assert result.reply == "recovered"
    assert cs._consecutive_thread_failures == 0


TIMEOUT_SCRIPT = '''\
absorb_initialize()

msg = r()
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "t1"}}})

# ACK every request (turn/start, turn/interrupt) but never complete a turn.
while True:
    msg = r()
    if msg is None:
        break
    if msg.get("id") is not None:
        w({"jsonrpc": "2.0", "id": msg["id"], "result": None})
'''


def test_timeout_reply_reset_claim_matches_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path / "home"))
    fake = _write_fake(tmp_path, TIMEOUT_SCRIPT)
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
        task_timeout_seconds=1.0,
    )

    async def _run():
        await cs.warm("sys")
        try:
            r1 = await cs.run_turn("hi", "sys")
            r2 = await cs.run_turn("still there?", "sys")
            return r1, r2
        finally:
            await cs.aclose()

    r1, r2 = asyncio.run(_run())
    # First timeout: below the wedged threshold — no rotation, no reset claim.
    assert "1-second timeout" in r1.reply
    assert r1.metadata["codex_turn_timeout"] is True
    assert r1.metadata["codex_thread_rotated"] is False
    assert "reset" not in r1.reply
    # Second consecutive timeout rotates; the reply may now claim the reset.
    assert r2.metadata["codex_thread_rotated"] is True
    assert "reset for the next turn" in r2.reply


# ─────────────────────────────────────────────────────────────────────────────
# reload() updates current_instructions without restarting the process
# ─────────────────────────────────────────────────────────────────────────────

RELOAD_SCRIPT = '''\
absorb_initialize()

# thread/start no longer carries instructions — codex reads AGENTS.md
# directly. The reload path mutates current_instructions but that
# field is now used only for future ``personality`` overrides; tests
# verify the call shape doesn't regress to passing instructions in
# thread/start or turn/start.
msg = r()
assert msg["method"] == "thread/start"
assert "instructions" not in msg["params"]
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})

# First turn
msg = r()
assert msg["method"] == "turn/start"
assert "instructions" not in msg["params"]
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "turn1"}})
w({"jsonrpc": "2.0", "id": msg["id"], "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

# Second turn after reload() — same call shape; reload mutated
# current_instructions but turn/start no longer carries it.
msg = r()
assert msg["method"] == "turn/start"
assert "instructions" not in msg["params"]
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u2", "itemId": "m2", "delta": "turn2"}})
w({"jsonrpc": "2.0", "id": msg["id"], "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_reload_tears_down_process_for_respawn(tmp_path):
    """reload() must tear the app-server process down so the next
    turn respawns it with a fresh ``config.toml`` read. Without that,
    new MCP entries (``install_host_mcp`` → ``sync_host_mcp`` flow)
    never reach the running codex thread."""
    fake = _write_fake(tmp_path, RELOAD_SCRIPT)
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("v1")
        assert cs._proc is not None
        proc_before = cs._proc
        await cs.reload("v2")
        # After reload the app-server process is gone — the next
        # run_turn will spawn a fresh one.
        assert cs._proc is None
        # current_instructions snapshot still flips so the next
        # sendUserTurn carries the v2 prompt without an extra
        # round-trip.
        assert cs.current_instructions == "v2"
        await cs.aclose()
        return proc_before

    proc_before = asyncio.run(_run())
    # Returncode populates synchronously after _teardown_locked
    # awaits the proc's exit.
    assert proc_before.returncode is not None


# ─────────────────────────────────────────────────────────────────────────────
# Teardown — stdin close → graceful exit (no TerminateProcess)
# ─────────────────────────────────────────────────────────────────────────────


_GRACEFUL_TEARDOWN_SCRIPT = '''\
import sys

absorb_initialize()
msg = r()  # thread/start
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})
while True:
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
'''


def test_aclose_closes_stdin_and_subprocess_self_exits(tmp_path):
    """aclose must close stdin so the subprocess sees EOF and Drops its
    own resources, NOT bypass via TerminateProcess."""
    fake = _write_fake(tmp_path, _GRACEFUL_TEARDOWN_SCRIPT)
    cs = CodexSession(
        agent_id="alice-teardown-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        await cs.aclose()

    asyncio.run(_run())


def test_aclose_falls_back_to_terminate_if_subprocess_ignores_eof(tmp_path):
    """Misbehaving server that ignores EOF must not pin the archive
    path; aclose escalates within bounded time."""
    fake = _write_fake(tmp_path, '''\
import time

absorb_initialize()
msg = r()  # thread/start
w({"jsonrpc": "2.0", "id": msg["id"], "result": {"thread": {"id": "c1"}}})
while True:
    time.sleep(0.5)
''')
    cs = CodexSession(
        agent_id="alice-teardown-stubborn-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        t0 = time.monotonic()
        await cs.aclose()
        return time.monotonic() - t0

    # 10s graceful + 3s terminate window + slack for slow CI.
    elapsed = asyncio.run(_run())
    assert elapsed < 20.0, f"aclose hung for {elapsed:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Recovery when conversation_id loads empty (silent-wedge guard)
# ─────────────────────────────────────────────────────────────────────────────


def test_load_conversation_id_failsoft_on_corrupt_json(tmp_path):
    """Corrupt session JSON loads as ``""`` — the signal
    ``_ensure_running`` keys on to recover."""
    session_file = tmp_path / "codex_session.json"
    session_file.write_text("not-valid-json{", encoding="utf-8")
    cs = CodexSession.__new__(CodexSession)
    cs.session_file = session_file
    assert cs._load_conversation_id() == ""


def test_ensure_running_with_empty_cid_and_alive_proc_respawns(tmp_path):
    """Alive proc but empty cid (corrupt load + warm-spawn race): tear
    the proc down and respawn so the thread is re-established."""
    cs = CodexSession.__new__(CodexSession)
    cs.agent_id = "empty-cid"
    cs._conversation_id = ""
    cs.current_instructions = None

    class _FakeProc:
        returncode = None

    cs._proc = _FakeProc()

    calls = {"teardown": 0, "spawn": 0}

    async def _stub_teardown():
        calls["teardown"] += 1
        cs._proc = None

    async def _stub_spawn():
        calls["spawn"] += 1
        cs._conversation_id = "conv_fresh"
        cs._proc = _FakeProc()

    cs._teardown_locked = _stub_teardown  # type: ignore[assignment]
    cs._spawn = _stub_spawn  # type: ignore[assignment]

    asyncio.run(cs._ensure_running("sys"))

    assert calls["teardown"] == 1
    assert calls["spawn"] == 1
    assert cs._conversation_id == "conv_fresh"


def test_ensure_running_with_non_empty_cid_and_alive_proc_is_noop(tmp_path):
    cs = CodexSession.__new__(CodexSession)
    cs.agent_id = "warm-noop"
    cs._conversation_id = "conv_existing"
    cs.current_instructions = None

    class _FakeProc:
        returncode = None

    cs._proc = _FakeProc()

    calls = {"teardown": 0, "spawn": 0}

    async def _stub_teardown():
        calls["teardown"] += 1

    async def _stub_spawn():
        calls["spawn"] += 1

    cs._teardown_locked = _stub_teardown  # type: ignore[assignment]
    cs._spawn = _stub_spawn  # type: ignore[assignment]

    asyncio.run(cs._ensure_running("sys"))

    assert calls["teardown"] == 0
    assert calls["spawn"] == 0
    assert cs._conversation_id == "conv_existing"


def test_ensure_running_with_dead_proc_spawns_without_teardown(tmp_path):
    cs = CodexSession.__new__(CodexSession)
    cs.agent_id = "cold-start"
    cs._conversation_id = ""
    cs.current_instructions = None
    cs._proc = None

    calls = {"teardown": 0, "spawn": 0}

    async def _stub_teardown():
        calls["teardown"] += 1

    async def _stub_spawn():
        calls["spawn"] += 1
        cs._conversation_id = "conv_new"

    cs._teardown_locked = _stub_teardown  # type: ignore[assignment]
    cs._spawn = _stub_spawn  # type: ignore[assignment]

    asyncio.run(cs._ensure_running("sys"))

    assert calls["teardown"] == 0
    assert calls["spawn"] == 1
    assert cs._conversation_id == "conv_new"


def test_run_turn_raises_when_cid_stays_empty(tmp_path):
    """Defence-in-depth: if ``_ensure_running`` ever returns without a
    cid, ``run_turn`` raises rather than sending ``threadId=""``."""
    cs = CodexSession.__new__(CodexSession)
    cs.agent_id = "fail-loud"
    cs._conversation_id = ""
    cs._lock = asyncio.Lock()
    cs._next_id = 1
    cs._active_turn = None
    cs.current_instructions = None

    async def _stub_ensure_running(_system_prompt):
        pass

    cs._ensure_running = _stub_ensure_running  # type: ignore[assignment]

    async def _run():
        return await cs.run_turn("hi", "sys")

    with pytest.raises(RuntimeError, match="empty conversation_id after _ensure_running"):
        asyncio.run(_run())


def test_corrupt_session_file_recovers_via_fresh_thread(tmp_path):
    """A corrupt session file + cold start recovers through the
    thread/start branch — no manual delete needed."""
    fake = _write_fake(tmp_path, '''\
absorb_initialize()

msg = r()
assert msg["method"] == "thread/start", f"unexpected first method {msg.get('method')!r}"
w({"jsonrpc": "2.0", "id": msg["id"],
   "result": {"thread": {"id": "conv_recovered", "createdAt": "2026-06-13T23:58:00Z"}}})

msg = r()  # turn/start
assert msg["method"] == "turn/start"
assert msg["params"]["threadId"] == "conv_recovered"
turn_id = msg["id"]
w({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
   "params": {"threadId": "t", "turnId": "u", "itemId": "m", "delta": "back"}})
w({"jsonrpc": "2.0", "id": turn_id, "result": None})
w({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    session_file = tmp_path / "codex_session.json"
    session_file.write_text("partial-corrupt{not-json", encoding="utf-8")

    cs = CodexSession(
        agent_id="corrupt-session",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        await cs.warm("sys")
        result = await cs.run_turn("are you there?", "sys")
        await cs.aclose()
        return result

    result = asyncio.run(_run())
    assert "back" in (result.reply or "")
    persisted = json.loads(session_file.read_text(encoding="utf-8"))
    assert persisted.get("conversation_id") == "conv_recovered"


def test_bootstrap_raises_when_thread_start_returns_no_id(tmp_path):
    """``_ensure_running``'s post-call invariant (alive proc + non-empty
    cid) relies on ``_bootstrap_session`` raising when thread/start
    returns no id — guard it so the session can't limp on empty."""
    fake = _write_fake(tmp_path, '''\
absorb_initialize()
msg = r()
assert msg["method"] == "thread/start"
# Structurally valid result, but no thread id under any key.
w({"jsonrpc": "2.0", "id": msg["id"],
   "result": {"thread": {"createdAt": "2026-06-13T00:00:00Z"}}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
''')
    session_file = tmp_path / "codex_session.json"
    cs = CodexSession(
        agent_id="bootstrap-no-id",
        session_file=session_file,
        argv=_argv_for(fake),
        cwd=str(tmp_path),
    )

    async def _run():
        try:
            await cs.warm("sys")
            return None
        except RuntimeError as exc:
            return str(exc)
        finally:
            await cs.aclose()

    err = asyncio.run(_run())
    assert err is not None, "warm should propagate the bootstrap failure"
    assert "no thread id" in err, err
    assert cs._proc is None
    assert cs._conversation_id == ""


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox policy — thread/start carries the configured sandbox; adapter
# sanitises unknown values
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_SCRIPT = '''\
absorb_initialize()

msg = r()
assert msg["method"] == "thread/start"
assert msg["params"]["sandbox"] == "workspace-write", \\
    f"sandbox was {msg['params'].get('sandbox')!r}"
assert msg["params"]["approvalPolicy"] == "never"
w({"jsonrpc": "2.0", "id": msg["id"],
   "result": {"thread": {"id": "conv_sb", "createdAt": "2026-05-15T00:00:00Z"}}})

while True:
    line = sys.stdin.readline()
    if not line:
        break
'''


def test_thread_start_carries_configured_sandbox(tmp_path):
    fake = _write_fake(tmp_path, SANDBOX_SCRIPT)
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=_argv_for(fake),
        cwd=str(tmp_path),
        sandbox="workspace-write",
    )

    async def _run():
        # The fake asserts sandbox/approvalPolicy on thread/start — a
        # mismatch crashes it, so warm() fails.
        await cs.warm("system prompt v1")
        await cs.aclose()

    asyncio.run(_run())


def test_sanitise_sandbox_falls_back_on_unknown():
    from puffo_agent.agent.adapters.local_cli import _sanitise_sandbox

    assert _sanitise_sandbox("workspace-write", "a") == "workspace-write"
    assert _sanitise_sandbox("read-only", "a") == "read-only"
    assert _sanitise_sandbox("danger-full-access", "a") == "danger-full-access"
    assert _sanitise_sandbox("bogus", "a") == "danger-full-access"
    assert _sanitise_sandbox("", "a") == "danger-full-access"


def test_codex_sandbox_change_resets_persisted_thread(tmp_path):
    sf = tmp_path / "codex_session.json"
    sf.write_text(
        json.dumps({"conversation_id": "old_thread", "sandbox": "danger-full-access"}),
        encoding="utf-8",
    )
    cs = CodexSession(
        agent_id="a", session_file=sf, argv=["x"], sandbox="workspace-write",
    )
    assert cs._conversation_id == ""  # changed → fresh thread next start


def test_codex_same_sandbox_keeps_persisted_thread(tmp_path):
    sf = tmp_path / "codex_session.json"
    sf.write_text(
        json.dumps({"conversation_id": "old_thread", "sandbox": "workspace-write"}),
        encoding="utf-8",
    )
    cs = CodexSession(
        agent_id="a", session_file=sf, argv=["x"], sandbox="workspace-write",
    )
    assert cs._conversation_id == "old_thread"  # unchanged → resume


def test_codex_legacy_session_file_treated_as_full_access(tmp_path):
    # Pre-feature file: only conversation_id, no sandbox → danger-full-access.
    sf = tmp_path / "codex_session.json"
    sf.write_text(json.dumps({"conversation_id": "old_thread"}), encoding="utf-8")
    keep = CodexSession(agent_id="a", session_file=sf, argv=["x"])
    assert keep._conversation_id == "old_thread"  # still full-access → resume
    reset = CodexSession(
        agent_id="a", session_file=sf, argv=["x"], sandbox="workspace-write",
    )
    assert reset._conversation_id == ""  # now differs → reset


def test_account_ratelimits_frame_pushes_to_reporter(tmp_path, monkeypatch):
    from puffo_agent.portal.control import reporter as reporter_mod

    rep = reporter_mod.AgentStatusReporter()
    monkeypatch.setattr(reporter_mod, "get_reporter", lambda: rep)
    cs = CodexSession(
        agent_id="alice-test-0001",
        session_file=tmp_path / "codex_session.json",
        argv=["x"],
        cwd=str(tmp_path),
    )
    raw = {"primary": {"usedPercent": 8, "windowDurationMins": 300, "resetsAt": 9}}
    asyncio.run(cs._handle_notification("account/rateLimits/updated", {"rateLimits": raw}))
    assert rep.latest_codex_rate_limits() == raw
