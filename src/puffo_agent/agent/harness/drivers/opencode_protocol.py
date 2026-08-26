"""Pinned OpenCode ``run --format json`` command and event shapes."""

from __future__ import annotations

from typing import Any

from ..driver import (
    HarnessEvent,
    HarnessEventType,
    RuntimeSpec,
    SessionRef,
    TurnRef,
)


def build_opencode_run_command(
    spec: RuntimeSpec,
    *,
    prompt: str,
    native_session_id: str = "",
) -> tuple[str, ...]:
    """Build one per-turn child command; OpenCode receives no stdin input."""
    command = [
        spec.executable or "opencode",
        "run",
        "--format",
        "json",
        "--dir",
        spec.workspace_dir,
    ]
    if spec.model:
        command.extend(("--model", spec.model))
    if native_session_id:
        command.extend(("--session", native_session_id))
    if spec.permission_mode == "bypassPermissions":
        command.append("--auto")
    command.extend(spec.launch_args)
    command.append(prompt)
    return tuple(command)


def normalize_opencode_frame(
    frame: dict[str, Any],
    *,
    session_ref: SessionRef,
    turn_ref: TurnRef,
) -> tuple[HarnessEvent, ...]:
    """Normalize one documented OpenCode JSON line without leaking payloads."""
    type_ = str(frame.get("type") or "")
    part = frame.get("part")
    part = part if isinstance(part, dict) else {}
    native_session_id = str(frame.get("sessionID") or part.get("sessionID") or "")
    native_turn_id = str(part.get("messageID") or "")

    def event(type_: HarnessEventType, data: dict[str, Any]) -> HarnessEvent:
        return HarnessEvent.normalized(
            type=type_,
            driver="opencode",
            session_ref=session_ref,
            turn_ref=turn_ref,
            native_session_id=native_session_id,
            native_turn_id=native_turn_id,
            data=data,
            native_payload=frame,
        )

    if type_ == "step_start":
        return (event(HarnessEventType.TURN_STARTED, {}),)
    if type_ == "text":
        block_id = str(part.get("id") or "")
        text = str(part.get("text") or "")
        return (
            event(
                HarnessEventType.ASSISTANT_DELTA,
                {"block_id": block_id, "delta": text},
            ),
            event(HarnessEventType.ASSISTANT_COMPLETED, {"block_id": block_id}),
        )
    if type_ == "tool_use":
        state = part.get("state")
        state = state if isinstance(state, dict) else {}
        status = str(state.get("status") or "")
        return (
            event(
                HarnessEventType.TOOL_COMPLETED,
                {
                    "tool_call_ref": str(part.get("id") or ""),
                    "label": _tool_label(part),
                    "outcome": "succeeded" if status == "completed" else "failed",
                },
            ),
        )
    if type_ == "step_finish":
        return (
            event(HarnessEventType.CONTEXT_UPDATED, _usage_data(part)),
        )
    if type_ == "error":
        return (
            event(
                HarnessEventType.RUNTIME_FAILED,
                {"code": "opencode_run_error"},
            ),
        )
    return (
        event(
            HarnessEventType.SESSION_UPDATED,
            {"record_type": type_ or "unknown"},
        ),
    )


def _tool_label(part: dict[str, Any]) -> str:
    name = str(part.get("tool") or "")
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


def _usage_data(part: dict[str, Any]) -> dict[str, int]:
    tokens = part.get("tokens")
    tokens = tokens if isinstance(tokens, dict) else {}
    cache = tokens.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    return {
        "input_tokens": _nonnegative_int(tokens.get("input")),
        "output_tokens": _nonnegative_int(tokens.get("output")),
        "reasoning_tokens": _nonnegative_int(tokens.get("reasoning")),
        "cache_read_tokens": _nonnegative_int(cache.get("read")),
        "cache_write_tokens": _nonnegative_int(cache.get("write")),
        # Measured on 1.18.16: step_finish tokens.total is the full context
        # occupancy of the step's provider call (input + cache.read +
        # output), not a session-cumulative counter — the one number
        # context tracking needs. With several tool steps in a turn the
        # last step_finish carries the current value.
        "total_tokens": _nonnegative_int(tokens.get("total")),
    }


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
