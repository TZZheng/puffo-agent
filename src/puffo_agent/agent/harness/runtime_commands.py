"""Validated, idempotent delivery of Runtime Manager commands."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping

from .._logging import log_runtime_event
from .driver import (
    PermissionDecision,
    PermissionRef,
    TurnRef,
    UnsupportedCapability,
)
from .runtime_manager import RuntimeStateError, get_runtime_manager

logger = logging.getLogger(__name__)

CommandResult = dict[str, object]
RESULT_CACHE_CAPACITY = 256
_RESULTS: OrderedDict[
    tuple[str, str], tuple[dict[str, object], CommandResult]
] = OrderedDict()
_TERMINAL_VALIDATION_ERRORS = frozenset({
    "unsupported_version",
    "invalid_command",
    "foreign_agent",
    "unsupported_operation",
    "invalid_target",
    "invalid_decision",
})


def _failure(code: str) -> CommandResult:
    return {"ok": False, "error": code, "error_code": code}


def _required_text(command: Mapping[str, object], name: str) -> str:
    value = command.get(name)
    return value if isinstance(value, str) and value.strip() else ""


def _cache_result(
    key: tuple[str, str],
    command: dict[str, object],
    result: CommandResult,
) -> None:
    if not (
        result.get("delivered") is True
        or result.get("error_code") in _TERMINAL_VALIDATION_ERRORS
    ):
        return
    _RESULTS[key] = (command, dict(result))
    _RESULTS.move_to_end(key)
    while len(_RESULTS) > RESULT_CACHE_CAPACITY:
        _RESULTS.popitem(last=False)


async def execute_runtime_command(
    command: Mapping[str, object],
    *,
    expected_agent_id: str,
    manager_agent_id: str,
    require_version: bool,
    permission_turn_from_active: bool,
    cloud_wire: bool = False,
) -> CommandResult:
    """Validate and deliver one command; completion remains event-driven."""
    command_id = _required_text(command, "command_id")
    wire_op = _required_text(command, "op")
    agent_id = _required_text(command, "agent_id")
    session_ref = _required_text(command, "session_ref")
    operator_slug = _required_text(command, "operator_slug")
    cache_key = (manager_agent_id, command_id)
    command_copy = dict(command)

    common = {"command_id", "agent_id", "session_ref", "op"}
    if require_version:
        common.add("version")
    if cloud_wire:
        common.add("operator_slug")
    expected_cancel_op = "runtime.cancel_turn" if cloud_wire else "cancel_turn"
    expected_permission_op = (
        "runtime.resolve_permission" if cloud_wire else "resolve_permission"
    )
    if wire_op == expected_cancel_op:
        allowed = common | {"turn_ref"}
    elif wire_op == expected_permission_op:
        allowed = common | {"permission_ref", "decision"}
        if not permission_turn_from_active:
            allowed.add("turn_ref")
    else:
        allowed = common
    if require_version and command.get("version") != 1:
        result = _failure("unsupported_version")
    elif set(command) != allowed:
        result = _failure("invalid_command")
    elif (
        not command_id
        or not agent_id
        or not session_ref
        or (cloud_wire and not operator_slug)
    ):
        result = _failure("invalid_command")
    elif agent_id != expected_agent_id:
        result = _failure("foreign_agent")
    elif wire_op not in {expected_cancel_op, expected_permission_op}:
        result = _failure("unsupported_operation")
    elif command_id and cache_key in _RESULTS:
        cached_command, cached_result = _RESULTS[cache_key]
        _RESULTS.move_to_end(cache_key)
        result = (
            dict(cached_result)
            if cached_command == command_copy
            else _failure("command_id_conflict")
        )
    else:
        op = wire_op.removeprefix("runtime.") if cloud_wire else wire_op
        manager = get_runtime_manager(manager_agent_id)
        if manager is None:
            result = _failure("runtime_unavailable")
        elif session_ref != str(manager.session_ref):
            result = _failure("stale_session_ref")
        else:
            turn_text = _required_text(command, "turn_ref")
            active_missing = False
            if op == "resolve_permission" and permission_turn_from_active:
                active = manager.active_turn_ref
                if active is None:
                    active_missing = True
                    turn_text = ""
                else:
                    turn_text = str(active)
            target = (
                _required_text(command, "turn_ref")
                if op == "cancel_turn"
                else _required_text(command, "permission_ref")
            )
            decision = _required_text(command, "decision")
            if active_missing:
                result = _failure("runtime_unavailable")
            elif not turn_text or not target:
                result = _failure("invalid_target")
            elif op == "resolve_permission" and decision not in {
                PermissionDecision.APPROVE.value,
                PermissionDecision.DENY.value,
            }:
                result = _failure("invalid_decision")
            else:
                try:
                    if op == "cancel_turn":
                        receipt = await manager.cancel_turn(TurnRef(turn_text))
                    else:
                        receipt = await manager.resolve_permission(
                            TurnRef(turn_text),
                            PermissionRef(target),
                            PermissionDecision(decision),
                        )
                    result = (
                        _failure("unsupported_capability")
                        if isinstance(receipt, UnsupportedCapability)
                        else {"ok": True, "delivered": True, "completed": False}
                    )
                except (RuntimeStateError, ValueError):
                    result = _failure("stale_runtime_reference")

    if command_id and cache_key not in _RESULTS:
        _cache_result(cache_key, command_copy, result)
    log_runtime_event(
        logger,
        "runtime.command",
        agent_id=manager_agent_id,
        session_ref=session_ref,
        turn_ref=_required_text(command, "turn_ref"),
        permission_ref=_required_text(command, "permission_ref"),
        capability=wire_op,
        capability_decision="delivered" if result.get("delivered") else "rejected",
        error_code=str(result.get("error_code") or ""),
    )
    return result
