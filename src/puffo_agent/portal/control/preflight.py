"""Fast runtime readiness checks performed before identity materialization."""

from __future__ import annotations

from pathlib import Path

from ...agent.cli_bin import resolve_opencode_bin, resolve_pi_bin
from ...agent.opencode_auth import OpenCodeProbeError, opencode_model_is_available
from ...agent.pi_auth import PiAuthProbeError, check_pi_auth, pi_auth_target
from ..runtime_matrix import (
    RUNTIME_CLI_LOCAL,
    resolve_effective_harness,
    resolve_effective_provider,
)
from ..state import RuntimeConfig, agent_dir, select_pi_auth_home
from .provision import ProvisionError

_MAX_NATIVE_REASON_CHARS = 200


def _not_ready(
    harness: str, reason: str, *, native_reason: str = "",
) -> ProvisionError:
    fields = {
        "error_code": "harness_not_ready",
        "harness": harness,
        "reason": reason,
    }
    if native_reason:
        fields["native_reason"] = native_reason[:_MAX_NATIVE_REASON_CHARS]
    label = "Pi" if harness == "pi" else "OpenCode"
    message = {
        "not_installed": f"{label} is not installed",
        "need_login": f"{label} sign-in required",
        "credential_check_error": f"{label} credential check failed",
    }[reason]
    return ProvisionError(message, **fields)


def preflight_runtime(runtime: RuntimeConfig, *, agent_id: str = "") -> None:
    """Reject an unusable Pi/OpenCode selection before provisioning state."""
    provider = resolve_effective_provider(runtime.kind, runtime.provider)
    harness = resolve_effective_harness(runtime.kind, provider, runtime.harness)
    if runtime.kind != RUNTIME_CLI_LOCAL or harness not in {"pi", "opencode"}:
        return

    if harness == "opencode":
        executable = resolve_opencode_bin()
        if not executable:
            raise _not_ready("opencode", "not_installed")
        try:
            model_ready = opencode_model_is_available(executable, runtime.model)
        except OpenCodeProbeError as exc:
            raise _not_ready("opencode", "credential_check_error") from exc
        if not model_ready:
            raise _not_ready("opencode", "need_login")
        return

    executable = resolve_pi_bin()
    if not executable:
        raise _not_ready("pi", "not_installed")
    auth_provider, auth_model = pi_auth_target(provider, runtime.model)
    host_home = Path.home()
    config_dir = host_home / ".pi" / "agent"
    if agent_id:
        config_dir = select_pi_auth_home(
            host_home,
            agent_dir(agent_id) / ".pi" / "agent",
        )
    try:
        result = check_pi_auth(
            executable,
            provider=auth_provider,
            model=auth_model,
            config_dir=config_dir,
        )
    except PiAuthProbeError as exc:
        raise _not_ready("pi", "credential_check_error") from exc
    if result.status == "ready":
        return
    if result.status == "not_ready":
        raise _not_ready("pi", "need_login", native_reason=result.reason)
    raise _not_ready("pi", "credential_check_error", native_reason=result.reason)
