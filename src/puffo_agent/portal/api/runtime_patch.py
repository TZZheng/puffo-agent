"""Validation and serialization for bridge runtime edits."""

from __future__ import annotations

from typing import Any

from ..runtime_matrix import validate_triple
from ..state import RuntimeConfig

SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")


def apply_runtime_patch(runtime: RuntimeConfig, payload: dict[str, Any]) -> str | None:
    """Apply the existing bridge-editable fields and validate the result."""
    for field in (
        "kind",
        "provider",
        "harness",
        "model",
        "api_key",
        "permission_mode",
        "docker_image",
    ):
        if field in payload:
            setattr(runtime, field, str(payload[field]))

    if "sandbox" in payload:
        sandbox = str(payload["sandbox"])
        if sandbox not in SANDBOX_MODES:
            return "sandbox must be one of: " + ", ".join(SANDBOX_MODES)
        runtime.sandbox = sandbox

    result = validate_triple(runtime.kind, runtime.provider, runtime.harness)
    return None if result.ok else f"runtime: {result.error}"


def runtime_response(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "kind": runtime.kind,
        "provider": runtime.provider,
        "model": runtime.model,
        "api_key_set": bool(runtime.api_key),
        "permission_mode": runtime.permission_mode,
        "sandbox": runtime.sandbox,
        "harness": runtime.harness,
        "docker_image": runtime.docker_image,
    }
