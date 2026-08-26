"""Protocol Drivers used by host-local and Docker Puffo runtimes."""

from dataclasses import dataclass
from typing import Any

from .driver import (
    Driver,
    RuntimeRef,
    SessionRef,
    TurnRef,
    PermissionRef,
    UnsupportedCapability,
)
from .codex_driver import CodexAppServerDriver, CodexDriver
from .claude_code_driver import ClaudeCodeCliDriver, ClaudeDriver
from .opencode_driver import OpenCodeCliDriver, OpenCodeDriver
from .acp_driver import AcpDriver, GenericAcpDriver
from .pi_driver import (
    PI_CAPABILITIES,
    PiDriver,
    PiToolBridgeUnavailableError,
    verify_pi_tool_bridge,
)

@dataclass(frozen=True)
class UnsupportedDriver:
    harness: str
    diagnostic: str = "no local Driver implementation for this harness"


def build_driver(name: str, **kwargs: Any) -> Driver | UnsupportedDriver:
    """Construct only Driver implementations admitted for production use.

    Process placement is supplied separately through ``process_factory``.
    """
    if name == "codex":
        return CodexAppServerDriver(**kwargs)
    if name == "opencode":
        return OpenCodeDriver(**kwargs)
    if name == "acp":
        return AcpDriver(**kwargs)
    if name == "pi":
        # Admission remains fail-closed in PiDriver.open(): every spawned Pi
        # process must load and attest the shipped Puffo tool bridge.
        return PiDriver(**kwargs)
    if not name or name == "claude-code":
        return ClaudeCodeCliDriver(**kwargs)
    return UnsupportedDriver(name)


__all__ = [
    "Driver",
    "RuntimeRef",
    "SessionRef",
    "TurnRef",
    "PermissionRef",
    "UnsupportedCapability",
    "UnsupportedDriver",
    "CodexAppServerDriver",
    "CodexDriver",
    "ClaudeCodeCliDriver",
    "ClaudeDriver",
    "OpenCodeCliDriver",
    "OpenCodeDriver",
    "AcpDriver",
    "GenericAcpDriver",
    "PiDriver",
    "PI_CAPABILITIES",
    "PiToolBridgeUnavailableError",
    "verify_pi_tool_bridge",
    "build_driver",
]
