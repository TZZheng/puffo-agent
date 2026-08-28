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
from .registry import (
    CatalogService,
    DEFAULT_HARNESS_REGISTRY,
    HarnessRegistry,
    SelectionPolicy,
    ValidatedSelection,
)

@dataclass(frozen=True)
class UnsupportedDriver:
    harness: str
    diagnostic: str = "no local Driver implementation for this harness"


def build_driver(name: str, **kwargs: Any) -> Driver | UnsupportedDriver:
    """Construct only Driver implementations admitted for production use.

    Process placement is supplied separately through ``process_factory``.
    """
    effective = name or "claude-code"
    driver = DEFAULT_HARNESS_REGISTRY.build_driver(effective, **kwargs)
    return driver if driver is not None else UnsupportedDriver(effective)


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
    "CatalogService",
    "HarnessRegistry",
    "SelectionPolicy",
    "ValidatedSelection",
    "build_driver",
]
