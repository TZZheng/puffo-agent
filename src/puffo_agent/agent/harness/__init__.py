"""Harness abstraction — which agent engine runs inside a runtime.

Runtime answers WHERE the agent executes; harness answers WHAT.
Only meaningful for the CLI runtimes; ``chat-local`` / ``sdk-local``
ignore the field. Docker retains three declarative Harness types.
The host-local runtime uses the long-lived Driver implementations for
``claude-code`` and ``codex`` only.
"""

from .base import Harness, HarnessTurn
from .claude_code import ClaudeCodeHarness
from .gemini_cli import GeminiCLIHarness
from .hermes import HermesHarness
from .driver import (
    Driver,
    HarnessDriver,
    RuntimeRef,
    SessionRef,
    TurnRef,
    PermissionRef,
    UnsupportedCapability,
)
from .codex_driver import CodexAppServerDriver, CodexDriver
from .claude_code_driver import ClaudeCodeCliDriver, ClaudeDriver
from dataclasses import dataclass
from typing import Any


def build_harness(name: str) -> Harness:
    """Resolve a harness name from agent.yml. Default Claude Code so
    agents without the field keep existing behaviour.
    """
    if not name or name == "claude-code":
        return ClaudeCodeHarness()
    if name == "hermes":
        return HermesHarness()
    if name == "gemini-cli":
        return GeminiCLIHarness()
    raise ValueError(
        f"unknown harness {name!r}: expected one of "
        "'claude-code', 'hermes', 'gemini-cli'"
    )


@dataclass(frozen=True)
class UnsupportedDriver:
    harness: str
    diagnostic: str = "no local Driver implementation for this harness"


def build_driver(name: str, **kwargs: Any) -> HarnessDriver | UnsupportedDriver:
    """Construct only the two ratified Driver implementations.

    This factory is deliberately separate from :func:`build_harness`, which
    remains the declarative registry used by the Docker runtime.
    """
    if name == "codex":
        return CodexAppServerDriver(**kwargs)
    if not name or name == "claude-code":
        return ClaudeCodeCliDriver(**kwargs)
    return UnsupportedDriver(name)


__all__ = [
    "Harness",
    "HarnessTurn",
    "ClaudeCodeHarness",
    "GeminiCLIHarness",
    "HermesHarness",
    "build_harness",
    "Driver", "HarnessDriver",
    "RuntimeRef", "SessionRef", "TurnRef", "PermissionRef",
    "UnsupportedCapability", "UnsupportedDriver",
    "CodexAppServerDriver", "CodexDriver",
    "ClaudeCodeCliDriver", "ClaudeDriver",
    "build_driver",
]
