"""Harness abstraction — which agent engine runs inside a runtime.

Runtime answers WHERE the agent executes; harness answers WHAT.
Only meaningful for the CLI runtimes; ``chat-local`` / ``sdk-local``
ignore the field. Four harnesses ship: ``claude-code`` (Anthropic),
``hermes`` (Anthropic + OpenAI), ``gemini-cli`` (Google),
``codex`` (OpenAI — opt-in, default for openai is still hermes).
Each declares ``supported_providers`` so the runtime matrix can
reject mismatched triples at load time.
"""

from .base import Harness, HarnessTurn
from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness
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
    if name == "codex":
        return CodexHarness()
    raise ValueError(
        f"unknown harness {name!r}: expected one of "
        "'claude-code', 'hermes', 'gemini-cli', 'codex'"
    )


@dataclass(frozen=True)
class UnsupportedDriver:
    harness: str
    diagnostic: str = "no Driver implementation for this legacy harness"


def build_driver(name: str, **kwargs: Any) -> HarnessDriver | UnsupportedDriver:
    """Construct only the two ratified Driver implementations.

    This factory is deliberately separate from :func:`build_harness`; legacy
    adapters continue accepting Hermes, Gemini, Docker, SDK, and chat paths.
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
    "CodexHarness",
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
