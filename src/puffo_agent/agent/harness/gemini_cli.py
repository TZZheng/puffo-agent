"""Gemini CLI harness.

Wraps Google's ``gemini`` CLI. Runtime support is currently
``cli-docker`` only; runtime validation rejects it for ``cli-local``.
"""

from __future__ import annotations

from .base import Harness


class GeminiCLIHarness(Harness):
    def name(self) -> str:
        return "gemini-cli"

    def supported_providers(self) -> frozenset[str]:
        return frozenset({"google"})
