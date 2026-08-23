"""Shared, operator-safe provider failure semantics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    message: str
    retryable: bool = False
    is_auth: bool = False


PROVIDER_FAILURES: Mapping[str, ProviderFailure] = MappingProxyType(
    {
        "authentication": ProviderFailure(
            "Provider authentication failed; sign in again and retry.",
            is_auth=True,
        ),
        "permission_denied": ProviderFailure(
            "The requested operation was not permitted.",
        ),
        "quota_exhausted": ProviderFailure(
            "The selected provider model has reached its usage limit; "
            "switch models or wait for the limit to reset.",
        ),
        "rate_limit": ProviderFailure(
            "The provider rate-limited this turn; retry later.",
            retryable=True,
        ),
        "provider_unavailable": ProviderFailure(
            "The provider is temporarily unavailable; retry later.",
            retryable=True,
        ),
        "provider_error": ProviderFailure(
            "The provider could not complete the turn.",
        ),
    }
)


def provider_failure(error_code: str) -> ProviderFailure | None:
    return PROVIDER_FAILURES.get(error_code)


def provider_failure_message(error_code: str, *, outcome: str | None = None) -> str:
    failure = provider_failure(error_code)
    if failure is not None:
        return failure.message
    if outcome is not None:
        return f"provider turn ended with outcome {outcome} (error_code={error_code})"
    return "The Agent runtime could not complete the turn."
