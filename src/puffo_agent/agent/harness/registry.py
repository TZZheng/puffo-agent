"""Immutable harness composition and model-selection admission.

Execution Drivers deliberately remain stateful protocol adapters.  This module
owns the static facts around them: how a harness obtains a catalog, how a
selection is admitted, and which Driver factory implements the harness.

UI catalog snapshots are display data only.  ``admit_selection`` always asks
``CatalogService`` for a new snapshot, so a stale Web choice never becomes
admission truth merely because it was previously displayed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .. import model_catalog
from ..model_catalog import ModelOption
from .acp_driver import AcpDriver
from .claude_code_driver import ClaudeCodeCliDriver
from .codex_driver import CodexAppServerDriver
from .driver import Driver
from .opencode_driver import OpenCodeDriver
from .pi_driver import PiDriver


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    harness: str
    models: tuple[ModelOption, ...]
    observed_at: float
    admission_refresh: bool


@dataclass(frozen=True, slots=True)
class ValidatedSelection:
    """A model/inference pair admitted against one catalog snapshot."""

    harness: str
    model: str
    inference_level: str
    supported_inference_levels: tuple[str, ...]
    admitted_at: float


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    supported_levels: tuple[str, ...] = ()
    unknown_model_levels: tuple[str, ...] = ()
    model_specific_levels: bool = False

    def validate(
        self,
        snapshot: CatalogSnapshot,
        *,
        model: str,
        inference_level: str,
    ) -> ValidatedSelection:
        levels = self.supported_levels
        if self.model_specific_levels:
            selected = next(
                (option for option in snapshot.models if option.id == model),
                None,
            )
            levels = (
                selected.supported_inference_levels
                if selected is not None
                else self.unknown_model_levels
            )
        if inference_level and inference_level not in levels:
            raise ValueError(
                f"inference_level={inference_level!r} not supported by "
                f"model={model!r} on harness={snapshot.harness!r}; "
                f"expected one of {list(levels)}"
            )
        return ValidatedSelection(
            harness=snapshot.harness,
            model=model,
            inference_level=inference_level,
            supported_inference_levels=levels,
            admitted_at=snapshot.observed_at,
        )


class CatalogService:
    """Single catalog read boundary for both display and admission."""

    def display_snapshot(
        self, harness: str, *, warm: bool = False,
    ) -> CatalogSnapshot:
        return CatalogSnapshot(
            harness=harness,
            # A warm display read may refresh an expired backend cache but
            # never bypasses its TTL. Admission below always does.
            models=tuple(model_catalog.provider_models(harness, fetch=warm)),
            observed_at=time.time(),
            admission_refresh=False,
        )

    def refresh(self, harness: str) -> CatalogSnapshot:
        return CatalogSnapshot(
            harness=harness,
            models=tuple(
                model_catalog.provider_models(
                    harness,
                    fetch=True,
                    force_refresh=True,
                )
            ),
            observed_at=time.time(),
            admission_refresh=True,
        )


@dataclass(frozen=True, slots=True)
class HarnessDefinition:
    selection_policy: SelectionPolicy
    driver_factory: Callable[..., Driver] | None


_PI_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

# Reviewed literal registry: intentionally no registration or mutation API.
HARNESS_DEFINITIONS: Mapping[str, HarnessDefinition] = MappingProxyType({
    "claude-code": HarnessDefinition(
        SelectionPolicy(("low", "medium", "high", "xhigh")),
        ClaudeCodeCliDriver,
    ),
    "codex": HarnessDefinition(
        SelectionPolicy(("minimal", "low", "medium", "high")),
        CodexAppServerDriver,
    ),
    "pi": HarnessDefinition(
        SelectionPolicy(
            supported_levels=_PI_LEVELS,
            model_specific_levels=True,
            unknown_model_levels=_PI_LEVELS,
        ),
        PiDriver,
    ),
    "opencode": HarnessDefinition(
        SelectionPolicy(
            supported_levels=("minimal", "low", "medium", "high", "xhigh", "max"),
            model_specific_levels=True,
        ),
        OpenCodeDriver,
    ),
    "acp": HarnessDefinition(SelectionPolicy(), AcpDriver),
    "gemini-cli": HarnessDefinition(SelectionPolicy(), None),
    "hermes": HarnessDefinition(SelectionPolicy(), None),
})


class HarnessRegistry:
    def __init__(
        self,
        definitions: Mapping[str, HarnessDefinition] = HARNESS_DEFINITIONS,
        catalog_service: CatalogService | None = None,
    ) -> None:
        self._definitions = MappingProxyType(dict(definitions))
        self.catalog_service = catalog_service or CatalogService()

    @property
    def definitions(self) -> Mapping[str, HarnessDefinition]:
        return self._definitions

    def admit_selection(
        self,
        harness: str,
        *,
        model: str,
        inference_level: str,
    ) -> ValidatedSelection:
        definition = self._definitions.get(harness)
        if definition is None:
            raise ValueError(f"unknown harness {harness!r}")
        snapshot = self.catalog_service.refresh(harness)
        return definition.selection_policy.validate(
            snapshot,
            model=model,
            inference_level=inference_level,
        )

    def build_driver(self, harness: str, **kwargs: Any) -> Driver | None:
        definition = self._definitions.get(harness)
        factory = definition.driver_factory if definition is not None else None
        return factory(**kwargs) if factory is not None else None


DEFAULT_HARNESS_REGISTRY = HarnessRegistry()
