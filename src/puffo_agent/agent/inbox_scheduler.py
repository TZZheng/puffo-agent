from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from .message_store import StoredMessage

MAX_MESSAGES = 50
MAX_ESTIMATED_TOKENS = 32_000
MAX_FORMATTED_BYTES = 96_000
COALESCE_SECONDS = 0.100

TargetProjection = tuple[str, ...]
Formatter = Callable[[StoredMessage], str]
Estimator = Callable[[str], int]


@dataclass(frozen=True)
class PlannedBatch:
    items: tuple[StoredMessage, ...]
    message_ids: tuple[str, ...]
    formatted_messages: tuple[str, ...]
    target_projections: tuple[TargetProjection, ...]
    pending_target_projections: tuple[TargetProjection, ...]
    estimated_tokens: int
    formatted_bytes: int
    more_available: bool
    unfit_head_id: str | None = None
    unfit_reason: str | None = None


class InboxPlanner:
    """Pure, policy-neutral leading-prefix planner."""

    @staticmethod
    def target_projection(item: StoredMessage) -> TargetProjection:
        if item.envelope_kind == "dm":
            return (
                "dm",
                item.sender_slug,
                item.recipient_slug or "",
            )
        if item.thread_root_id:
            return (
                "thread",
                item.space_id or "",
                item.channel_id or "",
                item.thread_root_id,
            )
        return ("channel", item.space_id or "", item.channel_id or "")

    def plan(
        self,
        items: Iterable[StoredMessage],
        *,
        formatter: Formatter,
        estimator: Estimator | None = None,
    ) -> PlannedBatch:
        unique: list[StoredMessage] = []
        seen_ids: set[str] = set()
        for item in items:
            if item.envelope_id in seen_ids:
                continue
            seen_ids.add(item.envelope_id)
            unique.append(item)

        selected: list[StoredMessage] = []
        formatted_selected: list[str] = []
        tokens = 0
        byte_count = 0
        unfit_head_id: str | None = None
        unfit_reason: str | None = None

        for item in unique:
            formatted = formatter(item)
            if not isinstance(formatted, str):
                raise TypeError("formatter must return str")
            formatted_bytes = len(formatted.encode("utf-8"))
            estimate = (
                estimator(formatted)
                if estimator is not None
                else max(1, formatted_bytes)
            )
            if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
                raise ValueError("estimator must return a non-negative integer")

            reasons: list[str] = []
            if len(selected) + 1 > MAX_MESSAGES:
                reasons.append("message count cap")
            if tokens + estimate > MAX_ESTIMATED_TOKENS:
                reasons.append("estimated token cap")
            if byte_count + formatted_bytes > MAX_FORMATTED_BYTES:
                reasons.append("formatted byte cap")
            if reasons:
                if not selected:
                    unfit_head_id = item.envelope_id
                    unfit_reason = ", ".join(reasons)
                break
            selected.append(item)
            formatted_selected.append(formatted)
            tokens += estimate
            byte_count += formatted_bytes

        projections: list[TargetProjection] = []
        seen_targets: set[TargetProjection] = set()
        for item in selected:
            target = self.target_projection(item)
            if target not in seen_targets:
                seen_targets.add(target)
                projections.append(target)

        consumed = len(selected)
        pending_projections: list[TargetProjection] = []
        pending_targets: set[TargetProjection] = set()
        for item in unique[consumed:]:
            target = self.target_projection(item)
            if target not in pending_targets:
                pending_targets.add(target)
                pending_projections.append(target)
        return PlannedBatch(
            items=tuple(selected),
            message_ids=tuple(item.envelope_id for item in selected),
            formatted_messages=tuple(formatted_selected),
            target_projections=tuple(projections),
            pending_target_projections=tuple(pending_projections),
            estimated_tokens=tokens,
            formatted_bytes=byte_count,
            more_available=consumed < len(unique),
            unfit_head_id=unfit_head_id,
            unfit_reason=unfit_reason,
        )

    async def resolve_unfit_head(
        self,
        batch: PlannedBatch,
        *,
        policy: Callable[..., bool | Awaitable[bool]],
        quarantine: Callable[..., bool | Awaitable[bool]],
    ) -> bool:
        """Run the injected oversize policy once, then guarded quarantine."""
        if batch.unfit_head_id is None:
            return False
        reason = batch.unfit_reason or "oldest item cannot fit"
        policy_signature = inspect.signature(policy)
        policy_parameters = policy_signature.parameters.values()
        policy_accepts_reason = any(
            parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for parameter in policy_parameters
        ) or len(policy_signature.parameters) >= 2
        still_unfit = (
            policy(batch.unfit_head_id, reason)
            if policy_accepts_reason
            else policy(batch.unfit_head_id)
        )
        if inspect.isawaitable(still_unfit):
            still_unfit = await still_unfit
        if not still_unfit:
            return False
        result = quarantine(batch.unfit_head_id, reason=reason)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)


class InboxCoalescer:
    """Metadata-free, non-resetting fixed-window wake coalescer."""

    def __init__(
        self,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        window_seconds: float = COALESCE_SECONDS,
    ):
        self._sleep = sleep
        self._monotonic = monotonic
        self._window_seconds = window_seconds
        self._wake = asyncio.Event()
        self._deadlines: deque[float] = deque()
        self._wait_lock = asyncio.Lock()

    def notify(self) -> None:
        now = self._monotonic()
        if not self._deadlines or now >= self._deadlines[-1]:
            self._deadlines.append(now + self._window_seconds)
        self._wake.set()

    async def wait_for_burst(self) -> None:
        async with self._wait_lock:
            await self._wake.wait()
            deadline = self._deadlines[0]
            remaining = max(0.0, deadline - self._monotonic())
            await self._sleep(remaining)
            self._deadlines.popleft()
            if self._deadlines:
                self._wake.set()
            else:
                self._wake.clear()
