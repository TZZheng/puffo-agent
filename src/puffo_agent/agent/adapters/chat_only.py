"""Chat-only adapter.

Wraps the message-completion providers (Anthropic/OpenAI) so existing
agents keep working without the SDK or CLI runtimes. Does not run
tools, does not touch the filesystem, ignores workspace/claude dirs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .base import Adapter, TurnContext, TurnResult
from ..context_controller import (
    ContextCapabilities,
    ContextSnapshot,
    ProviderAdmissionEvent,
    normalize_context_snapshot,
)


class ChatOnlyAdapter(Adapter):
    def __init__(self, provider):
        # ``provider`` exposes blocking
        # ``complete(system_prompt, messages) -> (str, int, int)``.
        self._provider = provider

    async def run_turn(self, ctx: TurnContext) -> TurnResult:
        reply, input_tokens, output_tokens = await asyncio.to_thread(
            self._provider.complete, ctx.system_prompt, ctx.messages,
        )
        await self._fire_admission_callback(ProviderAdmissionEvent(
            planning_cycle_key=getattr(
                self, "_context_admission_planning_cycle_key", "",
            ),
            provider_session_id=None,
            admitted_at=datetime.now(timezone.utc),
        ))
        return TurnResult(
            reply=reply,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=0,
        )

    async def get_context_snapshot(self) -> ContextSnapshot:
        return normalize_context_snapshot(
            used_tokens=0,
            estimated_source="chat_stateless_fallback_200000",
        )

    def get_context_capabilities(self) -> ContextCapabilities:
        return ContextCapabilities(
            diagnostic="chat completion is stateless; context control unsupported",
        )
