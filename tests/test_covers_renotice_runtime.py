"""Finalize-time cover reconciliation and one-shot renotice."""

import pytest

from puffo_agent.agent.global_inbox_runtime import GlobalInboxRuntime
from puffo_agent.agent.message_store_models import ProcessingState

from test_global_inbox_runtime import Adapter, make_store, receipt


def _human_content(text: str) -> dict:
    return {"text": text, "sender_type": "human"}


class _ReadingRunner:
    """Admit the turn, read the whole Inbox, optionally declare covers."""

    def __init__(self, adapter, covers=()):
        self.adapter = adapter
        self.covers = tuple(covers)
        self.runtime = None

    async def __call__(self, _planned):
        await self.adapter.admit()
        await self.runtime.read_inbox(limit=50, tool_arguments={"limit": 50})
        if self.covers:
            await self.runtime.store.add_message_covers(
                self.covers, source="send", by_envelope_id="reply-1",
            )


def _build_runtime(store, tmp_path, *, covers=()):
    adapter = Adapter()
    runner = _ReadingRunner(adapter, covers=covers)
    runtime = GlobalInboxRuntime(
        store=store,
        adapter=adapter,
        run_turn=runner,
        workspace=tmp_path,
    )
    runner.runtime = runtime
    return runtime


@pytest.mark.asyncio
async def test_uncovered_human_message_is_observed_not_renoticed_by_default(
    tmp_path,
):
    store = await make_store(tmp_path)
    await receipt(store, "h1", 1, content=_human_content("question"))
    runtime = _build_runtime(store, tmp_path)
    assert not runtime.covers_renotice_enabled
    assert await runtime.process_once()
    row = await store.get_message_by_envelope("h1")
    assert row.processing_state is ProcessingState.PROCESSED
    assert not row.renotified
    await store.close()


@pytest.mark.asyncio
async def test_uncovered_human_message_is_renoticed_exactly_once(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("PUFFO_COVERS_RENOTICE", "1")
    store = await make_store(tmp_path)
    await receipt(store, "h1", 1, content=_human_content("question"))
    runtime = _build_runtime(store, tmp_path)
    assert runtime.covers_renotice_enabled

    assert await runtime.process_once()
    row = await store.get_message_by_envelope("h1")
    assert row.processing_state is ProcessingState.PENDING
    assert row.renotified

    # The redelivered row goes uncovered again; the one-shot bit means the
    # second completed turn is terminal.
    assert await runtime.process_once()
    row = await store.get_message_by_envelope("h1")
    assert row.processing_state is ProcessingState.PROCESSED
    await store.close()


@pytest.mark.asyncio
async def test_covered_and_agent_messages_complete_without_renotice(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("PUFFO_COVERS_RENOTICE", "1")
    store = await make_store(tmp_path)
    await receipt(store, "human-covered", 1, content=_human_content("ping"))
    await receipt(store, "agent-note", 2, content={
        "text": "fyi", "sender_type": "agent",
    })
    runtime = _build_runtime(store, tmp_path, covers=("human-covered",))
    assert await runtime.process_once()
    for envelope_id in ("human-covered", "agent-note"):
        row = await store.get_message_by_envelope(envelope_id)
        assert row.processing_state is ProcessingState.PROCESSED
        assert not row.renotified
    await store.close()


@pytest.mark.asyncio
async def test_partial_coverage_renotices_only_the_uncovered_rows(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("PUFFO_COVERS_RENOTICE", "1")
    store = await make_store(tmp_path)
    await receipt(store, "answered", 1, content=_human_content("thread A"))
    await receipt(store, "forgotten", 2, content=_human_content("thread B"))
    runtime = _build_runtime(store, tmp_path, covers=("answered",))
    assert await runtime.process_once()
    answered = await store.get_message_by_envelope("answered")
    assert answered.processing_state is ProcessingState.PROCESSED
    forgotten = await store.get_message_by_envelope("forgotten")
    assert forgotten.processing_state is ProcessingState.PENDING
    assert forgotten.renotified
    await store.close()
