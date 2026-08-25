from pathlib import Path

from puffo_agent.evaluation.amb_provider import PuffoMemoryProvider


def test_bm25_retrieval_ranks_relevant_note_and_returns_full_body(tmp_path: Path):
    """Guards against returning search snippets instead of note-body context."""
    provider = PuffoMemoryProvider(retrieval_mode="bm25")
    provider.prepare(tmp_path)
    provider.ingest([
        _document("other", "alex collects stamps"),
        _document("target", "alex plans a hiking trip\nwith detailed gear notes"),
    ])

    documents, raw = provider.retrieve("What gear does Alex need for hiking?", k=1)

    assert documents[0].id == "target"
    assert documents[0].content == "alex plans a hiking trip\nwith detailed gear notes"
    assert raw == {"retrieval_mode": "bm25", "matched_note_paths": ["notes/target.md"]}


def test_retrieval_scopes_documents_to_the_requested_user(tmp_path: Path):
    """Guards against one LoCoMo conversation leaking into another's context."""
    provider = PuffoMemoryProvider(retrieval_mode="substring")
    provider.prepare(tmp_path)
    provider.ingest([
        _document("alice", "Morgan's favorite food is ramen", user_id="alice"),
        _document("bob", "Morgan's favorite food is curry", user_id="bob"),
    ])

    documents, _ = provider.retrieve("favorite food", k=2, user_id="bob")

    assert [document.id for document in documents] == ["bob"]


def test_substring_mode_uses_case_insensitive_matching_and_honors_limit(tmp_path: Path):
    """Guards the baseline mode from silently changing current substring semantics."""
    provider = PuffoMemoryProvider(retrieval_mode="substring")
    provider.prepare(tmp_path)
    provider.ingest([
        _document("first", "Needle appears in this note"),
        _document("second", "Another needle appears here"),
    ])

    documents, _ = provider.retrieve("nEeDlE", k=1)

    assert [document.id for document in documents] == ["first"]


def _document(document_id: str, content: str, user_id: str | None = None):
    from dataclasses import make_dataclass

    document_type = make_dataclass("Document", [("id", str), ("content", str), ("user_id", str | None, None)])
    return document_type(document_id, content, user_id)
