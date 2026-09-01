"""In-process adapter for evaluating Puffo's note store with AMB."""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from puffo_agent.agent.memory_store import NOTES_FILE_LIMIT, MemoryStore


class AmbDocument(Protocol):
    id: str
    content: str
    user_id: str | None


@dataclass(frozen=True)
class _StoredNote:
    document_id: str
    path: str
    user_id: str | None


@dataclass(frozen=True)
class RetrievedDocument:
    """The AMB document fields used by the generic RAG pipeline."""

    id: str
    content: str
    user_id: str | None = None


class PuffoMemoryProvider:
    """AMB-compatible note-store provider with the current substring baseline."""

    name = "puffo"
    description = "Puffo note-store substring retrieval baseline."
    kind = "local"
    concurrency = 1

    def __init__(self, retrieval_mode: str = "substring") -> None:
        if retrieval_mode not in {"substring", "bm25"}:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode
        self._store: MemoryStore | None = None
        self._notes: list[_StoredNote] = []

    def initialize(self) -> None:
        """Satisfy AMB's pre-prepare lifecycle hook without side effects."""

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        del unit_ids
        memory_root = store_dir / "puffo-memory"
        if reset and memory_root.exists():
            shutil.rmtree(memory_root)
        self._store = MemoryStore(memory_root)
        self._notes = []

    def ingest(self, documents: list[AmbDocument]) -> None:
        store = self._require_store()
        self._notes = []
        for document in documents:
            for index, chunk in enumerate(_chunk_note(document.content), start=1):
                path = f"notes/{_safe_name(document.id)}-{index:04d}.md"
                store.create_memory_file(path, chunk)
                self._notes.append(_StoredNote(document.id, path, document.user_id))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[AmbDocument], dict | None]:
        del query_timestamp
        if k < 1:
            return [], {"retrieval_mode": self.retrieval_mode, "matched_note_paths": []}
        notes = [note for note in self._notes if user_id is None or note.user_id == user_id]
        ranked = self._rank(query, notes)[:k]
        documents = [self._read_note(note) for note in ranked]
        return documents, {
            "retrieval_mode": self.retrieval_mode,
            "matched_note_paths": [note.path for note in ranked],
        }

    def _rank(self, query: str, notes: list[_StoredNote]) -> list[_StoredNote]:
        if self.retrieval_mode == "substring":
            needle = query.casefold()
            return [note for note in notes if needle in self._read_body(note).casefold()]
        return _rank_bm25(query, notes, self._read_body)

    def _read_note(self, note: _StoredNote) -> AmbDocument:
        return RetrievedDocument(
            id=note.document_id,
            content=self._read_body(note),
            user_id=note.user_id,
        )

    def _read_body(self, note: _StoredNote) -> str:
        return self._require_store().read_memory_file(note.path)["body"]

    def _require_store(self) -> MemoryStore:
        if self._store is None:
            raise RuntimeError("prepare() must be called before ingest() or retrieve()")
        return self._store


class PuffoBM25MemoryProvider(PuffoMemoryProvider):
    """AMB registration for Puffo notes ranked by an in-process BM25 index."""

    name = "puffo-bm25"
    description = "Puffo note store with deterministic in-process BM25 ranking."

    def __init__(self) -> None:
        super().__init__(retrieval_mode="bm25")


def _chunk_note(content: str) -> list[str]:
    encoded = content.encode("utf-8")
    return [
        encoded[offset : offset + NOTES_FILE_LIMIT].decode("utf-8", errors="ignore")
        for offset in range(0, len(encoded), NOTES_FILE_LIMIT)
    ] or [""]


def _safe_name(document_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", document_id).strip(".-") or "document"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _rank_bm25(
    query: str,
    notes: list[_StoredNote],
    read_body: Callable[[_StoredNote], str],
) -> list[_StoredNote]:
    tokenized_notes = [_tokens(read_body(note)) for note in notes]
    query_tokens = _tokens(query)
    if not notes or not query_tokens:
        return []
    average_length = sum(map(len, tokenized_notes)) / len(tokenized_notes)
    document_frequency = {
        token: sum(token in set(tokens) for tokens in tokenized_notes)
        for token in set(query_tokens)
    }
    scores = [
        _bm25_score(query_tokens, tokens, document_frequency, len(notes), average_length)
        for tokens in tokenized_notes
    ]
    return [notes[index] for index in sorted(range(len(notes)), key=lambda index: scores[index], reverse=True)]


def _bm25_score(
    query_tokens: list[str],
    document_tokens: list[str],
    document_frequency: dict[str, int],
    document_count: int,
    average_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token in query_tokens:
        frequency = document_tokens.count(token)
        if not frequency:
            continue
        inverse_frequency = math.log(
            1 + (document_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
        )
        denominator = frequency + k1 * (1 - b + b * len(document_tokens) / average_length)
        score += inverse_frequency * frequency * (k1 + 1) / denominator
    return score
