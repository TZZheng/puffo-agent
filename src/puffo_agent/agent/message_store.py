from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import time
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import aiosqlite

from ..portal.state import home_dir

_SCHEMA_INIT_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()

# Supplementary context is deliberately smaller than a content-bearing Inbox
# page.  The runtime applies the formatted-byte guard as well; these bounds
# keep the durable lookup itself finite before formatting adds metadata.
PRIOR_CONTEXT_MAX_ITEMS = 20
PRIOR_CONTEXT_MAX_BYTES = 48_000


def _schema_init_lock(db_path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _SCHEMA_INIT_LOCKS.setdefault(loop, {})
    return locks.setdefault(str(db_path.resolve()), asyncio.Lock())


_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    envelope_id TEXT PRIMARY KEY,
    envelope_kind TEXT NOT NULL,
    sender_slug TEXT NOT NULL,
    channel_id TEXT,
    space_id TEXT,
    recipient_slug TEXT,
    content_type TEXT NOT NULL DEFAULT 'text/plain',
    content TEXT NOT NULL,
    sent_at INTEGER NOT NULL,
    received_at INTEGER NOT NULL,
    thread_root_id TEXT,
    reply_to_id TEXT,
    is_encrypted INTEGER NOT NULL DEFAULT 1
);

-- Per-thread cursor used by the thread-batched priority queue. After
-- ``on_message_batch`` finishes successfully, the consumer advances
-- this to the ``sent_at`` of the last message in the dispatched
-- batch. The listen handler then drops any inbound message whose
-- ``sent_at`` is <= the stored cursor, so server-side pending-message
-- redeliveries after a daemon restart don't re-trigger the agent on
-- already-processed threads.
CREATE TABLE IF NOT EXISTS thread_processing_state (
    root_id TEXT PRIMARY KEY,
    last_processed_sent_at INTEGER NOT NULL
);

-- One row per channel the agent has been auto-prompted to introduce
-- itself in. Gate set in ``_accept_invite`` so a daemon restart (or a
-- server-side invite redelivery) can't trigger a second intro.
CREATE TABLE IF NOT EXISTS channel_intro_prompted (
    channel_id TEXT PRIMARY KEY,
    prompted_at INTEGER NOT NULL
);

-- Out-of-band channel→space mappings discovered without an inbound
-- message. The /messages table inference is enough for channels the
-- agent has received traffic from, but the intro-nudge path needs
-- the mapping BEFORE the first real message — agent calls
-- send_message against the freshly-joined channel, MCP asks
-- lookup_channel_space, and without this table the daemon-side
-- query returns 404 → MCP falls back to agent.yml's home space,
-- which is the wrong space when the agent is now multi-space.
-- Populated by ``_find_public_general_channel`` (and any future
-- channel-discovery hook). ``lookup_channel_space`` checks this
-- table first before the /messages fallback.
CREATE TABLE IF NOT EXISTS channel_space_map (
    channel_id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    learned_at INTEGER NOT NULL
);

-- Last "FYI, X is DMing me" notice per foreign sender, so the 72h
-- throttle survives daemon restarts.
CREATE TABLE IF NOT EXISTS dm_notices (
    sender_slug TEXT PRIMARY KEY,
    last_notified_at INTEGER NOT NULL
);
"""

_DEPENDENT_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_messages_channel
    ON messages (channel_id, sent_at) WHERE channel_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_dm
    ON messages (sender_slug, sent_at) WHERE envelope_kind = 'dm';
CREATE INDEX IF NOT EXISTS idx_messages_dm_recipient
    ON messages (recipient_slug, sent_at) WHERE envelope_kind = 'dm';
CREATE INDEX IF NOT EXISTS idx_messages_received ON messages (received_at);
CREATE INDEX IF NOT EXISTS idx_messages_thread_root
    ON messages (thread_root_id, sent_at) WHERE thread_root_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_server_seq_unique
    ON messages (server_seq) WHERE server_seq IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_pending_fifo
    ON messages (processing_state, server_seq, after_server_seq, local_ordinal);
CREATE INDEX IF NOT EXISTS idx_messages_turn
    ON messages (processing_turn_id, processing_state);
CREATE INDEX IF NOT EXISTS idx_messages_channel_pending
    ON messages (space_id, channel_id, processing_state, server_seq);

CREATE TABLE IF NOT EXISTS channel_context_state (
    space_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    context_baseline_seq INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (space_id, channel_id)
);
CREATE TABLE IF NOT EXISTS turn_runs (
    turn_id TEXT PRIMARY KEY,
    provider_session_id TEXT,
    state TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER
);
CREATE TABLE IF NOT EXISTS turn_run_messages (
    turn_id TEXT NOT NULL,
    envelope_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (turn_id, envelope_id),
    UNIQUE (turn_id, ordinal),
    FOREIGN KEY (turn_id) REFERENCES turn_runs(turn_id)
);
CREATE TABLE IF NOT EXISTS local_ordinal_allocator (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_ordinal INTEGER NOT NULL
);
INSERT OR IGNORE INTO local_ordinal_allocator(singleton, next_ordinal) VALUES (1, 1);
CREATE TABLE IF NOT EXISTS inbox_notice_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    generation INTEGER NOT NULL,
    pending_count INTEGER NOT NULL,
    first_pending_deadline_ms INTEGER,
    last_delivered_generation INTEGER NOT NULL,
    last_delivered_provider_session_id TEXT
);
INSERT OR IGNORE INTO inbox_notice_state(
    singleton, generation, pending_count, first_pending_deadline_ms,
    last_delivered_generation
) VALUES (1, 0, 0, NULL, 0);

-- One row is both the immutable one-shot reminder intent and its only
-- occurrence. This table intentionally represents a single trigger only.
CREATE TABLE IF NOT EXISTS reminder_occurrences (
    reminder_id TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL UNIQUE,
    target TEXT NOT NULL,
    content TEXT NOT NULL,
    intended_at_ms INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('scheduled','claimed','cancelled','delivered')),
    created_at_ms INTEGER NOT NULL,
    claimed_at_ms INTEGER,
    actual_fire_at_ms INTEGER,
    cancelled_at_ms INTEGER,
    delivered_at_ms INTEGER,
    delivered_event_id TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_reminder_occurrences_due
    ON reminder_occurrences (state, intended_at_ms, occurrence_id);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


class DataNotFound(Exception):
    """Raised by reads that need to distinguish "this channel /
    thread has never been seen" from "seen, but the requested
    window is empty after filters". The MCP tool layer surfaces a
    different user-facing message for each.
    """


class ReceiptDisposition(str, Enum):
    ELIGIBLE = "eligible"
    TERMINAL = "terminal"
    FOREIGN_DM_GATED = "foreign_dm_gated"
    LOCAL_RUNTIME = "local_runtime"


class ProcessingState(str, Enum):
    PENDING = "pending"
    IN_TURN = "in_turn"
    PROCESSED = "processed"


class ReceiptWriteStatus(str, Enum):
    COMMITTED = "committed"
    IDEMPOTENT = "idempotent"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ReceiptResult:
    status: ReceiptWriteStatus
    disposition: ReceiptDisposition
    reason: str
    acknowledge: bool


@dataclass(frozen=True)
class PendingPage:
    items: tuple["StoredMessage", ...]
    through_seq: int | None
    more_available: bool


@dataclass(frozen=True)
class InboxNoticeState:
    generation: int
    pending_count: int
    first_pending_deadline_ms: int | None
    last_delivered_generation: int
    last_delivered_provider_session_id: str | None

    @property
    def delivery_pending(self) -> bool:
        """Whether the current generation lacks a session-aware acceptance.

        New scheduling must use :meth:`is_due_for`, because one accepted
        generation is suppressed only for the native provider session that
        accepted it.  This compatibility projection remains useful to receipt
        callers that only need to know whether a new generation exists.
        """
        return (
            self.pending_count > 0
            and (
                self.generation != self.last_delivered_generation
                or not self.last_delivered_provider_session_id
            )
        )

    def is_due_for(self, provider_session_id: str | None) -> bool:
        """Return whether this session needs one metadata-only notice.

        Some provider transports learn their native session id only in the
        acceptance receipt for their first offered turn.  Before that receipt,
        a notice is due only when this generation has no session-aware
        acceptance at all.  Once an accepting session is durable, an unknown
        session must remain suppressed rather than recreating an equivalent
        notice loop.
        """
        if self.pending_count <= 0:
            return False
        if not provider_session_id:
            return self.delivery_pending
        return (
            self.generation != self.last_delivered_generation
            or provider_session_id != self.last_delivered_provider_session_id
        )


@dataclass(frozen=True)
class InboxPage:
    items: tuple["StoredMessage", ...]
    next_cursor: str
    has_more: bool
    remaining_count: int
    snapshot_generation: int
    target: str


@dataclass(frozen=True)
class TurnRun:
    turn_id: str
    provider_session_id: str | None
    state: str
    message_ids: tuple[str, ...]
    started_at: int
    completed_at: int | None


@dataclass(frozen=True)
class ReminderOccurrence:
    """The durable Agent-owned fact for one single-fire reminder."""

    reminder_id: str
    occurrence_id: str
    target: str
    content: str
    intended_at_ms: int
    state: str
    created_at_ms: int
    claimed_at_ms: int | None
    actual_fire_at_ms: int | None
    cancelled_at_ms: int | None
    delivered_at_ms: int | None
    delivered_event_id: str | None

    def as_dict(self) -> dict[str, Any]:
        """The stable, provider-neutral reminder tool result."""
        return {
            "reminder_id": self.reminder_id,
            "occurrence_id": self.occurrence_id,
            "state": self.state,
            "target": self.target,
            "content": self.content,
            "intended_at": reminder_time_to_rfc3339(self.intended_at_ms),
            "actual_fire_at": (
                reminder_time_to_rfc3339(self.actual_fire_at_ms)
                if self.actual_fire_at_ms is not None else None
            ),
            "created_at": reminder_time_to_rfc3339(self.created_at_ms),
            "cancelled_at": (
                reminder_time_to_rfc3339(self.cancelled_at_ms)
                if self.cancelled_at_ms is not None else None
            ),
            "delivered_at": (
                reminder_time_to_rfc3339(self.delivered_at_ms)
                if self.delivered_at_ms is not None else None
            ),
        }


REMINDER_STATES = frozenset({"scheduled", "claimed", "cancelled", "delivered"})
MAX_REMINDER_LIST_LIMIT = 100


def reminder_time_to_rfc3339(value: int) -> str:
    """Render epoch milliseconds in one canonical UTC representation."""
    return (
        datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def parse_reminder_target(target: str) -> tuple[str, str, str, str]:
    """Validate and decompose a canonical local Inbox target.

    Returns ``(kind, space_id, channel_id, thread_root_id_or_peer)``.  The
    one-shot contract deliberately accepts only the target forms which the
    Inbox already projects; it does not infer a provider route from prose.
    """
    if not isinstance(target, str) or not target or target != target.strip():
        raise ValueError("target must be a non-empty canonical Inbox target")
    parts = target.split(":")
    valid_segment = lambda value: bool(value) and value == value.strip()
    if parts[0] == "dm" and len(parts) == 2 and valid_segment(parts[1]):
        return ("dm", "", "", parts[1])
    if (
        parts[0] == "channel"
        and len(parts) == 3
        and valid_segment(parts[1])
        and valid_segment(parts[2])
    ):
        return ("channel", parts[1], parts[2], "")
    if (
        parts[0] == "channel"
        and len(parts) == 5
        and valid_segment(parts[1])
        and valid_segment(parts[2])
        and parts[3] == "thread"
        and valid_segment(parts[4])
    ):
        return ("channel", parts[1], parts[2], parts[4])
    raise ValueError("invalid reminder target")


class LifecycleConflict(Exception):
    """An exact Inbox lifecycle transition could not be applied."""


@dataclass
class StoredMessage:
    envelope_id: str
    envelope_kind: str
    sender_slug: str
    channel_id: Optional[str]
    space_id: Optional[str]
    recipient_slug: Optional[str]
    content_type: str
    content: Any
    sent_at: int
    received_at: int
    thread_root_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    # False only for a plaintext (non-E2EE) message; absent/legacy rows are True.
    is_encrypted: bool = True
    server_seq: int | None = None
    receipt_disposition: ReceiptDisposition | None = None
    receipt_reason: str | None = None
    processing_state: ProcessingState | None = None
    processing_turn_id: str | None = None
    model_visible_at: int | None = None
    processed_at: int | None = None
    local_ordinal: int | None = None
    after_server_seq: int | None = None


@dataclass
class ChannelRoot:
    """One root post in a channel plus how many replies it accrued.
    Used by ``get_channel_roots`` to surface thread heads without
    blasting every reply into the agent's context window — the agent
    sees N=reply_count and calls ``get_thread_messages`` only on
    threads it actually wants to read into.
    """
    message: StoredMessage
    reply_count: int


class MessageStore:
    NOTICE_WINDOW_MS = 3_000

    def __init__(
        self,
        db_path: str | Path,
        *,
        now_ms: Any = _now_ms,
    ):
        self.db_path = Path(db_path)
        self._now_ms = now_ms
        self._db: Optional[aiosqlite.Connection] = None
        self._open_lock = asyncio.Lock()
        # aiosqlite multiplexes one connection. Keep every Inbox transaction
        # and lifecycle-sensitive read outside another coroutine's transaction.
        self._inbox_lock = asyncio.Lock()

    @staticmethod
    def for_agent(agent_id: str) -> MessageStore:
        return MessageStore(home_dir() / "agents" / agent_id / "messages.db")

    async def open(self) -> None:
        async with self._open_lock:
            if self._db is not None:
                return

            async with _schema_init_lock(self.db_path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                db = await aiosqlite.connect(str(self.db_path), timeout=30.0)
                db.row_factory = aiosqlite.Row
                try:
                    await self._configure_connection(db)
                    # Phase 1: tables that do not depend on additive Inbox columns.
                    await self._execute_locked_script(db, _BASE_SCHEMA)

                    # Acquire the database write lock before inspecting columns.
                    # Separate stores may initialize the same file concurrently.
                    await db.execute("BEGIN IMMEDIATE")
                    try:
                        cur = await db.execute("PRAGMA table_info(messages)")
                        cols = {row[1] for row in await cur.fetchall()}
                        additions = {
                            "is_encrypted": "is_encrypted INTEGER NOT NULL DEFAULT 1",
                            "server_seq": "server_seq INTEGER",
                            "receipt_disposition": (
                                "receipt_disposition TEXT CHECK (receipt_disposition IS NULL OR "
                                "receipt_disposition IN ('eligible','terminal','foreign_dm_gated','local_runtime'))"
                            ),
                            "receipt_reason": "receipt_reason TEXT",
                            "processing_state": (
                                "processing_state TEXT CHECK (processing_state IS NULL OR "
                                "processing_state IN ('pending','in_turn','processed'))"
                            ),
                            "processing_turn_id": "processing_turn_id TEXT",
                            "model_visible_at": "model_visible_at INTEGER",
                            "processed_at": "processed_at INTEGER",
                            "local_ordinal": "local_ordinal INTEGER",
                            "after_server_seq": "after_server_seq INTEGER",
                        }
                        for name, declaration in additions.items():
                            if name not in cols:
                                await db.execute(
                                    f"ALTER TABLE messages ADD COLUMN {declaration}"
                                )
                        await db.commit()
                    except BaseException:
                        await db.rollback()
                        raise

                    # Phase 3: indexes and tables whose definitions use the new columns.
                    await self._execute_locked_script(db, _DEPENDENT_SCHEMA)
                    await self._migrate_nullable_provider_session(db)
                    await self._migrate_notice_provider_session(db)
                    self._db = db
                except BaseException:
                    await db.close()
                    raise

    @staticmethod
    async def _configure_connection(db: aiosqlite.Connection) -> None:
        # Setting WAL on a brand-new database can fail immediately when
        # another process is doing the same. Retry that one-time transition;
        # later schema work is serialized by BEGIN IMMEDIATE.
        await db.execute("PRAGMA busy_timeout=1000")
        deadline = time.monotonic() + 30.0
        delay = 0.01
        while True:
            try:
                async with db.execute("PRAGMA journal_mode=WAL") as cursor:
                    row = await cursor.fetchone()
                if row is None or str(row[0]).lower() != "wal":
                    raise RuntimeError("SQLite refused WAL journal mode")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.25)

        await db.execute("PRAGMA busy_timeout=30000")
        # synchronous=NORMAL is WAL-safe (crash may lose the last group
        # commit, never corrupt) and halves write latency.
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA temp_store=MEMORY")
        await db.execute("PRAGMA cache_size=-20000")
        await db.execute("PRAGMA mmap_size=268435456")

    @staticmethod
    async def _execute_locked_script(
        db: aiosqlite.Connection, script: str
    ) -> None:
        try:
            await db.executescript(f"BEGIN IMMEDIATE;\n{script}\nCOMMIT;")
        except BaseException:
            await db.rollback()
            raise

    async def _migrate_nullable_provider_session(
        self, db: aiosqlite.Connection
    ) -> None:
        """Upgrade databases created by the first Package 1 implementation."""
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute("PRAGMA table_info(turn_runs)") as cursor:
                columns = {row["name"]: row for row in await cursor.fetchall()}
            provider_column = columns.get("provider_session_id")
            if provider_column is None or not provider_column["notnull"]:
                await db.commit()
                return

            await db.execute(
                "ALTER TABLE turn_run_messages "
                "RENAME TO turn_run_messages_nonnullable_migration"
            )
            await db.execute(
                "ALTER TABLE turn_runs RENAME TO turn_runs_nonnullable_migration"
            )
            statements = (
                """CREATE TABLE turn_runs (
                    turn_id TEXT PRIMARY KEY,
                    provider_session_id TEXT,
                    state TEXT NOT NULL,
                    started_at INTEGER NOT NULL,
                    completed_at INTEGER
                )""",
                """CREATE TABLE turn_run_messages (
                    turn_id TEXT NOT NULL,
                    envelope_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (turn_id, envelope_id),
                    UNIQUE (turn_id, ordinal),
                    FOREIGN KEY (turn_id) REFERENCES turn_runs(turn_id)
                )""",
                """INSERT INTO turn_runs
                    (turn_id, provider_session_id, state, started_at, completed_at)
                SELECT turn_id, provider_session_id, state, started_at, completed_at
                FROM turn_runs_nonnullable_migration""",
                """INSERT INTO turn_run_messages (turn_id, envelope_id, ordinal)
                SELECT turn_id, envelope_id, ordinal
                FROM turn_run_messages_nonnullable_migration""",
                "DROP TABLE turn_run_messages_nonnullable_migration",
                "DROP TABLE turn_runs_nonnullable_migration",
            )
            for statement in statements:
                await db.execute(statement)
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    async def _migrate_notice_provider_session(
        self, db: aiosqlite.Connection
    ) -> None:
        """Add the accepting native session without rewriting notice state.

        The first global-Inbox schema stored only a generation-level delivery
        marker.  Existing accepted generations keep that marker and get a
        ``NULL`` session, which makes them eligible for one bounded discovery
        by the next known native session instead of pretending the old
        acceptance belongs to an arbitrary resumed transcript.
        """
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute("PRAGMA table_info(inbox_notice_state)") as cursor:
                columns = {row["name"] for row in await cursor.fetchall()}
            if "last_delivered_provider_session_id" not in columns:
                await db.execute(
                    "ALTER TABLE inbox_notice_state "
                    "ADD COLUMN last_delivered_provider_session_id TEXT"
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    async def close(self) -> None:
        async with self._open_lock:
            if self._db:
                await self._db.close()
                self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.open()
        assert self._db is not None
        return self._db

    async def _refresh_notice_state(
        self,
        db: aiosqlite.Connection,
        *,
        activated_at: int | None = None,
    ) -> InboxNoticeState:
        """Refresh the singleton notice row inside the caller's transaction."""
        async with db.execute(
            "SELECT generation, pending_count, first_pending_deadline_ms, "
            "last_delivered_generation, last_delivered_provider_session_id "
            "FROM inbox_notice_state WHERE singleton = 1"
        ) as cursor:
            old = await cursor.fetchone()
        async with db.execute(
            "SELECT COUNT(*) FROM messages WHERE processing_state = ?",
            (ProcessingState.PENDING.value,),
        ) as cursor:
            count_row = await cursor.fetchone()
        pending_count = int(count_row[0])
        old_count = int(old["pending_count"])
        generation = int(old["generation"]) + (pending_count != old_count)
        deadline = old["first_pending_deadline_ms"]
        if pending_count == 0:
            deadline = None
        elif old_count == 0:
            started = (
                int(activated_at)
                if activated_at is not None
                else int(self._now_ms())
            )
            deadline = started + self.NOTICE_WINDOW_MS
        await db.execute(
            "UPDATE inbox_notice_state SET generation = ?, pending_count = ?, "
            "first_pending_deadline_ms = ? WHERE singleton = 1",
            (generation, pending_count, deadline),
        )
        return InboxNoticeState(
            generation,
            pending_count,
            int(deadline) if deadline is not None else None,
            int(old["last_delivered_generation"]),
            old["last_delivered_provider_session_id"],
        )

    async def get_notice_state(self) -> InboxNoticeState:
        async with self._inbox_lock:
            db = await self._ensure_db()
            async with db.execute(
                "SELECT generation, pending_count, first_pending_deadline_ms, "
                "last_delivered_generation, last_delivered_provider_session_id "
                "FROM inbox_notice_state WHERE singleton = 1"
            ) as cursor:
                row = await cursor.fetchone()
            return InboxNoticeState(
                int(row["generation"]),
                int(row["pending_count"]),
                (
                    int(row["first_pending_deadline_ms"])
                    if row["first_pending_deadline_ms"] is not None
                    else None
                ),
                int(row["last_delivered_generation"]),
                row["last_delivered_provider_session_id"],
            )

    @staticmethod
    def _valid_notice_acceptance(
        generation: int, provider_session_id: str | None,
    ) -> bool:
        return (
            not isinstance(generation, bool)
            and isinstance(generation, int)
            and isinstance(provider_session_id, str)
            and bool(provider_session_id)
        )

    async def _mark_notice_delivered_unlocked(
        self,
        db: aiosqlite.Connection,
        generation: int,
        provider_session_id: str,
    ) -> bool:
        """Persist one native acceptance while the caller owns the write lock."""
        cursor = await db.execute(
            "UPDATE inbox_notice_state "
            "SET last_delivered_generation = ?, "
            "last_delivered_provider_session_id = ? "
            "WHERE singleton = 1 AND generation = ? AND pending_count > 0 "
            "AND NOT (last_delivered_generation = ? "
            "AND last_delivered_provider_session_id IS ?)",
            (
                generation,
                provider_session_id,
                generation,
                generation,
                provider_session_id,
            ),
        )
        return cursor.rowcount == 1

    async def mark_notice_delivered(
        self, generation: int, provider_session_id: str | None,
    ) -> bool:
        """Record a notice only after its native provider accepts it.

        Acceptance is deduplicated by ``(generation, provider_session_id)``.
        The same pending generation may therefore be rediscovered once by a
        replacement session, while a resumed session never receives its own
        accepted notice again.
        """
        if not self._valid_notice_acceptance(generation, provider_session_id):
            return False
        assert isinstance(provider_session_id, str)
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                accepted = await self._mark_notice_delivered_unlocked(
                    db, generation, provider_session_id,
                )
                await db.commit()
                return accepted
            except Exception:
                await db.rollback()
                raise

    async def store(self, payload: Any, *, received_at: int | None = None) -> None:
        async with self._inbox_lock:
            await self._store_unlocked(payload, received_at=received_at)

    async def _store_unlocked(
        self, payload: Any, *, received_at: int | None = None
    ) -> None:
        db = await self._ensure_db()
        if isinstance(payload, dict):
            envelope_id = payload.get("envelope_id", "")
            envelope_kind = payload.get("envelope_kind", "channel")
            sender_slug = payload.get("sender_slug", "")
            channel_id = payload.get("channel_id")
            space_id = payload.get("space_id")
            recipient_slug = payload.get("recipient_slug")
            content_type = payload.get("content_type", "text/plain")
            content = payload.get("content", "")
            sent_at = payload.get("sent_at", _now_ms())
            thread_root_id = payload.get("thread_root_id")
            reply_to_id = payload.get("reply_to_id")
            is_encrypted = payload.get("is_encrypted", True)
        else:
            envelope_id = payload.envelope_id
            envelope_kind = payload.envelope_kind
            sender_slug = payload.sender_slug
            channel_id = payload.channel_id
            space_id = payload.space_id
            recipient_slug = payload.recipient_slug
            content_type = payload.content_type
            content = payload.content
            sent_at = payload.sent_at
            thread_root_id = getattr(payload, "thread_root_id", None)
            reply_to_id = getattr(payload, "reply_to_id", None)
            is_encrypted = getattr(payload, "is_encrypted", True)

        content_str = json.dumps(content) if not isinstance(content, str) else content
        if received_at is None:
            received_at = _now_ms()

        await db.execute(
            """INSERT OR IGNORE INTO messages
            (envelope_id, envelope_kind, sender_slug, channel_id, space_id,
             recipient_slug, content_type, content, sent_at, received_at,
             thread_root_id, reply_to_id, is_encrypted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                envelope_id, envelope_kind, sender_slug, channel_id, space_id,
                recipient_slug, content_type, content_str, sent_at, received_at,
                thread_root_id, reply_to_id, 1 if is_encrypted else 0,
            ),
        )
        await db.commit()

    @staticmethod
    def _payload_values(payload: Any, received_at: int | None) -> tuple[Any, ...]:
        def value(name: str, default: Any = None) -> Any:
            if isinstance(payload, dict):
                return payload.get(name, default)
            return getattr(payload, name, default)

        envelope_id = value("envelope_id", "")
        content = value("content", "")
        content_str = json.dumps(content) if not isinstance(content, str) else content
        return (
            envelope_id,
            value("envelope_kind", "channel"),
            value("sender_slug", ""),
            value("channel_id"),
            value("space_id"),
            value("recipient_slug"),
            value("content_type", "text/plain"),
            content_str,
            value("sent_at", _now_ms()),
            received_at if received_at is not None else _now_ms(),
            value("thread_root_id"),
            value("reply_to_id"),
            1 if value("is_encrypted", True) else 0,
        )

    @staticmethod
    def _receipt_ack(disposition: ReceiptDisposition) -> bool:
        return disposition in (
            ReceiptDisposition.ELIGIBLE,
            ReceiptDisposition.TERMINAL,
        )

    async def store_receipt(
        self,
        payload: Any,
        *,
        server_seq: int,
        disposition: ReceiptDisposition,
        reason: str,
        received_at: int | None = None,
    ) -> ReceiptResult:
        async with self._inbox_lock:
            return await self._store_receipt_unlocked(
                payload,
                server_seq=server_seq,
                disposition=disposition,
                reason=reason,
                received_at=received_at,
            )

    async def _store_receipt_unlocked(
        self,
        payload: Any,
        *,
        server_seq: int,
        disposition: ReceiptDisposition,
        reason: str,
        received_at: int | None = None,
    ) -> ReceiptResult:
        """Persist one durably classified Server receipt.

        Envelope/sequence association conflicts fail closed without mutation.
        A history-only legacy row may acquire its real sequence, but is never
        silently activated as Inbox work.
        """
        if isinstance(server_seq, bool) or not isinstance(server_seq, int) or server_seq <= 0:
            raise ValueError("server_seq must be a positive integer")
        disposition = ReceiptDisposition(disposition)
        values = self._payload_values(payload, received_at)
        envelope_id = values[0]
        if not isinstance(envelope_id, str) or not envelope_id:
            raise ValueError("payload must contain a non-empty envelope_id")
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                "SELECT server_seq, receipt_disposition, receipt_reason, "
                "processing_state, local_ordinal, after_server_seq "
                "FROM messages WHERE envelope_id = ?",
                (envelope_id,),
            ) as cursor:
                by_id = await cursor.fetchone()
            async with db.execute(
                "SELECT envelope_id FROM messages WHERE server_seq = ?",
                (server_seq,),
            ) as cursor:
                by_seq = await cursor.fetchone()

            if by_id is not None:
                existing_seq = by_id["server_seq"]
                if existing_seq is None:
                    if by_seq is not None and by_seq["envelope_id"] != envelope_id:
                        await db.rollback()
                        return ReceiptResult(
                            ReceiptWriteStatus.CONFLICT, disposition,
                            "server sequence belongs to another envelope", False,
                        )
                    if (
                        by_id["receipt_disposition"]
                        == ReceiptDisposition.LOCAL_RUNTIME.value
                        and by_id["processing_state"]
                        in {
                            ProcessingState.PENDING.value,
                            ProcessingState.IN_TURN.value,
                            ProcessingState.PROCESSED.value,
                        }
                        and disposition is ReceiptDisposition.ELIGIBLE
                    ):
                        cursor = await db.execute(
                            """UPDATE messages
                               SET server_seq = ?, receipt_disposition = ?,
                                   receipt_reason = ?, local_ordinal = NULL,
                                   after_server_seq = NULL
                               WHERE envelope_id = ? AND server_seq IS NULL
                                 AND receipt_disposition = ?""",
                            (
                                server_seq,
                                ReceiptDisposition.ELIGIBLE.value,
                                reason,
                                envelope_id,
                                ReceiptDisposition.LOCAL_RUNTIME.value,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise LifecycleConflict(
                                "local receipt promotion lost its association"
                            )
                        await db.commit()
                        return ReceiptResult(
                            ReceiptWriteStatus.COMMITTED,
                            ReceiptDisposition.ELIGIBLE,
                            reason,
                            True,
                        )
                    if (
                        by_id["receipt_disposition"] is None
                        and by_id["processing_state"] is None
                        and disposition is ReceiptDisposition.FOREIGN_DM_GATED
                    ):
                        await db.execute(
                            """UPDATE messages
                               SET server_seq = ?, receipt_disposition = ?,
                                   receipt_reason = ?
                               WHERE envelope_id = ? AND server_seq IS NULL
                                 AND receipt_disposition IS NULL
                                 AND processing_state IS NULL""",
                            (
                                server_seq,
                                disposition.value,
                                reason,
                                envelope_id,
                            ),
                        )
                        await db.commit()
                        return ReceiptResult(
                            ReceiptWriteStatus.COMMITTED,
                            disposition,
                            reason,
                            False,
                        )
                    await db.execute(
                        "UPDATE messages SET server_seq = ? WHERE envelope_id = ? "
                        "AND server_seq IS NULL",
                        (server_seq, envelope_id),
                    )
                    await db.commit()
                    return ReceiptResult(
                        ReceiptWriteStatus.COMMITTED, disposition,
                        "legacy sequence backfilled without Inbox activation",
                        self._receipt_ack(disposition),
                    )
                if int(existing_seq) != server_seq:
                    await db.rollback()
                    return ReceiptResult(
                        ReceiptWriteStatus.CONFLICT, disposition,
                        "envelope id belongs to another server sequence", False,
                    )
                stored_raw = by_id["receipt_disposition"]
                stored = ReceiptDisposition(stored_raw) if stored_raw else disposition
                await db.rollback()
                return ReceiptResult(
                    ReceiptWriteStatus.IDEMPOTENT,
                    stored,
                    by_id["receipt_reason"] or reason,
                    self._receipt_ack(stored),
                )

            if by_seq is not None:
                await db.rollback()
                return ReceiptResult(
                    ReceiptWriteStatus.CONFLICT, disposition,
                    "server sequence belongs to another envelope", False,
                )

            processing = (
                ProcessingState.PENDING.value
                if disposition is ReceiptDisposition.ELIGIBLE
                else None
            )
            await db.execute(
                """INSERT INTO messages
                   (envelope_id, envelope_kind, sender_slug, channel_id, space_id,
                    recipient_slug, content_type, content, sent_at, received_at,
                    thread_root_id, reply_to_id, is_encrypted, server_seq,
                    receipt_disposition, receipt_reason, processing_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values + (server_seq, disposition.value, reason, processing),
            )
            if processing == ProcessingState.PENDING.value:
                await self._refresh_notice_state(
                    db, activated_at=int(values[9])
                )
            await db.commit()
            return ReceiptResult(
                ReceiptWriteStatus.COMMITTED,
                disposition,
                reason,
                self._receipt_ack(disposition),
            )
        except Exception:
            await db.rollback()
            raise

    async def promote_gated_receipt(
        self,
        envelope_id: str,
        server_seq: int,
        *,
        reason: str,
    ) -> ReceiptResult:
        async with self._inbox_lock:
            return await self._promote_gated_receipt_unlocked(
                envelope_id, server_seq, reason=reason
            )

    async def _promote_gated_receipt_unlocked(
        self,
        envelope_id: str,
        server_seq: int,
        *,
        reason: str,
    ) -> ReceiptResult:
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                "SELECT server_seq, receipt_disposition, processing_state "
                "FROM messages WHERE envelope_id = ?",
                (envelope_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or row["server_seq"] != server_seq:
                await db.rollback()
                return ReceiptResult(
                    ReceiptWriteStatus.CONFLICT,
                    ReceiptDisposition.FOREIGN_DM_GATED,
                    "gated receipt association mismatch",
                    False,
                )
            if (
                row["receipt_disposition"] == ReceiptDisposition.ELIGIBLE.value
                and row["processing_state"] == ProcessingState.PENDING.value
            ):
                await db.rollback()
                return ReceiptResult(
                    ReceiptWriteStatus.IDEMPOTENT,
                    ReceiptDisposition.ELIGIBLE,
                    reason,
                    True,
                )
            if (
                row["receipt_disposition"] != ReceiptDisposition.FOREIGN_DM_GATED.value
                or row["processing_state"] is not None
            ):
                await db.rollback()
                return ReceiptResult(
                    ReceiptWriteStatus.CONFLICT,
                    ReceiptDisposition.FOREIGN_DM_GATED,
                    "receipt is not approval-gated",
                    False,
                )
            await db.execute(
                """UPDATE messages
                   SET receipt_disposition = ?, receipt_reason = ?, processing_state = ?
                   WHERE envelope_id = ? AND server_seq = ?
                     AND receipt_disposition = ? AND processing_state IS NULL""",
                (
                    ReceiptDisposition.ELIGIBLE.value,
                    reason,
                    ProcessingState.PENDING.value,
                    envelope_id,
                    server_seq,
                    ReceiptDisposition.FOREIGN_DM_GATED.value,
                ),
            )
            await self._refresh_notice_state(db)
            await db.commit()
            return ReceiptResult(
                ReceiptWriteStatus.COMMITTED,
                ReceiptDisposition.ELIGIBLE,
                reason,
                True,
            )
        except Exception:
            await db.rollback()
            raise

    async def store_local_event(
        self,
        payload: Any,
        *,
        reason: str,
        intro_channel_id: str | None = None,
        received_at: int | None = None,
    ) -> StoredMessage:
        async with self._inbox_lock:
            return await self._store_local_event_unlocked(
                payload,
                reason=reason,
                intro_channel_id=intro_channel_id,
                received_at=received_at,
            )

    async def _store_local_event_unlocked(
        self,
        payload: Any,
        *,
        reason: str,
        intro_channel_id: str | None = None,
        received_at: int | None = None,
    ) -> StoredMessage:
        values = self._payload_values(payload, received_at)
        envelope_id = values[0]
        if not isinstance(envelope_id, str) or not envelope_id:
            raise ValueError("payload must contain a non-empty envelope_id")
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            if intro_channel_id:
                cursor = await db.execute(
                    "INSERT OR IGNORE INTO channel_intro_prompted(channel_id, prompted_at) "
                    "VALUES (?, ?)",
                    (intro_channel_id, _now_ms()),
                )
                if cursor.rowcount == 0:
                    async with db.execute(
                        "SELECT receipt_disposition FROM messages WHERE envelope_id = ?",
                        (envelope_id,),
                    ) as existing_cursor:
                        existing = await existing_cursor.fetchone()
                    if (
                        existing is None
                        or existing["receipt_disposition"]
                        != ReceiptDisposition.LOCAL_RUNTIME.value
                    ):
                        raise LifecycleConflict(
                            "introduction marker already belongs to another local event"
                        )
                    await db.rollback()
                    message = await self._get_message_by_envelope_unlocked(
                        db, envelope_id
                    )
                    assert message is not None
                    return message

            await self._insert_local_event_in_transaction(
                db, values=values, reason=reason,
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        message = await self._get_message_by_envelope_unlocked(db, envelope_id)
        assert message is not None
        return message

    async def _insert_local_event_in_transaction(
        self,
        db: aiosqlite.Connection,
        *,
        values: tuple[Any, ...],
        reason: str,
    ) -> None:
        """Insert one local pending event while a caller owns ``BEGIN IMMEDIATE``.

        Reminder delivery needs this exact insertion plus an occurrence state
        transition in one SQLite transaction.  Keeping allocation here avoids
        nesting ``store_local_event`` transactions while preserving the same
        local frontier, ordinal, and notice-generation semantics for every
        local event type.
        """
        async with db.execute(
            """SELECT MAX(server_seq) FROM messages
               WHERE receipt_disposition = ? AND processing_state = ?
                 AND server_seq IS NOT NULL""",
            (ReceiptDisposition.ELIGIBLE.value, ProcessingState.PENDING.value),
        ) as cursor:
            frontier_row = await cursor.fetchone()
        frontier = int(frontier_row[0]) if frontier_row[0] is not None else None
        async with db.execute(
            "SELECT next_ordinal FROM local_ordinal_allocator WHERE singleton = 1"
        ) as cursor:
            ordinal_row = await cursor.fetchone()
        ordinal = int(ordinal_row[0])
        await db.execute(
            "UPDATE local_ordinal_allocator SET next_ordinal = ? WHERE singleton = 1",
            (ordinal + 1,),
        )
        await db.execute(
            """INSERT INTO messages
               (envelope_id, envelope_kind, sender_slug, channel_id, space_id,
                recipient_slug, content_type, content, sent_at, received_at,
                thread_root_id, reply_to_id, is_encrypted, receipt_disposition,
                receipt_reason, processing_state, local_ordinal, after_server_seq)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values + (
                ReceiptDisposition.LOCAL_RUNTIME.value,
                reason,
                ProcessingState.PENDING.value,
                ordinal,
                frontier,
            ),
        )
        await self._refresh_notice_state(db, activated_at=int(values[9]))

    @staticmethod
    def _reminder_from_row(row: aiosqlite.Row) -> ReminderOccurrence:
        return ReminderOccurrence(
            reminder_id=str(row["reminder_id"]),
            occurrence_id=str(row["occurrence_id"]),
            target=str(row["target"]),
            content=str(row["content"]),
            intended_at_ms=int(row["intended_at_ms"]),
            state=str(row["state"]),
            created_at_ms=int(row["created_at_ms"]),
            claimed_at_ms=(
                int(row["claimed_at_ms"])
                if row["claimed_at_ms"] is not None else None
            ),
            actual_fire_at_ms=(
                int(row["actual_fire_at_ms"])
                if row["actual_fire_at_ms"] is not None else None
            ),
            cancelled_at_ms=(
                int(row["cancelled_at_ms"])
                if row["cancelled_at_ms"] is not None else None
            ),
            delivered_at_ms=(
                int(row["delivered_at_ms"])
                if row["delivered_at_ms"] is not None else None
            ),
            delivered_event_id=(
                str(row["delivered_event_id"])
                if row["delivered_event_id"] is not None else None
            ),
        )

    @staticmethod
    def _valid_reminder_time(value: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    async def create_reminder(
        self,
        *,
        content: str,
        target: str,
        intended_at_ms: int,
        reminder_id: str | None = None,
        occurrence_id: str | None = None,
        created_at_ms: int | None = None,
    ) -> ReminderOccurrence:
        """Durably create one immutable, single-fire reminder intent."""
        if not isinstance(content, str) or not content:
            raise ValueError("reminder content must be a non-empty string")
        parse_reminder_target(target)
        if not self._valid_reminder_time(intended_at_ms):
            raise ValueError("intended_at_ms must be an integer epoch milliseconds")
        if created_at_ms is None:
            created_at_ms = int(self._now_ms())
        if not self._valid_reminder_time(created_at_ms):
            raise ValueError("created_at_ms must be an integer epoch milliseconds")
        reminder_id = reminder_id or f"reminder-{uuid.uuid4()}"
        occurrence_id = occurrence_id or f"occurrence-{uuid.uuid4()}"
        if (
            not isinstance(reminder_id, str)
            or not reminder_id
            or not isinstance(occurrence_id, str)
            or not occurrence_id
            or reminder_id == occurrence_id
        ):
            raise ValueError("reminder and occurrence identities must be distinct")
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    """INSERT INTO reminder_occurrences
                       (reminder_id, occurrence_id, target, content, intended_at_ms,
                        state, created_at_ms)
                       VALUES (?, ?, ?, ?, ?, 'scheduled', ?)""",
                    (
                        reminder_id,
                        occurrence_id,
                        target,
                        content,
                        intended_at_ms,
                        created_at_ms,
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            reminder = await self._get_reminder_unlocked(db, reminder_id)
            assert reminder is not None
            return reminder

    async def _get_reminder_unlocked(
        self, db: aiosqlite.Connection, reminder_id: str,
    ) -> ReminderOccurrence | None:
        async with db.execute(
            "SELECT * FROM reminder_occurrences WHERE reminder_id = ?",
            (reminder_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._reminder_from_row(row) if row is not None else None

    async def get_reminder(self, reminder_id: str) -> ReminderOccurrence | None:
        if not isinstance(reminder_id, str) or not reminder_id:
            raise ValueError("reminder_id must be a non-empty string")
        async with self._inbox_lock:
            return await self._get_reminder_unlocked(
                await self._ensure_db(), reminder_id,
            )

    async def list_reminders(
        self, *, state: str = "", limit: int = 50,
    ) -> tuple[ReminderOccurrence, ...]:
        if not isinstance(state, str) or (state and state not in REMINDER_STATES):
            raise ValueError("state must be empty or a reminder state")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_REMINDER_LIST_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {MAX_REMINDER_LIST_LIMIT}"
            )
        async with self._inbox_lock:
            db = await self._ensure_db()
            if state:
                query = (
                    "SELECT * FROM reminder_occurrences WHERE state = ? "
                    "ORDER BY intended_at_ms, occurrence_id LIMIT ?"
                )
                params: tuple[Any, ...] = (state, limit)
            else:
                query = (
                    "SELECT * FROM reminder_occurrences "
                    "ORDER BY intended_at_ms, occurrence_id LIMIT ?"
                )
                params = (limit,)
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return tuple(self._reminder_from_row(row) for row in rows)

    async def next_reminder_deadline(
        self, *, now_ms: int | None = None,
    ) -> int | None:
        """Return the next scheduled deadline, or now when recovery is claimed."""
        now = int(self._now_ms()) if now_ms is None else now_ms
        if not self._valid_reminder_time(now):
            raise ValueError("now_ms must be an integer epoch milliseconds")
        async with self._inbox_lock:
            db = await self._ensure_db()
            async with db.execute(
                "SELECT 1 FROM reminder_occurrences WHERE state = 'claimed' LIMIT 1"
            ) as cursor:
                if await cursor.fetchone() is not None:
                    return now
            async with db.execute(
                """SELECT intended_at_ms FROM reminder_occurrences
                   WHERE state = 'scheduled'
                   ORDER BY intended_at_ms, occurrence_id LIMIT 1"""
            ) as cursor:
                row = await cursor.fetchone()
            return int(row["intended_at_ms"]) if row is not None else None

    async def claim_due_reminders(
        self, *, now_ms: int | None = None,
    ) -> tuple[ReminderOccurrence, ...]:
        """Claim every due scheduled occurrence without changing its identity."""
        now = int(self._now_ms()) if now_ms is None else now_ms
        if not self._valid_reminder_time(now):
            raise ValueError("now_ms must be an integer epoch milliseconds")
        async with self._inbox_lock:
            return await self._claim_due_reminders_unlocked(now)

    async def _claim_due_reminders_unlocked(
        self, now_ms: int,
    ) -> tuple[ReminderOccurrence, ...]:
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                """SELECT reminder_id FROM reminder_occurrences
                   WHERE state = 'scheduled' AND intended_at_ms <= ?
                   ORDER BY intended_at_ms, occurrence_id""",
                (now_ms,),
            ) as cursor:
                rows = await cursor.fetchall()
            claimed_ids: list[str] = []
            for row in rows:
                reminder_id = str(row["reminder_id"])
                cursor = await db.execute(
                    """UPDATE reminder_occurrences
                       SET state = 'claimed', claimed_at_ms = ?, actual_fire_at_ms = ?
                       WHERE reminder_id = ? AND state = 'scheduled'""",
                    (now_ms, now_ms, reminder_id),
                )
                if cursor.rowcount == 1:
                    claimed_ids.append(reminder_id)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        claimed: list[ReminderOccurrence] = []
        for reminder_id in claimed_ids:
            occurrence = await self._get_reminder_unlocked(db, reminder_id)
            assert occurrence is not None
            claimed.append(occurrence)
        return tuple(claimed)

    @staticmethod
    def _reminder_event_payload(reminder: ReminderOccurrence) -> dict[str, Any]:
        kind, space_id, channel_id, tail = parse_reminder_target(reminder.target)
        actual_fire_at_ms = reminder.actual_fire_at_ms
        if actual_fire_at_ms is None:
            raise LifecycleConflict("claimed reminder is missing actual fire time")
        content = {
            "event_type": "reminder",
            "reminder_id": reminder.reminder_id,
            "occurrence_id": reminder.occurrence_id,
            "target": reminder.target,
            "content": reminder.content,
            "intended_at": reminder_time_to_rfc3339(reminder.intended_at_ms),
            "actual_fire_at": reminder_time_to_rfc3339(actual_fire_at_ms),
        }
        if kind == "dm":
            return {
                "envelope_id": f"reminder-occurrence:{reminder.occurrence_id}",
                "envelope_kind": "dm",
                # ``target_projection`` and ``route_for`` use this peer.
                "sender_slug": tail,
                "recipient_slug": "",
                "content_type": "application/puffo-reminder+json",
                "content": content,
                "sent_at": actual_fire_at_ms,
            }
        return {
            "envelope_id": f"reminder-occurrence:{reminder.occurrence_id}",
            "envelope_kind": "channel",
            "sender_slug": "reminder",
            "channel_id": channel_id,
            "space_id": space_id,
            "content_type": "application/puffo-reminder+json",
            "content": content,
            "sent_at": actual_fire_at_ms,
            "thread_root_id": tail or None,
        }

    async def _deliver_claimed_reminder_unlocked(
        self,
        reminder_id: str,
        *,
        now_ms: int,
    ) -> ReminderOccurrence | None:
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            reminder = await self._get_reminder_unlocked(db, reminder_id)
            if reminder is None or reminder.state != "claimed":
                await db.rollback()
                return None
            event_id = f"reminder-occurrence:{reminder.occurrence_id}"
            async with db.execute(
                "SELECT receipt_disposition, content FROM messages WHERE envelope_id = ?",
                (event_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is None:
                payload = self._reminder_event_payload(reminder)
                await self._insert_local_event_in_transaction(
                    db,
                    values=self._payload_values(payload, now_ms),
                    reason="reminder occurrence",
                )
            else:
                # This state is impossible after a normal crash because the
                # insert and terminal occurrence update commit together.  If
                # a pre-existing row is encountered, fail closed unless it is
                # exactly a local reminder event for this occurrence.
                try:
                    existing_content = json.loads(existing["content"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    existing_content = None
                if (
                    existing["receipt_disposition"]
                    != ReceiptDisposition.LOCAL_RUNTIME.value
                    or not isinstance(existing_content, dict)
                    or existing_content.get("event_type") != "reminder"
                    or existing_content.get("occurrence_id")
                    != reminder.occurrence_id
                ):
                    raise LifecycleConflict(
                        "reminder delivery event identity belongs to another row"
                    )
            cursor = await db.execute(
                """UPDATE reminder_occurrences
                   SET state = 'delivered', delivered_event_id = ?, delivered_at_ms = ?
                   WHERE reminder_id = ? AND state = 'claimed'""",
                (event_id, now_ms, reminder_id),
            )
            if cursor.rowcount != 1:
                raise LifecycleConflict("reminder delivery lost claimed ownership")
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        delivered = await self._get_reminder_unlocked(db, reminder_id)
        assert delivered is not None
        return delivered

    async def deliver_due_reminders(
        self, *, now_ms: int | None = None,
    ) -> tuple[ReminderOccurrence, ...]:
        """Atomically enqueue every claimed/due occurrence at most once.

        A claimed occurrence is included even when it was claimed by a worker
        that restarted before delivery.  Every individual delivery commits the
        local Inbox row and terminal occurrence state together.
        """
        now = int(self._now_ms()) if now_ms is None else now_ms
        if not self._valid_reminder_time(now):
            raise ValueError("now_ms must be an integer epoch milliseconds")
        async with self._inbox_lock:
            await self._claim_due_reminders_unlocked(now)
            db = await self._ensure_db()
            async with db.execute(
                """SELECT reminder_id FROM reminder_occurrences WHERE state = 'claimed'
                   ORDER BY intended_at_ms, occurrence_id"""
            ) as cursor:
                rows = await cursor.fetchall()
            delivered: list[ReminderOccurrence] = []
            for row in rows:
                occurrence = await self._deliver_claimed_reminder_unlocked(
                    str(row["reminder_id"]), now_ms=now,
                )
                if occurrence is not None:
                    delivered.append(occurrence)
            return tuple(delivered)

    async def cancel_reminder(
        self, reminder_id: str, *, cancelled_at_ms: int | None = None,
    ) -> ReminderOccurrence:
        """Idempotently cancel work that has not terminally delivered."""
        if not isinstance(reminder_id, str) or not reminder_id:
            raise ValueError("reminder_id must be a non-empty string")
        cancelled_at = int(self._now_ms()) if cancelled_at_ms is None else cancelled_at_ms
        if not self._valid_reminder_time(cancelled_at):
            raise ValueError("cancelled_at_ms must be an integer epoch milliseconds")
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                reminder = await self._get_reminder_unlocked(db, reminder_id)
                if reminder is None:
                    raise ValueError("unknown reminder_id")
                if reminder.state in {"scheduled", "claimed"}:
                    cursor = await db.execute(
                        """UPDATE reminder_occurrences
                           SET state = 'cancelled', cancelled_at_ms = ?
                           WHERE reminder_id = ? AND state IN ('scheduled', 'claimed')""",
                        (cancelled_at, reminder_id),
                    )
                    if cursor.rowcount != 1:
                        raise LifecycleConflict("reminder cancellation lost ownership")
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            result = await self._get_reminder_unlocked(db, reminder_id)
            assert result is not None
            return result

    async def quarantine_pending(self, envelope_id: str, *, reason: str) -> bool:
        async with self._inbox_lock:
            return await self._quarantine_pending_unlocked(
                envelope_id, reason=reason
            )

    async def _quarantine_pending_unlocked(
        self, envelope_id: str, *, reason: str
    ) -> bool:
        db = await self._ensure_db()
        await db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await db.execute(
                """UPDATE messages
                   SET receipt_disposition = ?, receipt_reason = ?,
                       processing_state = NULL, processing_turn_id = NULL,
                       model_visible_at = NULL
                   WHERE envelope_id = ? AND processing_state = ?""",
                (
                    ReceiptDisposition.TERMINAL.value,
                    reason,
                    envelope_id,
                    ProcessingState.PENDING.value,
                ),
            )
            changed = cursor.rowcount == 1
            if changed:
                await self._refresh_notice_state(db)
            await db.commit()
            return changed
        except Exception:
            await db.rollback()
            raise

    async def get_pending(self, *, limit: int | None = None) -> tuple[StoredMessage, ...]:
        async with self._inbox_lock:
            return await self._get_pending_unlocked(limit=limit)

    async def _get_pending_unlocked(
        self, *, limit: int | None = None
    ) -> tuple[StoredMessage, ...]:
        db = await self._ensure_db()
        sql = """
            SELECT * FROM messages
            WHERE processing_state = ?
              AND (
                (receipt_disposition = ? AND server_seq IS NOT NULL)
                OR
                (receipt_disposition = ? AND server_seq IS NULL
                 AND local_ordinal IS NOT NULL)
              )
            ORDER BY COALESCE(server_seq, after_server_seq, 0) ASC,
                     CASE WHEN server_seq IS NOT NULL THEN 0 ELSE 1 END ASC,
                     local_ordinal ASC, envelope_id ASC
        """
        params: list[Any] = [
            ProcessingState.PENDING.value,
            ReceiptDisposition.ELIGIBLE.value,
            ReceiptDisposition.LOCAL_RUNTIME.value,
        ]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            params.append(limit)
        async with db.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return tuple(self._row_to_msg(row) for row in rows)

    @staticmethod
    def target_projection(item: StoredMessage) -> str:
        """Return the canonical Inbox projection for one durable row."""
        if item.envelope_kind == "dm":
            return f"dm:{item.sender_slug}"
        base = f"channel:{item.space_id or ''}:{item.channel_id or ''}"
        if item.thread_root_id:
            return f"{base}:thread:{item.thread_root_id}"
        return base

    @staticmethod
    def _inbox_order(item: StoredMessage) -> tuple[int, int, int, str]:
        return (
            int(
                item.server_seq
                if item.server_seq is not None
                else item.after_server_seq or 0
            ),
            0 if item.server_seq is not None else 1,
            int(item.local_ordinal or 0),
            item.envelope_id,
        )

    @staticmethod
    def _encode_inbox_cursor(value: dict[str, Any]) -> str:
        raw = json.dumps(
            value, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_inbox_cursor(cursor: str) -> dict[str, Any]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        except Exception as exc:
            raise ValueError("invalid Inbox cursor") from exc
        if (
            not isinstance(value, dict)
            or value.get("v") != 1
            or not isinstance(value.get("target"), str)
            or not isinstance(value.get("generation"), int)
            or not isinstance(value.get("ceiling"), list)
            or not isinstance(value.get("last"), list)
        ):
            raise ValueError("invalid Inbox cursor")
        return value

    async def read_inbox_page(
        self,
        *,
        target: str = "",
        cursor: str = "",
        limit: int = 50,
    ) -> InboxPage:
        """Read one stable pending snapshot without changing lifecycle state."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        if not isinstance(target, str):
            raise ValueError("target must be a string")
        target = target.strip()
        if target and not (
            target.startswith("channel:") or target.startswith("dm:")
        ):
            raise ValueError("invalid Inbox target")
        async with self._inbox_lock:
            state = await self._notice_state_unlocked()
            pending = tuple(
                item
                for item in await self._get_pending_unlocked()
                if not target or self.target_projection(item) == target
            )
            if cursor:
                decoded = self._decode_inbox_cursor(cursor)
                if decoded["target"] != target:
                    raise ValueError("Inbox cursor belongs to another target")
                generation = int(decoded["generation"])
                ceiling = tuple(decoded["ceiling"])
                last = tuple(decoded["last"])
            else:
                generation = state.generation
                ceiling = self._inbox_order(pending[-1]) if pending else (0, 0, 0, "")
                last = (-1, -1, -1, "")
            snapshot = [
                item
                for item in pending
                if last < self._inbox_order(item) <= ceiling
            ]
            selected = tuple(snapshot[:limit])
            remaining = len(snapshot) - len(selected)
            has_more = remaining > 0
            next_cursor = ""
            if has_more:
                next_cursor = self._encode_inbox_cursor(
                    {
                        "v": 1,
                        "target": target,
                        "generation": generation,
                        "ceiling": list(ceiling),
                        "last": list(self._inbox_order(selected[-1])),
                    }
                )
            return InboxPage(
                selected,
                next_cursor,
                has_more,
                remaining,
                generation,
                target,
            )

    async def _notice_state_unlocked(self) -> InboxNoticeState:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT generation, pending_count, first_pending_deadline_ms, "
            "last_delivered_generation, last_delivered_provider_session_id "
            "FROM inbox_notice_state WHERE singleton = 1"
        ) as cursor:
            row = await cursor.fetchone()
        return InboxNoticeState(
            int(row["generation"]),
            int(row["pending_count"]),
            (
                int(row["first_pending_deadline_ms"])
                if row["first_pending_deadline_ms"] is not None
                else None
            ),
            int(row["last_delivered_generation"]),
            row["last_delivered_provider_session_id"],
        )

    async def get_prior_context(
        self,
        anchor: StoredMessage,
        *,
        limit: int = PRIOR_CONTEXT_MAX_ITEMS,
        max_bytes: int = PRIOR_CONTEXT_MAX_BYTES,
    ) -> tuple[StoredMessage, ...]:
        """Return a bounded, read-only slice before an Inbox page anchor.

        The route is derived from the durable row itself.  Only rows already
        completed or terminally classified are eligible, so pending and
        in-turn work can never be smuggled into a later provider decision.
        Ordering uses the same durable Inbox tuple as paging and local-event
        frontiers; ``sent_at`` and provider-session state are not authority.
        """
        if not isinstance(anchor, StoredMessage):
            raise TypeError("anchor must be a StoredMessage")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be non-negative")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        limit = min(limit, PRIOR_CONTEXT_MAX_ITEMS)
        max_bytes = min(max_bytes, PRIOR_CONTEXT_MAX_BYTES)
        if not limit or not max_bytes:
            return ()

        position = self._inbox_order(anchor)
        order_sql = (
            "COALESCE(m.server_seq, m.after_server_seq, 0)",
            "CASE WHEN m.server_seq IS NOT NULL THEN 0 ELSE 1 END",
            "COALESCE(m.local_ordinal, 0)",
            "m.envelope_id",
        )
        clauses = [
            f"({', '.join(order_sql)}) < (?, ?, ?, ?)",
            "(m.processing_state = ? OR m.receipt_disposition = ?)",
        ]
        params: list[Any] = [
            *position,
            ProcessingState.PROCESSED.value,
            ReceiptDisposition.TERMINAL.value,
        ]

        if anchor.envelope_kind == "dm":
            peer = anchor.sender_slug or anchor.recipient_slug or ""
            if not peer:
                return ()
            clauses.extend([
                "m.envelope_kind = 'dm'",
                "(m.sender_slug = ? OR m.recipient_slug = ?)",
            ])
            params.extend((peer, peer))
        else:
            clauses.extend([
                "m.envelope_kind != 'dm'",
                "m.space_id = ?",
                "m.channel_id = ?",
            ])
            params.extend((anchor.space_id, anchor.channel_id))
            is_intro_prompt = (
                anchor.sender_slug == "system"
                and anchor.envelope_id.startswith("intro-prompt-")
                and anchor.thread_root_id == anchor.envelope_id
            )
            if anchor.thread_root_id and not is_intro_prompt:
                clauses.append("(m.envelope_id = ? OR m.thread_root_id = ?)")
                params.extend((anchor.thread_root_id, anchor.thread_root_id))
            else:
                clauses.append("m.thread_root_id IS NULL")

        async with self._inbox_lock:
            db = await self._ensure_db()
            sql = (
                "SELECT m.* FROM messages m WHERE "
                + " AND ".join(clauses)
                + f" ORDER BY {', '.join(expression + ' DESC' for expression in order_sql)}"
                + " LIMIT ?"
            )
            params.append(limit)
            async with db.execute(sql, tuple(params)) as cursor:
                rows = await cursor.fetchall()

            # The query selects newest-before-anchor first so the bounded
            # slice retains the newest rows, then returns them chronological
            # to match every other history/context read. Count durable body
            # bytes here; GlobalInboxRuntime applies the exact formatter-byte
            # bound before exposing the result.
            selected_rows: list[aiosqlite.Row] = []
            used_bytes = 0
            for row in rows:
                body_bytes = len(str(row["content"]).encode("utf-8"))
                if used_bytes + body_bytes > max_bytes:
                    continue
                selected_rows.append(row)
                used_bytes += body_bytes
            return tuple(
                self._row_to_msg(row) for row in reversed(selected_rows)
            )

    async def get_channel_pending(
        self,
        space_id: str,
        channel_id: str,
        *,
        after_seq: int | None = None,
        through_seq: int | None = None,
        limit: int = 50,
    ) -> PendingPage:
        async with self._inbox_lock:
            return await self._get_channel_pending_unlocked(
                space_id,
                channel_id,
                after_seq=after_seq,
                through_seq=through_seq,
                limit=limit,
            )

    async def _get_channel_pending_unlocked(
        self,
        space_id: str,
        channel_id: str,
        *,
        after_seq: int | None = None,
        through_seq: int | None = None,
        limit: int = 50,
    ) -> PendingPage:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        candidates = [
            item
            for item in await self._get_pending_unlocked()
            if item.space_id == space_id and item.channel_id == channel_id
        ]

        def included(item: StoredMessage) -> bool:
            position = (
                item.server_seq
                if item.server_seq is not None
                else item.after_server_seq
            )
            if position is None:
                position = 0
            if after_seq is not None and position < after_seq:
                return False
            if (
                after_seq is not None
                and position == after_seq
                and item.server_seq is not None
            ):
                return False
            if through_seq is not None and position > through_seq:
                return False
            return True

        filtered = [item for item in candidates if included(item)]
        selected = tuple(filtered[:limit])
        selected_server = [
            item.server_seq for item in selected if item.server_seq is not None
        ]
        return PendingPage(
            selected,
            max(selected_server) if selected_server else None,
            len(filtered) > len(selected),
        )

    @staticmethod
    def _exact_ids(message_ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(message_ids)
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise LifecycleConflict("message ID set must be non-empty")
        if len(ids) != len(set(ids)):
            raise LifecycleConflict("message ID set contains duplicates")
        return ids

    async def admit_messages(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
        provider_session_id: str | None,
        model_visible_at: int | None = None,
    ) -> TurnRun:
        async with self._inbox_lock:
            return await self._admit_messages_unlocked(
                message_ids,
                turn_id=turn_id,
                provider_session_id=provider_session_id,
                model_visible_at=model_visible_at,
            )

    async def start_turn(
        self,
        *,
        turn_id: str,
        provider_session_id: str | None,
        started_at: int | None = None,
        notice_generation: int | None = None,
    ) -> TurnRun:
        """Create an active Turn, atomically accepting a notice when present."""
        if not turn_id:
            raise LifecycleConflict("turn ID is required")
        if notice_generation is not None and not self._valid_notice_acceptance(
            notice_generation, provider_session_id,
        ):
            raise LifecycleConflict(
                "notice acceptance requires a generation and provider session"
            )
        async with self._inbox_lock:
            db = await self._ensure_db()
            started = started_at if started_at is not None else int(self._now_ms())
            try:
                await db.execute("BEGIN IMMEDIATE")
                if notice_generation is not None:
                    assert isinstance(provider_session_id, str)
                    if not await self._mark_notice_delivered_unlocked(
                        db, notice_generation, provider_session_id,
                    ):
                        raise LifecycleConflict(
                            "Inbox notice is stale or already accepted by this session"
                        )
                await db.execute(
                    "INSERT INTO turn_runs(turn_id, provider_session_id, state, "
                    "started_at, completed_at) VALUES (?, ?, ?, ?, NULL)",
                    (
                        turn_id,
                        provider_session_id,
                        ProcessingState.IN_TURN.value,
                        started,
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            run = await self._get_turn_run_unlocked(db, turn_id)
            assert run is not None
            return run

    async def finalize_empty_turn(
        self,
        *,
        turn_id: str,
        state: str = ProcessingState.PROCESSED.value,
        rearm_notice: bool = False,
    ) -> TurnRun:
        """Finalize a notice Turn which admitted no Inbox rows."""
        if state not in (ProcessingState.PROCESSED.value, "requeued"):
            raise LifecycleConflict("invalid terminal Turn state")
        async with self._inbox_lock:
            db = await self._ensure_db()
            completed = int(self._now_ms())
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "UPDATE turn_runs SET state = ?, completed_at = ? "
                    "WHERE turn_id = ? AND state = ? AND NOT EXISTS "
                    "(SELECT 1 FROM turn_run_messages WHERE turn_id = ?)",
                    (
                        state,
                        completed,
                        turn_id,
                        ProcessingState.IN_TURN.value,
                        turn_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise LifecycleConflict("Turn is not active and empty")
                if rearm_notice:
                    # A notice is delivery metadata, not an Inbox ACK. If the
                    # Turn admitted no rows, create a fresh delivery generation
                    # for the unchanged pending set so work cannot become
                    # stranded after an ignored tool call, failure, or restart.
                    await db.execute(
                        "UPDATE inbox_notice_state "
                        "SET generation = generation + 1, "
                        "first_pending_deadline_ms = ? "
                        "WHERE singleton = 1 AND pending_count > 0 "
                        "AND generation = last_delivered_generation",
                        (completed + self.NOTICE_WINDOW_MS,),
                    )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            run = await self._get_turn_run_unlocked(db, turn_id)
            assert run is not None
            return run

    async def _admit_messages_unlocked(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
        provider_session_id: str | None,
        model_visible_at: int | None = None,
    ) -> TurnRun:
        ids = self._exact_ids(message_ids)
        if not turn_id:
            raise LifecycleConflict("turn ID is required")
        visible_at = model_visible_at if model_visible_at is not None else _now_ms()
        db = await self._ensure_db()
        placeholders = ",".join("?" for _ in ids)
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                f"SELECT envelope_id, processing_state FROM messages "
                f"WHERE envelope_id IN ({placeholders})",
                ids,
            ) as cursor:
                rows = await cursor.fetchall()
            states = {row["envelope_id"]: row["processing_state"] for row in rows}
            if len(states) != len(ids) or any(
                states.get(item) != ProcessingState.PENDING.value for item in ids
            ):
                raise LifecycleConflict("every exact message ID must be pending")
            async with db.execute(
                "SELECT provider_session_id, state FROM turn_runs WHERE turn_id = ?",
                (turn_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None:
                if existing["provider_session_id"] != provider_session_id:
                    raise LifecycleConflict("turn belongs to another provider session")
                if existing["state"] != ProcessingState.IN_TURN.value:
                    raise LifecycleConflict("turn is not active")
                async with db.execute(
                    f"""SELECT envelope_id FROM turn_run_messages
                        WHERE turn_id = ? AND envelope_id IN ({placeholders})""",
                    (turn_id, *ids),
                ) as cursor:
                    duplicate_members = await cursor.fetchall()
                if duplicate_members:
                    raise LifecycleConflict(
                        "message ID is already a member of this turn"
                    )
                async with db.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) FROM turn_run_messages "
                    "WHERE turn_id = ?",
                    (turn_id,),
                ) as cursor:
                    ordinal_row = await cursor.fetchone()
                first_ordinal = int(ordinal_row[0]) + 1
            else:
                await db.execute(
                    """INSERT INTO turn_runs
                       (turn_id, provider_session_id, state, started_at, completed_at)
                       VALUES (?, ?, ?, ?, NULL)""",
                    (
                        turn_id,
                        provider_session_id,
                        ProcessingState.IN_TURN.value,
                        visible_at,
                    ),
                )
                first_ordinal = 0
            for offset, envelope_id in enumerate(ids):
                await db.execute(
                    "INSERT INTO turn_run_messages(turn_id, envelope_id, ordinal) "
                    "VALUES (?, ?, ?)",
                    (turn_id, envelope_id, first_ordinal + offset),
                )
            cursor = await db.execute(
                f"""UPDATE messages
                    SET processing_state = ?, processing_turn_id = ?, model_visible_at = ?
                    WHERE envelope_id IN ({placeholders}) AND processing_state = ?""",
                (
                    ProcessingState.IN_TURN.value,
                    turn_id,
                    visible_at,
                    *ids,
                    ProcessingState.PENDING.value,
                ),
            )
            if cursor.rowcount != len(ids):
                raise LifecycleConflict("admission changed fewer rows than requested")
            await self._refresh_notice_state(db)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        run = await self._get_turn_run_unlocked(db, turn_id)
        assert run is not None
        return run

    async def mark_processed(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
        processed_at: int | None = None,
    ) -> TurnRun:
        async with self._inbox_lock:
            return await self._mark_processed_unlocked(
                message_ids, turn_id=turn_id, processed_at=processed_at
            )

    async def _mark_processed_unlocked(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
        processed_at: int | None = None,
    ) -> TurnRun:
        ids = self._exact_ids(message_ids)
        completed = processed_at if processed_at is not None else _now_ms()
        db = await self._ensure_db()
        placeholders = ",".join("?" for _ in ids)
        await db.execute("BEGIN IMMEDIATE")
        try:
            expected = await self._turn_message_ids(db, turn_id)
            if len(expected) != len(ids) or set(expected) != set(ids):
                raise LifecycleConflict("completion IDs do not exactly match the turn")
            async with db.execute(
                f"""SELECT envelope_id FROM messages
                    WHERE envelope_id IN ({placeholders})
                      AND processing_state = ? AND processing_turn_id = ?""",
                (*ids, ProcessingState.IN_TURN.value, turn_id),
            ) as cursor:
                rows = await cursor.fetchall()
            if len(rows) != len(ids):
                raise LifecycleConflict("every exact message ID must be in this turn")
            cursor = await db.execute(
                f"""UPDATE messages SET processing_state = ?, processed_at = ?
                    WHERE envelope_id IN ({placeholders})
                      AND processing_state = ? AND processing_turn_id = ?""",
                (
                    ProcessingState.PROCESSED.value,
                    completed,
                    *ids,
                    ProcessingState.IN_TURN.value,
                    turn_id,
                ),
            )
            if cursor.rowcount != len(ids):
                raise LifecycleConflict("completion changed fewer rows than requested")
            cursor = await db.execute(
                "UPDATE turn_runs SET state = ?, completed_at = ? "
                "WHERE turn_id = ? AND state = ?",
                (
                    ProcessingState.PROCESSED.value,
                    completed,
                    turn_id,
                    ProcessingState.IN_TURN.value,
                ),
            )
            if cursor.rowcount != 1:
                raise LifecycleConflict("turn is not active")
            await self._refresh_notice_state(db)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        run = await self._get_turn_run_unlocked(db, turn_id)
        assert run is not None
        return run

    async def requeue_messages(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
    ) -> TurnRun:
        async with self._inbox_lock:
            return await self._requeue_messages_unlocked(
                message_ids, turn_id=turn_id
            )

    async def _requeue_messages_unlocked(
        self,
        message_ids: Iterable[str],
        *,
        turn_id: str,
    ) -> TurnRun:
        ids = self._exact_ids(message_ids)
        db = await self._ensure_db()
        placeholders = ",".join("?" for _ in ids)
        await db.execute("BEGIN IMMEDIATE")
        try:
            expected = await self._turn_message_ids(db, turn_id)
            if len(expected) != len(ids) or set(expected) != set(ids):
                raise LifecycleConflict("requeue IDs do not exactly match the turn")
            cursor = await db.execute(
                f"""UPDATE messages
                    SET processing_state = ?, processing_turn_id = NULL,
                        model_visible_at = NULL, processed_at = NULL
                    WHERE envelope_id IN ({placeholders})
                      AND processing_state = ? AND processing_turn_id = ?""",
                (
                    ProcessingState.PENDING.value,
                    *ids,
                    ProcessingState.IN_TURN.value,
                    turn_id,
                ),
            )
            if cursor.rowcount != len(ids):
                raise LifecycleConflict("every exact message ID must be in this turn")
            cursor = await db.execute(
                "UPDATE turn_runs SET state = ?, completed_at = ? "
                "WHERE turn_id = ? AND state = ?",
                ("requeued", _now_ms(), turn_id, ProcessingState.IN_TURN.value),
            )
            if cursor.rowcount != 1:
                raise LifecycleConflict("turn is not active")
            await self._refresh_notice_state(db)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        run = await self._get_turn_run_unlocked(db, turn_id)
        assert run is not None
        return run

    async def _turn_message_ids(
        self, db: aiosqlite.Connection, turn_id: str
    ) -> tuple[str, ...]:
        async with db.execute(
            "SELECT envelope_id FROM turn_run_messages WHERE turn_id = ? "
            "ORDER BY ordinal",
            (turn_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return tuple(row["envelope_id"] for row in rows)

    async def get_turn_run(self, turn_id: str) -> TurnRun | None:
        async with self._inbox_lock:
            db = await self._ensure_db()
            return await self._get_turn_run_unlocked(db, turn_id)

    async def get_active_turn_runs(self) -> tuple[TurnRun, ...]:
        """Return durable Turns that a restarted provider process cannot own."""
        async with self._inbox_lock:
            db = await self._ensure_db()
            async with db.execute(
                "SELECT turn_id FROM turn_runs WHERE state = ? "
                "ORDER BY started_at, turn_id",
                (ProcessingState.IN_TURN.value,),
            ) as cursor:
                rows = await cursor.fetchall()
            active: list[TurnRun] = []
            for row in rows:
                run = await self._get_turn_run_unlocked(db, row["turn_id"])
                if run is not None:
                    active.append(run)
            return tuple(active)

    async def _get_turn_run_unlocked(
        self, db: aiosqlite.Connection, turn_id: str
    ) -> TurnRun | None:
        async with db.execute(
            "SELECT * FROM turn_runs WHERE turn_id = ?", (turn_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        ids = await self._turn_message_ids(db, turn_id)
        return TurnRun(
            row["turn_id"],
            row["provider_session_id"],
            row["state"],
            ids,
            int(row["started_at"]),
            int(row["completed_at"]) if row["completed_at"] is not None else None,
        )

    async def get_in_turn_messages(
        self, turn_id: str, provider_session_id: str | None
    ) -> tuple[StoredMessage, ...]:
        # A stateless provider has no durable identity to authenticate recovery.
        if provider_session_id is None:
            return ()
        async with self._inbox_lock:
            db = await self._ensure_db()
            run = await self._get_turn_run_unlocked(db, turn_id)
            if (
                run is None
                or run.provider_session_id != provider_session_id
                or run.state != ProcessingState.IN_TURN.value
            ):
                return ()
            async with db.execute(
                """SELECT m.* FROM turn_run_messages trm
                   JOIN messages m ON m.envelope_id = trm.envelope_id
                   WHERE trm.turn_id = ? AND m.processing_state = ?
                     AND m.processing_turn_id = ?
                   ORDER BY trm.ordinal""",
                (turn_id, ProcessingState.IN_TURN.value, turn_id),
            ) as cursor:
                rows = await cursor.fetchall()
            return tuple(self._row_to_msg(row) for row in rows)

    async def get_context_baseline(
        self, space_id: str, channel_id: str
    ) -> int | None:
        async with self._inbox_lock:
            db = await self._ensure_db()
            async with db.execute(
                "SELECT context_baseline_seq FROM channel_context_state "
                "WHERE space_id = ? AND channel_id = ?",
                (space_id, channel_id),
            ) as cursor:
                row = await cursor.fetchone()
            return int(row[0]) if row is not None else None

    async def set_context_baseline(
        self,
        space_id: str,
        channel_id: str,
        context_baseline_seq: int,
        *,
        updated_at: int | None = None,
    ) -> None:
        if context_baseline_seq < 0:
            raise ValueError("context baseline must be non-negative")
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                """INSERT INTO channel_context_state
                   (space_id, channel_id, context_baseline_seq, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(space_id, channel_id) DO UPDATE SET
                     context_baseline_seq = excluded.context_baseline_seq,
                     updated_at = excluded.updated_at""",
                (
                    space_id,
                    channel_id,
                    context_baseline_seq,
                    updated_at if updated_at is not None else _now_ms(),
                ),
            )
            await db.commit()

    async def get_model_visible_through_seq(
        self, turn_id: str, space_id: str, channel_id: str,
        *, candidate_seq: int | None = None,
    ) -> int | None:
        """Return the highest *safe* model-visible channel prefix.

        Server sequence numbers are global, so numeric adjacency is not a
        channel invariant.  A later row is nevertheless not safe to expose
        as a freshness boundary while an earlier locally-known channel row is
        still pending or belongs to another unfinished turn.
        """
        async with self._inbox_lock:
            db = await self._ensure_db()
            # The persisted context baseline is a separate, already trusted
            # history boundary.  It must not be re-proven by (or blocked on)
            # old locally retained Inbox rows.
            async with db.execute(
                "SELECT context_baseline_seq FROM channel_context_state "
                "WHERE space_id = ? AND channel_id = ?",
                (space_id, channel_id),
            ) as cursor:
                baseline_row = await cursor.fetchone()
            baseline = (
                int(baseline_row["context_baseline_seq"])
                if baseline_row is not None else None
            )
            async with db.execute(
                """SELECT server_seq, processing_turn_id, processing_state,
                          receipt_disposition, model_visible_at
                   FROM messages
                   WHERE space_id = ? AND channel_id = ?
                     AND envelope_kind != 'dm' AND server_seq IS NOT NULL
                   ORDER BY server_seq ASC""",
                (
                    space_id,
                    channel_id,
                ),
            ) as cursor:
                rows = await cursor.fetchall()
            safe: int | None = None
            for row in rows:
                sequence = int(row["server_seq"])
                if baseline is not None and sequence <= baseline:
                    continue
                # A history candidate proves at most itself.  In particular,
                # a later active-turn row cannot turn that read into an
                # unbounded freshness advance.
                if candidate_seq is not None and sequence > candidate_seq:
                    break
                state = row["processing_state"]
                # A history tool's just-admitted watermark is model-visible
                # even though it is not moved into the Inbox turn.  It is
                # still bounded by every earlier locally-known blocker.
                is_candidate = candidate_seq is not None and sequence == candidate_seq
                is_visible = row["model_visible_at"] is not None or is_candidate
                is_this_turn = row["processing_turn_id"] == turn_id
                terminal = state == ProcessingState.PROCESSED.value
                terminal_receipt = (
                    row["receipt_disposition"] == ReceiptDisposition.TERMINAL.value
                )
                admissible = terminal_receipt or is_candidate or (
                    is_visible and (terminal or (is_this_turn and state == ProcessingState.IN_TURN.value))
                )
                if not admissible:
                    # This is a known same-channel blocker; do not leapfrog it.
                    break
                safe = sequence
            return safe

    async def has_known_channel_rows_above_baseline(
        self, space_id: str, channel_id: str,
    ) -> bool:
        """Whether a candidate advance has local channel evidence to respect."""
        async with self._inbox_lock:
            db = await self._ensure_db()
            async with db.execute(
                """SELECT 1 FROM messages
                   WHERE space_id = ? AND channel_id = ?
                     AND envelope_kind != 'dm' AND server_seq IS NOT NULL
                     AND server_seq > COALESCE((
                       SELECT context_baseline_seq FROM channel_context_state
                       WHERE space_id = ? AND channel_id = ?
                     ), -1)
                   LIMIT 1""",
                (space_id, channel_id, space_id, channel_id),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def channel_exists(self, channel_id: str) -> bool:
        """True iff the store has ever recorded a message in
        ``channel_id``. Used by the data service to return 404 when
        the caller asks for history on a channel the agent has never
        seen — distinguishes "unknown channel" from "known channel,
        empty window after filters" (200 + empty list).
        """
        if not channel_id:
            return False
        db = await self._ensure_db()
        async with db.execute(
            "SELECT 1 FROM messages WHERE channel_id = ? LIMIT 1",
            (channel_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def has_message(self, envelope_id: str) -> bool:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT 1 FROM messages WHERE envelope_id = ?", (envelope_id,)
        ) as cursor:
            return await cursor.fetchone() is not None

    async def lookup_channel_space(self, channel_id: str) -> str | None:
        """Return the ``space_id`` known for ``channel_id``, or
        ``None`` when neither the explicit map nor any prior message
        gives one. Used by the MCP subprocess (which can't read the
        daemon's in-memory ``_channel_space`` map) as a cross-space
        fallback.

        Two-source lookup, in order:

        1. ``channel_space_map`` — explicit mappings recorded by
           out-of-band discovery (``_find_public_general_channel`` and
           friends). Lets send_message resolve a channel BEFORE the
           first inbound message lands on it — the case the intro
           nudge needs.
        2. ``messages`` — last ``space_id`` seen on an envelope in
           that channel. Steady-state fallback that doesn't need
           explicit bookkeeping; works automatically once any message
           arrives.
        """
        if not channel_id:
            return None
        db = await self._ensure_db()
        async with db.execute(
            "SELECT space_id FROM channel_space_map WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None and row[0]:
            return row[0]
        async with db.execute(
            "SELECT space_id FROM messages WHERE channel_id = ? "
            "AND space_id IS NOT NULL AND space_id != '' "
            "ORDER BY sent_at DESC LIMIT 1",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return row[0] if row[0] else None

    async def mark_channel_space(self, channel_id: str, space_id: str) -> None:
        """Record an explicit channel→space mapping. Called by
        out-of-band channel-discovery paths (``_find_public_general_channel``)
        so ``lookup_channel_space`` can resolve the channel before
        the first inbound message lands."""
        if not channel_id or not space_id:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                """INSERT INTO channel_space_map (channel_id, space_id, learned_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(channel_id) DO UPDATE SET
                     space_id = excluded.space_id,
                     learned_at = excluded.learned_at""",
                (channel_id, space_id, _now_ms()),
            )
            await db.commit()

    async def unmark_channel_space(self, channel_id: str) -> None:
        """Drop a single channel→space mapping. Called by the per-
        channel eviction path (``_on_kicked_from_channel`` /
        ``_on_left_channel`` in puffo_core_client) so subsequent
        MCP-tool resolution doesn't keep routing into a channel
        we're no longer in. Idempotent — non-existent rows are
        silently skipped."""
        if not channel_id:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                "DELETE FROM channel_space_map WHERE channel_id = ?",
                (channel_id,),
            )
            await db.commit()

    async def unmark_channel_space_for_space(self, space_id: str) -> None:
        """Per-space companion to ``unmark_channel_space`` — drops
        every channel→space mapping whose space matches. Called by
        the space-level eviction path (``_on_left_space`` /
        ``_on_kicked_from_space``) when the agent has been removed
        from a whole space, so MCP-tool resolution for any of that
        space's channels fast-fails locally instead of round-tripping
        the server for a guaranteed-403."""
        if not space_id:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                "DELETE FROM channel_space_map WHERE space_id = ?",
                (space_id,),
            )
            await db.commit()

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 50,
        before: int | None = None,
    ) -> list[StoredMessage]:
        db = await self._ensure_db()
        if before is not None:
            sql = """SELECT * FROM messages
                     WHERE channel_id = ? AND sent_at < ?
                     ORDER BY sent_at DESC, envelope_id DESC LIMIT ?"""
            params: tuple = (channel_id, before, limit)
        else:
            sql = """SELECT * FROM messages
                     WHERE channel_id = ?
                     ORDER BY sent_at DESC, envelope_id DESC LIMIT ?"""
            params = (channel_id, limit)

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_msg(r) for r in reversed(rows)]

    async def get_held_reconsideration_rows(
        self,
        *,
        space_id: str,
        channel_id: str,
        after_seq: int,
        through_seq: int,
        limit: int = 50,
    ) -> list[StoredMessage]:
        """Return the bounded locally-decrypted channel interval for a held send.

        The caller still proves the terminal envelope separately.  Keeping this
        query in the store makes it impossible for the server response to be
        mistaken for plaintext context.
        """
        db = await self._ensure_db()
        async with db.execute(
            """SELECT * FROM messages
               WHERE space_id = ? AND channel_id = ?
                 AND server_seq IS NOT NULL AND server_seq > ? AND server_seq <= ?
               ORDER BY server_seq ASC LIMIT ?""",
            (space_id, channel_id, after_seq, through_seq, max(1, min(limit, 200))),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_msg(row) for row in rows]

    async def get_dm_history(
        self,
        peer_slug: str,
        limit: int = 50,
        before: int | None = None,
    ) -> list[StoredMessage]:
        db = await self._ensure_db()
        if before is not None:
            sql = """SELECT * FROM messages
                     WHERE envelope_kind = 'dm'
                       AND (sender_slug = ? OR recipient_slug = ?)
                       AND sent_at < ?
                     ORDER BY sent_at DESC, envelope_id DESC LIMIT ?"""
            params: tuple = (peer_slug, peer_slug, before, limit)
        else:
            sql = """SELECT * FROM messages
                     WHERE envelope_kind = 'dm'
                       AND (sender_slug = ? OR recipient_slug = ?)
                     ORDER BY sent_at DESC, envelope_id DESC LIMIT ?"""
            params = (peer_slug, peer_slug, limit)

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_msg(r) for r in reversed(rows)]

    async def get_message_by_envelope(
        self, envelope_id: str,
    ) -> Optional["StoredMessage"]:
        """Single-row lookup by envelope_id. Returns ``None`` when
        the agent never saw the message. Lets this class act as an
        in-process drop-in for ``mcp.data_client.DataClient`` in hosts
        that skip the loopback HTTP round-trip.
        """
        if not envelope_id:
            return None
        async with self._inbox_lock:
            db = await self._ensure_db()
            return await self._get_message_by_envelope_unlocked(db, envelope_id)

    async def _get_message_by_envelope_unlocked(
        self, db: aiosqlite.Connection, envelope_id: str
    ) -> Optional["StoredMessage"]:
        async with db.execute(
            "SELECT * FROM messages WHERE envelope_id = ?", (envelope_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_msg(row)

    async def get_thread_batch(
        self,
        root_id: str,
        since_sent_at: int,
    ) -> list[StoredMessage]:
        """Root message + every reply in its thread with
        ``sent_at > since_sent_at``, ordered ascending. The root row
        itself has ``thread_root_id IS NULL`` and matches via the
        ``envelope_id = root_id`` arm of the OR.
        """
        if not root_id:
            return []
        db = await self._ensure_db()
        async with db.execute(
            """SELECT * FROM messages
               WHERE (envelope_id = ? OR thread_root_id = ?)
                 AND sent_at > ?
               ORDER BY sent_at ASC, envelope_id ASC""",
            (root_id, root_id, since_sent_at),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_msg(r) for r in rows]

    async def _resolve_since_sent_at(self, since_envelope_id: str | None) -> tuple[int, str] | None:
        """Look up the ``sent_at`` of a reference envelope. Used by
        ``get_channel_roots`` / ``get_thread_messages`` to translate
        a ``since=<envelope_id>`` filter into an exclusive sent_at
        lower bound. Returns ``None`` when the envelope isn't in the
        store (caller treats that as "no since filter")."""
        if not since_envelope_id:
            return None
        db = await self._ensure_db()
        async with db.execute(
            "SELECT sent_at, envelope_id FROM messages WHERE envelope_id = ?",
            (since_envelope_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return (int(row[0]), str(row[1])) if row else None

    async def get_channel_roots(
        self,
        channel_id: str,
        limit: int = 20,
        since_envelope_id: str | None = None,
        before_envelope_id: str | None = None,
        before_ts: int | None = None,
        after_ts: int | None = None,
    ) -> list[ChannelRoot]:
        """Recent root posts in ``channel_id`` (``thread_root_id``
        IS NULL) with the count of replies that point at each.

        Replies in any thread are excluded from the result — the
        agent gets one bullet per conversation head and can drill
        into the threads it actually cares about via
        ``get_thread_messages``. Filtering options:

        - ``since_envelope_id`` — return roots whose ``sent_at`` is
          strictly greater than that envelope's. Convenient when
          the agent already knows the latest root it processed.
        - ``before_envelope_id`` — return roots strictly before that
          composite ``(sent_at, envelope_id)`` cursor. This permits stable
          reverse pagination when several envelopes share a timestamp.
        - ``after_ts`` (ms-epoch) — exclusive lower bound on
          ``sent_at``. Combined with ``since`` we take the larger.
        - ``before_ts`` (ms-epoch) — exclusive upper bound on
          ``sent_at``.

        Returned newest-first up to ``limit``. Raises
        ``DataNotFound`` if the channel has never had any message
        stored — so the MCP layer can distinguish "unknown channel"
        from "known but empty window".
        """
        if not await self.channel_exists(channel_id):
            raise DataNotFound(f"channel not found: {channel_id}")
        db = await self._ensure_db()
        since_resolved = await self._resolve_since_sent_at(since_envelope_id)
        before_resolved = await self._resolve_since_sent_at(before_envelope_id)

        # ``reply_count`` is a correlated subquery on the same
        # ``messages`` table; the WAL writer is the only producer, so
        # the count is point-in-time consistent.
        clauses = ["m.channel_id = ?", "m.thread_root_id IS NULL"]
        params: list = [channel_id]
        if since_resolved is not None:
            clauses.append("(m.sent_at > ? OR (m.sent_at = ? AND m.envelope_id > ?))")
            params.extend([since_resolved[0], since_resolved[0], since_resolved[1]])
        if before_resolved is not None:
            clauses.append("(m.sent_at < ? OR (m.sent_at = ? AND m.envelope_id < ?))")
            params.extend([before_resolved[0], before_resolved[0], before_resolved[1]])
        if after_ts is not None:
            clauses.append("m.sent_at > ?")
            params.append(int(after_ts))
        if before_ts is not None:
            clauses.append("m.sent_at < ?")
            params.append(int(before_ts))
        where = " AND ".join(clauses)
        ascending_cursor = since_resolved is not None and before_resolved is None
        order = "m.sent_at ASC, m.envelope_id ASC" if ascending_cursor else "m.sent_at DESC, m.envelope_id DESC"
        sql = (
            "SELECT m.*, "
            "(SELECT COUNT(*) FROM messages r "
            " WHERE r.thread_root_id = m.envelope_id) AS reply_count "
            f"FROM messages m WHERE {where} ORDER BY {order} LIMIT ?"
        )
        params.append(max(1, min(int(limit), 200)))

        async with db.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        # Cursor pages already select oldest-first; uncursorred history is
        # selected newest-first then reversed for the public oldest-first view.
        return [
            ChannelRoot(
                message=self._row_to_msg(r),
                reply_count=int(r["reply_count"]),
            )
            for r in (rows if ascending_cursor else reversed(rows))
        ]

    async def get_thread_messages(
        self,
        root_id: str,
        limit: int = 50,
        since_envelope_id: str | None = None,
        before_envelope_id: str | None = None,
        before_ts: int | None = None,
        after_ts: int | None = None,
    ) -> list[StoredMessage]:
        """Messages belonging to a thread (the root itself plus
        every reply pointing at it), filtered the same way as
        ``get_channel_roots``. Newest-first selection, then
        reversed to oldest-first in the returned list — matches
        ``get_channel_history``'s shape so the MCP tool can format
        either the same way. Raises ``DataNotFound`` when no
        message with that envelope_id has been stored — same
        rationale as ``get_channel_roots``.
        """
        if not root_id:
            raise DataNotFound("thread root not found: (empty)")
        if not await self.has_message(root_id):
            raise DataNotFound(f"thread root not found: {root_id}")
        db = await self._ensure_db()
        since_resolved = await self._resolve_since_sent_at(since_envelope_id)
        before_resolved = await self._resolve_since_sent_at(before_envelope_id)

        clauses = ["(envelope_id = ? OR thread_root_id = ?)"]
        params: list = [root_id, root_id]
        if since_resolved is not None:
            clauses.append("(sent_at > ? OR (sent_at = ? AND envelope_id > ?))")
            params.extend([since_resolved[0], since_resolved[0], since_resolved[1]])
        if before_resolved is not None:
            clauses.append("(sent_at < ? OR (sent_at = ? AND envelope_id < ?))")
            params.extend([before_resolved[0], before_resolved[0], before_resolved[1]])
        if after_ts is not None:
            clauses.append("sent_at > ?")
            params.append(int(after_ts))
        if before_ts is not None:
            clauses.append("sent_at < ?")
            params.append(int(before_ts))
        where = " AND ".join(clauses)
        ascending_cursor = since_resolved is not None and before_resolved is None
        order = "sent_at ASC, envelope_id ASC" if ascending_cursor else "sent_at DESC, envelope_id DESC"
        sql = f"SELECT * FROM messages WHERE {where} ORDER BY {order} LIMIT ?"
        params.append(max(1, min(int(limit), 200)))

        async with db.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_msg(r) for r in (rows if ascending_cursor else reversed(rows))]

    async def get_last_processed_sent_at(self, root_id: str) -> int:
        """``sent_at`` of the last message in the most recently
        dispatched batch for this thread, or ``0`` if the agent has
        never processed it. Used at enqueue time to drop redelivered
        messages whose work has already been done.
        """
        if not root_id:
            return 0
        db = await self._ensure_db()
        async with db.execute(
            "SELECT last_processed_sent_at FROM thread_processing_state "
            "WHERE root_id = ?",
            (root_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def mark_thread_processed(
        self, root_id: str, sent_at: int,
    ) -> None:
        """Upsert the per-thread cursor. ``MAX(existing, new)`` so
        out-of-order writes (extremely unlikely but cheap to guard)
        never regress the cursor.
        """
        if not root_id:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                """INSERT INTO thread_processing_state (root_id, last_processed_sent_at)
                   VALUES (?, ?)
                   ON CONFLICT(root_id) DO UPDATE SET
                     last_processed_sent_at = MAX(
                       thread_processing_state.last_processed_sent_at,
                       excluded.last_processed_sent_at
                     )""",
                (root_id, sent_at),
            )
            await db.commit()

    async def has_channel_intro_been_prompted(self, channel_id: str) -> bool:
        """True iff the agent already had a self-introduction prompted
        for ``channel_id``. Used by ``_accept_invite`` to gate the
        synthetic system-message enqueue so a restart-time replay of
        the same pending invite doesn't fire a second intro."""
        if not channel_id:
            return False
        db = await self._ensure_db()
        async with db.execute(
            "SELECT 1 FROM channel_intro_prompted WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def get_dm_notice(self, sender_slug: str) -> int | None:
        """Epoch-ms of the last FYI notice for ``sender_slug``, or None."""
        if not sender_slug:
            return None
        db = await self._ensure_db()
        async with db.execute(
            "SELECT last_notified_at FROM dm_notices WHERE sender_slug = ?",
            (sender_slug,),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else None

    async def has_dm_from(self, sender_slug: str) -> bool:
        """Any stored inbound DM from ``sender_slug``."""
        if not sender_slug:
            return False
        db = await self._ensure_db()
        async with db.execute(
            "SELECT 1 FROM messages WHERE envelope_kind = 'dm' "
            "AND sender_slug = ? LIMIT 1",
            (sender_slug,),
        ) as cur:
            return await cur.fetchone() is not None

    async def set_dm_notice(self, sender_slug: str, notified_at: int) -> None:
        if not sender_slug:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                """INSERT INTO dm_notices (sender_slug, last_notified_at)
                   VALUES (?, ?)
                   ON CONFLICT(sender_slug) DO UPDATE SET
                     last_notified_at = excluded.last_notified_at""",
                (sender_slug, notified_at),
            )
            await db.commit()

    async def mark_channel_intro_prompted(self, channel_id: str) -> None:
        """Record that an intro nudge has been enqueued for
        ``channel_id``. Idempotent."""
        if not channel_id:
            return
        async with self._inbox_lock:
            db = await self._ensure_db()
            await db.execute(
                """INSERT INTO channel_intro_prompted (channel_id, prompted_at)
                   VALUES (?, ?)
                   ON CONFLICT(channel_id) DO NOTHING""",
                (channel_id, _now_ms()),
            )
            await db.commit()

    async def cleanup(self, retention_days: int = 90) -> int:
        async with self._inbox_lock:
            db = await self._ensure_db()
            cutoff = _now_ms() - retention_days * 86_400_000
            async with db.execute(
                """DELETE FROM messages
                   WHERE received_at < ?
                     AND (processing_state IS NULL OR processing_state = 'processed')
                     AND (
                       receipt_disposition IS NULL
                       OR receipt_disposition != 'foreign_dm_gated'
                     )""",
                (cutoff,),
            ) as cursor:
                count = cursor.rowcount
            await db.commit()
            return count

    def _row_to_msg(self, row: aiosqlite.Row) -> StoredMessage:
        content_raw = row["content"]
        try:
            content = json.loads(content_raw)
        except (json.JSONDecodeError, ValueError):
            content = content_raw

        return StoredMessage(
            envelope_id=row["envelope_id"],
            envelope_kind=row["envelope_kind"],
            sender_slug=row["sender_slug"],
            channel_id=row["channel_id"],
            space_id=row["space_id"],
            recipient_slug=row["recipient_slug"],
            content_type=row["content_type"],
            content=content,
            sent_at=row["sent_at"],
            received_at=row["received_at"],
            thread_root_id=row["thread_root_id"],
            reply_to_id=row["reply_to_id"],
            is_encrypted=bool(row["is_encrypted"]),
            server_seq=row["server_seq"],
            receipt_disposition=(
                ReceiptDisposition(row["receipt_disposition"])
                if row["receipt_disposition"] is not None
                else None
            ),
            receipt_reason=row["receipt_reason"],
            processing_state=(
                ProcessingState(row["processing_state"])
                if row["processing_state"] is not None
                else None
            ),
            processing_turn_id=row["processing_turn_id"],
            model_visible_at=row["model_visible_at"],
            processed_at=row["processed_at"],
            local_ordinal=row["local_ordinal"],
            after_server_seq=row["after_server_seq"],
        )
