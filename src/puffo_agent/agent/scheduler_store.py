"""PUF-394: SQLite persistence for the daemon-layer per-agent scheduler.

One `scheduler.db` per agent dir (peer of `messages.db`), owned by the daemon
Worker. Survives session drop / daemon restart / worker respawn — the whole
point vs the session-local Claude Code SDK Cron* tools. Mirrors the aiosqlite
+ WAL idiom of `message_store.py`. All access is on the daemon event loop and
awaited serially (single writer); `busy_timeout` covers the rare overlap with
the co-located MCP/RPC path.

Timestamps are ms-epoch ints. `jobs` is the schedule; `runs` is history + the
no-overlap guard (an open row — `ended_at IS NULL` — means a fire is still in
flight).
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    cron_expr       TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    channel_id      TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    last_run_at     INTEGER,
    last_run_status TEXT,
    next_run_at     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER,
    status      TEXT,
    envelope_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id);
CREATE INDEX IF NOT EXISTS idx_runs_open ON runs(job_id) WHERE ended_at IS NULL;
"""

# Fields a caller may mutate via update_job (id/created_at are immutable).
_MUTABLE = frozenset(
    {
        "name",
        "cron_expr",
        "prompt",
        "channel_id",
        "enabled",
        "next_run_at",
        "last_run_at",
        "last_run_status",
    }
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class SchedulerStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(
            "PRAGMA journal_mode=WAL;"
            "PRAGMA synchronous=NORMAL;"
            "PRAGMA busy_timeout=5000;"
        )
        await self._db.executescript(_SCHEMA)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.open()
        assert self._db is not None
        return self._db

    # ── jobs CRUD ────────────────────────────────────────────────────────

    async def create_job(
        self,
        *,
        name: str,
        cron_expr: str,
        prompt: str,
        channel_id: str | None,
        next_run_at: int,
    ) -> dict:
        db = await self._ensure_db()
        now = _now_ms()
        job_id = _new_id("cronjob")
        await db.execute(
            "INSERT INTO jobs (id, name, cron_expr, prompt, channel_id, enabled, "
            "created_at, updated_at, last_run_at, last_run_status, next_run_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?, NULL, NULL, ?)",
            (job_id, name, cron_expr, prompt, channel_id, now, now, next_run_at),
        )
        await db.commit()
        job = await self.get_job(job_id)
        assert job is not None
        return job

    async def get_job(self, job_id: str) -> dict | None:
        db = await self._ensure_db()
        cur = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_jobs(self) -> list[dict]:
        db = await self._ensure_db()
        cur = await db.execute("SELECT * FROM jobs ORDER BY created_at")
        return [dict(r) for r in await cur.fetchall()]

    async def update_job(self, job_id: str, **fields) -> dict | None:
        sets = {k: v for k, v in fields.items() if k in _MUTABLE and v is not None}
        if not sets:
            return await self.get_job(job_id)
        sets["updated_at"] = _now_ms()
        db = await self._ensure_db()
        assignments = ", ".join(f"{k} = ?" for k in sets)
        cur = await db.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*sets.values(), job_id),
        )
        await db.commit()
        if cur.rowcount == 0:
            return None
        return await self.get_job(job_id)

    async def delete_job(self, job_id: str) -> bool:
        db = await self._ensure_db()
        cur = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
        return cur.rowcount > 0

    # ── engine helpers ───────────────────────────────────────────────────

    async def due_jobs(self, now_ms: int) -> list[dict]:
        """Enabled jobs whose next_run_at has arrived, oldest-due first."""
        db = await self._ensure_db()
        cur = await db.execute(
            "SELECT * FROM jobs WHERE enabled = 1 AND next_run_at <= ? "
            "ORDER BY next_run_at",
            (now_ms,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def has_open_run(self, job_id: str) -> bool:
        """True if a prior fire of this job is still in flight (no-overlap)."""
        db = await self._ensure_db()
        cur = await db.execute(
            "SELECT 1 FROM runs WHERE job_id = ? AND ended_at IS NULL LIMIT 1",
            (job_id,),
        )
        return (await cur.fetchone()) is not None

    async def job_for_open_envelope(self, envelope_id: str) -> str | None:
        """job_id of the still-open run for a fired envelope, if any."""
        db = await self._ensure_db()
        cur = await db.execute(
            "SELECT job_id FROM runs WHERE envelope_id = ? AND ended_at IS NULL",
            (envelope_id,),
        )
        row = await cur.fetchone()
        return row["job_id"] if row else None

    async def open_run(self, job_id: str, envelope_id: str, started_at: int) -> str:
        db = await self._ensure_db()
        run_id = _new_id("cronrun")
        await db.execute(
            "INSERT INTO runs (id, job_id, started_at, ended_at, status, envelope_id) "
            "VALUES (?, ?, ?, NULL, NULL, ?)",
            (run_id, job_id, started_at, envelope_id),
        )
        await db.commit()
        return run_id

    async def close_run(self, envelope_id: str, status: str, ended_at: int) -> bool:
        """Close the open run for a fired envelope (called on turn completion)."""
        db = await self._ensure_db()
        cur = await db.execute(
            "UPDATE runs SET ended_at = ?, status = ? "
            "WHERE envelope_id = ? AND ended_at IS NULL",
            (ended_at, status, envelope_id),
        )
        await db.commit()
        return cur.rowcount > 0

    async def record_fire(self, job_id: str, last_run_at: int, next_run_at: int) -> None:
        """After firing: stamp last_run_at + advance next_run_at."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE jobs SET last_run_at = ?, next_run_at = ?, updated_at = ? "
            "WHERE id = ?",
            (last_run_at, next_run_at, _now_ms(), job_id),
        )
        await db.commit()

    async def set_last_status(self, job_id: str, status: str) -> None:
        db = await self._ensure_db()
        await db.execute(
            "UPDATE jobs SET last_run_status = ?, updated_at = ? WHERE id = ?",
            (status, _now_ms(), job_id),
        )
        await db.commit()
