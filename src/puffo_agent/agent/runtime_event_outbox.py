"""Bounded, restart-safe SQLite outbox for Runtime Events v1."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from ._logging import log_runtime_event
from .harness.driver import HarnessEvent, HarnessEventType
from .runtime_events import (
    DeltaCoalescer,
    LifecycleValidator,
    RuntimeEvent,
    RuntimeEventProjector,
)

APPEND_PATH = "/v2/agent-runtime/events:append"
# These responses describe the upload channel, not a specific event row.  In
# particular, a rolling Server deployment can briefly expose an authenticated
# Agent to a Server without the append endpoint.  Retain FIFO evidence until
# the operator repairs that boundary rather than treating the head as bad.
_DEGRADED_CHANNEL_HTTP_STATUSES = frozenset({401, 403, 404, 405})


def runtime_event_outbox_path(agent_state_dir: str | Path) -> Path:
    """Return the daemon-owned event DB beside the Agent message DB."""
    return Path(agent_state_dir) / "runtime_events.db"


class OutboxCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OutboxRow:
    sequence: int
    event_id: str
    event_type: str
    event_json: bytes
    retry_count: int

    @property
    def event(self) -> dict[str, Any]:
        return json.loads(self.event_json)


@dataclass(frozen=True, slots=True)
class UploadResult:
    state: str
    count: int = 0
    error_code: str = ""


class RuntimeEventOutbox:
    """One per Agent. All writes use immediate SQLite transactions."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = 10_000,
        max_bytes: int = 16 * 1024 * 1024,
        terminal_reserve_rows: int = 1,
        terminal_reserve_bytes: int = 2048,
        logger=None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.terminal_reserve_rows = terminal_reserve_rows
        self.terminal_reserve_bytes = terminal_reserve_bytes
        self.logger = logger
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id TEXT NOT NULL UNIQUE,
              event_type TEXT NOT NULL,
              event_json BLOB NOT NULL,
              byte_count INTEGER NOT NULL,
              retry_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS state (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        self._db.commit()
        self._lock = asyncio.Lock()

    def close(self) -> None:
        self._db.close()

    async def aclose(self) -> None:
        self.close()

    @staticmethod
    def canonical_bytes(event: RuntimeEvent | dict[str, Any]) -> bytes:
        value = event.as_dict() if isinstance(event, RuntimeEvent) else event
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")

    def usage(self) -> tuple[int, int]:
        row = self._db.execute(
            "SELECT COUNT(*) AS rows, COALESCE(SUM(byte_count), 0) AS bytes "
            "FROM events"
        ).fetchone()
        return int(row["rows"]), int(row["bytes"])

    def can_start_turn(self, *, estimated_start_bytes: int = 0) -> bool:
        rows, bytes_ = self.usage()
        return (
            rows + 2 + self.terminal_reserve_rows <= self.max_rows
            and bytes_ + estimated_start_bytes
            + self.terminal_reserve_bytes <= self.max_bytes
        )

    def require_turn_capacity(self, *, estimated_start_bytes: int = 0) -> None:
        if not self.can_start_turn(estimated_start_bytes=estimated_start_bytes):
            self._log("runtime.capacity", state="blocked", error_code="capacity")
            raise OutboxCapacityError("runtime event outbox is at capacity")

    async def enqueue(
        self, event: RuntimeEvent, *, terminal: bool | None = None
    ) -> int:
        encoded = self.canonical_bytes(event)
        is_terminal = (
            event.type == "turn.finished" if terminal is None else terminal
        )
        async with self._lock:
            rows, bytes_ = self.usage()
            row_limit = self.max_rows
            byte_limit = self.max_bytes
            if not is_terminal:
                row_limit -= self.terminal_reserve_rows
                byte_limit -= self.terminal_reserve_bytes
            if rows + 1 > row_limit or bytes_ + len(encoded) > byte_limit:
                self._log(
                    "runtime.capacity", state="blocked",
                    event_id=event.event_id, event_type=event.type,
                    error_code="capacity",
                )
                raise OutboxCapacityError(
                    "enqueue would exceed runtime event outbox capacity"
                )
            try:
                cursor = self._db.execute(
                    "INSERT INTO events"
                    "(event_id,event_type,event_json,byte_count) VALUES(?,?,?,?)",
                    (event.event_id, event.type, encoded, len(encoded)),
                )
                self._db.commit()
            except sqlite3.IntegrityError:
                row = self._db.execute(
                    "SELECT sequence,event_json FROM events WHERE event_id=?",
                    (event.event_id,),
                ).fetchone()
                if row is None or bytes(row["event_json"]) != encoded:
                    raise ValueError("event_id reused with different content")
                return int(row["sequence"])
            sequence = int(cursor.lastrowid)
            self._log(
                "runtime.enqueued", event_id=event.event_id,
                event_type=event.type, outbox_sequence=sequence,
                agent_id=event.agent_id, session_ref=event.session_ref,
                turn_ref=event.turn_ref,
            )
            return sequence

    def prefix(
        self, *, max_rows: int = 50, max_bytes: int = 256 * 1024
    ) -> list[OutboxRow]:
        result: list[OutboxRow] = []
        # Account for the complete JSON envelope, including wrapper and commas.
        total = len(b'{"events":[]}')
        for row in self._db.execute(
            "SELECT sequence,event_id,event_type,event_json,retry_count "
            "FROM events ORDER BY sequence LIMIT ?", (max_rows,)
        ):
            encoded = bytes(row["event_json"])
            added = len(encoded) + (1 if result else 0)
            if result and total + added > max_bytes:
                break
            if not result and total + added > max_bytes:
                # Return the oversized head alone so the uploader can
                # deterministically quarantine it and unblock later rows.
                result.append(OutboxRow(int(row["sequence"]), str(row["event_id"]), str(row["event_type"]), encoded, int(row["retry_count"])))
                break
            result.append(OutboxRow(
                int(row["sequence"]), str(row["event_id"]),
                str(row["event_type"]), encoded, int(row["retry_count"]),
            ))
            total += added
        return result

    def increment_retries(self, sequences: Iterable[int]) -> None:
        values = tuple(int(value) for value in sequences)
        if not values:
            return
        placeholders = ",".join("?" for _ in values)
        self._db.execute(
            f"UPDATE events SET retry_count=retry_count+1 "
            f"WHERE sequence IN ({placeholders})", values,
        )
        self._db.commit()

    def acknowledge(self, rows: list[OutboxRow], accepted_ids: list[str]) -> None:
        if accepted_ids != [row.event_id for row in rows]:
            raise ValueError("append response does not exactly match batch")
        if not rows:
            return
        first, last = rows[0].sequence, rows[-1].sequence
        actual = [
            str(row["event_id"]) for row in self._db.execute(
                "SELECT event_id FROM events WHERE sequence BETWEEN ? AND ? "
                "ORDER BY sequence", (first, last)
            )
        ]
        if actual != accepted_ids:
            raise ValueError("outbox changed while acknowledging batch")
        self._db.execute(
            "DELETE FROM events WHERE sequence BETWEEN ? AND ?", (first, last)
        )
        self._db.commit()

    def discard(self, row: OutboxRow, *, error_code: str) -> None:
        """Quarantine a permanently rejected head by removing it from FIFO."""
        self._db.execute("DELETE FROM events WHERE sequence = ?", (row.sequence,))
        self._db.commit()
        self._log("runtime.discarded", outbox_sequence=row.sequence,
                  event_id=row.event_id, error_code=error_code)

    def set_active_turn(
        self, turn_ref: str | None, *, session_ref: str = "",
        native_session_id: str = "",
    ) -> None:
        values = {
            "active_turn_ref": turn_ref or "",
            "session_ref": session_ref,
            "native_session_id": native_session_id,
        }
        with self._db:
            for key, value in values.items():
                self._db.execute(
                    "INSERT INTO state(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )

    def state(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self._db.execute("SELECT key,value FROM state")
        }

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            log_runtime_event(self.logger, event, **fields)


class RuntimeEventUploader:
    """Serial uploader which lets a permanent bad head unblock later rows."""

    def __init__(
        self,
        outbox: RuntimeEventOutbox,
        transport: Callable[..., Awaitable[Any]],
        *,
        batch_rows: int = 50,
        batch_bytes: int = 256 * 1024,
    ):
        self.outbox = outbox
        self.transport = transport
        self.batch_rows = batch_rows
        self.batch_bytes = batch_bytes
        self.degraded_error = ""
        self._lock = asyncio.Lock()
        self._isolate_head = False

    async def upload_once(self) -> UploadResult:
        async with self._lock:
            rows = self.outbox.prefix(
                max_rows=1 if self._isolate_head else self.batch_rows,
                max_bytes=self.batch_bytes
            )
            if not rows:
                return UploadResult("idle")
            body = b'{"events":[' + b",".join(
                row.event_json for row in rows
            ) + b"]}"
            # An oversized singleton is quarantined locally; it must never be
            # sent as an over-limit HTTP request.
            if len(body) > self.batch_bytes:
                self.degraded_error = "body_too_large"
                self.outbox.discard(rows[0], error_code=self.degraded_error)
                return UploadResult("discarded", error_code=self.degraded_error)
            self.outbox._log(
                "runtime.batch_attempt",
                first_sequence=rows[0].sequence,
                last_sequence=rows[-1].sequence,
                retry_count=max(row.retry_count for row in rows),
                event_count=len(rows),
            )
            try:
                response = self.transport(APPEND_PATH, body)
                if inspect.isawaitable(response):
                    response = await response
                status, payload = await _decode_response(response)
            except Exception:
                self.outbox.increment_retries(row.sequence for row in rows)
                self.outbox._log(
                    "runtime.retry", outbox_sequence=rows[0].sequence,
                    retry_count=rows[0].retry_count + 1,
                    error_code="transport",
                )
                return UploadResult("retry", error_code="transport")
            if status == 429 or status >= 500:
                self.outbox.increment_retries(row.sequence for row in rows)
                self.outbox._log(
                    "runtime.retry", outbox_sequence=rows[0].sequence,
                    retry_count=rows[0].retry_count + 1,
                    error_code=f"http_{status}",
                )
                return UploadResult("retry", error_code=f"http_{status}")
            if status in _DEGRADED_CHANNEL_HTTP_STATUSES:
                self.degraded_error = f"http_{status}"
                self.outbox._log(
                    "runtime.retry", outbox_sequence=rows[0].sequence,
                    retry_count=rows[0].retry_count,
                    state="degraded", error_code=self.degraded_error,
                )
                return UploadResult("degraded", error_code=self.degraded_error)
            if status < 200 or status >= 300:
                # A batch rejection does not identify its bad member.  Retry
                # its head alone before discarding anything, preserving FIFO
                # evidence and letting valid followers progress.
                if len(rows) > 1:
                    self.degraded_error = f"http_{status}"
                    self._isolate_head = True
                    return UploadResult("retry", error_code=self.degraded_error)
                self.degraded_error = f"http_{status}"
                self.outbox._log(
                    "runtime.retry", outbox_sequence=rows[0].sequence,
                    retry_count=rows[0].retry_count,
                    state="degraded", error_code=self.degraded_error,
                )
                self.outbox.discard(rows[0], error_code=self.degraded_error)
                self._isolate_head = False
                return UploadResult("discarded", error_code=self.degraded_error)
            try:
                accepted = payload["accepted"]
                accepted_ids = [str(item["event_id"]) for item in accepted]
                if len(accepted_ids) != len(rows):
                    raise ValueError("partial acknowledgement")
                self.outbox.acknowledge(rows, accepted_ids)
            except (KeyError, TypeError, ValueError):
                if len(rows) > 1:
                    self.degraded_error = "malformed_response"
                    self._isolate_head = True
                    return UploadResult("retry", error_code=self.degraded_error)
                self.degraded_error = "malformed_response"
                self.outbox._log(
                    "runtime.retry", outbox_sequence=rows[0].sequence,
                    retry_count=rows[0].retry_count,
                    state="degraded", error_code=self.degraded_error,
                )
                self.outbox.discard(rows[0], error_code=self.degraded_error)
                self._isolate_head = False
                return UploadResult("discarded", error_code=self.degraded_error)
            self.outbox._log(
                "runtime.acknowledged", first_sequence=rows[0].sequence,
                last_sequence=rows[-1].sequence, event_count=len(rows),
            )
            self.degraded_error = ""
            self._isolate_head = False
            return UploadResult("uploaded", len(rows))


class RuntimeEventProjectingSink:
    """Durably project the Runtime Manager's normalized event stream.

    Token-sized deltas are coalesced before immutable public event IDs are
    allocated. The lifecycle validator runs on the exact rows that enter the
    outbox, never on provider-native timing or IDs.
    """

    def __init__(
        self,
        outbox: RuntimeEventOutbox,
        projector: RuntimeEventProjector,
        *,
        delta_bytes: int = 4096,
    ):
        self.outbox = outbox
        self.projector = projector
        self.validator = LifecycleValidator()
        self.delta_bytes = delta_bytes
        self._coalescers: dict[tuple[str, str], DeltaCoalescer] = {}

    async def __call__(self, event: HarnessEvent) -> None:
        kind = (
            event.type.value
            if isinstance(event.type, HarnessEventType)
            else str(event.type)
        )
        self.outbox._log(
            "runtime.normalized_event",
            agent_id=self.projector.agent_id,
            session_ref=self.projector.session_ref,
            turn_ref=str(event.turn_ref) if event.turn_ref is not None else "",
            event_type=kind,
        )
        if kind == "turn.assistant_delta" and event.turn_ref is not None:
            text = event.data.get("text")
            if not isinstance(text, str) or not text:
                return
            key = (
                str(event.turn_ref),
                str(event.data.get("block_id") or "result"),
            )
            coalescer = self._coalescers.setdefault(
                key, DeltaCoalescer(self.delta_bytes)
            )
            for chunk in coalescer.add(text):
                await self._project(replace(
                    event, data={**event.data, "text": chunk}
                ))
            return
        if kind == "turn.assistant_completed" and event.turn_ref is not None:
            key = (
                str(event.turn_ref),
                str(event.data.get("block_id") or "result"),
            )
            coalescer = self._coalescers.pop(key, None)
            if coalescer is not None:
                for chunk in coalescer.flush():
                    await self._project(replace(
                        event,
                        type=HarnessEventType.ASSISTANT_DELTA,
                        data={**event.data, "text": chunk},
                    ))
            await self._project(event)
            return
        if kind in {"turn.completed", "turn.abandoned"}:
            await self._flush_turn(event)
        await self._project(event)

    async def _flush_turn(self, terminal: HarnessEvent) -> None:
        turn = str(terminal.turn_ref) if terminal.turn_ref is not None else ""
        for key, coalescer in tuple(self._coalescers.items()):
            if key[0] != turn:
                continue
            for chunk in coalescer.flush():
                try:
                    await self._project(replace(
                        terminal,
                        type=HarnessEventType.ASSISTANT_DELTA,
                        data={"text": chunk, "block_id": key[1]},
                    ))
                except OutboxCapacityError:
                    break
            try:
                await self._project(replace(
                    terminal,
                    type=HarnessEventType.ASSISTANT_COMPLETED,
                    data={"block_id": key[1]},
                ))
            except OutboxCapacityError:
                pass
            self._coalescers.pop(key, None)

    async def _project(self, event: HarnessEvent) -> None:
        for projected in self.projector.project_all(event):
            self.outbox._log(
                "runtime.projected",
                agent_id=projected.agent_id,
                session_ref=projected.session_ref,
                turn_ref=projected.turn_ref,
                event_id=projected.event_id,
                event_type=projected.type,
            )
            self.validator.accept(projected)
            await self.outbox.enqueue(
                projected, terminal=projected.type == "turn.finished"
            )


async def _decode_response(response: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(response, tuple) and len(response) == 2:
        status, payload = response
    elif isinstance(response, dict) and "status" in response:
        status, payload = response["status"], response.get("body", {})
    else:
        status = response.status
        payload = response.json()
        if inspect.isawaitable(payload):
            payload = await payload
    if isinstance(payload, (bytes, bytearray, str)):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("append response body must be an object")
    return int(status), payload
