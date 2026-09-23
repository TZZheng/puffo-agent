"""Attach to a LingTai Agent that is already running, instead of starting one.

LingTai Agents are resident: the process and its ``.agent.lock`` belong to
LingTai, and Puffo owns only the connection. So this driver has no spawn code
at all. ``open`` connects to the Agent's local socket and ``close`` disconnects;
every path that would restart the runtime under ``AcpDriver`` (reopen, resume
fallback, rollover, reload) becomes a reconnect to the same process.

The admission channel is the one ``AcpDriver`` gives a child it spawns, handed
over differently. A spawned child inherits one end of an authority socketpair
at exec time; an attached Agent receives it over the connection instead, as
SCM_RIGHTS on the first frame. The authority is therefore scoped to this
connection: when the connection ends, the Agent has no Puffo authority left.

Handshake (agreed with LingTai Dev, Lingtai<>Puffo 219779):

- Puffo sends one newline-terminated JSON line carrying exactly one descriptor:
  ``{"type": "puffo.attach/1", "runtime_id", "registry", "launch_id"}``.
- The Agent checks ``runtime_id`` against its own registry record and its own
  directory, then answers one line: ``{"ok": true, ...}`` or
  ``{"ok": false, "reason": ...}``.
- After ``ok`` the same socket carries ACP.

Any refusal or malformed answer fails the open. There is no fallback to
starting a process: a second instance on the same directory is exactly what
attach mode exists to prevent.
"""

from __future__ import annotations

import array
import asyncio
import json
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..driver import RuntimeSpec
from ..driver_authority_server import DriverAuthorityServer
from .acp import AcpDriver, ValidatedLaunchPlan

ATTACH_PROTOCOL = "puffo.attach/1"
HANDSHAKE_TIMEOUT_SECONDS = 10.0
# The answer is one short JSON line; anything longer is not a handshake reply.
MAX_HANDSHAKE_REPLY_BYTES = 4096
_STREAM_LIMIT = 16 * 1024 * 1024


class AttachRefused(RuntimeError):
    """The Agent answered the handshake and declined this connection."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"LingTai refused the attach: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class AttachTarget:
    """Which running Agent to attach to, and the binding Puffo claims for it.

    ``runtime_id`` and ``registry`` are what provision recorded. The Agent
    verifies them against its own records; sending them is a claim, not proof.
    """

    socket_path: Path
    runtime_id: str
    registry: Path


class _AttachedConnection:
    """What ``AcpDriver`` expects from ``_spawn``, backed by a socket.

    ``stderr`` is None because an attached Agent's diagnostics stay with the
    Agent. ``wait`` returns when the Agent side closes the connection; there is
    no exit status because nothing exited as far as Puffo can tell.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.stdin = writer
        self.stdout = reader
        self.stderr = None

    async def wait(self) -> None:
        try:
            await self.stdin.wait_closed()
        except (ConnectionError, OSError):
            pass
        return None

    async def disconnect(self) -> None:
        self.stdin.close()
        try:
            await self.stdin.wait_closed()
        except (ConnectionError, OSError):
            pass


class AcpAttachDriver(AcpDriver):
    """ACP over a connection to a resident LingTai Agent. Never spawns."""

    def __init__(self, target: AttachTarget, **kwargs: Any) -> None:
        if kwargs.get("process_factory") is not None:
            raise TypeError("attach mode does not start processes")
        super().__init__(**kwargs)
        self.target = target

    def _validate_launch_plan(self, spec: RuntimeSpec) -> ValidatedLaunchPlan:
        launch = super()._validate_launch_plan(spec)
        # Model flags are launch arguments, and nothing is launched here. The
        # resident Agent keeps whatever model it runs, so do not report a
        # selection that never reached it.
        if spec.model:
            self._model_selection = ""
            self._spawn_warnings = (
                "attach mode cannot select a model; the running LingTai Agent "
                f"keeps its own, and spec.model {spec.model!r} was not applied",
            )
        return launch

    async def _spawn(self, launch: ValidatedLaunchPlan) -> Any:
        if not isinstance(launch, ValidatedLaunchPlan):
            raise TypeError("attach requires a ValidatedLaunchPlan")
        authority = DriverAuthorityServer()
        launch_id = f"launch_{uuid.uuid4().hex}"
        try:
            endpoint = authority.issue_root(launch_id=launch_id)
            try:
                sock = await asyncio.to_thread(
                    _handshake, self.target, launch_id, endpoint.fileno()
                )
            finally:
                # The Agent holds its own copy now, or the attach failed; ours
                # must not keep the channel alive either way.
                endpoint.close()
            try:
                reader, writer = await _open_stream(sock)
            except BaseException:
                sock.close()
                raise
        except BaseException:
            authority.close()
            raise
        self._driver_authority = authority
        return _AttachedConnection(reader, writer)

    async def _watch_process(self, proc: Any) -> None:
        await proc.wait()
        # The Agent side hung up. Withdraw the authority now rather than when
        # someone gets round to closing the driver: the Agent still holds its
        # end, and it must not be able to get provider calls approved on a
        # connection that no longer exists.
        authority = self._driver_authority
        if authority is not None:
            authority.close()
        await super()._watch_process(_Ended())

    async def _release_transport(self, proc: Any) -> None:
        # Disconnect only. The Agent process, its lock and its directory are
        # LingTai's; closing Puffo's side is the whole of Puffo's part.
        if proc is not None:
            await proc.disconnect()


class _Ended:
    """A transport that has already ended, for the base watcher's report."""

    async def wait(self) -> None:
        return None


class _CloseOnEof(asyncio.StreamReaderProtocol):
    # The stock protocol keeps a unix socket half-open after the peer's EOF, so
    # the connection never counts as lost and nothing notices the Agent left.
    def eof_received(self) -> bool:
        super().eof_received()
        return False


async def _open_stream(
    sock: socket.socket,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_STREAM_LIMIT, loop=loop)
    protocol = _CloseOnEof(reader, loop=loop)
    transport, _ = await loop.create_unix_connection(lambda: protocol, sock=sock)
    return reader, asyncio.StreamWriter(transport, protocol, reader, loop)


def _handshake(target: AttachTarget, launch_id: str, authority_fd: int) -> socket.socket:
    """Connect, hand over the authority descriptor, and read the verdict.

    Blocking, bounded by ``HANDSHAKE_TIMEOUT_SECONDS``; run off the loop.
    """
    hello = {
        "type": ATTACH_PROTOCOL,
        "runtime_id": target.runtime_id,
        "registry": str(target.registry),
        "launch_id": launch_id,
    }
    frame = (json.dumps(hello, separators=(",", ":")) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
        sock.connect(str(target.socket_path))
        rights = array.array("i", [authority_fd])
        sent = sock.sendmsg([frame], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
        if sent < len(frame):
            sock.sendall(frame[sent:])
        reply = _read_line(sock)
        _check_reply(reply)
        sock.settimeout(None)
        sock.setblocking(False)
        return sock
    except BaseException:
        sock.close()
        raise


def _read_line(sock: socket.socket) -> bytes:
    # Byte at a time so nothing past the reply is consumed: after "ok" the
    # stream belongs to ACP.
    line = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise AttachRefused("connection closed before the handshake answer")
        if chunk == b"\n":
            return bytes(line)
        line += chunk
        if len(line) > MAX_HANDSHAKE_REPLY_BYTES:
            raise AttachRefused("handshake answer exceeds its bound")


def _check_reply(raw: bytes) -> None:
    try:
        reply = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise AttachRefused("handshake answer is not JSON") from None
    if not isinstance(reply, dict):
        raise AttachRefused("handshake answer is not an object")
    if reply.get("ok") is True:
        return
    reason = reply.get("reason")
    raise AttachRefused(reason if isinstance(reason, str) and reason else "no reason given")
