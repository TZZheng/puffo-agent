"""Attach mode: connect to a resident LingTai Agent, never start one.

The Agent side here is a stand-in written against the agreed handshake
(Lingtai<>Puffo 219779): one JSON line carrying one descriptor, one JSON line
back, then ACP on the same socket. It uses the received descriptor the way the
kernel will, as a Driver-authority client, so the audit these tests read is the
Puffo authority server's own record of what arrived over that descriptor.
"""

from __future__ import annotations

import array
import asyncio
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import uuid
from pathlib import Path

import pytest

from puffo_agent.agent.harness.driver import RuntimeSpec
from puffo_agent.agent.harness.drivers import acp as acp_module
from puffo_agent.agent.harness.drivers.acp_attach import (
    AcpAttachDriver,
    AttachRefused,
    AttachTarget,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "SCM_RIGHTS"),
    reason="attach hands the authority over as SCM_RIGHTS",
)


def _authority_request(sock: socket.socket, payload: dict) -> dict:
    encoded = json.dumps(payload).encode()
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    return json.loads(_read_exact(sock, size))


def _read_exact(sock: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


class ResidentAgentStandIn:
    """Accepts attaches one at a time and records what each one delivered."""

    def __init__(self, path: Path, *, verdict=None, hang_up_after_session=False) -> None:
        self.path = path
        self.hang_up_after_session = hang_up_after_session
        self.verdict = verdict or (lambda hello: {"ok": True})
        self.hellos: list[dict] = []
        self.descriptor_counts: list[int] = []
        self.provider_decisions: list[dict] = []
        self.after_disconnect: list[str] = []
        self.disconnects = 0
        # Set by the test when the probe after a disconnect should run, so it
        # asks about a state the test chose rather than racing the driver.
        self.probe_now = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(4)
        self._listener.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.probe_now.set()
        self._stop.set()
        self._thread.join(timeout=5)
        self._listener.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.settimeout(5)
            try:
                self._serve_one(client)
            finally:
                client.close()

    def _serve_one(self, client: socket.socket) -> None:
        line, fds = self._read_hello(client)
        hello = json.loads(line)
        self.hellos.append(hello)
        self.descriptor_counts.append(len(fds))
        authority = socket.socket(fileno=fds[0]) if fds else None
        for extra in fds[1:]:
            os.close(extra)
        try:
            verdict = self.verdict(hello)
            if verdict.get("ok") and authority is not None:
                claimed = _authority_request(authority, {"version": 1, "op": "hello"})
                self.provider_decisions.append(
                    _authority_request(
                        authority,
                        {
                            "version": 1,
                            "op": "authorize_provider_call",
                            "call_id": str(uuid.uuid4()),
                            "launch_id": claimed["launch_id"],
                            "provider": "llm",
                            "capability": "root",
                        },
                    )
                )
            client.sendall((json.dumps(verdict) + "\n").encode())
            if verdict.get("ok"):
                self._serve_acp(client, hang_up=self.hang_up_after_session)
            self.disconnects += 1
            if authority is not None and self.probe_now.wait(timeout=5):
                self.after_disconnect.append(self._probe(authority))
        finally:
            if authority is not None:
                authority.close()

    @staticmethod
    def _read_hello(client: socket.socket) -> tuple[bytes, list[int]]:
        data = b""
        fds: list[int] = []
        while b"\n" not in data:
            chunk, ancdata, _flags, _ = client.recvmsg(4096, socket.CMSG_SPACE(64))
            if not chunk:
                raise EOFError
            for level, kind, raw in ancdata:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    items = array.array("i")
                    items.frombytes(raw[: len(raw) - len(raw) % items.itemsize])
                    fds.extend(items.tolist())
            data += chunk
        line, _, rest = data.partition(b"\n")
        assert rest == b"", "client wrote past the handshake before the answer"
        return line, fds

    @staticmethod
    def _serve_acp(client: socket.socket, *, hang_up: bool) -> None:
        stream = client.makefile("rwb")
        for raw in stream:
            message = json.loads(raw)
            if "id" not in message:
                continue
            method = message.get("method")
            if method == "initialize":
                result = {"protocolVersion": 1, "agentCapabilities": {}}
            elif method == "session/new":
                result = {"sessionId": "resident-session"}
            else:
                result = {}
            stream.write(
                (json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n").encode()
            )
            stream.flush()
            if hang_up and method == "session/new":
                client.shutdown(socket.SHUT_RDWR)
                return

    @staticmethod
    def _probe(authority: socket.socket) -> str:
        """Ask for a provider call after Puffo's side went away."""
        try:
            decision = _authority_request(
                authority,
                {
                    "version": 1,
                    "op": "authorize_provider_call",
                    "call_id": str(uuid.uuid4()),
                    "launch_id": "any",
                    "provider": "llm",
                    "capability": "root",
                },
            )
        except (EOFError, OSError):
            return "channel_closed"
        return decision.get("state", "unknown")


@pytest.fixture
def socket_dir():
    # AF_UNIX paths are short on macOS; pytest's tmp_path can exceed the limit.
    path = Path(tempfile.mkdtemp(prefix="attach-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def no_spawn(monkeypatch):
    calls: list[tuple] = []

    async def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("attach mode started a process")

    monkeypatch.setattr(acp_module.asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    return calls


def _target(socket_dir: Path) -> AttachTarget:
    return AttachTarget(
        socket_path=socket_dir / "agent.sock",
        runtime_id="puffo-runtime-1",
        registry=Path("/home/me/.puffo/lingtai/runtime-registry.json"),
    )


def _spec(**overrides) -> RuntimeSpec:
    fields = {"workspace_dir": "/workspace", "executable": "lingtai-agent"}
    fields.update(overrides)
    return RuntimeSpec(**fields)


@pytest.mark.asyncio
async def test_attach_opens_a_session_and_hands_over_the_authority(socket_dir, no_spawn):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(target.socket_path)
    driver = AcpAttachDriver(target)
    try:
        opened = await driver.open(_spec())
        authority = driver._driver_authority
        assert authority is not None
        audits = authority.audit_records()
    finally:
        await driver.close()
        agent.close()

    assert opened.native_session_id == "resident-session"
    [hello] = agent.hellos
    assert hello["type"] == "puffo.attach/1"
    assert hello["runtime_id"] == "puffo-runtime-1"
    assert hello["registry"] == str(target.registry)
    assert agent.descriptor_counts == [1]
    # The provider call the Agent made over the received descriptor landed in
    # Puffo's own authority, bound to this connection's launch.
    [decision] = agent.provider_decisions
    assert decision["state"] == "granted"
    granted = [a for a in audits if a.operation == "authorize_provider_call"]
    assert [a.state for a in granted] == ["granted"]
    assert granted[0].launch_id == hello["launch_id"]
    assert no_spawn == []


@pytest.mark.asyncio
async def test_closing_disconnects_and_takes_the_authority_with_it(socket_dir, no_spawn):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(target.socket_path)
    driver = AcpAttachDriver(target)
    await driver.open(_spec())
    await driver.close()
    agent.probe_now.set()
    for _ in range(100):
        if agent.after_disconnect:
            break
        await asyncio.sleep(0.02)
    agent.close()

    assert agent.disconnects == 1
    assert agent.after_disconnect == ["channel_closed"]
    assert no_spawn == []


@pytest.mark.asyncio
async def test_the_agent_hanging_up_ends_the_authority_without_puffo_closing(
    socket_dir, no_spawn
):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(target.socket_path, hang_up_after_session=True)
    driver = AcpAttachDriver(target)
    try:
        await driver.open(_spec())
        watcher = driver._watcher
        assert watcher is not None
        await asyncio.wait_for(asyncio.shield(watcher), timeout=5)
        # Puffo has not closed anything; only the Agent side went away.
        agent.probe_now.set()
        for _ in range(100):
            if agent.after_disconnect:
                break
            await asyncio.sleep(0.02)
        observed = list(agent.after_disconnect)
    finally:
        await driver.close()
        agent.close()

    assert observed == ["channel_closed"]
    assert no_spawn == []


@pytest.mark.asyncio
async def test_a_refusal_fails_the_open_and_starts_nothing(socket_dir, no_spawn):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(
        target.socket_path,
        verdict=lambda hello: {"ok": False, "reason": "runtime_id is not registered here"},
    )
    driver = AcpAttachDriver(target)
    try:
        with pytest.raises(AttachRefused) as refused:
            await driver.open(_spec())
    finally:
        await driver.close()
    agent.probe_now.set()
    for _ in range(100):
        if agent.after_disconnect:
            break
        await asyncio.sleep(0.02)
    agent.close()

    assert refused.value.reason == "runtime_id is not registered here"
    assert driver._driver_authority is None
    assert agent.after_disconnect == ["channel_closed"]
    assert no_spawn == []


@pytest.mark.asyncio
async def test_no_listening_agent_is_an_error_not_a_launch(socket_dir, no_spawn):
    driver = AcpAttachDriver(_target(socket_dir))
    try:
        with pytest.raises(OSError):
            await driver.open(_spec())
    finally:
        await driver.close()
    assert driver._driver_authority is None
    assert no_spawn == []


@pytest.mark.asyncio
async def test_each_connection_gets_its_own_launch(socket_dir, no_spawn):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(target.socket_path)
    driver = AcpAttachDriver(target)
    try:
        await driver.open(_spec())
        await driver.close()
        await driver.open(_spec())
    finally:
        await driver.close()
        agent.close()

    first, second = agent.hellos
    assert first["runtime_id"] == second["runtime_id"]
    assert first["launch_id"] != second["launch_id"]
    assert no_spawn == []


@pytest.mark.asyncio
async def test_a_model_is_not_reported_as_selected(socket_dir, no_spawn):
    target = _target(socket_dir)
    agent = ResidentAgentStandIn(target.socket_path)
    driver = AcpAttachDriver(target)
    try:
        # An executable the spawn path knows a model flag for, so the only
        # thing keeping the selection out of the report is attach mode.
        opened = await driver.open(_spec(executable="gemini", model="some-model"))
    finally:
        await driver.close()
        agent.close()

    assert not any(
        name.startswith("model_selection/")
        for name in opened.diagnostics.native_capabilities
    )
    assert any("attach mode cannot select a model" in w for w in opened.diagnostics.warnings)


def test_attach_driver_refuses_a_process_factory(socket_dir):
    with pytest.raises(TypeError):
        AcpAttachDriver(_target(socket_dir), process_factory=lambda *a: None)
