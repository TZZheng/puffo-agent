"""Tests for worker/config integration.

Covers: PuffoCoreConfig, PuffoCoreMessageClient, puffo_core_server,
and config builders.
"""

import os
import sys
import tempfile
import time
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from puffo_agent.agent.message_store import MessageStore
from puffo_agent.crypto.keystore import KeyStore, Session, StoredIdentity, encode_secret
from puffo_agent.crypto.primitives import Ed25519KeyPair, KemKeyPair
from puffo_agent.mcp.config import (
    PUFFO_CORE_TOOL_NAMES,
    PUFFO_CORE_TOOL_FQNS,
    puffo_core_mcp_env,
    puffo_core_stdio_sdk_config,
)
from puffo_agent.portal.state import AgentConfig, PuffoCoreConfig


def _now_ms():
    return int(time.time() * 1000)


def _make_keystore():
    d = tempfile.mkdtemp()
    ks_dir = os.path.join(d, "keys")
    ks = KeyStore(ks_dir)
    device_key = Ed25519KeyPair.generate()
    kem_kp = KemKeyPair.generate()
    identity = StoredIdentity(
        slug="bot-0001",
        device_id="dev_test",
        root_secret_key=encode_secret(Ed25519KeyPair.generate().secret_bytes()),
        device_signing_secret_key=encode_secret(device_key.secret_bytes()),
        kem_secret_key=encode_secret(kem_kp.secret_bytes()),
        server_url="http://localhost:3000",
    )
    ks.save_identity(identity)
    subkey = Ed25519KeyPair.generate()
    session = Session(
        slug="bot-0001",
        subkey_id="sk_test",
        subkey_secret_key=encode_secret(subkey.secret_bytes()),
        expires_at=_now_ms() + 3_600_000,
    )
    ks.save_session(session)
    return ks, ks_dir, d, kem_kp


# ── PuffoCoreConfig tests ──────────────────────────────────────────


def test_puffo_core_config_is_configured():
    cfg = PuffoCoreConfig()
    assert not cfg.is_configured()

    cfg = PuffoCoreConfig(server_url="http://localhost", slug="bot-0001", device_id="dev_1", space_id="sp_1")
    assert cfg.is_configured()

    cfg = PuffoCoreConfig(server_url="http://localhost", slug="", device_id="dev_1", space_id="sp_1")
    assert not cfg.is_configured()

    cfg = PuffoCoreConfig(server_url="http://localhost", slug="bot-0001", device_id="dev_1", space_id="")
    assert not cfg.is_configured()


def test_puffo_core_config_in_agent_config():
    cfg = AgentConfig(
        id="test-agent",
        puffo_core=PuffoCoreConfig(
            server_url="http://localhost:3000",
            slug="bot-0001",
            device_id="dev_1",
            space_id="sp_test",
        ),
    )
    assert cfg.puffo_core.is_configured()
    assert cfg.puffo_core.space_id == "sp_test"


def test_agent_config_default_puffo_core():
    cfg = AgentConfig(id="test-agent")
    assert not cfg.puffo_core.is_configured()


@pytest.mark.asyncio
async def test_supported_worker_driver_registers_live_encrypted_controls_and_events(
    tmp_path, monkeypatch,
):
    from puffo_agent.agent.harness.driver import (
        CancelReceipt,
        CompactRequest,
        HarnessDriver,
        HarnessEvent,
        PermissionReceipt,
        RuntimeOpened,
        RuntimeRef,
        RuntimeSpec,
        SessionRef,
        TurnInput,
        TurnRef,
        TurnStarted,
        UnsupportedCapability,
    )
    from puffo_agent.agent.harness.codex_driver import CODEX_CAPABILITIES
    from puffo_agent.agent.runtime_event_outbox import RuntimeEventOutbox
    from puffo_agent.portal.control.client import execute_command
    from puffo_agent.portal.state import DaemonConfig, RuntimeConfig
    from puffo_agent.portal.worker import (
        _promote_supported_local_driver,
        build_adapter,
    )

    class FakeDriver(HarnessDriver):
        def __init__(self):
            self.queue = asyncio.Queue()
            self.driver_turn = TurnRef("driver-turn")

        async def open(self, spec: RuntimeSpec, resume=None):
            return RuntimeOpened(
                RuntimeRef("runtime"), SessionRef("native-session"),
                "native-session", bool(resume), CODEX_CAPABILITIES,
                SimpleNamespace(),
            )

        async def start_turn(self, input: TurnInput):
            return TurnStarted(self.driver_turn, "native-turn")

        async def steer_turn(self, turn, input):
            return UnsupportedCapability("steer")

        async def cancel_turn(self, turn):
            return CancelReceipt(True, turn)

        async def context_status(self):
            return UnsupportedCapability("context_status")

        async def compact(self, request: CompactRequest):
            return UnsupportedCapability("compact")

        async def resolve_permission(self, request, decision):
            await self.queue.put(HarnessEvent(
                type="turn.permission_updated",
                driver="codex",
                session_ref=SessionRef("native-session"),
                turn_ref=self.driver_turn,
                data={
                    "permission_ref": str(request),
                    "state": decision.value,
                },
            ))
            return PermissionReceipt(True, request)

        def events(self):
            async def iterate():
                while True:
                    event = await self.queue.get()
                    if event is None:
                        return
                    yield event
            return iterate()

        async def close(self):
            await self.queue.put(None)

    cfg = AgentConfig(
        id="worker-driver-control",
        runtime=RuntimeConfig(kind="cli-local", harness="codex"),
    )
    adapter = build_adapter(DaemonConfig(), cfg)
    adapter.workspace_dir = str(tmp_path)
    adapter._verify = lambda: None

    async def no_install():
        return None

    adapter._install_desired = no_install
    adapter._ensure_codex_session = lambda: SimpleNamespace(
        argv=["codex", "app-server"],
        env={},
        get_provider_session_id=lambda: None,
    )
    fake = FakeDriver()
    import puffo_agent.agent.harness as harness_module
    monkeypatch.setattr(harness_module, "build_driver", lambda _name: fake)

    outbox = RuntimeEventOutbox(tmp_path / "runtime_events.db")
    facade = await _promote_supported_local_driver(
        adapter,
        cfg,
        system_prompt="system",
        outbox=outbox,
        logical_session_ref="logical-session",
    )
    try:
        started = await facade.manager.start_turn(TurnInput("hello"))
        await fake.queue.put(HarnessEvent(
            type="turn.started",
            driver="codex",
            session_ref=SessionRef("native-session"),
            turn_ref=fake.driver_turn,
        ))
        await fake.queue.put(HarnessEvent(
            type="turn.permission_requested",
            driver="codex",
            session_ref=SessionRef("native-session"),
            turn_ref=fake.driver_turn,
            data={
                "permission_ref": "permission-one",
                "state": "pending",
            },
        ))
        for _ in range(20):
            if any(
                row.event_type == "permission.updated"
                and row.event["payload"]["state"] == "pending"
                for row in outbox.prefix()
            ):
                break
            await asyncio.sleep(0)

        refs = {
            "session_ref": "logical-session",
            "turn_ref": str(started.turn_ref),
        }
        permission = await execute_command(
            "runtime.resolve_permission",
            cfg.id,
            {
                **refs,
                "permission_ref": "permission-one",
                "decision": "approved",
            },
            command_id="worker-permission",
        )
        cancelled = await execute_command(
            "runtime.cancel_turn",
            cfg.id,
            refs,
            command_id="worker-cancel",
        )
        assert permission == {
            "ok": True, "delivered": True, "completed": False,
        }
        assert cancelled == {
            "ok": True, "delivered": True, "completed": False,
        }

        await fake.queue.put(HarnessEvent(
            type="turn.completed",
            driver="codex",
            session_ref=SessionRef("native-session"),
            turn_ref=fake.driver_turn,
            data={"outcome": "cancelled"},
        ))
        await facade.manager.wait_terminal(started.turn_ref)
        rows = [row.event for row in outbox.prefix()]
        assert rows
        assert all(row["scope"] == {"kind": "operator"} for row in rows)
        assert any(
            row["type"] == "permission.updated"
            and row["payload"]["state"] == "approved"
            for row in rows
        )
        assert rows[-1]["type"] == "turn.finished"
        assert rows[-1]["payload"]["outcome"] == "cancelled"
    finally:
        await facade.aclose()
        outbox.close()


@pytest.mark.parametrize(
    ("kind", "harness", "provider", "expected"),
    [
        ("cli-local", "hermes", "anthropic", "LocalCLIAdapter"),
        ("cli-docker", "claude-code", "anthropic", "DockerCLIAdapter"),
        ("cli-docker", "codex", "openai", "DockerCLIAdapter"),
        ("cli-docker", "hermes", "anthropic", "DockerCLIAdapter"),
        ("cli-docker", "gemini-cli", "google", "DockerCLIAdapter"),
        ("sdk-local", "", "anthropic", "SDKAdapter"),
    ],
)
def test_unpromoted_runtime_matrix_retains_legacy_adapter_classes(
    tmp_path, monkeypatch, kind, harness, provider, expected,
):
    import types
    from puffo_agent.portal.state import DaemonConfig, RuntimeConfig
    from puffo_agent.portal.worker import build_adapter

    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    if kind == "sdk-local":
        sdk = types.ModuleType("claude_agent_sdk")
        sdk.query = lambda *_args, **_kwargs: None
        for name in (
            "ClaudeAgentOptions", "AssistantMessage", "TextBlock",
            "ToolUseBlock", "ResultMessage",
        ):
            setattr(sdk, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    cfg = AgentConfig(
        id=f"matrix-{kind}-{harness or 'sdk'}",
        runtime=RuntimeConfig(
            kind=kind, harness=harness, provider=provider, api_key="test-key",
        ),
    )
    daemon = DaemonConfig()
    daemon.google.api_key = "test-key"
    assert type(build_adapter(daemon, cfg)).__name__ == expected


def test_local_gemini_retains_existing_rejection(tmp_path, monkeypatch):
    from puffo_agent.portal.state import DaemonConfig, RuntimeConfig
    from puffo_agent.portal.worker import build_adapter

    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    cfg = AgentConfig(
        id="local-gemini",
        runtime=RuntimeConfig(
            kind="cli-local", harness="gemini-cli",
            provider="google", api_key="test-key",
        ),
    )
    with pytest.raises(RuntimeError, match="not supported.*cli-local"):
        build_adapter(DaemonConfig(), cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["codex", "claude-code"])
async def test_supported_local_promotion_registers_before_return_and_reload_aligns_identity(
    tmp_path, monkeypatch, harness,
):
    from puffo_agent.agent.harness.driver import (
        CancelReceipt,
        HarnessDriver,
        HarnessEvent,
        RuntimeOpened,
        RuntimeRef,
        RuntimeSpec,
        SessionRef,
        TurnInput,
        TurnRef,
        TurnStarted,
        UnsupportedCapability,
    )
    from puffo_agent.agent.harness.codex_driver import CODEX_CAPABILITIES
    from puffo_agent.agent.harness.runtime_manager import (
        RuntimeManagerAdapter,
        get_runtime_manager,
    )
    from puffo_agent.agent.runtime_event_outbox import RuntimeEventOutbox
    from puffo_agent.portal.control.client import execute_command
    from puffo_agent.portal.state import DaemonConfig, RuntimeConfig
    from puffo_agent.portal.worker import (
        Worker,
        _promote_supported_local_driver,
        build_adapter,
    )

    class ReloadableDriver(HarnessDriver):
        def __init__(self):
            self.open_count = 0
            self.start_calls = 0
            self.queue = asyncio.Queue()
            self.driver_turn = TurnRef("driver-turn")

        async def open(self, spec: RuntimeSpec, resume=None):
            self.open_count += 1
            self.queue = asyncio.Queue()
            native = f"native-session-{self.open_count}"
            await self.queue.put(HarnessEvent(
                type="session.opened",
                driver=harness,
                session_ref=SessionRef(native),
                native_session_id=native,
            ))
            return RuntimeOpened(
                RuntimeRef(f"runtime-{self.open_count}"),
                SessionRef(native),
                native,
                bool(resume),
                CODEX_CAPABILITIES,
                SimpleNamespace(),
            )

        async def start_turn(self, input: TurnInput):
            self.start_calls += 1
            self.driver_turn = TurnRef(f"driver-turn-{self.open_count}")
            return TurnStarted(
                self.driver_turn, f"native-turn-{self.open_count}"
            )

        async def steer_turn(self, turn, input):
            return UnsupportedCapability("steer")

        async def cancel_turn(self, turn):
            return CancelReceipt(True, turn)

        async def context_status(self):
            return UnsupportedCapability("context_status")

        async def compact(self, request):
            return UnsupportedCapability("compact")

        async def resolve_permission(self, request, decision):
            return UnsupportedCapability("permission")

        def events(self):
            async def iterate():
                while True:
                    event = await self.queue.get()
                    if event is None:
                        return
                    yield event
            return iterate()

        async def close(self):
            await self.queue.put(None)

    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    cfg = AgentConfig(
        id=f"promoted-{harness}",
        puffo_core=PuffoCoreConfig(
            server_url="https://example.test",
            slug="agent-test",
            device_id="device-test",
            space_id="space-test",
        ),
        runtime=RuntimeConfig(kind="cli-local", harness=harness),
    )

    def make_adapter():
        adapter = build_adapter(DaemonConfig(), cfg)
        adapter.workspace_dir = str(tmp_path)
        adapter._verify = lambda: None

        async def no_install():
            return None

        async def no_warm(_system_prompt):
            return None

        adapter._install_desired = no_install
        adapter.warm = no_warm
        prepared = SimpleNamespace(
            argv=[harness, "app-server"],
            env={},
            extra_args=[],
            get_provider_session_id=lambda: None,
            build_command=lambda *_args: [harness],
        )
        adapter._ensure_codex_session = lambda: prepared
        adapter._ensure_session = lambda: prepared
        return adapter

    import puffo_agent.agent.harness as harness_module
    import puffo_agent.portal.worker as worker_module

    # Exercise the real Worker readiness gate. Pause inside the gate after it
    # sets _warm_done so the assertion observes the exact adapter/registry
    # state exposed to wait_warm() callers.
    adapter = make_adapter()
    driver = ReloadableDriver()
    monkeypatch.setattr(harness_module, "build_driver", lambda _name: driver)
    monkeypatch.setattr(worker_module, "build_adapter", lambda *_args: adapter)
    monkeypatch.setattr(
        worker_module,
        "_build_puffo_core_client",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    worker = Worker(DaemonConfig(), cfg)
    gate_pause = asyncio.Event()
    original_gate = Worker._run_post_warm_gate

    async def pausing_gate(agent_id):
        await original_gate(worker, agent_id)
        await gate_pause.wait()

    worker._run_post_warm_gate = pausing_gate
    worker.start()
    assert await worker.wait_warm(timeout=1.0) is True
    assert isinstance(worker._adapter, RuntimeManagerAdapter)
    assert get_runtime_manager(cfg.id) is worker._adapter.manager
    await worker.stop()

    adapter = make_adapter()
    driver = ReloadableDriver()
    monkeypatch.setattr(harness_module, "build_driver", lambda _name: driver)
    outbox = RuntimeEventOutbox(tmp_path / f"{harness}.db")
    facade = await _promote_supported_local_driver(
        adapter,
        cfg,
        system_prompt="system",
        outbox=outbox,
        logical_session_ref="logical-session-1",
    )
    assert isinstance(facade, RuntimeManagerAdapter)
    assert get_runtime_manager(cfg.id) is facade.manager
    for _ in range(100):
        if outbox.state().get("session_ref") == "logical-session-1":
            break
        await asyncio.sleep(0)
    assert outbox.state()["native_session_id"] == "native-session-1"

    old_session = str(facade.manager.session_ref)
    await facade.reload("system", with_session=True)
    new_session = str(facade.manager.session_ref)
    assert new_session != old_session
    for _ in range(100):
        if outbox.state().get("session_ref") == new_session:
            break
        await asyncio.sleep(0)
    assert outbox.state()["session_ref"] == new_session
    assert outbox.state()["native_session_id"] == "native-session-2"

    started = await facade.manager.start_turn(TurnInput("hello"))
    await driver.queue.put(HarnessEvent(
        type="turn.started",
        driver=harness,
        session_ref=SessionRef("native-session-2"),
        turn_ref=driver.driver_turn,
        native_session_id="native-session-2",
        native_turn_id="native-turn-2",
    ))
    stale = await execute_command(
        "runtime.cancel_turn",
        cfg.id,
        {"session_ref": old_session, "turn_ref": str(started.turn_ref)},
        command_id=f"stale-{harness}",
    )
    current = await execute_command(
        "runtime.cancel_turn",
        cfg.id,
        {"session_ref": new_session, "turn_ref": str(started.turn_ref)},
        command_id=f"current-{harness}",
    )
    assert stale["error_code"] == "stale_session_ref"
    assert current == {"ok": True, "delivered": True, "completed": False}
    for _ in range(100):
        initial = [row.event for row in outbox.prefix()]
        if [row["type"] for row in initial[:2]] == [
            "turn.started", "activity.updated",
        ]:
            break
        await asyncio.sleep(0)
    assert [row["type"] for row in initial[:2]] == [
        "turn.started", "activity.updated",
    ]
    assert initial[1]["payload"] == {"text": "Working"}
    assert driver.start_calls == 1
    await facade.aclose()
    outbox.close()

    # The real RuntimeManager callback must reject before touching the Driver
    # or durable active-turn state when there are not enough rows for the
    # required start/activity/terminal lifecycle.
    blocked_driver = ReloadableDriver()
    monkeypatch.setattr(
        harness_module, "build_driver", lambda _name: blocked_driver,
    )
    blocked_outbox = RuntimeEventOutbox(
        tmp_path / f"{harness}-blocked.db", max_rows=2,
    )
    blocked = await _promote_supported_local_driver(
        make_adapter(), cfg, system_prompt="system", outbox=blocked_outbox,
        logical_session_ref="logical-session-blocked",
    )
    try:
        with pytest.raises(Exception, match="outbox is at capacity"):
            await blocked.manager.start_turn(TurnInput("blocked"))
        assert blocked_driver.start_calls == 0
        assert blocked_outbox.prefix() == []
        assert blocked_outbox.state().get("active_turn_ref", "") == ""
    finally:
        await blocked.aclose()
        blocked_outbox.close()


@pytest.mark.asyncio
async def test_legacy_worker_global_runtime_projects_and_uploads_start_activity_terminal(
    tmp_path, monkeypatch,
):
    """The UnsupportedDriver fallback retains the Worker-owned projection bridge.

    This intentionally reaches the nested bridge through the same
    ``client.global_runtime.run_turn`` callback used by the live Inbox rather
    than extracting or duplicating it in a unit helper.
    """
    from puffo_agent.agent.core import PuffoAgent
    from puffo_agent.agent.harness import UnsupportedDriver
    from puffo_agent.agent.message_store import MessageStore
    from puffo_agent.agent.runtime_event_outbox import runtime_event_outbox_path
    from puffo_agent.portal.state import DaemonConfig, RuntimeConfig
    from puffo_agent.portal.worker import Worker, build_adapter
    import puffo_agent.agent.harness as harness_module
    import puffo_agent.portal.profile_sync as profile_sync
    import puffo_agent.portal.worker as worker_module

    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    cfg = AgentConfig(
        id="legacy-runtime-events",
        puffo_core=PuffoCoreConfig(
            server_url="https://example.test",
            slug="legacy-agent",
            device_id="legacy-device",
            space_id="legacy-space",
        ),
        runtime=RuntimeConfig(kind="cli-local", harness="codex"),
    )

    release_append = asyncio.Event()
    append_started = asyncio.Event()
    append_bodies = []
    provider_calls = []

    class HoldingHttp:
        keyless = False

        async def post(self, path, body):
            append_bodies.append((path, body))
            append_started.set()
            await release_append.wait()
            return {
                "accepted": [
                    {"event_id": event["event_id"]}
                    for event in body["events"]
                ],
            }

    class Client:
        slug = "legacy-agent"
        keystore = SimpleNamespace()
        http = HoldingHttp()

        def __init__(self, name="legacy-messages"):
            self.store = MessageStore(str(tmp_path / f"{name}.db"))
            self.global_runtime = None

        async def listen(self):
            await asyncio.Event().wait()

        async def recover_pending_delivery(self, *_args, **_kwargs):
            return None

        async def stop(self):
            return None

    client = Client()

    def make_adapter():
        adapter = build_adapter(DaemonConfig(), cfg)
        adapter.workspace_dir = str(tmp_path)
        adapter._verify = lambda: None

        async def no_install():
            return None

        async def no_warm(_system_prompt):
            return None

        adapter._install_desired = no_install
        adapter.warm = no_warm
        adapter._ensure_codex_session = lambda: SimpleNamespace(
            argv=["codex", "app-server"], env={},
            get_provider_session_id=lambda: "native-legacy-session",
        )
        return adapter

    async def no_profile_sync(*_args, **_kwargs):
        return None

    async def no_status_loop():
        return None

    class NoopReporter:
        async def run_heartbeat_loop(self):
            await no_status_loop()

        def stop(self):
            return None

    async def spy_handle_global_inbox_turn(self, planned, on_progress=None):
        provider_calls.append(planned)
        if on_progress is not None:
            await on_progress("safe legacy output")
        return "safe legacy reply"

    monkeypatch.setattr(harness_module, "build_driver", lambda _name: UnsupportedDriver("codex"))
    monkeypatch.setattr(worker_module, "build_adapter", lambda *_args: make_adapter())
    monkeypatch.setattr(worker_module, "_build_puffo_core_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(profile_sync, "sync_full_profile", no_profile_sync)
    monkeypatch.setattr(PuffoAgent, "handle_global_inbox_turn", spy_handle_global_inbox_turn)

    worker = Worker(DaemonConfig(), cfg)
    worker._build_status_reporter = lambda _client: NoopReporter()
    worker.start()
    try:
        for _ in range(200):
            if client.global_runtime is not None:
                break
            await asyncio.sleep(0.01)
        assert client.global_runtime is not None

        await client.global_runtime.run_turn(SimpleNamespace(
            provider_input="legacy bridge input", targets=(),
        ))
        assert len(provider_calls) == 1

        outbox_path = runtime_event_outbox_path(
            worker_module.agent_dir(cfg.id),
        )
        for _ in range(200):
            if append_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert append_started.is_set()

        # The upload is deliberately held: a second connection observes the
        # durable start/activity/terminal sequence before acknowledgement may
        # delete it.
        with sqlite3.connect(outbox_path) as connection:
            rows = connection.execute(
                "SELECT event_json FROM events ORDER BY sequence"
            ).fetchall()
        durable = [json.loads(row[0]) for row in rows]
        assert [event["type"] for event in durable] == [
            "turn.started", "activity.updated", "output.updated",
            "output.updated", "output.updated", "turn.finished",
        ]
        assert durable[1]["payload"] == {"text": "Working"}
        assert durable[-1]["payload"]["outcome"] == "succeeded"

        release_append.set()
        for _ in range(200):
            if append_bodies:
                break
            await asyncio.sleep(0.01)
        flattened = [
            event for _path, body in append_bodies
            for event in body.get("events", [])
        ]
        assert [event["type"] for event in flattened] == [
            "turn.started", "activity.updated", "output.updated",
            "output.updated", "output.updated", "turn.finished",
        ]
        for _ in range(200):
            with sqlite3.connect(outbox_path) as connection:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM events"
                ).fetchone()[0]
            if remaining == 0:
                break
            await asyncio.sleep(0.01)
        assert remaining == 0
    finally:
        release_append.set()
        await worker.stop()
        await client.store.close()

    # Run the retained bridge a second time with only two rows available.
    # Admission occurs before the nested Provider call, so no durable row,
    # append request, or global active-turn state may be created.
    from puffo_agent.agent import runtime_event_outbox as outbox_module

    original_outbox = outbox_module.RuntimeEventOutbox
    monkeypatch.setattr(
        outbox_module,
        "RuntimeEventOutbox",
        lambda path: original_outbox(path, max_rows=2),
    )
    limited_client = Client("legacy-limited-messages")
    monkeypatch.setattr(
        worker_module,
        "_build_puffo_core_client",
        lambda *_args, **_kwargs: limited_client,
    )
    limited_worker = Worker(DaemonConfig(), cfg)
    limited_worker._build_status_reporter = lambda _client: NoopReporter()
    limited_worker.start()
    try:
        for _ in range(200):
            if limited_client.global_runtime is not None:
                break
            await asyncio.sleep(0.01)
        assert limited_client.global_runtime is not None
        with pytest.raises(Exception, match="outbox is at capacity"):
            await limited_client.global_runtime.run_turn(SimpleNamespace(
                provider_input="rejected legacy input", targets=(),
            ))
        assert len(provider_calls) == 1
        assert limited_client.global_runtime.active.turn_id == ""
        limited_path = runtime_event_outbox_path(
            worker_module.agent_dir(cfg.id),
        )
        with sqlite3.connect(limited_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM events").fetchone() == (0,)
        assert len(append_bodies) == 1
    finally:
        await limited_worker.stop()
        await limited_client.store.close()


# ── Config builder tests ───────────────────────────────────────────


def test_puffo_core_tool_names():
    assert "send_message" in PUFFO_CORE_TOOL_NAMES
    assert "whoami" in PUFFO_CORE_TOOL_NAMES
    assert "refresh" in PUFFO_CORE_TOOL_NAMES
    assert "reload_system_prompt" not in PUFFO_CORE_TOOL_NAMES
    assert "approve_permission" not in PUFFO_CORE_TOOL_NAMES
    assert len(PUFFO_CORE_TOOL_FQNS) == len(PUFFO_CORE_TOOL_NAMES)
    assert all(t.startswith("mcp__puffo__") for t in PUFFO_CORE_TOOL_FQNS)


def test_puffo_core_mcp_env():
    env = puffo_core_mcp_env(
        slug="bot-0001",
        device_id="dev_1",
        server_url="http://localhost:3000",
        space_id="sp_test",
        keystore_dir="/tmp/keys",
        workspace="/workspace",
        agent_id="bot-0001",
        runtime_kind="cli-local",
        harness="claude-code",
    )
    assert env["PUFFO_CORE_SLUG"] == "bot-0001"
    assert env["PUFFO_CORE_DEVICE_ID"] == "dev_1"
    assert env["PUFFO_CORE_SERVER_URL"] == "http://localhost:3000"
    assert env["PUFFO_CORE_SPACE_ID"] == "sp_test"
    assert env["PUFFO_CORE_KEYSTORE_DIR"] == "/tmp/keys"
    # MCP reads SQLite via the daemon's data service at
    # PUFFO_DATA_SERVICE_URL, not by opening the DB directly.
    assert "PUFFO_CORE_DB_PATH" not in env
    assert env["PUFFO_DATA_SERVICE_URL"] == "http://127.0.0.1:63386"
    assert env["PUFFO_AGENT_ID"] == "bot-0001"
    assert env["PUFFO_WORKSPACE"] == "/workspace"
    assert env["PUFFO_RUNTIME_KIND"] == "cli-local"
    assert env["PUFFO_HARNESS"] == "claude-code"


def test_puffo_core_mcp_env_optional_fields():
    env = puffo_core_mcp_env(
        slug="bot-0001",
        device_id="dev_1",
        server_url="http://localhost:3000",
        keystore_dir="/tmp/keys",
        workspace="/workspace",
    )
    assert "PUFFO_CORE_SPACE_ID" not in env
    assert "PUFFO_RUNTIME_KIND" not in env
    assert "PUFFO_HARNESS" not in env
    assert "PUFFO_AGENT_ID" not in env


def test_puffo_core_mcp_env_pins_python_user_base():
    """cli-local rewrites HOME on the claude subprocess, which
    would move Python's user-site to an empty per-agent path and
    hide ``mcp`` from the MCP subprocess. ``PYTHONUSERBASE`` pins
    user-site to the daemon's real base regardless of HOME."""
    import site
    env = puffo_core_mcp_env(
        slug="bot-0001",
        device_id="dev_1",
        server_url="http://localhost:3000",
        keystore_dir="/tmp/keys",
        workspace="/workspace",
        runtime_kind="cli-local",
    )
    assert env["PYTHONUSERBASE"] == site.getuserbase()


def test_puffo_core_mcp_env_skips_python_user_base_for_docker():
    """The container has its own Python install with baked-in deps,
    so the host's user-base path is meaningless inside it. We
    deliberately don't forward ``PYTHONUSERBASE`` into the docker
    env block to keep the contract semantically clean."""
    env = puffo_core_mcp_env(
        slug="bot-0001",
        device_id="dev_1",
        server_url="http://localhost:3000",
        keystore_dir="/tmp/keys",
        workspace="/workspace",
        runtime_kind="cli-docker",
    )
    assert "PYTHONUSERBASE" not in env


def test_puffo_core_stdio_sdk_config():
    cfg = puffo_core_stdio_sdk_config(
        python="/usr/bin/python3",
        slug="bot-0001",
        device_id="dev_1",
        server_url="http://localhost:3000",
        space_id="sp_test",
        keystore_dir="/tmp/keys",
        workspace="/workspace",
        agent_id="bot-0001",
    )
    assert "puffo" in cfg
    server = cfg["puffo"]
    assert server["type"] == "stdio"
    assert server["command"] == "/usr/bin/python3"
    assert server["args"] == ["-m", "puffo_agent.mcp.puffo_core_server"]
    assert server["env"]["PUFFO_CORE_SLUG"] == "bot-0001"
    assert server["env"]["PUFFO_AGENT_ID"] == "bot-0001"


# ── MCP server build test ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_puffo_core_server_builds():
    """The puffo-core MCP server should register both API and local tools."""
    ks, ks_dir, d, _ = _make_keystore()
    db_path = os.path.join(d, "messages.db")

    from puffo_agent.mcp.puffo_core_server import build_server

    mcp = build_server(
        slug="bot-0001",
        device_id="dev_test",
        server_url="http://localhost:3000",
        space_id="sp_test",
        keystore_dir=ks_dir,
        workspace=d,
        agent_id="bot-0001",
        # Test never makes a real call; value just needs to parse.
        data_service_url="http://127.0.0.1:0",
    )
    tool_names = {t.name for t in await mcp.list_tools()}
    assert "whoami" in tool_names
    assert "send_message" in tool_names
    assert "get_channel_history" in tool_names
    assert "refresh" in tool_names
    assert "reload_system_prompt" not in tool_names
    assert "install_skill" in tool_names
    assert "list_skills" in tool_names
    assert "install_mcp_server" in tool_names
    assert "list_mcp_servers" in tool_names


@pytest.mark.asyncio
async def test_puffo_core_server_whoami():
    ks, ks_dir, d, _ = _make_keystore()
    db_path = os.path.join(d, "messages.db")

    from puffo_agent.mcp.puffo_core_server import build_server

    mcp = build_server(
        slug="bot-0001",
        device_id="dev_test",
        server_url="http://localhost:3000",
        space_id="sp_test",
        keystore_dir=ks_dir,
        workspace=d,
        agent_id="bot-0001",
        # Test never makes a real call; value just needs to parse.
        data_service_url="http://127.0.0.1:0",
    )
    result = await mcp.call_tool("whoami", {})
    text = "".join(getattr(item, "text", str(item)) for item in result)
    assert "bot-0001" in text
    assert "dev_test" in text


# ── MessageStore WAL test ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_store_wal_mode():
    d = tempfile.mkdtemp()
    store = MessageStore(os.path.join(d, "messages.db"))
    await store.open()
    db = await store._ensure_db()
    async with db.execute("PRAGMA journal_mode") as cursor:
        row = await cursor.fetchone()
        assert row[0] == "wal"
    await store.close()


# ── PuffoCoreMessageClient unit tests ──────────────────────────────


@pytest.mark.asyncio
async def test_puffo_core_client_send_fallback_message_encrypts():
    """Smoke test: send_fallback_message resolves channel members and their
    device certs, encrypts the reply, and posts the bare envelope.

    Mocked endpoints reflect the real puffo-core wire shape:
      * ``/spaces/{space}/channels/{ch}/members`` -> ``{"members": [...]}``
      * ``/certs/sync?slugs=...`` -> entries with ``kind=device_cert``
      * ``POST /messages`` accepts the envelope at the top level
        (``Json<MessageEnvelope>``), not wrapped in ``{"envelope": ...}``.
    """
    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    ks, ks_dir, d, kem_kp = _make_keystore()
    db_path = os.path.join(d, "messages.db")
    ms = MessageStore(db_path)
    await ms.open()

    from puffo_agent.crypto.encoding import base64url_encode

    recipient_kem_pk_b64 = base64url_encode(
        KemKeyPair.generate().public_key_bytes()
    )

    class FakeHttp:
        def __init__(self):
            self.calls = []
            self.post_bodies = []

        async def get(self, path):
            self.calls.append(("GET", path))
            # Channel members: slugs only; client follows up with
            # /certs/sync to translate to device certs.
            if path.startswith("/spaces/") and "/channels/" in path and path.endswith("/members"):
                return {"members": [{"slug": "alice", "role": "member"}]}
            if path.startswith("/certs/sync"):
                return {
                    "entries": [{
                        "seq": 1,
                        "kind": "device_cert",
                        "slug": "alice",
                        "cert": {
                            "device_id": "dev_recipient",
                            "kem_public_key": recipient_kem_pk_b64,
                        },
                    }],
                    "has_more": False,
                }
            return {}

        async def post(self, path, body=None):
            self.calls.append(("POST", path))
            self.post_bodies.append(body)
            return {"ok": True, "envelope_id": body.get("envelope_id"), "devices_queued": 1}

        async def _ensure_subkey(self):
            pass

    http = FakeHttp()
    client = PuffoCoreMessageClient(
        slug="bot-0001",
        device_id="dev_test",
        space_id="sp_test",
        keystore=ks,
        http_client=http,
        message_store=ms,
    )
    # Pre-seed the channel→space cache: ``send_fallback_message``
    # no longer silently falls back to ``self.space_id`` (legacy
    # home space) when the cache misses. Production fills this
    # cache from inbound envelopes + membership events; the smoke
    # test seeds it directly.
    client._channel_space["ch_abc"] = "sp_test"
    class Delegate:
        def __init__(self):
            self.requests = []

        async def send(self, request):
            self.requests.append(request)
            return {"state": "sent", "attempted": True, "envelope_id": "coordinated"}

    delegate = Delegate()
    client.send_delegate = delegate
    result = await client.send_fallback_message(
        "ch_abc", "hello world", root_id=""
    )
    assert result["state"] == "sent"
    assert delegate.requests[0].destination == "ch_abc"
    assert delegate.requests[0].text == "hello world"
    assert http.calls == []
    await ms.close()


@pytest.mark.asyncio
async def test_send_fallback_message_drops_when_channel_space_unknown():
    """Regression for the ``self.space_id`` silent-fallback at
    line 2382 — the agent used to route an unknown-channel reply
    to its legacy home space, which under cross-space deployments
    (or after a kick / cascade that evicted the in-memory cache)
    sent the reply to the wrong place or to a space the agent had
    just been kicked from. Now the path drops the reply with a
    clear log line instead."""
    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    ks, ks_dir, d, kem_kp = _make_keystore()
    db_path = os.path.join(d, "messages.db")
    ms = MessageStore(db_path)
    await ms.open()

    class _NoHttp:
        def __init__(self):
            self.calls = []

        async def get(self, path):
            self.calls.append(("GET", path))
            raise AssertionError(
                f"send_fallback_message must NOT round-trip when "
                f"channel→space is unknown; got GET {path}"
            )

        async def post(self, path, body=None):
            self.calls.append(("POST", path))
            raise AssertionError(
                f"send_fallback_message must NOT POST when "
                f"channel→space is unknown; got POST {path}"
            )

        async def _ensure_subkey(self):
            pass

    http = _NoHttp()
    client = PuffoCoreMessageClient(
        slug="bot-0001",
        device_id="dev_test",
        space_id="sp_legacy_home",  # ← used to be the silent fallback
        keystore=ks,
        http_client=http,
        message_store=ms,
    )
    # Crucially: do NOT pre-seed ``_channel_space``. The pre-fix
    # behaviour would have read ``self.space_id`` and routed to
    # ``/spaces/sp_legacy_home/...``; the post-fix behaviour drops.
    await client.send_fallback_message("ch_unknown", "reply", root_id="")

    # No HTTP at all — the FakeHttp ``raise`` ensures it.
    assert http.calls == []
    await ms.close()


def test_worker_build_and_listener_use_global_runtime_contract():
    import inspect

    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    constructor_params = inspect.signature(PuffoCoreMessageClient).parameters
    listener_params = inspect.signature(PuffoCoreMessageClient.listen).parameters

    assert tuple(listener_params) == ("self", "on_message")
    assert tuple(constructor_params) == (
        "slug",
        "device_id",
        "space_id",
        "keystore",
        "http_client",
        "message_store",
        "operator_slug",
        "auto_accept_space_invitations",
        "auto_accept_dm",
        "workspace",
        "max_inline_chars",
        "segment_chars",
        "agent_created_at",
        "image_edge_px",
        "catchup_stale_hours",
        "agent_id",
        "bridge_client",
    )


@pytest.mark.asyncio
async def test_reminder_sync_reconnect_callbacks_and_shutdown_close_resources(tmp_path):
    """Both transport flavors expose the same signal-only reconnect seam."""
    from puffo_agent.agent.puffo_core_client import PuffoCoreMessageClient

    ks, _ks_dir, _directory, _kem = _make_keystore()

    class Http:
        keyless = False

        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    native_http = Http()
    native = PuffoCoreMessageClient(
        slug="bot-0001", device_id="dev_test", space_id="sp_test",
        keystore=ks, http_client=native_http,
        message_store=MessageStore(tmp_path / "native.db"),
    )
    native_calls: list[str] = []

    async def native_signal():
        native_calls.append("native")

    native.add_connected_callback(native_signal)
    await native._notify_connected_callbacks()
    assert native_calls == ["native"]
    await native.stop()
    assert native_http.close_calls == 1

    class Bridge:
        def __init__(self):
            self.callbacks = []
            self.close_calls = 0

        def add_connected_callback(self, callback):
            self.callbacks.append(callback)

        async def close(self):
            self.close_calls += 1

    bridge = Bridge()
    bridge_http = Http()
    bridge_http.keyless = True
    bridge_client = PuffoCoreMessageClient(
        slug="bot-0001", device_id="dev_test", space_id="sp_test",
        keystore=ks, http_client=bridge_http,
        message_store=MessageStore(tmp_path / "bridge.db"),
        bridge_client=bridge,
    )
    bridge_calls: list[str] = []

    async def bridge_signal():
        bridge_calls.append("bridge")

    bridge_client.add_connected_callback(bridge_signal)
    assert len(bridge.callbacks) == 1
    await bridge.callbacks[0]()
    assert bridge_calls == ["bridge"]
    await bridge_client.stop()
    assert bridge.close_calls == 1
    assert bridge_http.close_calls == 1
