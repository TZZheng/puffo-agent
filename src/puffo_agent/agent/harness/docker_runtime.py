"""Prepare and own the per-agent Docker Codex runtime.

Runs the pinned Codex CLI's ``codex app-server`` inside the per-Agent
container through a bounded ``docker exec -i`` transport consumed by
:class:`~puffo_agent.agent.harness.codex_driver.CodexAppServerDriver`.
The container is the sandbox, so codex runs with
``danger-full-access`` and the Driver auto-approves permissions.

Isolation: the agent's Codex home (``CODEX_HOME``), auth view, config,
skills, and MCPs are synchronized host-side and bind-mounted at the
container's ``/home/agent/.codex``. The operator's ``~/.codex`` is
never mounted; bearer/API values cross only as named ``-e`` variables,
never as argv literals. The owner starts the container on prepare and
stops it with a bounded ``docker stop -t 5`` after the Driver closes —
never ``rm -f`` on a live runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..._proc import no_window_kwargs
from ...mcp.config import puffo_core_mcp_env, write_codex_mcp_config
from ...portal.state import (
    AgentConfig,
    DaemonConfig,
    agent_codex_user_dir,
    agent_dir,
    agent_home_dir,
    read_host_codex_mcp_servers,
    shared_fs_dir,
    sync_host_codex_auth_view,
    sync_host_codex_skills,
)
from ...portal.workspace_layout import ensure_workspace_shared_link
from ..adapters.desired_install import run_spawn_install
from ..adapters.docker_cli import (
    DEFAULT_IMAGE,
    _puffo_agent_pkg_dir,
    _run_cmd,
    container_state,
    ensure_docker_image,
    CONTAINER_LAYOUT_VERSION,
)
from ..cli_bin import resolve_docker_bin
from .driver import RuntimeSpec
from .local_runtime import (
    PreparedLocalRuntime,
    _read_json_object,
    build_codex_gateway_provider,
    compute_session_fingerprint,
)
from ...portal.host_assets import filter_container_mcp_servers

logger = logging.getLogger(__name__)

# Container-local paths for the codex process, its auth/config home, and
# the puffo-core MCP's per-agent state (keystore + memory). ``agent_home``
# host dir is bind-mounted at CODEX_CONTAINER_STATE_DIR, exactly like the
# Claude Docker runtime.
CODEX_CONTAINER_CODEX_HOME = "/home/agent/.codex"
CODEX_CONTAINER_STATE_DIR = "/home/agent/.puffo-agent-state"
# The container is the filesystem boundary. A second bubblewrap sandbox
# cannot create user namespaces under Docker's default seccomp policy, so
# codex always runs fully open inside the container.
CODEX_CONTAINER_SANDBOX = "danger-full-access"

# Distinct from DockerCLIAdapter's plain layout marker: the extra ":codex"
# suffix forces a container rebuild when an agent flips between the Claude
# and Codex Docker harnesses (mount layouts differ), while a layout-version
# bump still rebuilds both.
_CODEX_LAYOUT_MARKER = f"{CONTAINER_LAYOUT_VERSION}:codex"

_DOCKER_LAYOUT_MARKER = ".docker-layout"


def _sanitise_permission_mode(mode: str, agent_id: str) -> str:
    if mode in {"bypassPermissions"}:
        return mode
    if mode:
        logger.warning(
            "agent %s: permission_mode %r is not supported; using "
            "'bypassPermissions'",
            agent_id,
            mode,
        )
    return "bypassPermissions"


class DockerCodexPreparer:
    """Prepare, start, and clean up one per-agent Docker Codex runtime.

    Implements the same :class:`RuntimePreparer` contract as
    :class:`LocalRuntimePreparer` so the durable Runtime Manager assembly
    in ``local_runtime.build_local_runtime_adapter`` is shared unchanged.
    The composition boundary additionally wires ``process_factory`` into
    ``CodexAppServerDriver`` and ``aclose`` as the bounded container stop.
    """

    harness_name = "codex"

    def __init__(self, daemon_cfg: DaemonConfig, agent_cfg: AgentConfig):
        self.daemon_cfg = daemon_cfg
        self.agent_cfg = agent_cfg
        self.agent_id = agent_cfg.id
        self.workspace_dir = agent_cfg.resolve_workspace_dir()
        self.claude_dir = agent_cfg.resolve_claude_dir()
        self.agent_home = agent_home_dir(self.agent_id)
        self.codex_home = agent_codex_user_dir(self.agent_id)
        self.shared_fs_dir = shared_fs_dir()
        self.image = agent_cfg.runtime.docker_image or DEFAULT_IMAGE
        self.container_name = f"puffo-{self.agent_id}"
        self.permission_mode = _sanitise_permission_mode(
            agent_cfg.runtime.permission_mode, self.agent_id
        )
        self.model = agent_cfg.runtime.model or daemon_cfg.openai.model or ""
        self.memory_limit = (
            agent_cfg.runtime.docker_memory_limit or daemon_cfg.docker_memory_limit
        )
        self.memory_reservation = (
            agent_cfg.runtime.docker_memory_reservation
            or daemon_cfg.docker_memory_reservation
        )
        self._docker_bin = "docker"
        self._desired_codex_extras: dict[str, dict] = {}
        self._desired_installed = False
        self._container_stopped = False

    async def prepare(
        self,
        *,
        system_prompt: str,
        persisted_native_session_id: str = "",
        persisted_session_fingerprint: str = "",
    ) -> PreparedLocalRuntime:
        spec = await self.refresh_spec(system_prompt)
        session_fingerprint = self.session_fingerprint(spec)
        legacy_path = self._legacy_session_path()
        legacy_id = self._load_legacy_session_id(legacy_path)
        discarded_persisted_session = bool(
            persisted_native_session_id
            and persisted_session_fingerprint != session_fingerprint
        )
        if persisted_native_session_id and not discarded_persisted_session:
            native_session_id = persisted_native_session_id
            source = "runtime_event_outbox"
        elif legacy_id:
            native_session_id = legacy_id
            source = "legacy_session_file"
        else:
            native_session_id = ""
            source = (
                "fresh_incompatible_persisted_session"
                if discarded_persisted_session
                else "fresh"
            )
        if discarded_persisted_session:
            logger.info(
                "agent %s: starting a fresh codex session because the durable "
                "session fingerprint is missing or incompatible",
                self.agent_id,
            )
        await self.ensure_container()
        return PreparedLocalRuntime(
            harness_name=self.harness_name,
            spec=spec,
            native_session_id=native_session_id,
            migration_source=source,
            legacy_session_path=legacy_path,
            preparer=self,
            session_fingerprint=session_fingerprint,
            discarded_persisted_session=discarded_persisted_session,
        )

    def session_fingerprint(self, spec: RuntimeSpec) -> str:
        """Fingerprint only inputs that make a native session incompatible."""
        from ...portal.runtime_matrix import resolve_effective_provider

        return compute_session_fingerprint(
            agent_cfg=self.agent_cfg,
            harness_name=self.harness_name,
            provider=resolve_effective_provider(
                "cli-docker", self.agent_cfg.runtime.provider
            ),
            spec=spec,
        )

    async def refresh_spec(self, system_prompt: str) -> RuntimeSpec:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.codex_home.mkdir(parents=True, exist_ok=True)
        agents_md = self.codex_home / "AGENTS.md"
        if not agents_md.exists():
            agents_md.write_text("", encoding="utf-8")
        await self._install_desired_once()
        return self._prepare_codex_spec(system_prompt)

    async def _install_desired_once(self) -> None:
        if self._desired_installed:
            return
        self._desired_installed = True
        extras = await run_spawn_install(
            agent_id=self.agent_id,
            agent_home=self.agent_home,
            workspace_dir=self.workspace_dir,
            harness_name=self.harness_name,
            desired_skills=self.agent_cfg.desired_skills,
            desired_mcps=self.agent_cfg.desired_mcps,
            server_url=self.agent_cfg.puffo_core.server_url,
            slug=self.agent_cfg.puffo_core.slug,
            keys_dir=str(agent_dir(self.agent_id) / "keys"),
            containerized=True,
        )
        if extras:
            self._desired_codex_extras = extras

    def _codex_gateway_provider(self) -> dict[str, str] | None:
        return build_codex_gateway_provider(
            model=self.model,
            llm_base_url=self.agent_cfg.runtime.llm_base_url,
            api_key=self.agent_cfg.runtime.api_key,
        )

    def _container_puffo_mcp_env(self) -> dict[str, str] | None:
        pc = self.agent_cfg.puffo_core
        if not pc.is_configured():
            return None
        env = puffo_core_mcp_env(
            slug=pc.slug,
            device_id=pc.device_id,
            server_url=pc.server_url,
            space_id=pc.space_id,
            keystore_dir=f"{CODEX_CONTAINER_STATE_DIR}/keys",
            workspace="/workspace",
            shared_workspace="/workspace/shared",
            agent_id=self.agent_id,
            data_service_url=(
                f"http://host.docker.internal:{self.daemon_cfg.data_service.port}"
            ),
            rpc_url=f"http://host.docker.internal:{self.daemon_cfg.rpc_service.port}",
            runtime_kind="cli-docker",
            harness=self.harness_name,
            memory_dir=f"{CODEX_CONTAINER_STATE_DIR}/memory",
            transport=pc.transport,
        )
        env["CODEX_HOME"] = CODEX_CONTAINER_CODEX_HOME
        env["PYTHONPATH"] = "/opt/puffoagent-pkg"
        return env

    def _prepare_codex_spec(self, system_prompt: str) -> RuntimeSpec:
        host_home = Path.home()
        host_mcps = read_host_codex_mcp_servers(host_home)
        reachable_mcps, unreachable = filter_container_mcp_servers(host_mcps)
        for name, command in unreachable:
            logger.warning(
                "agent %s: host codex MCP %r has host-local path %r that won't "
                "resolve inside the container — SKIPPED (not injected). "
                "Install the binary in the image or bind-mount it, then "
                "re-sync, to make this MCP available.",
                self.agent_id,
                name,
                command,
            )
        extras = dict(self._desired_codex_extras)
        extras.update(reachable_mcps)
        gateway = self._codex_gateway_provider()
        config_kwargs: dict[str, Any] = {
            "extra_servers": extras,
            "inference_level": self.agent_cfg.runtime.inference_level,
            "provider": gateway,
        }
        puffo_env = self._container_puffo_mcp_env()
        if puffo_env:
            config_kwargs.update({
                "command": "python3",
                "args": ["-m", "puffo_agent.mcp.puffo_core_server"],
                "env": puffo_env,
            })
        else:
            logger.warning(
                "agent %s: cli-docker codex MCP tools unavailable — "
                "puffo_core is not configured. populate `puffo_core:` in "
                "agent.yml so send_message / list_channels_in_all_spaces / "
                "etc. show up under codex's tool surface.",
                self.agent_id,
            )
        write_codex_mcp_config(self.codex_home / "config.toml", **config_kwargs)

        # Only codex-relevant values; never the operator's env wholesale.
        environment: dict[str, str] = {
            "HOME": "/home/agent",
            "CODEX_HOME": CODEX_CONTAINER_CODEX_HOME,
        }
        if gateway:
            environment["OPENAI_API_KEY"] = self.agent_cfg.runtime.api_key
        else:
            auth_mode = sync_host_codex_auth_view(host_home, self.codex_home)
            if auth_mode == "no-host-file":
                raise RuntimeError(
                    f"agent {self.agent_id!r}: codex needs auth; run "
                    "`codex login` on the host or configure "
                    "runtime.llm_base_url and runtime.api_key"
                )
            logger.info(
                "agent %s: refreshed Codex credential view (%s)",
                self.agent_id,
                auth_mode,
            )
        skill_count = sync_host_codex_skills(host_home, self.codex_home)
        if skill_count:
            logger.info(
                "agent %s: synced %d host codex skill(s) into %s",
                self.agent_id,
                skill_count,
                self.codex_home,
            )

        from ...portal.control.context_telemetry import configured_compact_pct

        compact_pct = configured_compact_pct(
            "codex", self.agent_cfg.env_overrides
        )
        return RuntimeSpec(
            workspace_dir="/workspace",
            model=self.model,
            system_prompt=system_prompt,
            environment=environment,
            permission_mode=self.permission_mode,
            sandbox=CODEX_CONTAINER_SANDBOX,
            task_timeout_seconds=self.agent_cfg.runtime.task_timeout_seconds,
            auto_compact_threshold_pct=compact_pct,
        )

    def _legacy_session_path(self) -> Path:
        return self.codex_home / "codex_session.json"

    def _load_legacy_session_id(self, path: Path) -> str:
        document = _read_json_object(path)
        persisted_sandbox = str(
            document.get("sandbox") or "danger-full-access"
        )
        if persisted_sandbox != CODEX_CONTAINER_SANDBOX:
            logger.info(
                "agent %s: not importing legacy Docker Codex session because "
                "sandbox changed from %s to %s",
                self.agent_id,
                persisted_sandbox,
                CODEX_CONTAINER_SANDBOX,
            )
            return ""
        return str(document.get("conversation_id") or "").strip()

    # ── Container lifecycle ────────────────────────────────────────────

    async def ensure_container(self) -> None:
        """Start the per-agent container, recreating it when the layout
        marker is stale (image tag bump or a harness flip)."""
        self._require_docker()
        state = await container_state(self._docker_bin, self.container_name)
        if state is None:
            raise RuntimeError(
                f"could not inspect Docker container {self.container_name!r}; "
                "refusing to create or replace it while Docker is unavailable"
            )
        layout_current = await self._layout_is_current()
        if state != "" and not layout_current:
            logger.warning(
                "agent %s: recreating Docker Codex container %r "
                "(layout=%s)",
                self.agent_id,
                self.container_name,
                layout_current,
            )
            if state == "running":
                # A live prior container must be stopped through the owned
                # bounded lifecycle before removal. A failed stop aborts the
                # replacement instead of falling through to forced removal.
                await _run_cmd(
                    [self._docker_bin, "stop", "-t", "5", self.container_name],
                )
            await _run_cmd(
                [self._docker_bin, "rm", self.container_name],
            )
            state = ""
        if state == "running":
            logger.info(
                "agent %s: reusing running Docker Codex container %r",
                self.agent_id,
                self.container_name,
            )
        elif state in ("exited", "created", "dead"):
            logger.info(
                "agent %s: starting existing container %r (was %s)",
                self.agent_id,
                self.container_name,
                state,
            )
            await _run_cmd([self._docker_bin, "start", self.container_name])
        elif state == "paused":
            logger.info(
                "agent %s: unpausing container %r",
                self.agent_id,
                self.container_name,
            )
            await _run_cmd([self._docker_bin, "unpause", self.container_name])
        elif state == "":
            await ensure_docker_image(
                self._docker_bin, self.image, agent_id=self.agent_id
            )
            await self._start_container()
        else:
            raise RuntimeError(
                f"Docker container {self.container_name!r} is in transient "
                f"state {state!r}; refusing to replace it"
            )
        self._write_layout_marker()

    def _require_docker(self) -> None:
        docker_bin = resolve_docker_bin()
        if docker_bin is None:
            raise RuntimeError(
                "docker binary not found. Tried $PUFFO_DOCKER_BIN, $PATH, "
                "the persistent user PATH, and known Docker Desktop install "
                "locations. Install Docker Desktop (Windows/macOS) or "
                "docker-ce (Linux) to use runtime kind 'cli-docker'."
            )
        self._docker_bin = docker_bin

    async def _layout_is_current(self) -> bool:
        marker = self.agent_home / _DOCKER_LAYOUT_MARKER
        try:
            return marker.read_text(encoding="utf-8").strip() == _CODEX_LAYOUT_MARKER
        except OSError:
            return False

    def _write_layout_marker(self) -> None:
        marker = self.agent_home / _DOCKER_LAYOUT_MARKER
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(_CODEX_LAYOUT_MARKER + "\n", encoding="utf-8")
        except OSError:
            pass

    async def _start_container(self) -> None:
        ensure_workspace_shared_link(self.workspace_dir, self.shared_fs_dir)
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.agent_home.mkdir(parents=True, exist_ok=True)
        command = [
            self._docker_bin,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-e",
            f"PUFFO_AGENT_ID={self.agent_id}",
            "-v",
            f"{self.workspace_dir}:/workspace",
            # Agent-scoped Codex home only — never the operator's ~/.codex.
            "-v",
            f"{self.codex_home}:{CODEX_CONTAINER_CODEX_HOME}",
            # Per-agent keystore + memory for the puffo-core MCP.
            "-v",
            f"{self.agent_home}:{CODEX_CONTAINER_STATE_DIR}",
            "-v",
            f"{self.shared_fs_dir}:/workspace/shared",
            # Compatibility for existing sessions and user-authored paths.
            "-v",
            f"{self.shared_fs_dir}:/workspace/.shared",
            "-v",
            f"{_puffo_agent_pkg_dir()}:/opt/puffoagent-pkg:ro",
            "--init",
        ]
        if self.memory_limit:
            command.extend(["--memory", self.memory_limit])
        if self.memory_reservation:
            command.extend(["--memory-reservation", self.memory_reservation])
        command.append(self.image)
        rc, _, stderr = await _run_cmd(command, check=False)
        if rc != 0:
            raise RuntimeError(
                f"docker run failed for {self.container_name}: "
                f"{stderr.decode('utf-8', errors='replace').strip()[:500]}"
            )

    # ── Driver transport ───────────────────────────────────────────────

    async def _exec_process(self, spec: RuntimeSpec) -> asyncio.subprocess.Process:
        """Spawn the bounded ``docker exec -i`` transport for the Driver.

        Only codex-relevant values are forwarded with explicit ``-e`` flags.
        The bearer/API key crosses by name (value read from the subprocess
        environment), never as an argv literal.
        """
        command = [self._docker_bin, "exec", "-i"]
        for key, value in spec.environment.items():
            if key == "OPENAI_API_KEY":
                command.extend(["-e", key])
            else:
                command.extend(["-e", f"{key}={value}"])
        command.extend([self.container_name, "codex", "app-server"])
        env = dict(os.environ)
        api_key = spec.environment.get("OPENAI_API_KEY")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        return await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # One provider frame can carry a large tool result; the default
            # 64 KiB stream limit would terminate the reader mid-session.
            limit=16 * 1024 * 1024,
            **no_window_kwargs(),
        )

    @property
    def process_factory(self) -> Callable[[RuntimeSpec], Any]:
        """Driver-compatible transport factory: ``docker exec -i`` spawn.

        Exposes the bounded exec transport so the composition boundary can
        inject it into ``CodexAppServerDriver`` without reaching into a
        private method. The Driver awaits the returned coroutine.
        """
        return self._exec_process

    # ── Bounded shutdown ───────────────────────────────────────────────

    async def aclose(self) -> None:
        """Stop the container once the Driver transport has terminated."""
        if self._container_stopped:
            return
        self._container_stopped = True
        # ``docker stop`` (not ``rm -f``) preserves the container's fs —
        # codex home, sessions, config — so the next start resumes cleanly.
        # ``-t 5`` bounds the SIGTERM grace inside Worker.stop's 30s budget.
        await _run_cmd(
            [self._docker_bin, "stop", "-t", "5", self.container_name],
            check=False,
        )


__all__ = [
    "DockerCodexPreparer",
    "CODEX_CONTAINER_CODEX_HOME",
    "CODEX_CONTAINER_STATE_DIR",
    "CODEX_CONTAINER_SANDBOX",
]
