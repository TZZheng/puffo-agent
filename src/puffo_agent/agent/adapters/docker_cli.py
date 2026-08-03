"""Docker CLI adapter for Claude Code and Codex.

Each agent runs its selected CLI in a dedicated container. Claude Code
uses its stream-json protocol; Codex uses its app-server JSON-RPC
protocol. The container is the sandbox, so Claude Code runs with
``--dangerously-skip-permissions``.

Per-agent ``.claude`` and ``.codex`` directories are bind-mounted into
the container. The daemon owns credential refresh and writes sanitized
credential views into those directories, matching ``cli-local``.

A second bind-mount exposes ``~/.puffo-agent/shared/`` at
``/workspace/.shared`` so all agents on this host can cooperate at
the filesystem level.

Lifecycle:
  - container: one per agent (``puffo-<id>``), started lazily,
    ``docker stop`` on ``aclose()``.
  - harness: one long-lived ClaudeSession or CodexSession subprocess.
  - session ids persist on the host so daemon/container restarts resume.

Image: bundled inline as a Dockerfile string, built on first use.
Users can override via ``runtime.docker_image`` to skip the build.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from ...mcp.config import (
    INFERENCE_LEVELS,
    write_cli_mcp_config,
    write_codex_mcp_config,
)
from ...portal.state import (
    filter_container_mcp_servers,
    read_host_codex_mcp_servers,
    seed_claude_home,
    sync_host_claude_code_auth_view,
    sync_host_codex_auth_view,
    sync_host_enabled_plugins,
    sync_host_mcp_servers,
    sync_host_skills,
)
from .base import Adapter, TurnContext, TurnResult
from .cli_session import AuditLog, ClaudeSession
from .codex_session import CodexSession


logger = logging.getLogger(__name__)


def _puffo_agent_pkg_dir() -> Path:
    """Host-side puffo_agent package import root, bind-mounted
    read-only into cli-docker containers at /opt/puffoagent-pkg so
    the in-container puffo-core MCP server can ``import puffo_agent.*``.
    """
    import puffo_agent
    return Path(puffo_agent.__file__).resolve().parent.parent


# Bump on Dockerfile changes so existing hosts rebuild without manual
# image-tag pruning. ``_ensure_image`` only builds when the tag is
# missing locally.
DEFAULT_IMAGE = "puffo/agent-runtime:v12"
CONTAINER_LAYOUT_VERSION = "12"

# Pinned Claude Code CLI version baked into the image. Floating would
# let an upstream release shift the stream-json protocol or
# ``--permission-mode`` semantics under us; bump deliberately after
# verification.
CLAUDE_CODE_NPM_VERSION = "2.1.117"

# Pinned Codex CLI version. Keep this aligned with the app-server
# protocol exercised by ``CodexSession``.
CODEX_NPM_VERSION = "0.145.0"

# Kept minimal. The claude CLI refuses --dangerously-skip-permissions
# as root, so we create a non-root ``agent`` user. UID doesn't need
# to match the host: Docker Desktop's VFS maps bind-mount perms.
#
# PID 1 tails the host-written audit log (via the workspace bind-
# mount) so ``docker logs <container>`` streams turn events. Without
# it the container would be a black box since the claude subprocess
# is spawned via docker-exec and its stdout returns to the host
# adapter, not container PID 1.
DOCKERFILE = """\
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
        git curl ca-certificates jq ripgrep \\
        python3 python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g \\
        @anthropic-ai/claude-code@__CLAUDE_CODE_VERSION__ \\
        @openai/codex@__CODEX_VERSION__

# Puffo MCP tools server deps. ``--break-system-packages`` is
# required on Debian bookworm (PEP 668); acceptable since the
# container is single-purpose and disposable. ``uv`` ships ``uvx``
# (Python counterpart of ``npx``) so agents can register stdio MCPs
# without per-server pip/npm install.
RUN pip3 install --break-system-packages --no-cache-dir \\
        "mcp>=1.0" "aiohttp>=3.9" "uv>=0.5" \\
        "cryptography>=43" "pyhpke>=0.6" "aiosqlite>=0.20" "pyyaml>=6.0"

RUN useradd -m -u 2000 -s /bin/bash agent
USER agent
WORKDIR /workspace

# GNU ``tail -F`` relies on inotify, and inotify doesn't propagate
# through Docker Desktop's host bind-mount on Windows / macOS.
# Instead we poll file size each second and emit newly-appended
# bytes to stdout for ``docker logs``. Start from EOF so we don't
# re-dump history on every restart.
CMD ["sh", "-c", "set -eu; mkdir -p /workspace/.puffo-agent; touch /workspace/.puffo-agent/audit.log; echo \\"[$(date -u +%FT%TZ)] puffo agent=${PUFFO_AGENT_ID:-unknown} container starting; polling /workspace/.puffo-agent/audit.log every 1s\\"; last=$(stat -c%s /workspace/.puffo-agent/audit.log 2>/dev/null || echo 0); while :; do size=$(stat -c%s /workspace/.puffo-agent/audit.log 2>/dev/null || echo 0); if [ \\"$size\\" -gt \\"$last\\" ]; then tail -c +$((last + 1)) /workspace/.puffo-agent/audit.log; last=$size; elif [ \\"$size\\" -lt \\"$last\\" ]; then last=0; fi; sleep 1; done"]
""".replace(
    "__CLAUDE_CODE_VERSION__", CLAUDE_CODE_NPM_VERSION,
).replace(
    "__CODEX_VERSION__", CODEX_NPM_VERSION,
)


class DockerCLIAdapter(Adapter):
    def __init__(
        self,
        agent_id: str,
        model: str,
        image: str,
        workspace_dir: str,
        claude_dir: str,
        session_file: str,
        agent_home_dir: str,
        shared_fs_dir: str,
        owner_username: str = "",
        permission_mode: str = "bypassPermissions",
        sandbox: str = "danger-full-access",
        inference_level: str = "",
        task_timeout_seconds: float = 1800.0,
        harness=None,
        memory_limit: str = "",
        memory_reservation: str = "",
        desired_skills: list[str] | None = None,
        desired_mcps: list[str] | None = None,
        puffo_core_server_url: str = "",
        puffo_core_slug: str = "",
        puffo_core_keys_dir: str = "",
    ):
        self.agent_id = agent_id
        self.model = model
        self.image = image or DEFAULT_IMAGE
        self.workspace_dir = workspace_dir
        self.claude_dir = claude_dir
        self.session_file = Path(session_file)
        self.container_name = f"puffo-{agent_id}"
        # Per-agent harness state is mounted into the container without
        # replacing its complete home directory.
        self.agent_home_dir = Path(agent_home_dir)
        self.claude_home_src = self.agent_home_dir / ".claude"
        # Cross-agent cooperation dir; same mount in every container
        # on this host — intentional escape hatch from per-agent
        # isolation.
        self.shared_fs_dir = Path(shared_fs_dir)
        self.owner_username = owner_username
        self.permission_mode = permission_mode
        self.sandbox = sandbox
        self.inference_level = inference_level
        self.task_timeout_seconds = task_timeout_seconds
        # Optional cgroup caps. ``--memory`` is a hard ceiling that
        # OOM-kills processes in this container only; ``--memory-
        # reservation`` is a soft floor. Bound a runaway claude so it
        # doesn't drain the VM and trigger ENOMEM on neighbours' small
        # reads. Empty = no flag = Docker default unbounded.
        self.memory_limit = memory_limit
        self.memory_reservation = memory_reservation
        # Which agent engine runs inside the container.
        if harness is None:
            from ..harness import ClaudeCodeHarness
            harness = ClaudeCodeHarness()
        if harness.name() not in {"claude-code", "codex"}:
            raise RuntimeError(
                f"agent {agent_id!r}: cli-docker supports only "
                "claude-code and codex harnesses"
            )
        self.harness = harness
        self.desired_skills = list(desired_skills or [])
        self.desired_mcps = list(desired_mcps or [])
        self.puffo_core_server_url = puffo_core_server_url
        self.puffo_core_slug = puffo_core_slug
        self.puffo_core_keys_dir = puffo_core_keys_dir
        self._desired_codex_extras: dict[str, dict] = {}
        self._desired_installed = False
        self._started_lock = asyncio.Lock()
        self._started = False
        self._session: ClaudeSession | None = None
        self._codex_session: CodexSession | None = None
        # Set post-construction by worker.py. When non-None, claude-
        # code is routed at ``puffo_core_server``. Values must be
        # CONTAINER-local paths since the MCP subprocess runs inside
        # the container.
        self.puffo_core_mcp_env: dict[str, str] | None = None

    async def run_turn(self, ctx: TurnContext) -> TurnResult:
        await self._ensure_started()
        user_message = ctx.messages[-1]["content"] if ctx.messages else ""
        if self.harness.name() == "codex":
            return await self._ensure_codex_session().run_turn(
                user_message, ctx.system_prompt,
            )
        session = self._ensure_session()
        return await session.run_turn(user_message, ctx.system_prompt)

    async def run_retry_turn(
        self,
        kick_text: str,
        fallback_user_message: str,
        ctx: TurnContext,
    ) -> TurnResult:
        await self._ensure_started()
        if self.harness.name() == "codex":
            return await self._ensure_codex_session().run_turn(
                fallback_user_message, ctx.system_prompt,
            )
        session = self._ensure_session()
        return await session.run_retry_turn(
            kick_text, fallback_user_message, ctx.system_prompt,
        )

    async def warm(self, system_prompt: str) -> None:
        """Start the container and resume a persisted harness session."""
        await self._ensure_started()
        if self.harness.name() == "codex":
            session = self._ensure_codex_session()
            if not session.has_persisted_session():
                logger.info(
                    "agent %s: no persisted codex conversation; deferring "
                    "spawn until first message", self.agent_id,
                )
                return
            await session.warm(system_prompt)
            return
        session = self._ensure_session()
        if not session.has_persisted_session():
            logger.info(
                "agent %s: no persisted session; deferring claude spawn until first message",
                self.agent_id,
            )
            return
        await session.warm(system_prompt)

    async def reload(
        self, new_system_prompt: str, *, with_session: bool = False,
    ) -> None:
        """Close the harness process so the next turn reloads config."""
        codex_session_file = (
            self._codex_session.session_file
            if self._codex_session is not None else self.codex_home / "codex_session.json"
        )
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        if self._codex_session is not None:
            await self._codex_session.aclose()
            self._codex_session = None
        if self.harness.name() == "codex" and self._started:
            self._prepare_codex_config(Path.home())
        if with_session:
            for path in (self.session_file, codex_session_file):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(
                        "agent %s: couldn't unlink session file %s: %s",
                        self.agent_id, path, exc,
                    )

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        if self._codex_session is not None:
            await self._codex_session.aclose()
            self._codex_session = None
        if not self._started:
            return
        # ``docker stop`` (not ``rm -f``) preserves the container's
        # fs — workspace, claude session files, MCP config — so the
        # next start resumes ``--resume <session_id>`` cleanly.
        # ``-t 5`` shortens docker's 10s SIGTERM grace; stays within
        # Worker.stop's 30s asyncio.wait_for even on slow Windows.
        await _run_cmd(
            ["docker", "stop", "-t", "5", self.container_name], check=False,
        )
        self._started = False

    async def health_probe(self) -> bool:
        if self._codex_session is not None:
            return await self._codex_session.health_probe()
        return True

    @property
    def codex_home(self) -> Path:
        return self.agent_home_dir / ".codex"

    def _ensure_codex_session(self) -> CodexSession:
        if self._codex_session is not None:
            return self._codex_session
        self._codex_session = CodexSession(
            agent_id=self.agent_id,
            session_file=self.codex_home / "codex_session.json",
            argv=[
                "docker", "exec", "-i",
                "-e", "CODEX_HOME=/home/agent/.codex",
                self.container_name, "codex", "app-server",
            ],
            cwd=None,
            permission_mode=self.permission_mode,
            sandbox=self.sandbox,
            model=self.model,
            task_timeout_seconds=self.task_timeout_seconds,
            audit=AuditLog(
                Path(self.workspace_dir) / ".puffo-agent" / "audit.log",
                self.agent_id,
            ),
        )
        return self._codex_session

    def _ensure_session(self) -> ClaudeSession:
        if self._session is not None:
            return self._session
        extra = self._prepare_mcp_args()
        self._session = ClaudeSession(
            agent_id=self.agent_id,
            session_file=self.session_file,
            build_command=self._build_command,
            # cwd is WORKDIR /workspace inside the container.
            cwd=None,
            # Host-side write; the workspace bind-mount delivers it
            # to the container's tail loop and ``docker logs``.
            audit=AuditLog(
                Path(self.workspace_dir) / ".puffo-agent" / "audit.log",
                self.agent_id,
            ),
            extra_args=extra,
            model=self.model,
        )
        return self._session

    def _build_command(
        self,
        extra_args: list[str],
        env_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        cmd: list[str] = ["docker", "exec", "-i"]
        # ``env_overrides`` flows in before the container name so
        # docker treats each ``-e KEY=VALUE`` as an exec flag.
        for key, value in (env_overrides or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([
            self.container_name,
            "claude", "--dangerously-skip-permissions",
        ])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.inference_level:
            if self.inference_level in INFERENCE_LEVELS:
                cmd.extend(["--effort", self.inference_level])
            else:
                logger.warning(
                    "agent %s: ignoring inference_level %r for claude-code "
                    "(expected one of %s)",
                    self.agent_id, self.inference_level,
                    ", ".join(INFERENCE_LEVELS),
                )
        cmd.extend(extra_args)
        return cmd

    def _prepare_mcp_args(self) -> list[str]:
        """Write the per-agent MCP config into the workspace and
        return the corresponding claude CLI flags. No
        ``--permission-prompt-tool`` — the container is the sandbox.
        """
        config_host = Path(self.workspace_dir) / ".puffo-agent" / "mcp-config.json"

        # Path values must be CONTAINER-local — override whatever the
        # worker put in the env dict from the host side.
        env = self._container_puffo_mcp_env()
        if env is not None:
            write_cli_mcp_config(
                config_host,
                command="python3",
                args=["-m", "puffo_agent.mcp.puffo_core_server"],
                env=env,
            )
            return ["--mcp-config", "/workspace/.puffo-agent/mcp-config.json"]

        logger.warning(
            "agent %s: cli-docker MCP tools unavailable — puffo_core is "
            "not configured. populate `puffo_core:` in agent.yml so "
            "send_message / list_channels_in_all_spaces / etc. show up under "
            "claude-code's tool surface.",
            self.agent_id,
        )
        return []

    async def _puffo_pkg_mount_is_current(self) -> bool:
        """``True`` iff the existing container's
        ``/opt/puffoagent-pkg`` bind mount still resolves to a
        directory containing the ``puffo_agent`` package.

        Implemented as a ``docker exec test -f`` rather than
        comparing ``docker inspect``'s Mount.Source against
        ``_puffo_agent_pkg_dir()`` because Docker Desktop on Windows
        rewrites the source path (``/run/desktop/mnt/host/c/...``)
        and a literal string compare wouldn't survive that. The
        in-container probe is authoritative: if claude-code's MCP
        subprocess can ``import puffo_agent`` from the bind mount,
        ``__init__.py`` must be visible — and if it isn't, the
        subprocess will crash and every puffo MCP tool will surface
        as "No such tool available".
        """
        rc, _, _ = await _run_cmd(
            [
                "docker", "exec", self.container_name,
                "test", "-f",
                "/opt/puffoagent-pkg/puffo_agent/__init__.py",
            ],
            check=False,
        )
        return rc == 0

    async def _container_harness_is_current(self) -> bool:
        harness = self.harness.name()
        command = "claude" if harness == "claude-code" else "codex"
        checks = [f"command -v {command} >/dev/null"]
        if harness == "codex":
            checks.append("test -f /home/agent/.codex/config.toml")
        rc, _, _ = await _run_cmd(
            ["docker", "exec", self.container_name, "sh", "-c", " && ".join(checks)],
            check=False,
        )
        return rc == 0

    async def _container_state(self) -> str:
        """Docker-reported container State.Status (``running``,
        ``exited``, ``paused``, ``created``, ``dead``), or ``""``
        when the container doesn't exist.
        """
        rc, out, _ = await _run_cmd(
            [
                "docker", "inspect",
                "-f", "{{.State.Status}}",
                self.container_name,
            ],
            check=False,
        )
        if rc != 0:
            return ""
        return out.decode("utf-8", errors="replace").strip()

    async def _install_desired(self) -> None:
        if self._desired_installed:
            return
        if not self.desired_skills and not self.desired_mcps:
            self._desired_installed = True
            return
        self._desired_installed = True
        from .desired_install import run_spawn_install
        codex_extras = await run_spawn_install(
            agent_id=self.agent_id,
            agent_home=self.agent_home_dir,
            workspace_dir=Path(self.workspace_dir),
            harness_name=self.harness.name(),
            desired_skills=self.desired_skills,
            desired_mcps=self.desired_mcps,
            server_url=self.puffo_core_server_url,
            slug=self.puffo_core_slug,
            keys_dir=self.puffo_core_keys_dir,
        )
        if codex_extras:
            self._desired_codex_extras = codex_extras

    def _container_puffo_mcp_env(self) -> dict[str, str] | None:
        if self.puffo_core_mcp_env is None:
            return None
        env = dict(self.puffo_core_mcp_env)
        env["PUFFO_CORE_KEYSTORE_DIR"] = "/home/agent/.puffo-agent-state/keys"
        env["PUFFO_WORKSPACE"] = "/workspace"
        env["PUFFO_RUNTIME_KIND"] = "cli-docker"
        env["PUFFO_HARNESS"] = self.harness.name()
        env["PYTHONPATH"] = "/opt/puffoagent-pkg"
        return env

    def _prepare_codex_config(self, host_home: Path) -> None:
        self.codex_home.mkdir(parents=True, exist_ok=True)
        agents_md = self.codex_home / "AGENTS.md"
        if not agents_md.exists():
            agents_md.write_text("", encoding="utf-8")

        auth_mode = sync_host_codex_auth_view(host_home, self.codex_home)
        if auth_mode == "no-host-file":
            raise RuntimeError(
                f"agent {self.agent_id!r}: codex needs auth; run `codex login` "
                "on the host so ~/.codex/auth.json exists"
            )

        host_mcps, unreachable = filter_container_mcp_servers(
            read_host_codex_mcp_servers(host_home),
        )
        for name, command in unreachable:
            logger.warning(
                "agent %s: skipping host Codex MCP %r because %r does not "
                "resolve inside the container", self.agent_id, name, command,
            )
        extras = dict(self._desired_codex_extras)
        extras.update(host_mcps)
        env = self._container_puffo_mcp_env()
        write_codex_mcp_config(
            self.codex_home / "config.toml",
            command="python3" if env is not None else None,
            args=["-m", "puffo_agent.mcp.puffo_core_server"] if env is not None else None,
            env=env,
            extra_servers=extras,
            inference_level=self.inference_level,
        )

    async def _ensure_started(self) -> None:
        async with self._started_lock:
            if self._started:
                return
            if shutil.which("docker") is None:
                raise RuntimeError(
                    "docker binary not found on PATH. install Docker Desktop "
                    "(Windows/macOS) or docker-ce (Linux) to use runtime "
                    "kind 'cli-docker'."
                )
            # Keep both harness homes ready so switching harnesses does not
            # require rebuilding the container.
            host_home = Path.home()
            seeded = seed_claude_home(host_home, self.agent_home_dir)
            if seeded:
                logger.info(
                    "agent %s: seeded per-agent virtual $HOME at %s from %s",
                    self.agent_id, self.agent_home_dir, host_home,
                )
            claude_auth_mode = sync_host_claude_code_auth_view(
                host_home, self.agent_home_dir,
            )
            logger.info(
                "agent %s: wrote host Claude credential view (%s)",
                self.agent_id, claude_auth_mode,
            )
            # One-way sync of host skills + MCP registrations into
            # the per-agent home. Runs every start so host edits
            # propagate without daemon restart.
            skill_count = sync_host_skills(host_home, self.agent_home_dir)
            if skill_count:
                logger.info(
                    "agent %s: synced %d host skill(s) into %s",
                    self.agent_id, skill_count,
                    self.agent_home_dir / ".claude" / "skills",
                )
            merged_mcp, unreachable = sync_host_mcp_servers(
                host_home, self.agent_home_dir,
            )
            if merged_mcp:
                logger.info(
                    "agent %s: merged %d host MCP server registration(s) "
                    "into per-agent .claude.json", self.agent_id, merged_mcp,
                )
            for name, cmd in unreachable:
                logger.warning(
                    "agent %s: host MCP %r has host-local path %r that won't "
                    "resolve inside the container — SKIPPED (not injected). "
                    "Install the binary in the image or bind-mount it, then "
                    "re-sync, to make this MCP available.",
                    self.agent_id, name, cmd,
                )
            await self._install_desired()
            if self.harness.name() == "codex":
                self._prepare_codex_config(host_home)
            # Plugins: the actual plugin tree is bind-mounted read-
            # only into the container by ``_start_container``
            # (see the ``-v {host_plugins}:/home/agent/.claude/plugins:ro``
            # line). Here we just propagate the ``enabledPlugins``
            # array from host settings.json so the container's Claude
            # knows which plugin names to load. The image bakes in
            # node/npm/python so most ``npx`` / ``uvx`` plugin
            # commands resolve naturally; native-binary plugins (e.g.
            # one that shells out to a host-only path) will still
            # fail and that surfaces as a runtime error.
            enabled_count = sync_host_enabled_plugins(
                host_home, self.agent_home_dir,
            )
            if enabled_count:
                logger.info(
                    "agent %s: propagated %d enabledPlugins entry/entries "
                    "from host settings.json", self.agent_id, enabled_count,
                )

            if (
                self.harness.name() == "claude-code"
                and not (self.agent_home_dir / ".claude" / ".credentials.json").exists()
            ):
                logger.warning(
                    "agent %s: agent has no %s — run `claude login` on the "
                    "host, then restart the agent. First turn will fail "
                    "with an auth error otherwise.",
                    self.agent_id,
                    self.agent_home_dir / ".claude" / ".credentials.json",
                )
            # Reuse the container left behind by a prior daemon run
            # (``aclose`` does ``docker stop``, not ``rm``) so
            # ``--resume <session_id>`` reattaches cleanly on the
            # next turn instead of paying container boot + image
            # pull every restart.
            state = await self._container_state()
            existed = state != ""
            if state == "running":
                logger.info(
                    "agent %s: reusing running container %r",
                    self.agent_id, self.container_name,
                )
            elif state in ("exited", "created", "dead"):
                logger.info(
                    "agent %s: starting existing container %r (was %s)",
                    self.agent_id, self.container_name, state,
                )
                await _run_cmd(["docker", "start", self.container_name])
            elif state == "paused":
                logger.info(
                    "agent %s: unpausing container %r",
                    self.agent_id, self.container_name,
                )
                await _run_cmd(["docker", "unpause", self.container_name])
            else:
                # state == "" — no container with this name.
                await self._ensure_image()
                await self._start_container()

            layout_marker = self.agent_home_dir / ".docker-layout"
            try:
                layout_current = (
                    layout_marker.read_text(encoding="utf-8").strip()
                    == CONTAINER_LAYOUT_VERSION
                )
            except OSError:
                layout_current = False
            package_current = (
                await self._puffo_pkg_mount_is_current() if existed else True
            )
            harness_current = (
                await self._container_harness_is_current() if existed else True
            )
            if existed and not (layout_current and package_current and harness_current):
                logger.warning(
                    "agent %s: recreating stale container %r "
                    "(layout=%s package=%s harness=%s)",
                    self.agent_id, self.container_name,
                    layout_current, package_current, harness_current,
                )
                await _run_cmd(
                    ["docker", "rm", "-f", self.container_name],
                    check=False,
                )
                await self._ensure_image()
                await self._start_container()
            if not await self._container_harness_is_current():
                raise RuntimeError(
                    f"docker image {self.image!r} does not provide a working "
                    f"{self.harness.name()} harness"
                )
            layout_marker.write_text(
                CONTAINER_LAYOUT_VERSION + "\n", encoding="utf-8",
            )
            self._started = True

    async def _ensure_image(self) -> None:
        if await _image_exists_locally(self.image):
            return
        if self.image != DEFAULT_IMAGE:
            raise RuntimeError(
                f"docker image {self.image!r} not found locally. "
                f"pull it (`docker pull {self.image}`) or clear "
                "runtime.docker_image to use the bundled default."
            )
        # Daemon-wide lock — concurrent ``docker build -t <tag>``
        # races in BuildKit's exporter and the loser crashes with
        # "image already exists". First wins; others wait and re-check.
        async with _BUILD_LOCK:
            if await _image_exists_locally(self.image):
                logger.info(
                    "agent %s: image %s was built by another worker "
                    "during our wait — skipping rebuild",
                    self.agent_id, self.image,
                )
                return
            logger.info(
                "agent %s: building docker image %s (first use — this may take a few minutes)",
                self.agent_id, self.image,
            )
            await self._build_image()

    async def _build_image(self) -> None:
        from ..._proc import no_window_kwargs
        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", self.image, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **no_window_kwargs(),
        )
        stdout, _ = await proc.communicate(DOCKERFILE.encode())
        if proc.returncode != 0:
            tail = stdout.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"docker build failed:\n{tail}")
        logger.info("agent %s: docker image %s built", self.agent_id, self.image)

    async def _start_container(self) -> None:
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        # Pre-create every bind-mount source as a real dir/file so
        # Docker doesn't auto-create one owned by root that the
        # non-root container user can't write to.
        self.agent_home_dir.mkdir(parents=True, exist_ok=True)
        (self.agent_home_dir / ".claude").mkdir(parents=True, exist_ok=True)
        self.codex_home.mkdir(parents=True, exist_ok=True)
        # .claude.json is a FILE (not a dir) — touch so the
        # bind-mount target is a file, not a dir.
        agent_claude_json = self.agent_home_dir / ".claude.json"
        agent_claude_json.touch(exist_ok=True)
        self.shared_fs_dir.mkdir(parents=True, exist_ok=True)

        # Bind-mounts per agent:
        #   1. workspace            — project root + cwd
        #   2. .claude dir          — per-agent identity
        #   3. .codex dir           — per-agent Codex identity/config
        #   4. .claude.json         — per-agent Claude CLI config
        #   5. shared_fs            — cross-agent cooperation
        #   6. puffoagent pkg       — host package for in-container imports
        #   7. .puffo-agent-state   — keystore + message DB
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-e", f"PUFFO_AGENT_ID={self.agent_id}",
            "-v", f"{self.workspace_dir}:/workspace",
            "-v", f"{self.claude_home_src}:/home/agent/.claude",
            "-v", f"{self.codex_home}:/home/agent/.codex",
            # Sibling .claude.json — without this it lands on the
            # container's ephemeral fs and is lost on restart.
            "-v", f"{agent_claude_json}:/home/agent/.claude.json",
            "-v", f"{self.shared_fs_dir}:/workspace/.shared",
            "-v", f"{_puffo_agent_pkg_dir()}:/opt/puffoagent-pkg:ro",
            # RW because subkey rotation rewrites <slug>.session.json.
            # Mounting :ro surfaced as [Errno 30] from MCP tool calls
            # past the subkey TTL. Whole agent_home_dir is mounted
            # rather than individual files because SQLite WAL files
            # (-wal, -shm) sit alongside the .db.
            "-v", f"{self.agent_home_dir}:/home/agent/.puffo-agent-state",
            "--init",  # reap zombies from claude's child processes
        ]
        # Host Claude Code plugins ride in via a read-only nested
        # bind-mount on top of the .claude dir. Plugin code (the
        # marketplace clones, cache, installed_plugins.json) lives in
        # ``~/.claude/plugins/`` and can be GB-scale; an extra mount
        # is cheaper than copying every worker start, and the agent
        # picks up host installs live. The image bakes in node/npm/
        # python+uv so most plugin commands (``npx``/``uvx``)
        # resolve. Sibling ``sync_host_enabled_plugins`` propagates
        # the ``enabledPlugins`` array via the per-agent
        # ``.claude/settings.json`` that lives under the existing
        # ``claude_home_src`` mount. Skipped when the host has no
        # plugins dir.
        host_plugins = Path.home() / ".claude" / "plugins"
        if host_plugins.is_dir():
            cmd.extend([
                "-v", f"{host_plugins}:/home/agent/.claude/plugins:ro",
            ])
        # ``--memory`` is a hard cgroup ceiling; ``--memory-reservation``
        # is a soft floor. Either may be empty (operator opt-out).
        if self.memory_limit:
            cmd.extend(["--memory", self.memory_limit])
        if self.memory_reservation:
            cmd.extend(["--memory-reservation", self.memory_reservation])
        cmd.extend([
            self.image,
            # No command override — the image's CMD tails the audit
            # log so ``docker logs`` streams turn events.
        ])
        rc, _, stderr = await _run_cmd(cmd, check=False)
        if rc != 0:
            raise RuntimeError(
                f"docker run failed for {self.container_name}: "
                f"{stderr.decode('utf-8', errors='replace').strip()[:500]}"
            )


# Serialises concurrent ``docker build -t <tag>`` across workers
# (right after an image-tag bump every cli-docker worker would
# otherwise race BuildKit's exporter).
_BUILD_LOCK = asyncio.Lock()


async def _image_exists_locally(tag: str) -> bool:
    rc, _, _ = await _run_cmd(
        ["docker", "image", "inspect", tag], check=False,
    )
    return rc == 0


async def _run_cmd(cmd: list[str], check: bool = True) -> tuple[int, bytes, bytes]:
    from ..._proc import no_window_kwargs
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **no_window_kwargs(),
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {stderr.decode('utf-8', errors='replace').strip()[:500]}"
        )
    return proc.returncode, stdout, stderr
