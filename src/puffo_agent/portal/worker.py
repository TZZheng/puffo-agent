"""Per-agent worker running one PuffoCoreMessageClient loop.

Owns the agent's adapter + WS listen loop + heartbeat task; written
into runtime.json so the CLI can read live stats without IPC.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from ..agent.adapters import Adapter
from ..agent.core import AgentAPIError, PuffoAgent
from ..limits import (
    DEFAULT_CATCHUP_STALE_HOURS,
    MAX_INLINE_MESSAGE_CHARS,
    MESSAGE_SEGMENT_CHARS,
)
from ..agent.status_reporter import StatusReporter
from .runtime_matrix import (
    HARNESS_PROVIDERS,
    RUNTIME_CLI_DOCKER,
    RUNTIME_CLI_LOCAL,
    RUNTIME_WS_LOCAL,
    resolve_effective_harness,
    resolve_effective_provider,
)
from .ws_local.hub import AttachPoint
from ..agent.shared_content import (
    looks_like_managed_claude_md,
    rebuild_agent_claude_md,
    rebuild_agent_codex_md,
)
from .state import (
    AgentConfig,
    DaemonConfig,
    PuffoCoreConfig,
    RuntimeConfig,
    RuntimeState,
    agent_claude_user_dir,
    agent_codex_user_dir,
    agent_dir,
    agent_home_dir,
    cli_session_json_path,
    docker_shared_dir,
    shared_fs_dir,
)


def _rebuild_managed_system_prompt(
    *,
    harness_name: str,
    agent_id: str,
    shared_path: Path,
    profile_path: str,
    memory_path: str,
    workspace_path: str,
    display_name: str = "",
    role: str = "",
    role_short: str = "",
) -> str:
    """Dispatch wrapper: write the right system-prompt file(s) for the
    agent's harness. Codex agents get ``$CODEX_HOME/AGENTS.md``; every
    other harness goes through the legacy claude-code path (which also
    writes GEMINI.md for the gemini-cli harness sharing the same body).
    Returns the assembled prompt body either way. The identity fields
    feed the managed ``memory/briefing/profile.md`` re-sync inside the
    rebuild.
    """
    if harness_name == "codex":
        return rebuild_agent_codex_md(
            shared_dir=shared_path,
            profile_path=Path(profile_path),
            memory_dir=Path(memory_path),
            workspace_dir=Path(workspace_path),
            codex_user_dir=agent_codex_user_dir(agent_id),
            agent_id=agent_id,
            display_name=display_name,
            role=role,
            role_short=role_short,
        )
    return rebuild_agent_claude_md(
        shared_dir=shared_path,
        profile_path=Path(profile_path),
        memory_dir=Path(memory_path),
        workspace_dir=Path(workspace_path),
        claude_user_dir=agent_claude_user_dir(agent_id),
        gemini_user_dir=agent_home_dir(agent_id) / ".gemini",
        agent_id=agent_id,
        display_name=display_name,
        role=role,
        role_short=role_short,
    )

logger = logging.getLogger(__name__)

RECONNECT_BACKOFF_SECONDS = 5.0


def build_adapter(daemon_cfg: DaemonConfig, agent_cfg: AgentConfig) -> Adapter:
    """Construct the adapter for ``runtime.kind``. Raises on unknown
    or misconfigured kinds."""
    kind = agent_cfg.runtime.kind or "chat-local"

    if kind == "chat-local":
        from ..agent.adapters.chat_only import ChatOnlyAdapter
        provider = _build_legacy_provider(daemon_cfg, agent_cfg.runtime)
        # Tool wiring is deferred: the in-process puffo_core dispatch
        # needs the live message client, which Worker._run() builds
        # after this adapter — it injects ``adapter.tool_dispatch``
        # there via _build_chat_local_tool_dispatch(). Mirrors how
        # sdk-local wires its MCP tools below, minus the subprocess.
        return ChatOnlyAdapter(provider)

    if kind == "sdk-local":
        from ..agent.adapters.sdk import SDKAdapter
        api_key = agent_cfg.runtime.api_key or daemon_cfg.anthropic.api_key
        model = agent_cfg.runtime.model or daemon_cfg.anthropic.model or "claude-sonnet-4-6"
        if not api_key:
            raise RuntimeError(
                f"agent {agent_cfg.id!r}: runtime kind 'sdk-local' requires an anthropic "
                "api_key in daemon.yml or agent.yml"
            )
        adapter = SDKAdapter(
            api_key=api_key,
            model=model,
            allowed_tools=agent_cfg.runtime.allowed_tools,
            agent_id=agent_cfg.id,
            workspace_dir=str(agent_cfg.resolve_workspace_dir()),
            max_turns=agent_cfg.runtime.max_turns,
            # Empty = vendor endpoint (unchanged); set = route the SDK's
            # model calls through the proxy via ANTHROPIC_BASE_URL.
            base_url=agent_cfg.runtime.llm_base_url,
        )
        if agent_cfg.puffo_core.is_configured():
            from ..mcp.config import puffo_core_stdio_sdk_config, default_python_executable
            pc = agent_cfg.puffo_core
            adapter.mcp_servers_override = puffo_core_stdio_sdk_config(
                python=default_python_executable(),
                slug=pc.slug,
                device_id=pc.device_id,
                server_url=pc.server_url,
                space_id=pc.space_id,
                keystore_dir=str(agent_dir(agent_cfg.id) / "keys"),
                workspace=str(agent_cfg.resolve_workspace_dir()),
                agent_id=agent_cfg.id,
                memory_dir=str(agent_cfg.resolve_memory_dir()),
            )
        return adapter

    # CLI adapters authenticate via the host's
    # ~/.claude/.credentials.json (set up by `claude login`); no
    # api_key is threaded through. Model overrides still flow.
    if kind == "cli-docker":
        # desired_skills install below; desired_mcps can't (their
        # launch commands don't resolve in-container) — reject loudly.
        if agent_cfg.desired_mcps:
            raise RuntimeError(
                f"agent {agent_cfg.id!r}: desired_mcps are not supported "
                "on the cli-docker runtime yet (the MCP launch command "
                "won't resolve inside the container). Clear them from "
                "agent.yml or switch runtime.kind to cli-local."
            )
        from ..agent.adapters.docker_cli import DockerCLIAdapter
        from ..agent.harness import build_harness
        harness = build_harness(agent_cfg.runtime.harness)
        # gemini-cli needs GEMINI_API_KEY per docker exec; claude-code
        # and hermes use the bind-mounted credentials file.
        google_key = ""
        if harness.name() == "gemini-cli":
            google_key = daemon_cfg.google.api_key
            if not google_key:
                raise RuntimeError(
                    f"agent {agent_cfg.id!r}: harness=gemini-cli requires a "
                    "google.api_key — pass --api-key on `agent create`, set "
                    "GEMINI_API_KEY in the environment, or run "
                    "`puffo-agent config` to save a daemon-wide default."
                )
        # Per-agent overrides win; empty falls through to daemon
        # defaults, then to "no cap".
        memory_limit = (
            agent_cfg.runtime.docker_memory_limit
            or daemon_cfg.docker_memory_limit
        )
        memory_reservation = (
            agent_cfg.runtime.docker_memory_reservation
            or daemon_cfg.docker_memory_reservation
        )
        adapter = DockerCLIAdapter(
            agent_id=agent_cfg.id,
            model=agent_cfg.runtime.model or daemon_cfg.anthropic.model or "",
            image=agent_cfg.runtime.docker_image,
            workspace_dir=str(agent_cfg.resolve_workspace_dir()),
            claude_dir=str(agent_cfg.resolve_claude_dir()),
            session_file=str(cli_session_json_path(agent_cfg.id)),
            agent_home_dir=str(agent_home_dir(agent_cfg.id)),
            shared_fs_dir=str(shared_fs_dir()),
            inference_level=getattr(agent_cfg.runtime, "inference_level", ""),
            harness=harness,
            google_api_key=google_key,
            memory_limit=memory_limit,
            memory_reservation=memory_reservation,
            desired_skills=agent_cfg.desired_skills,
            puffo_core_server_url=agent_cfg.puffo_core.server_url,
            puffo_core_slug=agent_cfg.puffo_core.slug,
            puffo_core_keys_dir=str(agent_dir(agent_cfg.id) / "keys"),
        )
        # Env for spawning the puffo_core MCP server; path-typed values are
        # rewritten to container bind-mount paths at config-write time.
        if agent_cfg.puffo_core.is_configured():
            from ..mcp.config import puffo_core_mcp_env
            pc = agent_cfg.puffo_core
            adapter.puffo_core_mcp_env = puffo_core_mcp_env(
                slug=pc.slug,
                device_id=pc.device_id,
                server_url=pc.server_url,
                space_id=pc.space_id,
                # Host paths; rewritten to container paths by
                # docker_cli's _write_cli_mcp_config.
                keystore_dir=str(agent_dir(agent_cfg.id) / "keys"),
                workspace=str(agent_cfg.resolve_workspace_dir()),
                agent_id=agent_cfg.id,
                # MCP runs inside the container; reach the host's
                # 127.0.0.1 data + rpc services via Docker's host alias.
                data_service_url=f"http://host.docker.internal:{daemon_cfg.data_service.port}",
                rpc_url=f"http://host.docker.internal:{daemon_cfg.rpc_service.port}",
                runtime_kind="cli-docker",
                harness=agent_cfg.runtime.harness,
                # MCP runs in-container; the agent dir is bind-mounted
                # at /home/agent/.puffo-agent-state (docker_cli also
                # pins this at its env-override sites).
                memory_dir="/home/agent/.puffo-agent-state/memory",
                # T23 phase-2: forward the transport so a keyless
                # ``bridge`` agent's subprocess MCP sets keyless=True
                # (unsigned /v2/cloud-agents/* path). Only "bridge" is
                # emitted downstream; native is inert.
                transport=pc.transport,
            )
        return adapter

    if kind == "cli-local":
        from ..agent.adapters.local_cli import LocalCLIAdapter
        # The legacy permission-proxy DM flow has not been ported to
        # puffo-core; the hook fail-opens when PUFFO_OPERATOR_USERNAME
        # is unset, so cli-local works without supervised approvals.
        operator = ""
        from ..agent.harness import build_harness
        harness = build_harness(agent_cfg.runtime.harness)
        if harness.name() == "codex":
            model = agent_cfg.runtime.model or daemon_cfg.openai.model or ""
        else:
            model = agent_cfg.runtime.model or daemon_cfg.anthropic.model or ""
        adapter = LocalCLIAdapter(
            agent_id=agent_cfg.id,
            model=model,
            workspace_dir=str(agent_cfg.resolve_workspace_dir()),
            claude_dir=str(agent_cfg.resolve_claude_dir()),
            session_file=str(cli_session_json_path(agent_cfg.id)),
            mcp_config_file=str(agent_dir(agent_cfg.id) / "mcp-config.json"),
            agent_home_dir=str(agent_home_dir(agent_cfg.id)),
            owner_username=operator,
            permission_mode=agent_cfg.runtime.permission_mode,
            sandbox=agent_cfg.runtime.sandbox,
            inference_level=getattr(agent_cfg.runtime, "inference_level", ""),
            task_timeout_seconds=getattr(
                agent_cfg.runtime,
                "task_timeout_seconds",
                600.0,
            ),
            harness=harness,
            desired_skills=agent_cfg.desired_skills,
            desired_mcps=agent_cfg.desired_mcps,
            puffo_core_server_url=agent_cfg.puffo_core.server_url,
            puffo_core_slug=agent_cfg.puffo_core.slug,
            puffo_core_keys_dir=str(agent_dir(agent_cfg.id) / "keys"),
            # Empty base URL → no env override (claude keeps its OAuth /
            # ~/.claude credential path). Set → ANTHROPIC_BASE_URL routes
            # the CLI's model calls through the proxy; the VK rides on
            # runtime.api_key (injected as ANTHROPIC_API_KEY only when
            # the base URL is also set — see LocalCLIAdapter._llm_env).
            llm_base_url=agent_cfg.runtime.llm_base_url,
            llm_api_key=agent_cfg.runtime.api_key,
        )
        if agent_cfg.puffo_core.is_configured():
            from ..mcp.config import puffo_core_mcp_env
            pc = agent_cfg.puffo_core
            adapter.puffo_core_mcp_env = puffo_core_mcp_env(
                slug=pc.slug,
                device_id=pc.device_id,
                server_url=pc.server_url,
                space_id=pc.space_id,
                keystore_dir=str(agent_dir(agent_cfg.id) / "keys"),
                workspace=str(agent_cfg.resolve_workspace_dir()),
                agent_id=agent_cfg.id,
                data_service_url=f"http://127.0.0.1:{daemon_cfg.data_service.port}",
                rpc_url=f"http://127.0.0.1:{daemon_cfg.rpc_service.port}",
                runtime_kind="cli-local",
                harness=agent_cfg.runtime.harness,
                memory_dir=str(agent_cfg.resolve_memory_dir()),
                # T23 phase-2: forward the transport so a keyless
                # ``bridge`` agent's subprocess MCP sets keyless=True
                # (unsigned /v2/cloud-agents/* path) instead of hitting
                # the signed keystore path and raising identity-not-found.
                # Only "bridge" is emitted downstream; native is inert.
                transport=pc.transport,
            )
        return adapter

    raise RuntimeError(
        f"agent {agent_cfg.id!r}: unknown runtime kind {kind!r} "
        "(valid: chat-local, sdk-local, cli-docker, cli-local)"
    )


def _build_legacy_provider(daemon_cfg: DaemonConfig, runtime: RuntimeConfig):
    """Anthropic/OpenAI message-completion provider for the
    chat-local adapter. Per-agent fields override daemon defaults."""
    provider_name = runtime.provider or daemon_cfg.default_provider
    # Empty base URL → None → vendor endpoint (today's behavior, byte-for-
    # byte unchanged). Set → route completions through the proxy (VK).
    base_url = runtime.llm_base_url or None

    if provider_name == "anthropic":
        try:
            from ..agent.providers.anthropic_provider import AnthropicProvider
        except ImportError as exc:
            # Only re-attribute to a missing extra when the *anthropic*
            # package itself is what's absent. An ImportError whose missing
            # module is something else (a broken transitive dep of an
            # installed anthropic SDK, a partial wheel, an internal
            # submodule) is a genuine failure — re-raise it unchanged rather
            # than misdirect the operator to reinstall an already-present
            # package.
            if (exc.name or "") != "anthropic" and not (
                exc.name or ""
            ).startswith("anthropic."):
                raise
            # The Anthropic SDK is a cloud-slim extra, absent from the base
            # (cli-local template) install. Surface the fix, not a raw
            # ModuleNotFoundError buried in the worker traceback.
            raise RuntimeError(
                "chat-local provider=anthropic needs the Anthropic SDK, "
                "which isn't in the base install — rebuild the template "
                "with \"pip install 'puffo-agent[anthropic]'\""
            ) from exc
        api_key = runtime.api_key or daemon_cfg.anthropic.api_key
        model = runtime.model or daemon_cfg.anthropic.model or "claude-sonnet-4-6"
        if not api_key:
            raise RuntimeError(
                "anthropic api_key is not set in daemon.yml or agent.yml"
            )
        return AnthropicProvider(api_key=api_key, model=model, base_url=base_url)

    if provider_name == "openai":
        try:
            from ..agent.providers.openai_provider import OpenAIProvider
        except ImportError as exc:
            # See the anthropic branch: only re-attribute when the *openai*
            # package itself is missing; re-raise any other ImportError.
            if (exc.name or "") != "openai" and not (
                exc.name or ""
            ).startswith("openai."):
                raise
            raise RuntimeError(
                "chat-local provider=openai needs the OpenAI SDK, which "
                "isn't in the base install — rebuild the template with "
                "\"pip install 'puffo-agent[openai]'\""
            ) from exc
        api_key = runtime.api_key or daemon_cfg.openai.api_key
        model = runtime.model or daemon_cfg.openai.model or "gpt-4o"
        if not api_key:
            raise RuntimeError(
                "openai api_key is not set in daemon.yml or agent.yml"
            )
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)

    raise RuntimeError(f"unknown provider {provider_name!r}")


def _build_chat_local_tool_dispatch(client) -> dict:
    """In-process puffo_core tool dispatch for the chat-local adapter.

    Mirrors ``ws_local.route._build_tool_dispatch``: the same
    PuffoCoreToolsConfig wired off the live message client, narrowed
    to the send tools the chat-local provider advertises as Anthropic
    tool schemas. Without this executor, the model's ``tool_use``
    blocks would parse but never post.
    """
    from ..mcp.puffo_core_tools import PuffoCoreToolsConfig
    from .ws_local.in_process_data_client import InProcessDataClient
    from .ws_local.tool_dispatch import build_dispatch

    cfg = PuffoCoreToolsConfig(
        slug=client.slug,
        device_id=client.device_id,
        keystore=client.keystore,
        http_client=client.http,
        data_client=InProcessDataClient(client.store, client),
        space_id=getattr(client, "space_id", None),
        workspace=getattr(client, "workspace", None),
        message_client=client,
        # T23: only the daemon-owned bridge WS may drive keyless sends;
        # None on native agents keeps the signed-crypto path.
        bridge_client=getattr(client, "_bridge", None),
    )
    return build_dispatch(
        cfg,
        allowed=frozenset({"send_message", "send_message_with_attachments"}),
    )


def _puffo_cli_keystore_dir() -> Path:
    """Where ``puffo-cli`` saves identities. Mirrors the Rust
    ``directories`` crate (``ProjectDirs::from("ai", "puffo",
    "puffo-cli").data_dir().join("keys")``).

    Windows: ``%APPDATA%\\puffo\\puffo-cli\\data\\keys``
    macOS:   ``~/Library/Application Support/ai.puffo.puffo-cli/keys``
    Linux:   ``$XDG_DATA_HOME/puffo/puffo-cli/keys``
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "puffo" / "puffo-cli" / "data" / "keys"
        return Path.home() / "AppData" / "Roaming" / "puffo" / "puffo-cli" / "data" / "keys"
    if sys.platform == "darwin":
        return (
            Path.home() / "Library" / "Application Support"
            / "ai.puffo.puffo-cli" / "keys"
        )
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "puffo" / "puffo-cli" / "keys"


def _ensure_agent_identity_imported(agent_id: str, slug: str) -> None:
    """Copy ``<slug>.json`` from puffo-cli's keystore into the
    per-agent puffo-agent keystore if missing. Without this the
    worker crash-loops on ``identity not found``. No-op when the
    destination exists or the source is missing."""
    if not slug:
        return
    dest = agent_dir(agent_id) / "keys" / f"{slug}.json"
    if dest.exists():
        return
    src = _puffo_cli_keystore_dir() / f"{slug}.json"
    if not src.exists():
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info(
            "agent %s: imported identity from puffo-cli keystore (%s → %s)",
            agent_id, src, dest,
        )
    except OSError as exc:
        logger.warning(
            "agent %s: failed to import identity from %s: %s",
            agent_id, src, exc,
        )


def _build_puffo_core_client(
    agent_cfg: AgentConfig,
    agent_id: str,
    daemon_cfg: "DaemonConfig | None" = None,
):
    """Construct a PuffoCoreMessageClient from the agent's config.
    ``daemon_cfg`` carries the host-wide tunables (currently the
    long-message redaction thresholds); accepts ``None`` so legacy
    test seeds that didn't have a daemon config keep working with
    the dataclass defaults.
    """
    from ..agent.message_store import MessageStore
    from ..agent.puffo_core_client import PuffoCoreMessageClient, max_image_edge_px
    from ..crypto.http_client import PuffoCoreHttpClient
    from ..crypto.keystore import KeyStore

    pc = agent_cfg.puffo_core
    bridge = None
    if pc.transport == "bridge":
        # T23 keyless transport: no identity file exists to import;
        # the bridge authenticates with the sandbox token instead.
        # Keystore/http below are still constructed (both lazy — they
        # never touch disk/network in __init__); phase 2 replaces
        # their call sites.
        from ..agent.bridge_client import CloudBridgeClient

        bridge = CloudBridgeClient(pc.server_url, pc.sandbox_token, pc.slug)
    else:
        _ensure_agent_identity_imported(agent_id, pc.slug)
    ks_dir = str(agent_dir(agent_id) / "keys")
    ks = KeyStore(ks_dir)
    # Bridge agents dispatch outbound tool work keyless over the unsigned
    # ``/v2/cloud-agents/*`` routes; ``route.py`` reuses ``client.http``, so
    # the in-process ws-local cfg's ``keyless`` accessor reads True here.
    http = PuffoCoreHttpClient(
        pc.server_url, ks, pc.slug, keyless=(pc.transport == "bridge"),
    )
    ms = MessageStore(str(agent_dir(agent_id) / "messages.db"))

    max_inline = (
        daemon_cfg.max_inline_message_chars if daemon_cfg is not None else MAX_INLINE_MESSAGE_CHARS
    )
    segment_chars = (
        daemon_cfg.segment_chars if daemon_cfg is not None else MESSAGE_SEGMENT_CHARS
    )
    catchup_stale_hours = (
        daemon_cfg.catchup_stale_hours
        if daemon_cfg is not None
        else DEFAULT_CATCHUP_STALE_HOURS
    )

    # The inbound-image downscale cap follows the harness's effective model
    # (Opus 4.7+ resolves 2576px, else 1568px).
    is_codex = (agent_cfg.runtime.harness or "claude-code") == "codex"
    if is_codex:
        model = agent_cfg.runtime.model or (daemon_cfg.openai.model if daemon_cfg else "")
    else:
        model = agent_cfg.runtime.model or (daemon_cfg.anthropic.model if daemon_cfg else "")

    return PuffoCoreMessageClient(
        slug=pc.slug,
        device_id=pc.device_id,
        space_id=pc.space_id,
        operator_slug=pc.operator_slug,
        auto_accept_space_invitations=pc.auto_accept_space_invitations,
        auto_accept_dm=getattr(pc, "auto_accept_dm", False),
        keystore=ks,
        http_client=http,
        message_store=ms,
        workspace=str(agent_cfg.resolve_workspace_dir()),
        max_inline_chars=max_inline,
        segment_chars=segment_chars,
        agent_created_at=agent_cfg.created_at,
        image_edge_px=max_image_edge_px(model),
        catchup_stale_hours=catchup_stale_hours,
        agent_id=agent_id,
        bridge_client=bridge,
    )


# Auth-class patterns: definitive OAuth/API-key failure only. Shared by
# the leak filter and the auth_failed health flip. High-FP markers
# (401, unauthorized, api_error) stay out; they collide with agent prose.
_AUTH_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Not logged in[\s\S]*Please run /login", re.IGNORECASE),
    re.compile(r"^\s*OAuth token (?:revoked|has expired)\b", re.IGNORECASE),
    re.compile(r"^\s*Invalid API key\b", re.IGNORECASE),
    re.compile(r"\bThis organization has been disabled\b", re.IGNORECASE),
    re.compile(r"\bauthentication_error\b", re.IGNORECASE),
)

# Worker-layer leak patterns NOT in the auth-class set. Sources:
# Claude Code error reference (CLI message-to-recovery table) +
# Claude API platform docs (canonical <type>_error identifiers).
_NON_AUTH_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Internal kick message echoed back as a reply.
    re.compile(
        r"^\s*\[puffo-agent system message\]\s+session errored on rate",
        re.IGNORECASE,
    ),
    # Subscription-plan quotas (the prod miss the reviewer surfaced).
    re.compile(r"^\s*You've hit your\b.*?\blimit\b", re.IGNORECASE),
    # Model-usage-cap fallback ("reached your <Model> limit …"); model-agnostic + apostrophe-robust (`.?`).
    re.compile(r"^\s*You.?ve reached your\b.*?\blimit\b", re.IGNORECASE),
    re.compile(r"^\s*Credit balance is too low\b", re.IGNORECASE),
    # CLI-emitted server 429 / 5xx.
    re.compile(r"\bAPI Error: Request rejected \(429\)", re.IGNORECASE),
    re.compile(r"\bAPI Error: Server is temporarily limiting requests\b", re.IGNORECASE),
    re.compile(r"\bAPI Error: Repeated 529 Overloaded errors\b", re.IGNORECASE),
    re.compile(r"\bAPI Error: 500\b[\s\S]*Internal server error\b", re.IGNORECASE),
    # API-canonical <type>_error identifiers, minus high-FP entries
    # (invalid_request_error / not_found_error / api_error) per audit.
    re.compile(r"\brate[_ -]limit[_ -]error\b", re.IGNORECASE),
    re.compile(r"\boverloaded_error\b", re.IGNORECASE),
    re.compile(r"\bbilling_error\b", re.IGNORECASE),
    re.compile(r"\bpermission_error\b", re.IGNORECASE),
    re.compile(r"\btimeout_error\b", re.IGNORECASE),
)

_WORKER_ERROR_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_AUTH_ERROR_PATTERNS,
    *_NON_AUTH_LEAK_PATTERNS,
)

# Post-leak backoff: random 15-60s samples sustained limits instead of
# hammering; single-batch leaks self-clear during the sleep.
_SUPPRESSION_BACKOFF_MIN_SECONDS = 15.0
_SUPPRESSION_BACKOFF_MAX_SECONDS = 60.0


def _looks_like_auth_error(reply: str) -> bool:
    """True iff ``reply`` is one of the definitive auth-class
    failure strings (Claude CLI re-login prompt, OAuth-token
    revoked/expired, invalid API key, disabled org, or the
    ``authentication_error`` API identifier). Drives the
    ``runtime.health=auth_failed`` flip alongside PUF-207's
    startup-paused signal."""
    if not reply:
        return False
    for pattern in _AUTH_ERROR_PATTERNS:
        if pattern.search(reply):
            return True
    return False


def _suppress_worker_error_leak(reply: str) -> str | None:
    """Return ``None`` when ``reply`` matches a worker-error
    pattern, signalling the caller to suppress the channel post and
    surface to the operator instead. Returns the reply unchanged
    otherwise. Mirrors ``_coerce_root_visibility``'s shape."""
    if not reply:
        return reply
    for pattern in _WORKER_ERROR_LEAK_PATTERNS:
        if pattern.search(reply):
            return None
    return reply


def _handle_suppressed_reply(
    reply: str,
    runtime: "RuntimeState",
    agent_id: str,
    *,
    scope: str,
    on_auth_failure: Optional[Callable[[], None]] = None,
    on_auth_failed_enter: Optional[Callable[[], None]] = None,
) -> tuple[bool, float]:
    """Shared landing for a suppressed worker-error leak. Returns
    ``(suppressed, backoff_seconds)``:

    - Clean prose: ``(False, 0.0)``; caller proceeds normally.
    - Leak detected: ``(True, uniform(15, 60))``; caller skips
      ``send_fallback_message`` and ``asyncio.sleep(backoff)`` so
      the next batch doesn't immediately re-leak in tight loops.

    On suppression: log the truncated payload, populate
    ``runtime.error`` with a scope-tagged + leak-class-tagged
    message, and (if auth-class) flip ``runtime.health="auth_failed"``
    — that signal is definitive regardless of which scope surfaced
    it. ``on_auth_failure`` fires on the auth-class branch only;
    PUF-221 hooks the daemon's ``CredentialRefresher.notify_refresh_needed``
    here so a 401-leak short-circuits the 2-min poll instead of
    waiting for the next tick. ``on_auth_failed_enter`` fires ONLY on
    the was-ok→auth_failed transition (not re-entries), giving the
    per-session DM dedup a natural firing edge."""
    safe_reply = _suppress_worker_error_leak(reply)
    if safe_reply is not None:
        return False, 0.0
    is_auth = _looks_like_auth_error(reply)
    backoff = random.uniform(
        _SUPPRESSION_BACKOFF_MIN_SECONDS,
        _SUPPRESSION_BACKOFF_MAX_SECONDS,
    )
    logger.warning(
        "agent %s: suppressed worker-error leak in %s reply (backoff %.1fs): %s",
        agent_id, scope, backoff, reply[:200],
    )
    if is_auth:
        was_ok = runtime.health != "auth_failed"
        runtime.health = "auth_failed"
        if on_auth_failure is not None:
            try:
                on_auth_failure()
            except Exception as exc:
                logger.warning(
                    "agent %s: on_auth_failure callback raised: %s",
                    agent_id, exc,
                )
        if was_ok and on_auth_failed_enter is not None:
            try:
                on_auth_failed_enter()
            except Exception as exc:
                logger.warning(
                    "agent %s: on_auth_failed_enter callback raised: %s",
                    agent_id, exc,
                )
    if scope == "api-error-retry":
        if is_auth:
            runtime.error = (
                "Claude Code sign-in expired. On the computer running "
                "puffo-agent, open a terminal and run `claude auth "
                "login`, then send this agent a message."
            )
        else:
            runtime.error = (
                "Rate-limit / quota / server error — usually self-"
                "recovers. Check the puffo-agent daemon log if it "
                "persists."
            )
    else:
        runtime.error = (
            "Worker emitted an auth / rate-limit / quota error string "
            "instead of a real reply. Check the puffo-agent daemon log."
        )
    runtime.save(agent_id)
    return True, backoff



class Worker:
    """Runs a single AI agent inside the daemon event loop."""

    @staticmethod
    def _clear_api_error_abandoned_if_recoverable(
        runtime: "RuntimeState",
        agent_id: str,
        root_id: str,
        log: logging.Logger,
    ) -> None:
        """Clear ``runtime.health = "api_error_abandoned"`` back to
        ``"ok"`` on the next successful turn. ``auth_failed`` is
        deliberately left alone — PUF-221's CredentialRefresher
        owns that lifecycle and a single lucky turn shouldn't
        substitute for the refresh-success-ping.

        Known granularity mismatch (PUF-253 design input):
        ``api_error_abandoned`` is a thread-level event but
        ``runtime.health`` is an agent-global flag, so a success
        on thread B clears it even if thread A is still stuck.
        Last-write-wins until ``runtime.error`` becomes a list.
        """
        if runtime.health != "api_error_abandoned":
            return
        runtime.health = "ok"
        runtime.error = ""
        runtime.save(agent_id)
        log.info(
            "agent %s: api-error-recovery on thread %s; "
            "runtime.health cleared back to ok",
            agent_id, root_id,
        )

    @staticmethod
    def _clear_auth_failed_if_recoverable(
        runtime: "RuntimeState",
        agent_id: str,
        log: logging.Logger,
    ) -> None:
        # Symmetric to _clear_api_error_abandoned_if_recoverable but
        # fired on refresh-success (not turn-success). Optimistic: if
        # the next request still 401s, _handle_suppressed_reply re-sets.
        if runtime.health != "auth_failed":
            return
        runtime.health = "ok"
        runtime.error = ""
        runtime.save(agent_id)
        log.info(
            "agent %s: credential-refresh-success; "
            "runtime.health cleared from auth_failed back to ok",
            agent_id,
        )

    def _maybe_wake_refresher_if_auth_failed(self, agent_id: str) -> None:
        """A new batch while auth_failed: wake the refresher to re-check
        for an operator re-login now instead of waiting for the poll."""
        if self.runtime.health != "auth_failed":
            return
        if self._notify_refresh_needed is None:
            return
        try:
            self._notify_refresh_needed()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent %s: notify_refresh_needed raised: %s", agent_id, exc,
            )

    async def _run_post_warm_gate(self, agent_id: str) -> None:
        """Probe the adapter's round-trip readiness after warm() succeeds,
        reassert ``auth_failed`` if the provider's still unreachable, THEN
        release ``_warm_done``. Order matters — releasing first would let a
        queued message dispatch against an unprobed runtime."""
        try:
            probe_ok = await self._adapter.health_probe()
        except Exception as exc:
            logger.warning(
                "agent %s: health_probe raised; treating as "
                "probe-fail: %s", agent_id, exc,
            )
            probe_ok = False
        if not probe_ok:
            Worker._reassert_auth_failed_after_failed_probe(
                self.runtime, agent_id, logger,
            )
        self._warm_done.set()

    @staticmethod
    def _reassert_auth_failed_after_failed_probe(
        runtime: "RuntimeState",
        agent_id: str,
        log: logging.Logger,
    ) -> None:
        """``on_refresh_success`` eagerly clears ``auth_failed`` before
        the respawn, so a still-broken provider warms up looking healthy.
        When the post-warm probe fails, re-assert the failed state so the
        next refresh cycle retries. No-op unless the runtime is in the
        eager-cleared ``ok`` state."""
        if runtime.health != "ok":
            return
        runtime.health = "auth_failed"
        runtime.error = (
            "Claude Code sign-in expired. On the computer running "
            "puffo-agent, open a terminal and run `claude auth "
            "login`, then send this agent a message."
        )
        runtime.save(agent_id)
        log.warning(
            "agent %s: post-warm health probe failed; reasserted "
            "runtime.health = auth_failed",
            agent_id,
        )

    def _enter_auth_failed(self, agent_id: str) -> None:
        """Flip ``auth_failed`` + fire recovery (refresher kick + operator
        DM). Used on a confirmed adapter auth error so we skip the
        pointless kick-retries and go straight to recover-via-relogin."""
        rt = self.runtime
        was_ok = rt.health != "auth_failed"
        rt.health = "auth_failed"
        rt.error = (
            "Claude Code sign-in expired. On the computer running "
            "puffo-agent, open a terminal and run `claude auth "
            "login`, then send this agent a message."
        )
        rt.save(agent_id)
        if self._notify_refresh_needed is not None:
            try:
                self._notify_refresh_needed()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent %s: notify_refresh_needed raised: %s", agent_id, exc,
                )
        if was_ok:
            self._on_auth_failed_enter()

    def _on_auth_failed_enter(self) -> None:
        """Fire the operator DM once per auth_failed episode. The flag
        re-arms on auth_failed CLEAR (daemon ``on_refresh_success``) and
        on a failed send, so a later genuine failure re-notifies."""
        if self._auth_failed_notification_sent:
            return
        self._auth_failed_notification_sent = True
        try:
            asyncio.create_task(self._notify_operator_of_auth_failed_oauth())
        except Exception as exc:  # noqa: BLE001
            # Re-arm so a schedule failure retries on the next ENTER.
            self._auth_failed_notification_sent = False
            logger.warning(
                "agent %s: couldn't schedule auth-failed DM: %s",
                self.agent_cfg.id, exc,
            )

    async def _notify_operator_of_auth_failed_oauth(self) -> None:
        """DM the operator the bilingual OAuth-expired recovery copy.
        Re-arms the dedup flag on a transient failure (client not warm,
        or send raised) so the next ENTER retries instead of staying
        silently gated."""
        client = self._client
        if client is None:
            # Still warming — re-arm so a later ENTER retries.
            self._auth_failed_notification_sent = False
            logger.warning(
                "agent %s: auth-failed DM skipped — client not yet warm",
                self.agent_cfg.id,
            )
            return
        operator_slug = getattr(client, "operator_slug", "") or ""
        if not operator_slug:
            # No operator to DM; the red-dot UI is the only signal. Stay
            # gated — re-arming would respin on every 401.
            logger.warning(
                "agent %s: auth-failed but no operator_slug — not DMing",
                self.agent_cfg.id,
            )
            return
        from ..agent._invite_strings import (
            format_codex_oauth_expired,
            format_oauth_expired,
        )
        display_name = (
            getattr(self.agent_cfg, "display_name", "") or self.agent_cfg.id
        )
        # Codex agents need the Codex recovery command, not the Claude
        # one; otherwise the operator runs the wrong CLI and assumes
        # the alert is broken. Harness is the cheapest signal we have.
        runtime = getattr(self.agent_cfg, "runtime", None)
        harness = getattr(runtime, "harness", "") if runtime is not None else ""
        if harness == "codex":
            text = format_codex_oauth_expired(self.agent_cfg.id, display_name)
        else:
            text = format_oauth_expired(self.agent_cfg.id, display_name)
        try:
            await client._send_dm(operator_slug, text, root_id="")
        except Exception as exc:
            # Transient send failure — re-arm for the next ENTER.
            self._auth_failed_notification_sent = False
            logger.exception(
                "agent %s: auth-failed DM to %s raised: %s",
                self.agent_cfg.id, operator_slug, exc,
            )
            return
        logger.info(
            "agent %s: notified operator @%s of OAuth-expired",
            self.agent_cfg.id, operator_slug,
        )

    @staticmethod
    def _flip_health_in_progress(
        runtime: "RuntimeState",
        agent_id: str,
        log: logging.Logger,
    ) -> None:
        """Override any sticky red with ``in_progress`` at batch-top."""
        if runtime.health == "in_progress":
            return
        runtime.health = "in_progress"
        runtime.error = ""
        runtime.save(agent_id)
        log.info("agent %s: runtime.health → in_progress", agent_id)

    @staticmethod
    def _resolve_health_on_success(
        runtime: "RuntimeState",
        agent_id: str,
        log: logging.Logger,
    ) -> None:
        """Transition ``in_progress`` → ``ok``; skip any in-turn red."""
        if runtime.health != "in_progress":
            return
        runtime.health = "ok"
        runtime.error = ""
        runtime.save(agent_id)
        log.info("agent %s: runtime.health in_progress → ok", agent_id)

    @staticmethod
    def _fallback_unhandled_error_if_stuck_in_progress(
        runtime: "RuntimeState",
        agent_id: str,
        turn_error: str | None,
        log: logging.Logger,
    ) -> None:
        """``in_progress`` → ``unhandled_error`` when a non-retry-able
        exception left the flip stuck. Distinct from ``unknown`` so the
        CLI / heartbeat can surface it.
        """
        if runtime.health != "in_progress":
            return
        runtime.health = "unhandled_error"
        runtime.error = turn_error or "turn raised; no category red set"
        runtime.save(agent_id)
        log.warning(
            "agent %s: runtime.health → unhandled_error (%s)",
            agent_id, runtime.error,
        )

    def __init__(
        self,
        daemon_cfg: DaemonConfig,
        agent_cfg: AgentConfig,
        *,
        notify_refresh_needed: Optional[Callable[[], None]] = None,
        ensure_fresh_token: Optional[Callable[[], Awaitable[bool]]] = None,
        ws_local_hub=None,
    ):
        self.daemon_cfg = daemon_cfg
        self.agent_cfg = agent_cfg
        # Set for ws-local agents; the Worker idles and registers an
        # attach point instead of running a harness consumer.
        self._ws_local_hub = ws_local_hub
        # CredentialRefresher hook: a 401 in a reply short-circuits the 2-min poll.
        self._notify_refresh_needed = notify_refresh_needed
        # Blocking pre-delivery refresh via the daemon mutex; True iff the token
        # has time left. None for runtimes without claude OAuth (api-puffo, ws-local).
        self._ensure_fresh_token = ensure_fresh_token
        # In-memory dedup for the auth_failed ENTER operator DM;
        # re-armed on credential refresh-success (daemon
        # on_refresh_success) and on a failed send.
        self._auth_failed_notification_sent = False
        self.runtime = RuntimeState(
            status="running",
            started_at=int(time.time()),
            msg_count=0,
        )
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._adapter: Adapter | None = None
        # Held here so ``stop()`` can close the SQLite + WS handles
        # the client owns. Required on Windows so ``messages.db*``
        # files release before ``puffo-agent agent archive`` renames.
        self._client = None
        # Signalled when warm() finishes (success, failure, or skipped).
        # Daemon awaits this to serialise heavy startup across workers.
        self._warm_done = asyncio.Event()
        # Proactive (between-turns) profile reload state. The refresh
        # watcher (see ``_run``) and the turn-start ``_process_refresh_flags``
        # call share this lock so a flag is consumed exactly once —
        # whichever path wins takes it; the loser hits the early-return
        # when the flags are already gone. ``_turn_active`` gates the
        # watcher so it never applies mid-generation (deferred to the
        # turn-start path). ``_refresh_now`` is a signal-driven wake so
        # SIGHUP can beat the 250 ms idle poll.
        self._reload_lock = asyncio.Lock()
        self._turn_active = False
        self._refresh_now = asyncio.Event()

    def notify_refresh(self) -> None:
        """Wake the proactive refresh watcher now (sub-poll latency).
        Called from the daemon's SIGHUP handler, which runs on the loop
        thread. No-op safe: setting an already-set Event is idempotent,
        and if the worker has no watcher yet (pre-``_run``) the flag is
        simply observed on the first watcher tick."""
        self._refresh_now.set()

    async def _proactive_refresh_tick(self, flag_paths, apply) -> bool:
        """One proactive-watcher iteration's decision + apply. Skips when
        a turn owns flag consumption (``_turn_active``) or no flag in
        ``flag_paths`` is pending; otherwise takes ``_reload_lock`` (the
        same lock the turn-start path holds → consume-once), re-checks
        ``_turn_active``, and awaits ``apply`` (the bound
        ``_process_refresh_flags`` call). Returns whether ``apply`` ran.
        Exceptions from ``apply`` are swallowed so the watcher never
        kills the worker — the turn-start path is the backstop."""
        if self._turn_active:
            # A turn owns flag consumption; never apply mid-turn.
            return False
        if not any(p.exists() for p in flag_paths):
            # Cheap exists() check before taking the lock.
            return False
        async with self._reload_lock:
            if self._turn_active:
                # A turn started between the check and the lock; defer to
                # the turn-start path.
                return False
            try:
                await apply()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent %s: proactive refresh failed: %s",
                    self.agent_cfg.id, exc,
                )
            return True

    async def _refresh_watcher_loop(
        self, flag_paths, apply, *, interval: float = 0.25,
    ) -> None:
        """Proactive between-turns profile reload loop. Polls
        ``flag_paths`` every ``interval`` s (waking immediately when
        ``notify_refresh()`` fires) so a ``profile.md`` / host-sync /
        session edit takes effect while the agent is IDLE — instead of
        only lazily at the next turn. Each pending flag is funneled into
        ``apply``, the bound turn-start ``_process_refresh_flags`` /
        ``adapter.reload`` primitive (adapter-only reload: the bridge WS
        on ``self._client`` and the worker are untouched). No-op unless a
        flag is present, so native/desktop runtimes see no behavioral
        change beyond applying an already-written flag sooner. Runs as a
        sibling task of ``heartbeat``; ends when ``_stop`` is set."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._refresh_now.wait(), timeout=interval,
                )
            except asyncio.TimeoutError:
                pass
            self._refresh_now.clear()
            if self._stop.is_set():
                break
            await self._proactive_refresh_tick(flag_paths, apply)

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        self._task = asyncio.ensure_future(self._run())
        return self._task

    async def wait_warm(self, timeout: float | None = None) -> bool:
        """Block until warm() finishes or the worker exits early.
        Returns True on completion, False on timeout."""
        try:
            await asyncio.wait_for(self._warm_done.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def set_profile_cache(
        self, slug: str, display_name: str, avatar_url: str,
    ) -> None:
        """Cross-process bridge for the MCP ``get_user_info`` tool —
        the subprocess fetches fresh from puffo-server then POSTs the
        result here via the data service, so the next daemon render
        sees the fresh display_name + avatar without waiting for the
        TTL. No-op until the worker's PuffoCoreMessageClient has been
        constructed (warm() hasn't completed yet)."""
        if self._client is not None:
            self._client.set_profile(slug, display_name, avatar_url)

    def host_mcp_context(self):
        """Build a ``HostMcpContext`` from this worker's live state.
        Returns None until ``warm()`` has built the message client."""
        client = self._client
        if client is None:
            return None
        from .host_mcp_handler import HostMcpContext
        from .state import agent_home_dir
        harness = self.agent_cfg.runtime.harness or "claude-code"
        return HostMcpContext(
            agent_id=self.agent_cfg.id,
            slug=client.slug,
            operator_slug=client.operator_slug,
            host_home=Path.home(),
            agent_home=agent_home_dir(self.agent_cfg.id),
            harness=harness,
            keystore=client.keystore,
            http_client=client.http,
            message_client=client,
            send_coordinator=getattr(client, "send_delegate", None),
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                # Bounded wait so a wedged WS recv / subprocess pipe
                # can't hang shutdown. Cancellation usually completes
                # in milliseconds.
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "agent %s: worker task didn't exit within 10s of cancel — "
                    "moving on to adapter cleanup anyway",
                    self.agent_cfg.id,
                )
            except (asyncio.CancelledError, Exception):
                pass
        if self._adapter is not None:
            try:
                # Bounded wait so a wedged ``docker stop`` can't
                # deadlock shutdown. 30s covers docker's 10s SIGTERM
                # grace plus our own ``docker stop -t 5`` and drains.
                await asyncio.wait_for(self._adapter.aclose(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "agent %s: adapter aclose timed out after 30s — "
                    "container may still be running, run `docker ps` to check",
                    self.agent_cfg.id,
                )
            except Exception as exc:
                logger.warning(
                    "agent %s: adapter aclose failed: %s", self.agent_cfg.id, exc,
                )
        if self._client is not None:
            # Release WS + SQLite handles. Required on Windows so
            # ``messages.db*`` is renamable by ``agent archive``.
            try:
                await asyncio.wait_for(self._client.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "agent %s: client.stop timed out after 10s",
                    self.agent_cfg.id,
                )
            except Exception as exc:
                logger.warning(
                    "agent %s: client.stop failed: %s", self.agent_cfg.id, exc,
                )
            self._client = None
        self.runtime.status = "stopped"
        self.runtime.save(self.agent_cfg.id)

    def _runtime_info(self) -> dict[str, str]:
        rt = self.agent_cfg.runtime
        kind = rt.kind or "chat-local"
        if kind == RUNTIME_WS_LOCAL:
            return {
                "kind": kind,
                "provider": "",
                "harness": "",
                "model": "",
            }
        provider = rt.provider
        if not provider and kind in (RUNTIME_CLI_LOCAL, RUNTIME_CLI_DOCKER) and rt.harness:
            supported = HARNESS_PROVIDERS.get(rt.harness, frozenset())
            if len(supported) == 1:
                provider = next(iter(supported))
        if not provider and kind == "chat-local":
            provider = self.daemon_cfg.default_provider
        provider = resolve_effective_provider(kind, provider)
        harness = resolve_effective_harness(kind, provider, rt.harness)
        provider_cfg = getattr(self.daemon_cfg, provider, None)
        model = rt.model or getattr(provider_cfg, "model", "")
        return {
            "kind": kind,
            "provider": provider,
            "harness": harness,
            "model": model,
        }

    def _build_status_reporter(self, client) -> StatusReporter:
        bridge = getattr(client, "_bridge", None)
        reporter = StatusReporter(
            client.http,
            runtime_health_provider=(
                None if bridge is not None else lambda: self.runtime.health
            ),
            runtime_provider=self._runtime_info,
            status_sender=bridge.send_status if bridge is not None else None,
        )
        if bridge is not None:
            bridge.add_connected_callback(reporter.report_current_status)
        return reporter

    async def _run_ws_local(self) -> None:
        """ws-local agents run no harness consumer. Build the client,
        register an attach point, and idle — the bridge's /v1/ws-local
        route brings the agent online when a tool connects."""
        agent_id = self.agent_cfg.id
        try:
            if not self.agent_cfg.puffo_core.is_configured():
                raise RuntimeError(
                    f"agent {agent_id!r}: puffo_core block in agent.yml is incomplete"
                )
            client = _build_puffo_core_client(
                self.agent_cfg, agent_id, daemon_cfg=self.daemon_cfg,
            )
            self._client = client
            reporter = self._build_status_reporter(client)
            point = AttachPoint(
                slug=self.agent_cfg.puffo_core.slug,
                agent_id=agent_id,
                agent_cfg=self.agent_cfg,
                client=client,
                reporter=reporter,
                ack_timeout_s=180.0,
                ping_interval_s=30.0,
            )
        except Exception as e:
            logger.error("agent %s: ws-local init failed: %s", agent_id, e, exc_info=True)
            self.runtime.status = "error"
            self.runtime.error = str(e)
            self.runtime.save(agent_id)
            self._warm_done.set()
            return

        self.runtime.status = "running"
        self.runtime.save(agent_id)
        self._warm_done.set()
        if self._ws_local_hub is not None:
            self._ws_local_hub.register(point)

        # Push display_name / avatar_url / role / soul to the server identity.
        # The regular runtimes do this post-warm; ws-local idle skips that path,
        # so without this a freshly-created ws-local agent shows blank in the UI.
        async def _ws_local_profile_sync() -> None:
            from .profile_sync import sync_full_profile

            try:
                await sync_full_profile(self.agent_cfg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent %s: ws-local profile sync failed: %s", agent_id, exc)

        asyncio.ensure_future(_ws_local_profile_sync())
        logger.info("agent %s: ws-local idle, awaiting tool attach", agent_id)
        try:
            await self._stop.wait()
        finally:
            if self._ws_local_hub is not None:
                self._ws_local_hub.unregister(point)
            self.runtime.status = "stopped"
            self.runtime.save(agent_id)

    async def _run(self) -> None:
        if (self.agent_cfg.runtime.kind or "") == RUNTIME_WS_LOCAL:
            await self._run_ws_local()
            return
        agent_id = self.agent_cfg.id
        try:
            self._adapter = build_adapter(self.daemon_cfg, self.agent_cfg)
            profile_path = str(self.agent_cfg.resolve_profile_path())
            memory_path = str(self.agent_cfg.resolve_memory_dir())
            workspace_path = str(self.agent_cfg.resolve_workspace_dir())
            claude_path = str(self.agent_cfg.resolve_claude_dir())
            Path(workspace_path).mkdir(parents=True, exist_ok=True)
            _seed_claude_dir(Path(claude_path))

            # Managed CLAUDE.md (primer + profile + memory) written user-level for
            # auto-discovery; project-level stays agent-editable. chat/sdk-local
            # get the same string as system_prompt.
            shared_path = docker_shared_dir()
            claude_md = _rebuild_managed_system_prompt(
                harness_name=(self.agent_cfg.runtime.harness or "").strip(),
                agent_id=agent_id,
                shared_path=shared_path,
                profile_path=profile_path,
                memory_path=memory_path,
                workspace_path=workspace_path,
                display_name=self.agent_cfg.display_name,
                role=self.agent_cfg.role,
                role_short=self.agent_cfg.role_short,
            )

            # One-time migration: remove an older project-level
            # managed CLAUDE.md, but only if it still carries our
            # managed-content marker — never clobber user content.
            old_managed = Path(claude_path) / "CLAUDE.md"
            if looks_like_managed_claude_md(old_managed):
                try:
                    old_managed.unlink()
                    logger.info(
                        "agent %s: migrated stale managed CLAUDE.md out of %s",
                        agent_id, old_managed,
                    )
                except OSError as exc:
                    logger.warning(
                        "agent %s: could not remove stale %s: %s",
                        agent_id, old_managed, exc,
                    )

            puffo = PuffoAgent(
                adapter=self._adapter,
                system_prompt=claude_md,
                memory_dir=memory_path,
                workspace_dir=workspace_path,
                claude_dir=claude_path,
                agent_id=agent_id,
            )

            if not self.agent_cfg.puffo_core.is_configured():
                raise RuntimeError(
                    f"agent {agent_id!r}: puffo_core block in agent.yml "
                    "is incomplete. Required fields: server_url, slug, "
                    "device_id, space_id."
                )
            client = _build_puffo_core_client(
                self.agent_cfg, agent_id, daemon_cfg=self.daemon_cfg,
            )
            self._client = client

            # chat-local: now that the message client exists, inject
            # the in-process puffo_core send-tool dispatch so the
            # model's structured send_message calls post for real
            # (the counterpart of sdk-local's mcp_servers_override).
            from ..agent.adapters.chat_only import ChatOnlyAdapter
            if isinstance(self._adapter, ChatOnlyAdapter):
                self._adapter.tool_dispatch = (
                    _build_chat_local_tool_dispatch(client)
                )
        except Exception as e:
            logger.error("agent %s: failed to initialise: %s", agent_id, e, exc_info=True)
            self.runtime.status = "error"
            self.runtime.error = str(e)
            self.runtime.save(agent_id)
            # Init crashed before warm() — release the startup gate.
            self._warm_done.set()
            return

        # Per-agent refresh_ping retired; daemon-level CredentialRefresher is
        # the single writer of ~/.claude/.credentials.json.

        # Warm the adapter so persisted-session agents re-spawn their
        # subprocess now rather than on the first DM. Non-fatal.
        warm_ok = False
        try:
            await self._adapter.warm(claude_md)
            warm_ok = True
        except Exception as exc:
            logger.warning(
                "agent %s: warm() failed (will retry on first turn): %s",
                agent_id, exc,
            )
        if warm_ok:
            await self._run_post_warm_gate(agent_id)
        else:
            # Warm failed; release the startup gate so the daemon's
            # wait_warm doesn't block forever. Probe would have nothing
            # to verify anyway since the adapter never came up.
            self._warm_done.set()

        # Per-agent counterpart to daemon-startup full-sync: covers
        # paused→running flips + restart.flag respawns. Fire-and-
        # forget; never blocks listen().
        async def _post_warm_sync() -> None:
            from .profile_sync import sync_full_profile
            try:
                await sync_full_profile(self.agent_cfg)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent %s: post-warm profile sync failed: %s",
                    agent_id, exc,
                )

        asyncio.ensure_future(_post_warm_sync())

        pa_dir = Path(workspace_path) / ".puffo-agent"
        refresh_agent_flag_path = pa_dir / "refresh_agent.flag"
        refresh_host_sync_flag_path = pa_dir / "refresh_host_sync.flag"
        refresh_session_flag_path = pa_dir / "refresh_session.flag"
        # One durable global scheduler and one persistent semantic coordinator
        # own every provider turn.
        from ..agent.global_inbox_runtime import (
            ActiveBoundaryAdapter,
            BaselineAdapter,
            GlobalInboxRuntime,
            TrackingSendDelegate,
            await_listener_with_runtime,
        )
        from ..agent.send_coordinator import SemanticSendRequest, SendCoordinator
        from .ws_local.in_process_data_client import InProcessDataClient

        global_runtime: GlobalInboxRuntime

        async def run_global_turn(planned):
            async def execute_provider_turn():
                self._turn_active = True
                try:
                    async with self._reload_lock:
                        await _process_refresh_flags(
                            agent_id=agent_id,
                            harness_name=(
                                self.agent_cfg.runtime.harness or ""
                            ).strip(),
                            shared_path=shared_path,
                            profile_path=profile_path,
                            memory_path=memory_path,
                            workspace_path=workspace_path,
                            display_name=self.agent_cfg.display_name,
                            role=self.agent_cfg.role,
                            role_short=self.agent_cfg.role_short,
                            puffo=puffo,
                            adapter=self._adapter,
                            refresh_agent_flag=refresh_agent_flag_path,
                            refresh_host_sync_flag=refresh_host_sync_flag_path,
                            refresh_session_flag=refresh_session_flag_path,
                        )
                    self._maybe_wake_refresher_if_auth_failed(agent_id)
                    if (
                        self._ensure_fresh_token is not None
                        and self.agent_cfg.runtime.kind in (
                            RUNTIME_CLI_LOCAL, RUNTIME_CLI_DOCKER,
                        )
                        and not await self._ensure_fresh_token()
                    ):
                        self._enter_auth_failed(agent_id)
                        raise AgentAPIError(
                            "credential refresh failed before delivery",
                            is_auth=True,
                        )
                    Worker._flip_health_in_progress(
                        self.runtime, agent_id, logger
                    )
                    return await puffo.handle_global_inbox_turn(planned)
                finally:
                    self._turn_active = False

            try:
                reply = await execute_provider_turn()
            except AgentAPIError as exc:
                if exc.is_auth:
                    self._enter_auth_failed(agent_id)
                raise
            else:
                Worker._resolve_health_on_success(
                    self.runtime, agent_id, logger
                )
            self.runtime.msg_count += 1
            self.runtime.last_event_at = int(time.time())
            if not reply:
                return
            # Plain output has exactly one legal implicit route. Every explicit,
            # failed, attachment, or held attempt suppresses it.
            if len(planned.targets) != 1 or global_runtime.attempts.attempts:
                logger.warning(
                    "agent %s: suppressed ambiguous global plain output "
                    "(targets=%d attempts=%d)",
                    agent_id, len(planned.targets), global_runtime.attempts.attempts,
                )
                return
            target = planned.targets[0]
            if target[0] == "dm":
                destination = f"@{target[1]}"
                root_id = ""
            else:
                destination = target[2]
                root_id = target[3] if target[0] == "thread" else ""
            await global_runtime.send_delegate.send(SemanticSendRequest(
                destination=destination,
                text=reply,
                root_id=root_id,
            ))

        async def retry_global_turn(planned):
            self._turn_active = True
            try:
                try:
                    reply = await puffo.handle_global_inbox_retry(planned)
                finally:
                    self._turn_active = False
            except AgentAPIError as exc:
                if exc.is_auth:
                    self._enter_auth_failed(agent_id)
                raise
            else:
                Worker._resolve_health_on_success(self.runtime, agent_id, logger)
            if not reply:
                return
            if len(planned.targets) != 1 or global_runtime.attempts.attempts:
                logger.warning(
                    "agent %s: suppressed ambiguous global retry output "
                    "(targets=%d attempts=%d)",
                    agent_id, len(planned.targets), global_runtime.attempts.attempts,
                )
                return
            target = planned.targets[0]
            destination = f"@{target[1]}" if target[0] == "dm" else target[2]
            root_id = target[3] if target[0] == "thread" else ""
            await global_runtime.send_delegate.send(SemanticSendRequest(
                destination=destination,
                text=reply,
                root_id=root_id,
            ))

        run_global_turn.handle_global_inbox_retry = retry_global_turn

        global_runtime = GlobalInboxRuntime(
            store=client.store,
            adapter=self._adapter,
            run_turn=run_global_turn,
            workspace=workspace_path,
            held_catchup=client.recover_pending_delivery,
            send_mode_keys=(agent_id, client.slug),
            agent_id=agent_id,
        )
        coordinator = SendCoordinator(
            slug=client.slug,
            keystore=client.keystore,
            http_client=client.http,
            data_client=InProcessDataClient(client.store, client),
            workspace=workspace_path,
            baseline_source=BaselineAdapter(client.store),
            active_turn_source=ActiveBoundaryAdapter(
                client.store, global_runtime.active
            ),
            held_recovery_source=global_runtime.held_recovery_source,
        )
        global_runtime.coordinator = coordinator
        global_runtime.send_delegate = TrackingSendDelegate(
            coordinator, global_runtime.attempts, global_runtime
        )
        client.global_runtime = global_runtime
        client.send_coordinator = coordinator
        client.send_delegate = global_runtime.send_delegate
        global_runtime_task = asyncio.ensure_future(global_runtime.run())

        async def heartbeat():
            interval = max(1.0, self.daemon_cfg.runtime_heartbeat_seconds)
            while not self._stop.is_set():
                self.runtime.save(agent_id)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass

        # Proactive between-turns reload: bind the turn-start
        # ``_process_refresh_flags`` primitive over this run's locals so
        # the watcher applies the SAME reload the turn-start path does.
        refresh_flag_paths = (
            refresh_agent_flag_path,
            refresh_host_sync_flag_path,
            refresh_session_flag_path,
        )

        async def apply_refresh() -> None:
            await _process_refresh_flags(
                agent_id=agent_id,
                harness_name=(self.agent_cfg.runtime.harness or "").strip(),
                shared_path=shared_path,
                profile_path=profile_path,
                memory_path=memory_path,
                workspace_path=workspace_path,
                display_name=self.agent_cfg.display_name,
                role=self.agent_cfg.role,
                role_short=self.agent_cfg.role_short,
                puffo=puffo,
                adapter=self._adapter,
                refresh_agent_flag=refresh_agent_flag_path,
                refresh_host_sync_flag=refresh_host_sync_flag_path,
                refresh_session_flag=refresh_session_flag_path,
            )

        # PUF-221: per-agent credential_refresh coroutine retired —
        # CredentialRefresher in portal/credential_refresh.py owns the
        # refresh loop daemon-wide. Single writer = no multi-process
        # rotation race on Anthropic's single-use refresh tokens.

        # Status reporter: heartbeat task + inline begin/end_turn; no-op
        # without an http client (tests). Lazy provider reads live runtime.health.
        reporter = (
            self._build_status_reporter(client)
            if hasattr(client, "http")
            else None
        )
        if reporter is None:  # pragma: no cover — defensive
            class _NoopReporter:
                async def begin_turn(self, _mid):
                    return None
                async def end_turn(self, *_a, **_kw):
                    return None
                async def end_turn_batch(self, *_a, **_kw):
                    return None
                async def report_error(self, _t):
                    return None
                async def run_heartbeat_loop(self):
                    return None
                def stop(self):
                    return None
            reporter = _NoopReporter()  # type: ignore[assignment]

        hb_task = asyncio.ensure_future(heartbeat())
        status_task = asyncio.ensure_future(reporter.run_heartbeat_loop())
        watch_task = asyncio.ensure_future(
            self._refresh_watcher_loop(refresh_flag_paths, apply_refresh)
        )
        try:
            while not self._stop.is_set():
                try:
                    await await_listener_with_runtime(
                        client.listen(),
                        global_runtime_task,
                        label=f"agent {agent_id} global inbox runtime",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if global_runtime_task.done():
                        logger.error(
                            "agent %s: stopping after global inbox runtime failure: %s",
                            agent_id,
                            exc,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                        self.runtime.error = f"{type(exc).__name__}: {exc}"
                        self.runtime.save(agent_id)
                        try:
                            await reporter.report_error(self.runtime.error)
                        except Exception:
                            pass
                        raise
                    logger.warning(
                        "agent %s: listen() crashed: %s: %s — reconnecting in %.1fs",
                        agent_id, type(exc).__name__, exc, RECONNECT_BACKOFF_SECONDS,
                    )
                    self.runtime.error = f"{type(exc).__name__}: {exc}"
                    self.runtime.save(agent_id)
                    # Surface the failure on the agent's row so the
                    # operator sees it without tailing logs.
                    try:
                        await reporter.report_error(self.runtime.error or "listen crashed")
                    except Exception:
                        pass
                if self._stop.is_set():
                    break
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=RECONNECT_BACKOFF_SECONDS)
                except asyncio.TimeoutError:
                    pass
        finally:
            global_runtime.stop()
            global_runtime_task.cancel()
            try:
                await global_runtime_task
            except (asyncio.CancelledError, Exception):
                pass
            reporter.stop()
            hb_task.cancel()
            status_task.cancel()
            watch_task.cancel()
            for task in (hb_task, status_task, watch_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self.runtime.status = "stopped"
            self.runtime.save(agent_id)


async def _process_refresh_flags(
    *,
    agent_id: str,
    harness_name: str,
    shared_path: Path,
    profile_path: str,
    memory_path: str,
    workspace_path: str,
    puffo,
    adapter,
    refresh_agent_flag: Path,
    refresh_host_sync_flag: Path,
    refresh_session_flag: Path,
    display_name: str = "",
    role: str = "",
    role_short: str = "",
) -> None:
    """Consume any worker-scope refresh flags into a single
    ``adapter.reload(prompt, with_session=…)`` call at turn start.
    Order: host sync → CLAUDE.md rebuild → session drop. A changed
    primer/profile slice drops the session too (``--resume`` would replay
    the stale baked prompt); memory-only or no-op rebuilds keep it."""
    host_sync_seen = refresh_host_sync_flag.exists()
    agent_seen = refresh_agent_flag.exists()
    session_seen = refresh_session_flag.exists()
    if not (host_sync_seen or agent_seen or session_seen):
        return

    if host_sync_seen:
        try:
            from .state import (
                agent_home_dir,
                sync_host_mcp_servers,
                sync_host_skills,
            )
            host_home = Path.home()
            ah = agent_home_dir(agent_id)
            skill_count = sync_host_skills(host_home, ah)
            merged_mcp, _unreach = sync_host_mcp_servers(host_home, ah)
            logger.info(
                "agent %s: refresh_host_sync (skills=%d mcp=%d)",
                agent_id, skill_count, merged_mcp,
            )
        except Exception as exc:
            logger.warning(
                "agent %s: refresh_host_sync failed: %s", agent_id, exc,
            )

    new_prompt: str | None = None
    prompt_changed = False
    if agent_seen:
        try:
            new_prompt = _rebuild_managed_system_prompt(
                harness_name=harness_name,
                agent_id=agent_id,
                shared_path=shared_path,
                profile_path=profile_path,
                memory_path=memory_path,
                workspace_path=workspace_path,
                display_name=display_name,
                role=role,
                role_short=role_short,
            )
            from ..agent.shared_content import MEMORY_SECTION_HEADER

            def _session_core(prompt: str) -> str:
                return prompt.split(MEMORY_SECTION_HEADER, 1)[0]

            prompt_changed = _session_core(new_prompt) != _session_core(
                puffo.system_prompt
            )
            puffo.system_prompt = new_prompt
            logger.info(
                "agent %s: system prompt rebuilt from disk (changed=%s)",
                agent_id, prompt_changed,
            )
        except Exception as exc:
            logger.warning(
                "agent %s: refresh_agent failed: %s", agent_id, exc,
            )

    try:
        await adapter.reload(
            new_prompt if new_prompt is not None else puffo.system_prompt,
            with_session=session_seen or prompt_changed,
        )
    except Exception as exc:
        logger.warning(
            "agent %s: adapter.reload after refresh failed: %s",
            agent_id, exc,
        )

    for flag in (refresh_host_sync_flag, refresh_agent_flag, refresh_session_flag):
        try:
            flag.unlink()
        except OSError:
            pass


_CLAUDE_DIR_SUBDIRS = ("agents", "commands", "skills", "hooks")


def _seed_claude_dir(claude_dir: Path) -> None:
    """Create the Claude Code project-level skeleton (agents/,
    commands/, skills/, hooks/). Idempotent — never overwrites."""
    claude_dir.mkdir(parents=True, exist_ok=True)
    for sub in _CLAUDE_DIR_SUBDIRS:
        (claude_dir / sub).mkdir(exist_ok=True)
