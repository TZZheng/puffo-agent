"""Per-provider model catalogs.

Each harness exposes selectable models = aliases (the CLI resolves
these to the latest model in the family at runtime, so they never go
stale) + concrete versions. claude-code refreshes its concrete list
from the live, account-authoritative ``/v1/models``; codex reads its
local CLI cache; Pi and OpenCode ask their installed CLI for the
operator-visible catalog; the rest are static.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOption:
    id: str  # the ``--model`` value; "" means the daemon default
    label: str  # combo-box display text
    is_alias: bool = False


_DAEMON_DEFAULT = ModelOption("", "(daemon default)")

# CLI aliases — claude-code resolves these to the latest model in the
# family at call time, so they track new releases with no edits here.
_CLAUDE_ALIASES: tuple[ModelOption, ...] = (
    ModelOption("opus", "opus — latest Opus", is_alias=True),
    ModelOption("sonnet", "sonnet — latest Sonnet", is_alias=True),
)

# Models filtered out of the live ``/v1/models`` result — old dated
# point-releases + the haiku tier — to keep the picker to opus/sonnet.
_BLOCKED_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
})

# Offline fallback for claude-code — only consulted when ``/v1/models``
# is unreachable (the aliases + the live refresh otherwise keep it
# current).
_CLAUDE_STATIC: tuple[ModelOption, ...] = (
    ModelOption("claude-opus-4-8", "Claude Opus 4.8"),
    ModelOption("claude-opus-4-7", "Claude Opus 4.7"),
    ModelOption("claude-opus-4-6", "Claude Opus 4.6"),
    ModelOption("claude-sonnet-4-6", "Claude Sonnet 4.6"),
)

# codex reads its own local model cache (see _codex_models); these are
# the fallback when that cache is unreadable.
_CODEX_STATIC: tuple[ModelOption, ...] = (
    ModelOption("gpt-5.5", "GPT-5.5"),
    ModelOption("gpt-5.4", "GPT-5.4"),
    ModelOption("gpt-5.4-mini", "GPT-5.4-Mini"),
)

# hermes / gemini-cli are static for now.
# TODO: a dynamic source for gemini (Google API) like claude / codex.
_STATIC: dict[str, tuple[ModelOption, ...]] = {
    "hermes": (
        ModelOption("gpt-5.5", "GPT-5.5"),
        ModelOption("gpt-5.4", "GPT-5.4"),
        ModelOption("opus", "opus — latest Opus", is_alias=True),
        ModelOption("sonnet", "sonnet — latest Sonnet", is_alias=True),
    ),
    "gemini-cli": (
        ModelOption("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ModelOption("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ),
}

# Harnesses the catalog can answer for. Pi/OpenCode deliberately use their
# own effective local catalogs instead of borrowing Codex's model list.
KNOWN_HARNESSES: tuple[str, ...] = (
    "claude-code",
    "codex",
    "pi",
    "opencode",
    "gemini-cli",
    "hermes",
)

_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_CACHE_TTL_S = 3600.0
_FETCH_TIMEOUT_S = 6.0
_CLI_FETCH_TIMEOUT_S = 5.0

# harness -> (fetched_at, concrete_models). Guarded by _lock.
_cache: dict[str, tuple[float, tuple[ModelOption, ...]]] = {}
_lock = threading.Lock()


def _anthropic_oauth_token() -> str | None:
    """The operator's claude-code OAuth access token, or None."""
    path = Path.home() / ".claude" / ".credentials.json"
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return (creds.get("claudeAiOauth") or {}).get("accessToken")


def _fetch_anthropic_models() -> tuple[ModelOption, ...] | None:
    """Account-authoritative model list from ``/v1/models``. Returns
    None on any failure (no creds, network, auth) so callers fall back.
    """
    token = _anthropic_oauth_token()
    if not token:
        return None
    req = urllib.request.Request(
        _ANTHROPIC_MODELS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("anthropic /v1/models fetch failed: %s", exc)
        return None
    out = [
        ModelOption(m["id"], m.get("display_name") or m["id"])
        for m in data.get("data", [])
        if m.get("id") and m["id"] not in _BLOCKED_MODELS
    ]
    return tuple(out) or None


def _claude_concrete(*, fetch: bool) -> tuple[ModelOption, ...]:
    now = time.time()
    with _lock:
        cached = _cache.get("claude-code")
    if cached and now - cached[0] < _CACHE_TTL_S:
        return cached[1]
    if fetch:
        live = _fetch_anthropic_models()
        if live is not None:
            with _lock:
                _cache["claude-code"] = (now, live)
            return live
    # Serve the last-known list even if stale; else the static fallback.
    return cached[1] if cached else _CLAUDE_STATIC


def _codex_models() -> tuple[ModelOption, ...]:
    """The codex CLI's own local model cache. ``visibility == "list"``
    drops internal entries (e.g. codex-auto-review); the CLI's
    ``priority`` sets the order. Falls back to the static list when the
    cache is missing / unreadable. No block-list needed — ``visibility``
    does the filtering."""
    path = Path.home() / ".codex" / "models_cache.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _CODEX_STATIC
    listed = sorted(
        (
            m for m in data.get("models", [])
            if m.get("slug") and m.get("visibility") == "list"
        ),
        key=lambda m: m.get("priority", 9999),
    )
    out = tuple(
        ModelOption(m["slug"], m.get("display_name") or m["slug"]) for m in listed
    )
    return out or _CODEX_STATIC


def _valid_cli_model_id(value: str) -> bool:
    """True for a bounded, single-token model id safe to put on the wire."""
    return bool(value) and len(value) <= 512 and bool(
        re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9@][A-Za-z0-9@._:+/-]*", value)
    )


def _parse_pi_models(output: str) -> tuple[ModelOption, ...]:
    """Parse ``pi --list-models``'s whitespace-delimited table.

    Pi's model selector accepts the provider-qualified spelling. Keeping the
    provider is required because different providers may expose the same
    model id.
    """
    out: list[ModelOption] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        columns = raw.split()
        if len(columns) < 6 or columns[:2] == ["provider", "model"]:
            continue
        if columns[4] not in {"yes", "no"} or columns[5] not in {"yes", "no"}:
            continue
        model_id = f"{columns[0]}/{columns[1]}"
        if _valid_cli_model_id(model_id) and model_id not in seen:
            seen.add(model_id)
            out.append(ModelOption(model_id, model_id))
    return tuple(out)


def _parse_opencode_models(output: str) -> tuple[ModelOption, ...]:
    """Parse ``opencode models``'s provider-qualified, one-id-per-line output."""
    out: list[ModelOption] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        model_id = raw.strip()
        if (
            "/" in model_id
            and _valid_cli_model_id(model_id)
            and model_id not in seen
        ):
            seen.add(model_id)
            out.append(ModelOption(model_id, model_id))
    return tuple(out)


def _fetch_cli_models(harness: str) -> tuple[ModelOption, ...] | None:
    """Ask an installed harness for its effective local model catalog.

    ``None`` distinguishes discovery failure from a valid result. Callers keep
    serving a last-known-good catalog on transient CLI failures.
    """
    from .cli_bin import (
        normalize_launch_argv,
        resolve_opencode_bin,
        resolve_pi_bin,
    )

    if harness == "pi":
        executable = resolve_pi_bin()
        arguments = ("--list-models",)
        parser = _parse_pi_models
    elif harness == "opencode":
        executable = resolve_opencode_bin()
        arguments = ("models",)
        parser = _parse_opencode_models
    else:
        return None
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [*normalize_launch_argv(executable), *arguments],
            capture_output=True,
            text=True,
            timeout=_CLI_FETCH_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("%s model catalog command failed: %s", harness, exc)
        return None
    if completed.returncode != 0:
        logger.debug(
            "%s model catalog exited %s: %s",
            harness,
            completed.returncode,
            completed.stderr.strip(),
        )
        return None
    parsed = parser(completed.stdout)
    return parsed or None


def _cli_concrete(harness: str, *, fetch: bool) -> tuple[ModelOption, ...]:
    now = time.time()
    with _lock:
        cached = _cache.get(harness)
    if cached and now - cached[0] < _CACHE_TTL_S:
        return cached[1]
    if fetch:
        live = _fetch_cli_models(harness)
        if live is not None:
            with _lock:
                _cache[harness] = (now, live)
            return live
    return cached[1] if cached else ()


def provider_models(harness: str, *, fetch: bool = False) -> list[ModelOption]:
    """Selectable models for ``harness``: daemon-default + aliases +
    concrete versions.

    ``fetch`` may perform synchronous discovery for claude-code, Pi, and
    OpenCode (use off the UI thread — see ``prefetch``). When False it serves
    cached/static data without blocking. codex reads its local cache; the
    remaining harnesses are static.
    """
    if harness == "claude-code":
        # General aliases (opus/sonnet) sort after the concrete versions.
        return [_DAEMON_DEFAULT, *_claude_concrete(fetch=fetch), *_CLAUDE_ALIASES]
    if harness == "codex":
        return [_DAEMON_DEFAULT, *_codex_models()]
    if harness in {"pi", "opencode"}:
        return [_DAEMON_DEFAULT, *_cli_concrete(harness, fetch=fetch)]
    return [_DAEMON_DEFAULT, *_STATIC.get(harness, ())]


def prefetch() -> threading.Thread:
    """Warm dynamic catalogs in a background thread (call once at UI/daemon
    start so later ``provider_models`` reads normally hit cache).
    Returns the thread; callers may ignore it."""
    def _warm() -> None:
        for harness in ("claude-code", "pi", "opencode"):
            provider_models(harness, fetch=True)

    t = threading.Thread(
        target=_warm,
        daemon=True,
    )
    t.start()
    return t
