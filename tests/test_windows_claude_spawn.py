"""PUF-420: the claude adapter spawned a bare ``["claude"]``.

On POSIX ``execvp`` searches PATH and it works. On Windows
``CreateProcess`` does not, and an npm-installed CLI — which ships
``claude.cmd`` / ``claude.ps1`` and no ``.exe`` — failed with WinError 2
even though the shell resolved ``claude`` fine. The resolver was already
being called in ``_verify``; its answer was discarded.

There is no Windows CI, so the platform is faked. That is a real limit:
these pin argv construction, not that CreateProcess accepts the result.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from puffo_agent.agent import cli_bin
from puffo_agent.agent.cli_bin import spawn_argv, windows_runnable


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setattr(cli_bin.sys, "platform", "win32")


@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(cli_bin.sys, "platform", "linux")


def test_posix_passes_the_path_through(posix):
    assert spawn_argv("/usr/local/bin/claude") == ["/usr/local/bin/claude"]


def test_posix_never_wraps_even_for_a_cmd_suffix(posix):
    # A file literally named ``claude.cmd`` on Linux is still just a file.
    assert spawn_argv("/opt/claude.cmd") == ["/opt/claude.cmd"]


def test_exe_is_argv0_directly(win):
    assert spawn_argv(r"C:\Programs\claude\claude.exe") == [
        r"C:\Programs\claude\claude.exe"
    ]


@pytest.mark.parametrize("ext", [".cmd", ".bat"])
def test_cmd_shims_go_through_cmd_exe(win, ext):
    target = rf"C:\Users\j\AppData\Roaming\npm\claude{ext}"
    assert spawn_argv(target) == ["cmd.exe", "/c", target]


def test_ps1_goes_through_powershell_non_interactive(win):
    target = r"C:\Users\j\AppData\Roaming\npm\claude.ps1"
    assert spawn_argv(target) == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        target,
    ]


def test_suffix_matching_is_case_insensitive(win):
    assert spawn_argv(r"C:\npm\claude.CMD")[:2] == ["cmd.exe", "/c"]


# The npm layout: three files side by side, only two of them startable.
def _npm_layout(tmp_path: Path, *exts: str) -> Path:
    (tmp_path / "claude").write_text("#!/bin/sh\n", encoding="utf-8")
    for ext in exts:
        (tmp_path / f"claude{ext}").write_text("shim", encoding="utf-8")
    return tmp_path / "claude"


def test_extensionless_npm_script_is_upgraded_to_the_cmd_shim(win, tmp_path):
    # This is Jeremy's exact failure: shutil.which can hand back the
    # extensionless shell script, which CreateProcess rejects with the same
    # WinError 2 that "missing entirely" produces.
    bare = _npm_layout(tmp_path, ".cmd", ".ps1")
    assert windows_runnable(str(bare)) == str(tmp_path / "claude.cmd")
    assert spawn_argv(str(bare)) == ["cmd.exe", "/c", str(tmp_path / "claude.cmd")]


def test_exe_wins_over_cmd_and_ps1(win, tmp_path):
    bare = _npm_layout(tmp_path, ".exe", ".cmd", ".ps1")
    assert windows_runnable(str(bare)) == str(tmp_path / "claude.exe")


def test_cmd_wins_over_ps1(win, tmp_path):
    bare = _npm_layout(tmp_path, ".cmd", ".ps1")
    assert windows_runnable(str(bare)) == str(tmp_path / "claude.cmd")


def test_ps1_only_still_resolves(win, tmp_path):
    bare = _npm_layout(tmp_path, ".ps1")
    assert windows_runnable(str(bare)) == str(tmp_path / "claude.ps1")


def test_no_sibling_shim_leaves_the_path_alone(win, tmp_path):
    bare = _npm_layout(tmp_path)
    assert windows_runnable(str(bare)) == str(bare)


def test_windows_bundle_paths_cover_the_npm_global_shims(win):
    # Compared as whole strings, not Path.name: these are Windows paths and
    # the test host is POSIX, so backslashes aren't separators here and
    # %APPDATA% stays unexpanded.
    names = [str(p).lower() for p in cli_bin._claude_bundle_paths()]
    assert any(n.endswith(r"npm\claude.cmd") for n in names)
    assert any(n.endswith(r"npm\claude.ps1") for n in names)

    def first(suffix: str) -> int:
        return next(i for i, n in enumerate(names) if n.endswith(suffix))

    # .exe before .cmd before .ps1 — _first_executable takes the first hit.
    assert first(r"npm\claude.exe") < first(r"npm\claude.cmd")
    assert first(r"npm\claude.cmd") < first(r"npm\claude.ps1")


def test_missing_binary_message_names_the_windows_shim_case():
    msg = cli_bin.CLAUDE_BIN_MISSING
    assert "PUFFO_CLAUDE_BIN" in msg
    assert "claude.cmd" in msg
    # The raw failure is WinError 2, which reads as "not installed" and sent
    # the reporter down the wrong path. The message has to say otherwise.
    assert "npm" in msg


class TestBuildCommandUsesTheResolver:
    """The bug itself: argv[0] must come from the resolver, not a bare name."""

    def _adapter(self):
        from puffo_agent.agent.adapters.local_cli import LocalCLIAdapter

        return LocalCLIAdapter.__new__(LocalCLIAdapter)

    def _configure(self, adapter, tmp_path):
        adapter.permission_mode = "bypassPermissions"
        adapter.model = ""
        adapter.inference_level = ""
        adapter.auto_compact_threshold_pct = None
        adapter.mcp_config_file = tmp_path / "mcp.json"
        adapter.session_file = tmp_path / "session.json"
        adapter.workspace_dir = str(tmp_path)
        return adapter

    def test_argv0_is_the_resolved_absolute_path(self, monkeypatch, tmp_path):
        from puffo_agent.agent.adapters import local_cli

        monkeypatch.setattr(local_cli, "resolve_claude_bin", lambda: "/opt/bin/claude")
        monkeypatch.setattr(local_cli, "spawn_argv", lambda b: [b])
        adapter = self._configure(self._adapter(), tmp_path)
        cmd = local_cli.LocalCLIAdapter._build_command(adapter, [])
        assert cmd[0] == "/opt/bin/claude"
        assert cmd[0] != "claude"

    def test_windows_shim_wrapper_keeps_every_following_argument(
        self, monkeypatch, tmp_path
    ):
        from puffo_agent.agent.adapters import local_cli

        shim = r"C:\npm\claude.cmd"
        monkeypatch.setattr(local_cli, "resolve_claude_bin", lambda: shim)
        monkeypatch.setattr(
            local_cli, "spawn_argv", lambda b: ["cmd.exe", "/c", b]
        )
        adapter = self._configure(self._adapter(), tmp_path)
        cmd = local_cli.LocalCLIAdapter._build_command(adapter, ["--extra", "x"])
        assert cmd[:3] == ["cmd.exe", "/c", shim]
        # Wrapping must not eat the adapter's own flags.
        assert "--dangerously-skip-permissions" in cmd
        assert cmd[-2:] == ["--extra", "x"]

    def test_resolver_miss_raises_the_friendly_error_not_winerror(
        self, monkeypatch, tmp_path
    ):
        from puffo_agent.agent.adapters import local_cli

        monkeypatch.setattr(local_cli, "resolve_claude_bin", lambda: None)
        adapter = self._configure(self._adapter(), tmp_path)
        with pytest.raises(RuntimeError, match="PUFFO_CLAUDE_BIN"):
            local_cli.LocalCLIAdapter._build_command(adapter, [])
