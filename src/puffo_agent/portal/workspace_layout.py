"""Filesystem layout shared by all daemon-owned Agent workspaces."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)
SHARED_WORKSPACE_NAME = "shared"


def _same_location(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _create_windows_junction(link: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        completed = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return False
    return completed.returncode == 0


def ensure_workspace_shared_link(workspace: Path, shared_root: Path) -> str:
    """Expose one Puffo-home shared root as ``<workspace>/shared``.

    Existing real files and directories are never replaced. A stale symlink is
    safe to replace because the link itself owns no content. Returns one of
    ``created``, ``existing``, ``conflict``, or ``unavailable``.
    """
    workspace = workspace.expanduser()
    shared_root = shared_root.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    shared_existed = shared_root.exists()
    shared_root.mkdir(parents=True, exist_ok=True)
    if not shared_existed and os.name != "nt":
        shared_root.chmod(0o700)

    link = workspace / SHARED_WORKSPACE_NAME
    if link.exists() and _same_location(link, shared_root):
        return "existing"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        logger.warning(
            "workspace shared path %s already contains local data; preserving it "
            "instead of replacing it with the Puffo shared workspace",
            link,
        )
        return "conflict"

    relative_target = os.path.relpath(shared_root, start=workspace)
    try:
        os.symlink(relative_target, link, target_is_directory=True)
    except OSError as exc:
        if _create_windows_junction(link, shared_root):
            return "created"
        logger.warning(
            "could not expose Puffo shared workspace %s at %s: %s",
            shared_root,
            link,
            exc,
        )
        return "unavailable"
    return "created"
