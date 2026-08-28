"""Native launch-argument supplements for ACP executables.

ACP v1 has no model-selection verb, but the CLIs we target accept one at
launch. The supplement lives in this table instead of per-CLI branches in
the driver: the driver keeps a single wire format, and a new ACP target
that wants model selection adds one verified row here, not code.

Rows are added only after the flag is verified against the actual CLI —
a guessed flag would either error the spawn or, worse, be swallowed and
leave the CLI on its default model while we report the requested one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# executable basename -> verified model flag.
_MODEL_FLAG_BY_EXECUTABLE = {
    "gemini": "-m",  # gemini-cli 0.57.0, verified
    "kimi": "-m",  # kimi-cli 1.49.0, verified
}

# Any of these in operator launch_args means the operator already pinned a
# model; the supplement must not fight explicit config.
_MODEL_FLAG_SPELLINGS = ("-m", "--model")


def _basename(executable: str) -> str:
    name = Path(executable).name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def operator_pinned_model(launch_args: Sequence[str]) -> bool:
    """True when launch_args already carry a model flag (any spelling)."""
    return any(
        arg in _MODEL_FLAG_SPELLINGS or arg.startswith("--model=")
        for arg in launch_args
    )


def model_launch_args(executable: str, model: str) -> tuple[str, ...]:
    """Launch args that start ``executable`` on ``model``.

    Empty when no model was requested or the target has no verified
    preset — the caller then declares model selection unsupported for
    this spawn instead of guessing a flag.
    """
    if not model:
        return ()
    flag = _MODEL_FLAG_BY_EXECUTABLE.get(_basename(executable))
    if flag is None:
        return ()
    return (flag, model)
