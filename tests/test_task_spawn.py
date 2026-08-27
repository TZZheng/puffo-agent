"""Worker tasks must not die unclaimed: spawn attaches an exception-logging
done-callback, and no bare ensure_future/create_task remains in the tree."""

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from puffo_agent.tasks import spawn

_SRC = Path(__file__).resolve().parent.parent / "src" / "puffo_agent"
_HELPER = _SRC / "tasks.py"
_SPAWNERS = {"ensure_future", "create_task"}
_SPAWNER_OWNERS = {"asyncio", "loop"}


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_clean_completion_logs_nothing(caplog):
    async def ok():
        return 7

    with caplog.at_level(logging.DEBUG, logger="puffo_agent.tasks"):
        task = spawn(ok(), name="ok")
        assert await task == 7
        await _settle()

    assert caplog.records == []


@pytest.mark.asyncio
async def test_unclaimed_exception_is_logged_with_name_and_traceback(caplog):
    async def boom():
        raise ValueError("message backup key has invalid length")

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        spawn(boom(), name="reminder_sync.run")
        await _settle()

    records = _errors(caplog)
    assert len(records) == 1
    record = records[0]
    assert "reminder_sync.run" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)
    assert "message backup key has invalid length" in str(record.exc_info[1])


@pytest.mark.asyncio
async def test_cancelled_task_is_not_reported(caplog):
    started = asyncio.Event()

    async def sleeper():
        started.set()
        await asyncio.sleep(3600)

    with caplog.at_level(logging.DEBUG, logger="puffo_agent.tasks"):
        task = spawn(sleeper(), name="sleeper")
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _settle()

    assert caplog.records == []


@pytest.mark.asyncio
async def test_returned_task_is_named_and_awaitable():
    async def ok():
        return "value"

    task = spawn(ok(), name="named")
    assert isinstance(task, asyncio.Task)
    assert task.get_name() == "named"
    assert await task == "value"


@pytest.mark.asyncio
async def test_spawn_without_name_still_reports(caplog):
    async def boom():
        raise RuntimeError("nameless")

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        spawn(boom())
        await _settle()

    records = _errors(caplog)
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], RuntimeError)


@pytest.mark.asyncio
async def test_plain_future_is_reported_without_set_name(caplog):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        returned = spawn(future, name="ignored-on-future")
        assert returned is future
        future.set_exception(OSError("disk"))
        await _settle()

    records = _errors(caplog)
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], OSError)


@pytest.mark.asyncio
async def test_existing_done_callback_still_runs(caplog):
    tasks: set[asyncio.Future] = set()

    async def boom():
        raise ValueError("both callbacks")

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        task = spawn(boom(), name="housekept")
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        await _settle()

    assert tasks == set()
    assert len(_errors(caplog)) == 1


@pytest.mark.asyncio
async def test_awaited_companion_still_receives_the_exception(caplog):
    async def boom():
        raise ValueError("claimed")

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        task = spawn(boom(), name="claimed")
        with pytest.raises(ValueError):
            await task
        await _settle()

    assert len(_errors(caplog)) == 1


@pytest.mark.asyncio
async def test_start_services_shaped_failure_surfaces(caplog):
    """Prof-Puffo shape: bring-up spawns a service task nobody awaits."""
    ready = asyncio.Event()

    async def prepare_reminder_sync():
        raise ValueError("message backup key has invalid length")

    async def start_services():
        spawn(prepare_reminder_sync(), name="reminder_sync.run")
        ready.set()

    with caplog.at_level(logging.ERROR, logger="puffo_agent.tasks"):
        await start_services()
        await ready.wait()
        await _settle()

    records = _errors(caplog)
    assert len(records) == 1
    assert "reminder_sync.run" in records[0].getMessage()


def test_spawn_without_running_loop_raises():
    async def never():
        return None

    coro = never()
    try:
        with pytest.raises(RuntimeError):
            spawn(coro, name="no-loop")
    finally:
        coro.close()


def _bare_spawn_sites(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _SPAWNERS:
            continue
        if isinstance(func.value, ast.Name) and func.value.id in _SPAWNER_OWNERS:
            hits.append(f"{path}:{node.lineno} {func.value.id}.{func.attr}")
    return hits


def test_no_bare_task_spawn_remains_in_tree():
    sources = sorted(p for p in _SRC.rglob("*.py") if p != _HELPER)
    assert sources, "source sweep found no files"
    hits = [site for path in sources for site in _bare_spawn_sites(path)]
    assert hits == []


def test_helper_is_the_only_ensure_future_owner():
    hits = _bare_spawn_sites(_HELPER)
    assert len(hits) == 1
    assert "asyncio.ensure_future" in hits[0]
