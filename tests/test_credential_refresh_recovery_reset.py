"""PUF-349: recovery must leave clean state, and a benign UNCHANGED
must not be read as a failure.

Two gaps, one observed symptom. ``_detect_external_rotation`` fired
refresh-success but left ``_consecutive_failed`` and the on-disk
``refresh_broken`` flag alone — only the probe-driven REFRESHED path
reset those. And the streak counter treated UNCHANGED like FAILED, even
though UNCHANGED with a fresh token means "nothing needed doing".
Together: operator re-login recovered the daemon, then the next benign
probe re-flagged every agent as broken.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from puffo_agent.portal.credential_refresh import (
    REFRESH_BROKEN_THRESHOLD,
    REFRESH_SAFETY_MARGIN_SECONDS,
    CredentialRefresher,
    RefreshOutcome,
)


def _write_creds(host_home: Path, *, expires_in_seconds: int) -> Path:
    creds_path = host_home / ".claude" / ".credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-test",
            "refreshToken": "sk-ant-ort01-test",
            "expiresAt": int((time.time() + expires_in_seconds) * 1000),
            "scopes": ["user:inference"],
        }
    }))
    return creds_path


def _refresher(tmp_path: Path, monkeypatch, *, expires_in_seconds: int,
               agent_id: str = "agent-puf349"):
    from puffo_agent.portal.state import RuntimeState, agent_home_dir

    monkeypatch.setenv("PUFFO_AGENT_HOME", str(tmp_path))
    _write_creds(tmp_path / "host", expires_in_seconds=expires_in_seconds)
    RuntimeState(status="running", started_at=int(time.time())).save(agent_id)

    r = CredentialRefresher(host_home=tmp_path / "host")
    r.register_agent(agent_home_dir(agent_id))
    return r, agent_id


def _break_it(r) -> None:
    for _ in range(REFRESH_BROKEN_THRESHOLD):
        r._propagate_outcome(RefreshOutcome.FAILED)


# ── gap 1: external-rotation recovery resets the same state ────────


def test_external_rotation_resets_the_failure_counter(tmp_path, monkeypatch):
    r, _aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    monkeypatch.setattr(r.backend, "fingerprint", lambda: (2, 11))
    r._last_cred_fingerprint = (1, 10)
    r._consecutive_failed = 1

    r._detect_external_rotation()

    assert r._consecutive_failed == 0


def test_external_rotation_clears_refresh_broken_on_disk(tmp_path, monkeypatch):
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    _break_it(r)
    assert RuntimeState.load(aid).health == "refresh_broken"

    monkeypatch.setattr(r.backend, "fingerprint", lambda: (2, 11))
    r._last_cred_fingerprint = (1, 10)
    r._detect_external_rotation()

    rs = RuntimeState.load(aid)
    assert rs.health == "ok"
    assert rs.error == ""


def test_external_rotation_still_fires_refresh_success(tmp_path, monkeypatch):
    # The reset must be additive: the daemon's restart-auth_failed-agents
    # callback still has to run.
    r, _aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    fired: list[int] = []
    r.register_on_refresh_success(lambda: fired.append(1))
    monkeypatch.setattr(r.backend, "fingerprint", lambda: (2, 11))
    r._last_cred_fingerprint = (1, 10)

    r._detect_external_rotation()

    assert fired == [1]


def test_unchanged_fingerprint_does_not_reset(tmp_path, monkeypatch):
    # No rotation → no recovery. A quiet tick must not clear a real
    # failure streak.
    r, _aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    monkeypatch.setattr(r.backend, "fingerprint", lambda: (1, 10))
    r._last_cred_fingerprint = (1, 10)
    r._consecutive_failed = 1

    r._detect_external_rotation()

    assert r._consecutive_failed == 1


# ── gap 2: UNCHANGED is only trouble when the token is not fresh ───


def test_unchanged_with_fresh_token_does_not_count_toward_broken(
    tmp_path, monkeypatch,
):
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)

    for _ in range(REFRESH_BROKEN_THRESHOLD * 3):
        r._propagate_outcome(RefreshOutcome.UNCHANGED)

    assert r._consecutive_failed == 0
    assert r._consecutive_unchanged == REFRESH_BROKEN_THRESHOLD * 3
    assert RuntimeState.load(aid).health != "refresh_broken"


def test_unchanged_with_expiring_token_still_flips_broken(tmp_path, monkeypatch):
    # Detection must not be lost. This is the FileBackend case that logs
    # "claude may not be rewriting credentials.json ... operator may need
    # `claude /login`" — a refresh we needed and didn't get.
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(
        tmp_path, monkeypatch,
        expires_in_seconds=REFRESH_SAFETY_MARGIN_SECONDS - 60,
    )

    for _ in range(REFRESH_BROKEN_THRESHOLD):
        r._propagate_outcome(RefreshOutcome.UNCHANGED)

    assert r._consecutive_failed == REFRESH_BROKEN_THRESHOLD
    assert RuntimeState.load(aid).health == "refresh_broken"


def test_missing_credential_counts_unchanged_as_trouble(tmp_path, monkeypatch):
    # expires_in_seconds() returns None with no host file at all. That is
    # not a fresh token, so it must not take the benign branch.
    r, _aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    (tmp_path / "host" / ".claude" / ".credentials.json").unlink()

    r._propagate_outcome(RefreshOutcome.UNCHANGED)

    assert r._consecutive_failed == 1


def test_benign_unchanged_does_not_clear_an_existing_break(tmp_path, monkeypatch):
    # Not counting toward the streak is not the same as proving recovery.
    # Only a REFRESHED or a detected rotation clears the flag.
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    _break_it(r)

    r._propagate_outcome(RefreshOutcome.UNCHANGED)

    assert RuntimeState.load(aid).health == "refresh_broken"


# ── the reported trace, end to end ─────────────────────────────────


def test_relogin_then_benign_probe_does_not_reflag(tmp_path, monkeypatch):
    """The 2026-07-02 08:41:31–08:41:36 sequence.

    Agents were refresh_broken; the operator re-logged in; five seconds
    later an already-pending probe returned UNCHANGED because the token
    it found was the fresh one. Before the fix the counter — never reset
    by the rotation path — crossed the threshold again immediately.
    """
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    _break_it(r)
    assert RuntimeState.load(aid).health == "refresh_broken"

    # Operator re-login lands.
    monkeypatch.setattr(r.backend, "fingerprint", lambda: (2, 11))
    r._last_cred_fingerprint = (1, 10)
    r._detect_external_rotation()
    assert RuntimeState.load(aid).health == "ok"

    # The probe that was already in flight reports back.
    r._propagate_outcome(RefreshOutcome.UNCHANGED)
    r._propagate_outcome(RefreshOutcome.UNCHANGED)

    assert RuntimeState.load(aid).health == "ok"
    assert r._consecutive_failed == 0


def test_genuine_failure_after_recovery_still_flags(tmp_path, monkeypatch):
    # The counter resets, it doesn't become sticky-clean.
    from puffo_agent.portal.state import RuntimeState
    r, aid = _refresher(tmp_path, monkeypatch, expires_in_seconds=3600)
    monkeypatch.setattr(r.backend, "fingerprint", lambda: (2, 11))
    r._last_cred_fingerprint = (1, 10)
    r._detect_external_rotation()

    for _ in range(REFRESH_BROKEN_THRESHOLD):
        r._propagate_outcome(RefreshOutcome.FAILED)

    assert RuntimeState.load(aid).health == "refresh_broken"
