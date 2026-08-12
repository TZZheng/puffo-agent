from puffo_agent.portal.workspace_layout import ensure_workspace_shared_link


def test_shared_workspace_link_is_common_and_preserves_conflicts(tmp_path):
    shared = tmp_path / "shared"
    alice = tmp_path / "agents" / "alice" / "workspace"
    bob = tmp_path / "agents" / "bob" / "workspace"

    assert ensure_workspace_shared_link(alice, shared) == "created"
    assert ensure_workspace_shared_link(bob, shared) == "created"
    assert ensure_workspace_shared_link(alice, shared) == "existing"
    (alice / "shared" / "handoff.txt").write_text("ready", encoding="utf-8")
    assert (bob / "shared" / "handoff.txt").read_text(encoding="utf-8") == "ready"

    carol = tmp_path / "agents" / "carol" / "workspace"
    local_shared = carol / "shared"
    local_shared.mkdir(parents=True)
    (local_shared / "keep.txt").write_text("keep", encoding="utf-8")
    assert ensure_workspace_shared_link(carol, shared) == "conflict"
    assert (local_shared / "keep.txt").read_text(encoding="utf-8") == "keep"
