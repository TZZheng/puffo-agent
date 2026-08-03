import tempfile
from pathlib import Path
from types import SimpleNamespace

from puffo_agent.mcp.puffo_core_tools import register_core_tools
from puffo_agent.agent.shared_content import (
    DEFAULT_SHARED_CLAUDE_MD,
    DEFAULT_SKILLS,
    HELD_SEND_RECONSIDERATION_GUIDANCE,
    PARTICIPATION_AND_CONTINUATION_GUIDANCE,
    ensure_shared_primer,
    rebuild_agent_claude_md,
    rebuild_agent_codex_md,
)


PARENT_CLAUDE_BYTES = 16109
PARENT_CODEX_BYTES = 15989


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _rebuild(root: Path) -> tuple[str, str]:
    profile = root / "profile.md"
    profile.write_text("# Soul\nSOUL-MARKER-7b", encoding="utf-8")
    memory = root / "memory"
    workspace = root / "workspace"
    workspace.mkdir()
    common = dict(
        shared_dir=root / "shared", profile_path=profile, memory_dir=memory,
        workspace_dir=workspace, agent_id="AGENT-ID-MARKER-8c",
        display_name="DISPLAY-NAME-MARKER-9d", role="LONG-ROLE-MARKER-ae",
        role_short="SHORT-ROLE-MARKER-bf",
    )
    claude = rebuild_agent_claude_md(
        **common, claude_user_dir=root / ".claude", gemini_user_dir=root / ".gemini",
    )
    (memory / "briefing" / "topic.md").write_text(
        "NON-PROFILE-TOPIC-MARKER-c0", encoding="utf-8",
    )
    claude = rebuild_agent_claude_md(
        **common, claude_user_dir=root / ".claude", gemini_user_dir=root / ".gemini",
    )
    codex = rebuild_agent_codex_md(**common, codex_user_dir=root / ".codex")
    return claude, codex


def test_standing_prompt_is_compact_and_identity_is_compiled_once():
    claude, codex = _rebuild(_tmp())
    for text in (claude, codex):
        for marker in (
            "DISPLAY-NAME-MARKER-9d", "AGENT-ID-MARKER-8c", "LONG-ROLE-MARKER-ae",
            "SHORT-ROLE-MARKER-bf", "SOUL-MARKER-7b", "NON-PROFILE-TOPIC-MARKER-c0",
        ):
            assert text.count(marker) == 1
        assert "# Your role" not in text
        assert "# Your memory" in text
    assert len(claude.encode()) < PARENT_CLAUDE_BYTES
    assert len(codex.encode()) < PARENT_CODEX_BYTES


def test_policy_has_one_detailed_owner_and_primer_retains_contract():
    assert "<global_inbox_notice>" in DEFAULT_SHARED_CLAUDE_MD
    assert '"content_included":false' in DEFAULT_SHARED_CLAUDE_MD
    assert "read_inbox" in DEFAULT_SHARED_CLAUDE_MD
    assert "send_message" in DEFAULT_SHARED_CLAUDE_MD
    assert "[SILENT]" in DEFAULT_SHARED_CLAUDE_MD
    for absent in (
        "prior_context", "visible_draft_basis", "new_channel_context",
        "context_ready", "same originating assignment", "send_anyway=True",
    ):
        assert absent not in DEFAULT_SHARED_CLAUDE_MD
    read = DEFAULT_SKILLS["read-inbox"][1]
    history = DEFAULT_SKILLS["channel-history"][1]
    post = DEFAULT_SKILLS["get-post"][1]
    send = DEFAULT_SKILLS["send-message"][1]
    for phrase in (
        "## target=dm peer_id=...",
        "## target=channel space_id=...\n  channel_id=...",
        "## target=thread space_id=... channel_id=...\n  thread_root_id=...",
        "seq", "absolute `time`", "type", "id", "self", "encrypted",
        "@slug", "optional `name`", "body",
    ):
        assert phrase in read
    for text in (history, post):
        assert "read-inbox" in text
        assert "seq" not in text
    normalized_read = " ".join(read.split())
    assert "messages" in read and "prior_context" in read and "strictly earlier" in normalized_read
    assert "do not acknowledge pending Inbox rows" in read
    decision_guidance = " ".join(
        PARTICIPATION_AND_CONTINUATION_GUIDANCE.split()
    )
    all_prompt_surfaces = " ".join(
        " ".join(body.split())
        for body in (DEFAULT_SHARED_CLAUDE_MD, *[body for _, body in DEFAULT_SKILLS.values()])
    )
    assert normalized_read.count(decision_guidance) == 1
    assert all_prompt_surfaces.count(decision_guidance) == 2
    assert "distinct turns" in normalized_read
    assert "shared result" in normalized_read
    assert "Agent messages may legitimately trigger" in normalized_read
    assert "ask one concise human clarification" in normalized_read
    assert "For conversation decisions, use the `read-inbox` skill." in DEFAULT_SHARED_CLAUDE_MD
    normalized_primer = " ".join(DEFAULT_SHARED_CLAUDE_MD.split())
    assert "a `target=channel` route uses its `channel_id` without `root_id`" in normalized_primer
    assert "Starting a new thread is a model-owned presentation choice" in normalized_primer
    assert "originating request and conversation intent" not in DEFAULT_SHARED_CLAUDE_MD
    assert "does not acknowledge pending Inbox work" in " ".join(history.split())
    assert "does not acknowledge pending Inbox work" in " ".join(post.split())
    normalized_send = " ".join(send.lower().split())
    for phrase in (
        'state="held"', "unchanged draft", "draft boundary/latest pair",
        "visible_draft_basis", "new_channel_context", "context_ready=true",
        "context_ready=false", "inspect", "revise against the latest context",
        "send with normal freshness", "attempted but not sent",
        "not visible participation", "new successful visible contribution",
        "reconsider the originating interaction", "context-dependent",
        "sequence position", "shared-state coordination",
        "do not use `send_anyway`", "use the unchanged draft", "send nothing",
        "distinct-participation mode",
        "shared-result mode",
        "meaningful or converging interaction",
        "self-propagate without meaningful progress",
        "one concise human clarification",
        "send_anyway=true", "is rare",
        "model-owned", "may be held again", "sequence watermark alone is not semantic context",
        "preserve the `read_inbox` target by default",
        "omit it for `target=channel`",
        "pass the supplied `thread_root_id` for `target=thread`",
    ):
        assert phrase.lower() in normalized_send
    assert "the canonical return route" in normalized_read
    assert "preserve it by default" in normalized_read
    for skill_id in ("attachments", "channel-history", "send-message-with-attachments"):
        body = DEFAULT_SKILLS[skill_id][1]
        assert "visible_draft_basis" not in body
        assert "same originating assignment" not in body
    assert "common held-send procedure" in DEFAULT_SKILLS["send-message-with-attachments"][1]
    for forbidden in (
        "<inbox_message>", "one line per root post", "nested `route`",
        "assignment-completion", "benchmark", "response quota", "counting-specific",
        "sender-type silence", "forced mention/DM", "automatically retries",
        "always requires a separate history read", "participant-count", "one-reply",
        "sender suppression", "destination rule", "parser", "deterministic silence",
        "runtime policy",
    ):
        assert forbidden not in read.lower()
    for obsolete in (
        "peer progress alone does not create a new obligation",
        "part stays fulfilled and peer progress does not reopen",
        "another participant's overlapping contribution does not satisfy",
    ):
        assert obsolete not in all_prompt_surfaces.lower()
    send_policy_text = "\n".join((send, DEFAULT_SKILLS["send-message-with-attachments"][1])).lower()
    for forbidden in (
        "**when to use:**", "**when not to use:**", "prefer this over",
        'prefer "human"', "spontaneous cross-posts",
    ):
        assert forbidden not in send_policy_text


def test_conversation_judgment_is_absent_from_mcp_descriptions_and_profiles():
    decision_guidance = " ".join(
        PARTICIPATION_AND_CONTINUATION_GUIDANCE.split()
    )

    class CapturingMCP:
        def __init__(self) -> None:
            self.tools = []

        def tool(self):
            def register(function):
                self.tools.append(function)
                return function
            return register

    mcp = CapturingMCP()
    register_core_tools(mcp, SimpleNamespace(bridge_client=None))
    mcp_descriptions = " ".join(
        "\n".join(tool.__doc__ or "" for tool in mcp.tools).split()
    )
    assert decision_guidance not in mcp_descriptions
    assert "distinct-participation mode" not in mcp_descriptions

    claude, codex = _rebuild(_tmp())
    for generated_profile_surface in (claude, codex):
        normalized = " ".join(generated_profile_surface.split())
        assert decision_guidance not in normalized
        assert "distinct-participation mode" not in normalized


def test_read_inbox_owns_participation_and_continuation_judgment():
    decision_guidance = " ".join(
        PARTICIPATION_AND_CONTINUATION_GUIDANCE.split()
    )
    read = " ".join(DEFAULT_SKILLS["read-inbox"][1].split())
    assert read.count(decision_guidance) == 1
    for phrase in (
        "message that originated the request",
        "Earlier unrelated activity is context",
        "not evidence of turn position, assignment, or participation",
        "interaction's participation shape",
        "how many turns each participant is expected to take",
        "one successful turn per identity",
        "Explicit repetition, multiple rounds, or later work may require more",
        "track successful visible participation by identity",
        "within that interaction",
        "does not count as your visible contribution",
        "normally completes your participation in that round",
        "later peer progress alone does not reopen the same contribution",
        "keep participant position separate from the content or value",
        "does not by itself reset later positions",
        "Agent messages may legitimately trigger further Agent work or replies",
        "genuinely new work or a still-open step",
        "Continue meaningful or converging interaction",
        "repeat, oscillate, or self-propagate without meaningful progress",
        "ask one concise human clarification",
        "equivalent clarification is already present",
    ):
        assert phrase in read
    for stale_generic_tail in (
        "Evaluate pending messages together with relevant",
        "useful new contribution",
        "otherwise choose silence",
        "peer activity is evidence to evaluate",
        "Peer progress alone does not create a new obligation",
        "peer progress does not reopen",
    ):
        assert stale_generic_tail not in read

    root = _tmp()
    claude, codex = _rebuild(root)
    claude_skill = (
        root / "workspace" / ".claude" / "skills" / "read-inbox" / "SKILL.md"
    ).read_text(encoding="utf-8")
    codex_skill = (
        root / "workspace" / ".agents" / "skills" / "read-inbox" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert decision_guidance in " ".join(claude_skill.split())
    assert decision_guidance in " ".join(codex_skill.split())
    assert codex_skill == claude_skill.replace("mcp__puffo__", "")

    other_prompt_surfaces = [" ".join(surface.split()) for surface in (claude, codex)]
    other_prompt_surfaces.extend(
        " ".join(body.split())
        for skill_id, (_, body) in DEFAULT_SKILLS.items()
        if skill_id not in {"read-inbox", "send-message"}
    )
    assert not any(decision_guidance in surface for surface in other_prompt_surfaces)

    class CapturingMCP:
        def __init__(self) -> None:
            self.tools = []

        def tool(self):
            def register(function):
                self.tools.append(function)
                return function
            return register

    mcp = CapturingMCP()
    register_core_tools(mcp, SimpleNamespace(bridge_client=None))
    mcp_descriptions = " ".join(
        "\n".join(tool.__doc__ or "" for tool in mcp.tools).split()
    )
    assert decision_guidance not in mcp_descriptions


def test_held_send_applies_the_shared_judgment_to_the_attempted_draft():
    held_method = " ".join(HELD_SEND_RECONSIDERATION_GUIDANCE.split()).lower()
    send = " ".join(DEFAULT_SKILLS["send-message"][1].split()).lower()
    assert send.count(held_method) == 1

    root = _tmp()
    _rebuild(root)
    for path in (
        root / "workspace" / ".claude" / "skills" / "send-message" / "SKILL.md",
        root / "workspace" / ".agents" / "skills" / "send-message" / "SKILL.md",
    ):
        skill = " ".join(path.read_text(encoding="utf-8").split()).lower()
        assert skill.count(held_method) == 1

    other_prompt_surfaces = [" ".join(DEFAULT_SHARED_CLAUDE_MD.split())]
    other_prompt_surfaces.extend(
        " ".join(body.split())
        for skill_id, (_, body) in DEFAULT_SKILLS.items()
        if skill_id != "send-message"
    )
    held_opening = "A held draft was attempted but not sent"
    assert not any(held_opening in surface for surface in other_prompt_surfaces)
    assert "new successful visible contribution from you" in send
    assert "earlier successful `self=true` row already fulfills it" in send
    assert "Treat a held draft as evidence of an attempted response" not in send


def test_harnesses_discover_managed_skills_with_correct_tool_names():
    root = _tmp()
    claude, codex = _rebuild(root)
    assert "mcp__puffo__" in claude
    assert "mcp__puffo__" not in codex
    assert "send_message" in codex
    ensure_shared_primer(root / "shared")
    for skill_id in ("read-inbox", "send-message"):
        claude_skill = root / "workspace" / ".claude" / "skills" / skill_id / "SKILL.md"
        codex_skill = root / "workspace" / ".agents" / "skills" / skill_id / "SKILL.md"
        for skill in (claude_skill, codex_skill):
            text = skill.read_text(encoding="utf-8")
            assert "name:" in text and "description:" in text
        assert "mcp__puffo__" in claude_skill.read_text(encoding="utf-8")
        assert "mcp__puffo__" not in codex_skill.read_text(encoding="utf-8")


def test_managed_refresh_rewrites_stale_skill():
    root = _tmp()
    shared = root / "shared"
    ensure_shared_primer(shared)
    skill = shared / "skills" / "read-inbox" / "SKILL.md"
    skill.write_text("stale", encoding="utf-8")
    actions = dict(ensure_shared_primer(shared))
    assert actions["skills/read-inbox/SKILL.md"] == "updated"
    assert "prior_context" in skill.read_text(encoding="utf-8")
