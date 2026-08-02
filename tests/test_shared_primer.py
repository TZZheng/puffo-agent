import tempfile
from pathlib import Path
from types import SimpleNamespace

from puffo_agent.mcp.puffo_core_tools import register_core_tools
from puffo_agent.agent.shared_content import (
    DEFAULT_SHARED_CLAUDE_MD,
    DEFAULT_SKILLS,
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
    contribution_guidance = "Send only when the new content makes a useful new contribution; otherwise choose silence."
    all_prompt_surfaces = " ".join(
        " ".join(body.split())
        for body in (DEFAULT_SHARED_CLAUDE_MD, *[body for _, body in DEFAULT_SKILLS.values()])
    )
    assert normalized_read.count(contribution_guidance) == 1
    assert all_prompt_surfaces.count(contribution_guidance) == 1
    assert "Arrival or peer activity is evidence to evaluate, not an automatic reply trigger." in normalized_read
    assert "For conversation decisions, use the `read-inbox` skill." in DEFAULT_SHARED_CLAUDE_MD
    assert "useful new contribution" not in DEFAULT_SHARED_CLAUDE_MD
    assert "does not acknowledge pending Inbox work" in " ".join(history.split())
    assert "does not acknowledge pending Inbox work" in " ".join(post.split())
    normalized_send = " ".join(send.lower().split())
    for phrase in (
        'state="held"', "unchanged draft", "draft boundary/latest pair",
        "visible_draft_basis", "new_channel_context", "context_ready=true",
        "context_ready=false", "inspect", "revise and send with normal freshness",
        "retry the unchanged draft", "or send nothing", "send_anyway=true", "is rare",
        "model-owned", "may be held again", "sequence watermark alone is not semantic context",
    ):
        assert phrase.lower() in normalized_send
    for skill_id in ("attachments", "channel-history", "send-message-with-attachments"):
        body = DEFAULT_SKILLS[skill_id][1]
        assert "visible_draft_basis" not in body
        assert "same originating assignment" not in body
    assert "common held-send procedure" in DEFAULT_SKILLS["send-message-with-attachments"][1]
    for forbidden in (
        "<inbox_message>", "one line per root post", "nested `route`",
        "assignment-completion", "benchmark", "response quota", "counting-specific",
        "sender-type silence", "forced mention/DM", "automatically retries",
        "always requires a separate history read",
    ):
        assert forbidden not in "\n".join((DEFAULT_SHARED_CLAUDE_MD, *[body for _, body in DEFAULT_SKILLS.values()]))
    send_policy_text = "\n".join((send, DEFAULT_SKILLS["send-message-with-attachments"][1])).lower()
    for forbidden in (
        "**when to use:**", "**when not to use:**", "prefer this over",
        'prefer "human"', "spontaneous cross-posts",
    ):
        assert forbidden not in send_policy_text


def test_contribution_guidance_is_absent_from_mcp_descriptions_and_profiles():
    contribution_guidance = (
        "Send only when the new content makes a useful new contribution; "
        "otherwise choose silence."
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
    mcp_descriptions = "\n".join(tool.__doc__ or "" for tool in mcp.tools)
    assert contribution_guidance not in mcp_descriptions
    assert "useful new contribution" not in mcp_descriptions

    claude, codex = _rebuild(_tmp())
    for generated_profile_surface in (claude, codex):
        assert contribution_guidance not in generated_profile_surface
        assert "useful new contribution" not in generated_profile_surface


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
