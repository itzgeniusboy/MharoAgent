"""Headless tests for the TUI.

Everything runs through Textual's Pilot (no real terminal), which is also how
the SVG preview in README.md is generated. If these pass, the app boots,
streams, renders tool calls and honours approvals.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from mharo_tui.agent import Agent, AgentConfig
from mharo_tui.agent.session import Message, Session, Text as TextBlock, ToolCall
from mharo_tui.agent.tools import TOOLS, repo_info, run_tool
from mharo_tui.app import MharoApp
from mharo_tui.commands import COMMANDS
from mharo_tui.widgets import (
    ApprovalPrompt,
    AssistantBlock,
    CompletionPopup,
    NoticeBlock,
    Palette,
    PromptInput,
    Sidebar,
    ToolView,
    TopBar,
    Transcript,
    UserBlock,
    Welcome,
    fuzzy,
)

REPO = Path(__file__).resolve().parents[1]


def config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(provider="demo", model="demo", cwd=tmp_path, auto_approve=True, max_turns=3)


async def settle(app: MharoApp, pilot, timeout: float = 8.0, until=None) -> None:
    """Wait until the turn worker has drained *and* `until()` is satisfied.

    `submit_text` spawns a Textual worker, so `_busy` flips one tick later than
    the call returns — a plain "not busy" check races it and reads an empty
    transcript. The predicate pins the thing each test actually asserts on.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        idle = not getattr(app, "_busy", False) and getattr(app, "_worker", None) is None
        satisfied = bool(until()) if until is not None else True
        if idle and satisfied:
            await pilot.pause(0.15)
            return
    pytest.fail("turn did not settle in time")


# --------------------------------------------------------------------------- core


def test_tools_run_against_real_fs(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    ctx_config = tmp_path
    from mharo_tui.agent.tools import ToolContext

    ctx = ToolContext(cwd=ctx_config)
    assert "print" in run_tool("read_file", {"path": "hello.py"}, ctx).output
    assert run_tool("write_file", {"path": "a.txt", "content": "x\n"}, ctx).ok
    res = run_tool("edit_file", {"path": "a.txt", "old_text": "x", "new_text": "y"}, ctx)
    assert res.ok and "+y" in res.output and "-x" in res.output
    assert run_tool("search", {"pattern": "print"}, ctx).ok
    assert not run_tool("read_file", {"path": "missing.txt"}, ctx).ok
    assert run_tool("totally_bogus", {}, ctx).output.startswith("unknown tool")
    assert len(TOOLS) == 6


def test_bash_refuses_obvious_destruction(tmp_path):
    from mharo_tui.agent.tools import ToolContext

    ctx = ToolContext(cwd=tmp_path)
    out = run_tool("bash", {"command": "rm -rf / --no-preserve-root"}, ctx)
    assert not out.ok and "refused" in out.output
    ok = run_tool("bash", {"command": "printf 'a\\nb\\n'"}, ctx)
    assert ok.ok and "exit 0" in ok.output


def test_paths_cannot_escape_cwd(tmp_path):
    from mharo_tui.agent.tools import ToolContext

    ctx = ToolContext(cwd=tmp_path, allow_outside=False)
    res = run_tool("read_file", {"path": "/etc/passwd"}, ctx)
    assert not res.ok and "outside the working dir" in res.output


@pytest.mark.asyncio
async def test_agent_loop_runs_tools(tmp_path):
    events: list[dict] = []
    agent = Agent(
        config(tmp_path),
        on_event=lambda evt: events.append(evt.get("type")),
    )
    (tmp_path / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")
    await agent.submit("list the files here")
    calls = [b for m in agent.session.messages for b in m.blocks if isinstance(b, ToolCall)]
    assert calls, "demo provider should have issued a tool call"
    assert calls[0].ok and "notes.md" in (calls[0].result or "")
    assert "tool_end" in events and "turn_end" in events
    assert agent.session.path.exists()


@pytest.mark.asyncio
async def test_readonly_tools_are_never_gated(tmp_path):
    """list_dir/read_file must run silently; only mutating tools ask."""
    cfg = config(tmp_path)
    cfg.auto_approve = False
    asked: list[str] = []

    async def ask(call):
        asked.append(call.tool)
        return False

    agent = Agent(cfg, request_approval=ask)
    await agent.submit("list the files")
    assert asked == [], f"read-only tool should not have asked for approval: {asked}"
    calls = [b for m in agent.session.messages for b in m.blocks if isinstance(b, ToolCall)]
    assert calls and calls[0].ok


@pytest.mark.asyncio
async def test_mutating_tools_are_gated(tmp_path):
    cfg = config(tmp_path)
    cfg.auto_approve = False
    asked: list[str] = []

    async def ask(call):
        asked.append(call.tool)
        return False

    agent = Agent(cfg, request_approval=ask)
    call = ToolCall(tool="write_file", args={"path": "x.txt", "content": "hi\n"})
    await agent._run_call(call)
    assert asked == ["write_file"]
    assert call.ok is False and "denied" in (call.result or "").lower()
    assert not (tmp_path / "x.txt").exists()


def test_requires_approval_reads_the_registry(tmp_path):
    agent = Agent(config(tmp_path))
    assert agent.requires_approval(ToolCall(tool="read_file")) is False
    assert agent.requires_approval(ToolCall(tool="bash")) is True
    assert agent.requires_approval(ToolCall(tool="edit_file")) is True
    assert agent.requires_approval(ToolCall(tool="mystery_tool")) is True


@pytest.mark.asyncio
async def test_denied_tool_is_reported(tmp_path):
    """A denied mutating tool writes back into the transcript, not onto disk."""
    cfg = config(tmp_path)
    cfg.auto_approve = False

    async def deny(call):
        return False

    agent = Agent(cfg, request_approval=deny)
    call = ToolCall(tool="bash", args={"command": "printf nope"})
    await agent._run_call(call)
    assert call.ok is False
    assert "denied" in (call.result or "").lower()
    assert call.duration_ms == 0
    # _run_call only records the outcome on the call; _loop owns the transcript
    assert not (tmp_path / "nope").exists()


@pytest.mark.asyncio
async def test_session_roundtrip(tmp_path):
    session = Session(cwd=str(tmp_path), model="demo", provider="demo")
    session.add(Message(role="user", blocks=[TextBlock("hello there")]))
    session.add(Message(role="assistant", blocks=[ToolCall(tool="bash", args={"command": "ls"}, result="ok", ok=True, duration_ms=5)]))
    path = session.persist()
    loaded = Session.load(path)
    assert loaded.messages[0].text == "hello there"
    assert isinstance(loaded.messages[1].blocks[0], ToolCall)
    assert loaded.messages[1].blocks[0].args["command"] == "ls"


def test_token_and_openai_shape(tmp_path):
    session = Session(cwd=".", model="demo", provider="demo")
    session.add(Message(role="user", blocks=[TextBlock("hi")]))
    call = ToolCall(tool="bash", args={"command": "ls"}, result="file", ok=True)
    session.add(Message(role="assistant", blocks=[TextBlock("ran"), call]))
    chat = session.to_openai()
    assert chat[1]["tool_calls"][0]["function"]["name"] == "bash"
    assert chat[2]["role"] == "tool"
    assert session.prompt_tokens > 0


# --------------------------------------------------------------------------- app


@pytest.mark.asyncio
async def test_app_boots_with_all_regions(tmp_path):
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause(0.4)
        assert app.query_one("#transcript", Transcript) is not None
        assert app.query_one(Welcome) is not None
        assert app.query_one(PromptInput) is not None
        assert app.query_one(TopBar).model == "demo"
        assert app.theme in app.mharo_themes
        assert len(app.query(ToolView)) == 0


@pytest.mark.asyncio
async def test_submit_streams_markdown_and_folds_tool(tmp_path):
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause(0.3)
        await app.submit_text("list the files")
        await settle(app, pilot, until=lambda: bool(list(app.query(AssistantBlock))))
        assert app.query(UserBlock), "user block missing"
        assert app.query(AssistantBlock), "assistant block missing"
        tools = list(app.query(ToolView))
        assert tools, "tool block missing"
        assert tools[0].state == "ok"
        assert app.query_one(Welcome).display is False
        text = app.agent.session.last_assistant().text
        assert "list_dir" in text or "notes" in text or "→" in text


@pytest.mark.asyncio
async def test_enter_key_sends_prompt(tmp_path):
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("h", "i")
        assert app.query_one(PromptInput).content_text == "hi"
        await pilot.press("enter")
        await settle(app, pilot, until=lambda: bool(list(app.query(UserBlock))))
        assert app.query(UserBlock)
        assert "hi" in app.query_one(UserBlock).body


@pytest.mark.asyncio
async def test_approval_overlay_blocks_then_allows(tmp_path):
    cfg = config(tmp_path)
    cfg.auto_approve = False
    app = MharoApp(cfg)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(0.3)
        task = asyncio.create_task(app.run_turn("run pytest -q"))  # bash is gated
        for _ in range(60):
            await pilot.pause(0.05)
            if app.query_one(ApprovalPrompt).display:
                break
        assert app.query_one(ApprovalPrompt).display, "approval overlay never appeared"
        assert app.query_one(ApprovalPrompt).call is not None
        app._resolve_approval("once")
        await asyncio.wait_for(task, timeout=15)
        await pilot.pause(0.2)
        calls = [b for m in app.agent.session.messages for b in m.blocks if isinstance(b, ToolCall)]
        assert calls and calls[0].ok, "tool should have run after approval"


@pytest.mark.asyncio
async def test_slash_commands_and_palette(tmp_path):
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause(0.3)
        await app.run_command("/context")
        await app.run_command("/help")
        await app.run_command("/todos fix the flaky test")
        await app.run_command("/todos done")
        await app.run_command("/theme mharo-ink")
        await app.run_command("/model gpt-4o")
        await app.run_command("/nope")
        await pilot.pause(0.2)
        notices = list(app.query(NoticeBlock))
        assert len(notices) >= 5, f"expected notice blocks, got {len(notices)}"
        assert app.agent.session.todos[0]["done"] is True  # /todos wired through
        stats = app.agent.stats()
        assert app.query_one(PromptInput) is not None
        assert stats["window"] > 0
        # palette
        app.action_palette()
        await pilot.pause(0.1)
        assert app.query_one(Palette).display
        app.palette_key("down")
        app.palette_key("enter")
        await pilot.pause(0.2)
        assert app._palette_open is False

        # palette type-to-filter
        app.action_palette()
        await pilot.pause(0.1)
        assert app.focused is app.query_one(Palette).results
        assert len(app.query_one(Palette).visible_rows) > 20, "all entries listed on open"
        await pilot.press("c", "o", "s", "t")
        await pilot.pause(0.1)
        rows = app.query_one(Palette).visible_rows
        names = [r[0] for r in rows]
        assert len(names) == 4, f"expected 4 fuzzy matches for cost, got {names}"
        assert names[0] == "/cost"
        assert app.query_one(Palette).current() == "/cost"
        await pilot.press("backspace")
        await pilot.pause(0.1)
        assert app._palette_query == "cos"
        await pilot.press("t")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert app._palette_open is False
        assert app._palette_query == "", "query must reset after close"
        # reopen: empty needle -> full list again
        await pilot.press("ctrl+p")
        await pilot.pause(0.1)
        assert len(app.query_one(Palette).visible_rows) > 20
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert app._palette_open is False
        assert app._palette_query == ""


@pytest.mark.asyncio
async def test_shell_bang_and_completion_popup(tmp_path):
    app = MharoApp(config(tmp_path))
    (tmp_path / "alpha.txt").write_text("1\n", encoding="utf-8")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.3)
        await app.run_shell("printf 'ran\\n'")
        await pilot.pause(0.3)
        assert list(app.query(ToolView))[0].state == "ok"
        prompt = app.query_one(PromptInput)
        prompt.load_text_safe("/con")
        await pilot.pause(0.1)
        assert app.query_one(CompletionPopup).display
        assert app.query_one(CompletionPopup).rows[0][0].startswith("/context")
        app.action_complete()
        await pilot.pause(0.1)
        assert prompt.content_text.startswith("/context")


@pytest.mark.asyncio
async def test_keybindings_toggle_chrome(tmp_path):
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("ctrl+b")
        assert app.query_one(Sidebar).shown
        await pilot.press("ctrl+b")
        assert not app.query_one(Sidebar).shown
        await pilot.press("ctrl+o")
        assert app._expand_all
        await pilot.press("ctrl+f")
        assert not app.agent.config.auto_approve
        before = app.theme
        await pilot.press("ctrl+t")
        assert app.theme != before
        await pilot.press("ctrl+l")
        assert app.query(Welcome)


@pytest.mark.asyncio
async def test_interrupt_stops_turn(tmp_path):
    cfg = config(tmp_path)
    cfg.max_turns = 40
    app = MharoApp(cfg)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.3)
        await app.submit_text("hello there friend")
        await pilot.pause(0.05)
        app.action_interrupt()
        await settle(app, pilot, timeout=5)
        assert app._busy is False


@pytest.mark.asyncio
async def test_mentions_expand_into_prompt(tmp_path):
    (tmp_path / "conf.py").write_text("VALUE = 1\n", encoding="utf-8")
    app = MharoApp(config(tmp_path))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.3)
        out = app.expand_mentions("explain @conf.py please")
        assert "VALUE = 1" in out and "```" in out


@pytest.mark.asyncio
async def test_render_snapshot_for_readme(tmp_path):
    """Renders a realistic frame to SVG so the design is reviewable headlessly."""
    app = MharoApp(config(tmp_path))
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    async with app.run_test(size=(118, 40)) as pilot:
        await pilot.pause(0.3)
        await app.submit_text("list the files in this repo")
        await settle(app, pilot)
        await app.submit_text("run pytest -q")
        await settle(app, pilot)
        app.query_one(Sidebar).shown = True
        app.query_one(Sidebar).display = True
        app.query_one(Sidebar).set_keys([("ctrl+p", "palette"), ("esc", "interrupt")])
        app.action_fold_all()
        await pilot.pause(0.4)
        svg = app.export_screenshot()
        out = REPO / "preview.svg"
        out.write_text(svg, encoding="utf-8")
        assert svg.startswith("<svg") or "<svg" in svg[:200]
        assert len(svg) > 5_000


# --------------------------------------------------------------------------- misc


def test_fuzzy_helper():
    assert fuzzy("ctx", "/context")
    assert not fuzzy("zzz", "/context")
    assert fuzzy("", "anything")


def test_repo_info_never_raises(tmp_path):
    assert repo_info(tmp_path)["git"] in (True, False)


def test_commands_registry_is_complete():
    for name, cmd in COMMANDS.items():
        assert cmd.name == name
        assert cmd.help and callable(cmd.run)
    for required in {"help", "context", "model", "theme", "quit", "todos"}:
        assert required in COMMANDS


# ----------------------------------------------------------------- free ladder
def _sse_body(text: str) -> str:
    """A minimal OpenAI-style SSE body, built by hand (no json import needed here)."""
    chunk = ('{"choices": [{"delta": {"content": "' + text + '"}, "finish_reason": null}], '
             '"usage": {"prompt_tokens": 9, "completion_tokens": 3}}')
    return f"data: {chunk}\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_tui_boots_on_the_free_ladder_with_no_key(tmp_path, monkeypatch):
    """No key in the environment: the TUI still talks to a model — a free rung."""
    import httpx

    from mharo_tui.agent.ladder import Ladder, Rung

    for name in ("OPENAI_API_KEY", "MHARO_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL",
                 "MHARO_PROVIDER", "MHARO_ALLOW_DEMO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert "authorization" not in {k.lower() for k in request.headers}, "a free rung must not send a key"
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=_sse_body("answered by a free provider"))

    ladder = Ladder(rungs=[Rung(name="fake", base_url="https://free.example/v1", model="gpt-oss-20b",
                                models=("gpt-oss-20b",), min_interval_s=0.0)], state_path=None)
    cfg = AgentConfig(provider="auto", model="", cwd=tmp_path, auto_approve=True, max_turns=2,
                      provider_opts={"transport": httpx.MockTransport(handler), "ladder": ladder})
    app = MharoApp(cfg)
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause(0.3)
        assert app.agent.provider.name == "auto"
        await app.submit_text("say hi")
        await settle(app, pilot, until=lambda: bool(list(app.query(AssistantBlock))))
        assert seen and "free.example" in seen[0]
        assert "answered by a free provider" in app.agent.session.last_assistant().text
        await app.run_command("/free")
        await pilot.pause(0.2)
        notice = " ".join(getattr(w, "markdown", "") for w in app.query(NoticeBlock))
        assert "`fake`" in notice, f"/free did not render the ladder table:\n{notice[:300]}"
        assert "free · no key" in notice
        assert app.agent.stats()["cost"] == 0.0, "a free rung must cost nothing"
