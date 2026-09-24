"""MharoApp — the Textual application: composition, keymap and event routing.

The UI is a thin renderer over `Agent`: every streaming event maps to exactly
one transcript mutation, which keeps repaints cheap and the whole app testable
with Textual's headless Pilot (see tests/test_tui.py).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import TextArea

from .agent import Agent, AgentConfig
from .agent.providers import ProviderError
from .agent.session import (
    ErrorNote,
    Message,
    Session,
    Text as TextBlock,
    Thinking,
    ToolCall,
)
from .agent.tools import preview as tool_preview
from .agent.tools import run_tool
from .commands import COMMANDS
from .themes import DEFAULT_THEME, THEMES, cycle, register_all
from .widgets import (
    ApprovalPrompt,
    PromptPlaceholder,
    AssistantBlock,
    CompletionPopup,
    ErrorBlock,
    Gauge,
    NoticeBlock,
    Palette,
    PromptInput,
    Sidebar,
    StatusLine,
    ToolView,
    TopBar,
    Transcript,
    UserBlock,
    Welcome,
)

TIPS = [
    "an agent terminal for your repo — built on Textual",
    "",
    "  /help        every command             ctrl+p   command palette",
    "  /context     token budget              ctrl+b   sidebar",
    "  !ls -la      run a shell command       ctrl+o   fold / unfold tools",
    "  @file        attach file as context    ctrl+f   auto-approve edits",
    "  esc          interrupt a turn           ctrl+t   cycle theme",
    "",
    "  provider: demo (offline) — switch with /provider openai|anthropic",
]

KEY_HELP = [
    ("enter", "send"),
    ("shift+enter", "newline"),
    ("ctrl+p", "palette"),
    ("tab", "complete"),
    ("esc", "interrupt"),
    ("ctrl+b", "sidebar"),
    ("ctrl+o", "fold tools"),
    ("ctrl+f", "auto-approve"),
    ("ctrl+t", "theme"),
    ("ctrl+y", "copy reply"),
    ("ctrl+r", "retry"),
    ("ctrl+n", "new session"),
    ("ctrl+q", "quit"),
]

# Palette entries that are actions rather than slash commands.
ACTIONS = [
    ("toggle sidebar", "sidebar"),
    ("fold / unfold tools", "fold_all"),
    ("auto-approve on off", "auto_approve"),
    ("next theme", "next_theme"),
    ("new session", "new_session"),
    ("clear transcript", "clear"),
    ("copy last reply", "copy_reply"),
    ("retry last prompt", "retry"),
    ("quit", "quit"),
]


class MharoApp(App):
    TITLE = "mharo"
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "smart_quit", "Interrupt or quit"),
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+p", "palette", "Command palette"),
        ("ctrl+b", "sidebar", "Toggle sidebar"),
        ("ctrl+o", "fold_all", "Fold/unfold tools"),
        ("ctrl+f", "auto_approve", "Auto-approve"),
        ("ctrl+t", "next_theme", "Cycle theme"),
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+r", "retry", "Retry last prompt"),
        ("ctrl+y", "copy_reply", "Copy last reply"),
        ("ctrl+l", "clear", "Clear transcript"),
        ("f1", "help", "Help"),
    ]

    def __init__(self, config: AgentConfig | None = None, *, session: Session | None = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()
        self.mharo_themes = list(THEMES)
        self.mharo_forced_theme: str | None = None
        self.agent: Agent | None = None
        self._pending_session = session
        self._tools: dict[str, ToolView] = {}
        self._active: AssistantBlock | None = None
        self._approval_future: asyncio.Future | None = None
        self._always_allow: set[str] = set()
        self._expand_all = False
        self._busy = False
        self._turn_started = 0.0
        self._turn_cancelled = False
        self._thinking = ""
        self._worker: Any = None
        self.mharo_css_vars: dict[str, str] = dict(THEMES[DEFAULT_THEME]["vars"])
        self._palette_open = False
        self._palette_query = ""

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield TopBar(id="top")
        with Vertical(id="main"):
            yield Transcript(id="transcript")
            yield Sidebar()
        yield Palette()
        yield ApprovalPrompt()
        with Vertical(id="prompt"):
            yield CompletionPopup()
            with Vertical(id="prompt-box"):
                yield PromptInput(id="prompt-input")
                yield PromptPlaceholder()
        yield StatusLine()

    async def on_mount(self) -> None:
        register_all(self)
        if self.mharo_forced_theme in THEMES:
            try:
                self.theme = self.mharo_forced_theme
            except Exception:
                pass
        self._build_agent(self._pending_session)
        self.apply_palette()
        self.query_one("#transcript", Transcript).mount(Welcome(TIPS))
        self.set_interval(0.12, self._tick)
        self.refresh_chrome()
        self.query_one("#prompt-input", PromptInput).focus()

    def _build_agent(self, session: Session | None = None) -> None:
        try:
            self.agent = Agent(
                self.config,
                on_event=self.handle_event,
                request_approval=self.request_approval,
                session=session,
            )
        except ProviderError as exc:
            # Bad/absent credentials must not prevent the UI from opening.
            self.config.provider = "demo"
            self.config.model = None
            self.agent = Agent(
                self.config,
                on_event=self.handle_event,
                request_approval=self.request_approval,
                session=session,
            )
            self._boot_error = str(exc)
        else:
            self._boot_error = None
        self._tools, self._active = {}, None
        if session and session.messages:
            self.replay(session)
        self.refresh_chrome()
        if self._boot_error:
            self.call_after_refresh(self._append, ErrorBlock(self._boot_error))

    # ------------------------------------------------------------------ events

    def handle_event(self, evt: dict) -> None:
        kind = evt.get("type")
        if kind == "user_message":
            self._hide_welcome()
            self._append(UserBlock(evt["message"].text))
        elif kind == "assistant_start":
            self._hide_welcome()
            self._active = AssistantBlock(model=evt.get("model", ""))
            self._append(self._active)
        elif kind == "text":
            if self._active:
                self._active.append(evt.get("text", ""))
        elif kind == "thinking":
            self._thinking = (self._thinking + evt.get("text", "")).strip()[-90:]
        elif kind in {"tool_call", "tool_start"}:
            call = evt.get("call") or ToolCall(tool=evt.get("tool", "bash"), args=evt.get("args", {}))
            view = ToolView(call)
            self._tools[call.call_id] = view
            self._append(view)
            if self._expand_all:
                view.collapsed = False
        elif kind == "tool_running":
            self._set_tool(evt["call"], "running")
        elif kind == "tool_end":
            call = evt["call"]
            state = (
                "denied"
                if (call.result or "").startswith("User denied")
                else ("ok" if call.ok else "error")
            )
            self._set_tool(call, state, call.result or "")
        elif kind == "error":
            self._append(ErrorBlock(evt.get("text", "")))
        elif kind == "notice":
            self.set_state(evt.get("text", ""))
        elif kind == "usage":
            self.refresh_chrome()
        elif kind == "turn_end":
            if self._active:
                if not self._active.buffer.strip():
                    # a turn that only issued tool calls leaves an empty
                    # header behind — drop it, headers are for prose.
                    self._active.remove()
                else:
                    self._active.finish(interrupted=self._turn_cancelled)
            self._active = None
            self._thinking = ""
            self._turn_cancelled = False
            self.refresh_chrome()

    def _set_tool(self, call: ToolCall, state: str, output: str = "") -> None:
        view = self._tools.get(call.call_id)
        if view is None:
            return
        diff: str | None = None
        if call.tool in {"write_file", "edit_file"} and call.result:
            head, _, rest = call.result.partition("\n\n")
            diff = rest.strip() or None
            output = head
        view.set_state(state, output, diff)
        if self._expand_all:
            view.collapsed = False

    # ------------------------------------------------------------------ helpers

    def _append(self, widget: Any) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.mount(widget)
        transcript.scroll_end(animate=False)

    def _hide_welcome(self) -> None:
        try:
            self.query_one(Welcome).display = False
        except Exception:
            pass

    def set_state(self, text: str) -> None:
        try:
            self.query_one(StatusLine).set_state(text, self._busy, self.hints_text())
        except Exception:
            pass

    def hints_text(self) -> str:
        flags = []
        if self.agent and self.agent.config.auto_approve:
            flags.append("auto-approve")
        if self._always_allow:
            flags.append("always:" + ",".join(sorted(self._always_allow)))
        if self._expand_all:
            flags.append("tools expanded")
        core = "press esc to interrupt this turn" if self._busy else "enter send · / commands · @ file · ! shell · ctrl+p"
        return " · ".join([core, *flags])

    def _tick(self) -> None:
        if not self._busy:
            return
        secs = time.time() - self._turn_started
        label = f"working… {secs:.1f}s"
        if self._thinking:
            label += f"  ·  {self._thinking}"
        self.set_state(label)
        if self._active:
            self._active.tick()

    def refresh_chrome(self) -> None:
        if self.agent is None:
            return
        stats = self.agent.stats()
        top = self.query_one(TopBar)
        top.backend = self.agent.provider.name
        top.model = self.agent.session.model
        top.branch = stats.get("branch", "no-git")
        top.dirty = stats.get("dirty", 0)
        top.path = _shorten(str(self.agent.ctx.cwd))
        top.sep = "│" if self.agent.repo.get("git") else "" 
        top.gauge = f"ctx {stats['prompt_tokens']:,}/{stats['window']:,} ({stats['used_pct']:.0f}%)"
        top.cost = f"${stats['cost']:.3f}" if stats["cost"] else ""
        top.state = "working" if self._busy else "" 
        sidebar = self.query_one(Sidebar)
        sidebar.set_usage(stats)
        sidebar.set_todos(self.agent.session.todos)
        sidebar.set_files(self.agent.session.files_touched)
        sidebar.set_modes(
            self.agent.session.model,
            self.agent.provider.name,
            self.agent.config.auto_approve,
            self._expand_all,
        )
        if not self._busy:
            self.set_state("ready")
        self.title = f"mharo · {self.agent.session.model}"
        self.sub_title = str(self.agent.ctx.cwd)

    def apply_palette(self) -> None:
        """Theme extras that plain CSS can't express safely get pushed to widgets."""
        name = self.theme if self.theme in THEMES else DEFAULT_THEME
        spec = THEMES[name]
        self.mharo_css_vars = dict(spec.get("vars", {}))
        for selector, key, attr in (
            ("UserBlock", "user-bg", "background"),
            ("ToolView", "tool-bg", "background"),
            ("NoticeBlock", "user-bg", "background"),
        ):
            colour = self.mharo_css_vars.get(key)
            if not colour:
                continue
            try:
                for widget in self.query(selector):
                    setattr(widget.styles, attr, colour)
            except Exception:
                pass

    # ------------------------------------------------------------------ prompts

    async def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        message.stop()
        text = (message.text or "").strip()
        try:
            self.query_one(PromptInput).styles.height = 3
            self.query_one(PromptPlaceholder).display = True
        except Exception:
            pass
        self.query_one(CompletionPopup).hide()
        if self._palette_open:
            self.action_palette_close()
        if not text:
            return
        await self.submit_text(text)

    async def submit_text(self, text: str) -> None:
        if self.agent is None:
            return
        if self._busy:
            self.set_state("still working — esc to interrupt")
            return
        if text.startswith("/"):
            await self.run_command(text)
            return
        if text.startswith("!"):
            await self.run_shell(text[1:].strip())
            return
        self._worker = self.run_worker(self.run_turn(self.expand_mentions(text)), group="turn", exclusive=True, exit_on_error=False)

    async def run_turn(self, text: str) -> None:
        assert self.agent is not None
        self._busy, self._turn_started, self._turn_cancelled = True, time.time(), False
        self.set_state("working…")
        try:
            await self.agent.submit(text)
        except ProviderError as exc:
            self._append(ErrorBlock(str(exc)))
        except asyncio.CancelledError:
            self._turn_cancelled = True
            if self._active:
                self._active.finish(interrupted=True)
        except Exception as exc:
            self._append(ErrorBlock(f"{type(exc).__name__}: {exc}"))
        finally:
            self._busy = False
            self._worker = None
            self.refresh_chrome()
            try:
                self.query_one("#prompt-input", PromptInput).focus()
            except Exception:
                pass

    async def run_shell(self, command: str) -> None:
        """`!cmd` runs a shell command straight through the tool layer."""
        assert self.agent is not None
        if not command:
            return
        self._hide_welcome()
        call = ToolCall(tool="bash", args={"command": command})
        view = ToolView(call)
        self._append(view)
        view.set_state("running")
        result = await asyncio.to_thread(run_tool, "bash", {"command": command}, self.agent.ctx)
        call.result, call.ok, call.duration_ms = result.output, result.ok, result.duration_ms
        view.set_state("ok" if result.ok else "error", result.output)
        self.agent.session.messages.append(Message(role="user", blocks=[TextBlock(f"!{command}")]))
        self.agent.session.messages.append(Message(role="assistant", blocks=[call]))
        self.agent.session.persist()

    # ------------------------------------------------------------------ approvals

    async def request_approval(self, call: ToolCall) -> bool:
        assert self.agent is not None
        if call.tool in self._always_allow or self.agent.config.auto_approve:
            return True
        loop = asyncio.get_running_loop()
        self._approval_future = loop.create_future()
        try:
            self.query_one(ApprovalPrompt).ask(call, self.approval_detail(call))
        except Exception:
            self._approval_future = None
            return False
        verdict = "deny"
        try:
            verdict = await asyncio.wait_for(self._approval_future, timeout=1800)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            verdict = "deny"
        finally:
            self._approval_future = None
            try:
                self.query_one(ApprovalPrompt).answer()
            except Exception:
                pass
        if verdict == "always":
            self._always_allow.add(call.tool)
            self.notify(f"{call.tool}: always allow", timeout=2.0)
        self.refresh_chrome()
        return verdict in {"once", "always"}

    def approval_detail(self, call: ToolCall) -> str:
        args = call.args or {}
        cwd = self.agent.ctx.cwd if self.agent else Path.cwd()
        if call.tool == "write_file":
            path = cwd / str(args.get("path", ""))
            try:
                before = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                after = str(args.get("content", ""))
                return "\n".join(
                    difflib.unified_diff(
                        before.splitlines(), after.splitlines(), str(args.get("path")), str(args.get("path")), lineterm="", n=1
                    )
                )[:3000]
            except OSError:
                return ""
        if call.tool == "edit_file":
            return "\n".join(
                difflib.unified_diff(
                    str(args.get("old_text", "")).splitlines(),
                    str(args.get("new_text", "")).splitlines(),
                    "before", "after", lineterm="", n=1,
                )
            )[:3000]
        if call.tool == "bash":
            return f"  runs in {cwd}:\n  {args.get('command', '')}"
        return "\n".join(f"  {k}: {v}" for k, v in args.items())[:1500]

    def on_approval_prompt_answered(self, message: ApprovalPrompt.Answered) -> None:
        message.stop()
        self._resolve_approval(message.verdict)

    def _resolve_approval(self, verdict: str) -> None:
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(verdict)
        try:
            self.query_one(ApprovalPrompt).answer()
        except Exception:
            pass

    # ------------------------------------------------------------------ keys

    def on_key(self, event) -> None:
        try:
            approval_open = self.query_one(ApprovalPrompt).display
        except Exception:
            approval_open = False
        if approval_open and event.key in {"y", "a", "n"}:
            event.stop()
            self._resolve_approval({"y": "once", "a": "always", "n": "deny"}[event.key])
            return
        if self._palette_open:
            key = event.key
            if key in {"up", "down", "enter", "escape", "tab", "shift+tab"}:
                event.stop()
                self.palette_key(key)
                return
            query = self._palette_query
            if key == "backspace":
                event.stop()
                self._palette_query = query[:-1]
                self.palette_refilter()
                return
            if key == "space":
                event.stop()
                self._palette_query = query + " "
                self.palette_refilter()
                return
            if len(key) == 1 and key.isprintable():
                event.stop()
                self._palette_query = query + key
                self.palette_refilter()
                return
        if event.key == "tab":
            event.stop()
            self.action_complete()
        elif event.key == "escape" and self._busy:
            event.stop()
            self.action_interrupt()

    # ------------------------------------------------------------------ actions

    def action_interrupt(self) -> None:
        if self._palette_open:
            self.action_palette_close()
            return
        if self._busy:
            if self.agent:
                self.agent.cancel()
            self._turn_cancelled = True
            if self._worker is not None:
                try:
                    self._worker.cancel()
                except Exception:
                    pass
            self.set_state("interrupting…")

    def action_smart_quit(self) -> None:
        if self._busy:
            self.action_interrupt()
        else:
            self.exit()

    def action_sidebar(self) -> None:
        side = self.query_one(Sidebar)
        side.shown = not side.shown
        side.display = side.shown
        side.set_keys(KEY_HELP)
        side.refresh(layout=True)

    def action_fold_all(self) -> None:
        self._expand_all = not self._expand_all
        for view in self._tools.values():
            view.collapsed = not self._expand_all
        self.set_state("tools expanded" if self._expand_all else "tools folded")
        self.refresh_chrome()

    def action_auto_approve(self) -> None:
        assert self.agent
        self.set_auto_approve(not self.agent.config.auto_approve)
        state = self.agent.config.auto_approve
        self.notify(
            "auto-approve ON — edits and commands run unattended" if state else "auto-approve OFF",
            severity="warning" if state else "information",
            timeout=2.5,
        )

    def action_next_theme(self) -> None:
        self.set_theme(cycle(self.theme))

    def action_clear(self) -> None:
        self.clear_transcript()

    def action_copy_reply(self) -> None:
        if self.copy_last_reply():
            self.notify("copied last reply to clipboard", timeout=2)

    def action_retry(self) -> None:
        self.retry()

    def action_new_session(self) -> None:
        self.new_session()

    def action_help(self) -> None:
        asyncio.create_task(self.run_command("/help"))

    # ---- palette -----------------------------------------------------

    def action_palette(self) -> None:
        palette = self.query_one(Palette)
        self._palette_query = ""
        entries = [(f"/{c.name} {c.args}".strip(), c.help) for c in COMMANDS.values()]
        entries += [(label, f"action · {name}") for label, name in ACTIONS]
        palette.populate(entries)
        self._palette_open = True
        try:
            palette.results.focus()
        except Exception:
            pass

    def palette_refilter(self) -> None:
        self.query_one(Palette).set_query(self._palette_query)

    def action_palette_close(self) -> None:
        self._palette_query = ""
        self.query_one(Palette).display = False
        self._palette_open = False
        try:
            self.query_one("#prompt-input", PromptInput).focus()
        except Exception:
            pass

    def palette_key(self, key: str) -> None:
        palette = self.query_one(Palette)
        if key in {"up", "shift+tab"}:
            palette.move(-1)
        elif key in {"down", "tab"}:
            palette.move(1)
        elif key == "escape":
            self.action_palette_close()
        elif key == "enter":
            value = palette.current()
            self.action_palette_close()
            if value:
                asyncio.create_task(self.palette_pick(value))

    async def palette_pick(self, value: str) -> None:
        for label, name in ACTIONS:
            if label == value:
                self.run_action(name)
                return
        await self.submit_text(value if value.startswith("/") else f"/{value}")

    def on_palette_picked(self, message: Palette.Picked) -> None:
        message.stop()
        asyncio.create_task(self.palette_pick(message.value))

    # ------------------------------------------------------------------ commands

    async def run_command(self, text: str) -> None:
        name, _, arg = text[1:].partition(" ")
        command = COMMANDS.get(name)
        if command is None:
            self._append(NoticeBlock(f"Unknown command `/{name}` — `/help` lists them all."))
            return
        try:
            result = command.run(self, arg.strip())
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            self._append(ErrorBlock(f"/{name} failed: {type(exc).__name__}: {exc}"))
            return
        if isinstance(result, str) and result.strip():
            self._hide_welcome()
            self._append(NoticeBlock(result))
        self.refresh_chrome()

    async def agent_compact(self) -> str:
        if self.agent is None:
            return ""
        return await self.agent.compact()

    # ------------------------------------------------------------------ completions

    def on_prompt_input_typed(self, message: PromptInput.Typed) -> None:
        self.update_prompt_ui(message.text)

    def on_text_area_changed(self, message: TextArea.Changed) -> None:
        self.update_prompt_ui(self.query_one(PromptInput).content_text)

    def update_prompt_ui(self, text: str) -> None:
        try:
            self.query_one(PromptPlaceholder).display = not text.strip()
        except Exception:
            pass
        widget = self.query_one(PromptInput)
        try:
            widget.styles.height = min(11, max(3, widget.safe_line_count + 2))
        except Exception:
            pass
        popup = self.query_one(CompletionPopup)
        if text.startswith("/") and "\n" not in text and " " not in text:
            needle = text[1:]
            rows = [(f"/{c.name}", c.help) for c in COMMANDS.values() if c.name.startswith(needle)]
            popup.set_rows(rows)
            return
        token = text.rsplit(" ", 1)[-1]
        if token.startswith("@") and self.agent is not None:
            needle = token[1:].lower()
            try:
                names = sorted(p.name for p in self.agent.ctx.cwd.iterdir() if not p.name.startswith("."))
            except OSError:
                names = []
            popup.set_rows([(f"@{n}", "attach as context") for n in names if needle in n.lower()][:8])
            return
        popup.hide()

    def action_complete(self) -> None:
        popup = self.query_one(CompletionPopup)
        if self._palette_open:
            self.palette_key("tab")
            return
        if not popup.rows:
            return
        widget = self.query_one(PromptInput)
        chosen = popup.rows[0][0]
        text = widget.content_text
        if text.startswith("/"):
            widget.load_text_safe(chosen + " ")
        elif text.rstrip().endswith(chosen) or "@" in text:
            head = text.rsplit("@", 1)[0]
            widget.load_text_safe(head + chosen + " ")
        else:
            widget.load_text_safe(text + chosen + " ")
        popup.hide()
        widget.focus()

    def expand_mentions(self, text: str) -> str:
        """Replace `@path` tokens with fenced file contents (bounded)."""
        tokens = re.findall(r"(?<![\w/])@([\w./-]+)", text)
        if not tokens or self.agent is None:
            return text
        blocks: list[str] = []
        for token in tokens[:6]:
            path = self.agent.ctx.cwd / token
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                blocks.append(f"@{token} — could not read")
                continue
            lang = path.suffix.lstrip(".") or "text"
            body = raw[:6000] + ("\n… truncated …" if len(raw) > 6000 else "")
            blocks.append(f"@{token}\n```{lang}\n{body}\n```")
            text = text.replace(f"@{token}", f"[see {token} below]")
        if not blocks:
            return text
        return text + "\n\n" + "\n\n".join(blocks)

    # ------------------------------------------------------------------ app api

    def clear_transcript(self) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.remove_children()
        transcript.mount(Welcome(TIPS))
        self._tools = {}
        self._active = None

    def new_session(self) -> None:
        self._pending_session = None
        self._build_agent()
        self.clear_transcript()
        self.notify("new session", timeout=1.5)

    def list_sessions(self) -> list[Path]:
        return Session.recent(20)

    @staticmethod
    def session_title(path: Path) -> str:
        try:
            with path.open(encoding="utf-8") as fh:
                fh.readline()
                line = fh.readline()
            data = json.loads(line) if line.strip() else {}
            blocks = data.get("blocks", [])
            head = (blocks[0].get("text", "") if blocks else "").strip().replace("\n", " ")
            return head[:60] or "(empty)"
        except Exception:
            return "(unreadable)"

    def load_session(self, path: Path) -> None:
        session = Session.load(path)
        self.config.cwd = session.cwd or self.config.cwd
        self._build_agent(session)

    def replay(self, session: Session) -> None:
        """Re-render a resumed transcript into blocks."""
        self.clear_transcript()
        for msg in session.messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                self._append(UserBlock(msg.text))
                continue
            block = AssistantBlock(model=msg.model or "")
            self._append(block)
            body = ""
            for item in msg.blocks:
                if isinstance(item, TextBlock):
                    body += item.text
                elif isinstance(item, Thinking):
                    continue
                elif isinstance(item, ToolCall):
                    view = ToolView(item)
                    view.set_state("ok" if item.ok else "error", item.result or "")
                    self._tools[item.call_id] = view
                    self._append(view)
                elif isinstance(item, ErrorNote):
                    self._append(ErrorBlock(item.text))
            block.append(body)
            block.finish()

    def add_todo(self, text: str) -> None:
        assert self.agent
        self.agent.session.todos.append({"text": text, "done": False})
        self.agent.session.persist()
        self.query_one(Sidebar).set_todos(self.agent.session.todos)

    def toggle_todo(self, mode: str) -> None:
        assert self.agent
        todos = self.agent.session.todos
        if mode == "clear":
            todos.clear()
        elif mode == "done":
            for item in reversed(todos):
                if not item["done"]:
                    item["done"] = True
                    break
        elif mode == "rm":
            for i, item in enumerate(todos):
                if not item["done"]:
                    todos.pop(i)
                    break
        self.agent.session.persist()
        self.query_one(Sidebar).set_todos(todos)

    def copy_last_reply(self) -> str:
        assert self.agent
        msg = self.agent.session.last_assistant()
        text = msg.text if msg else ""
        if not text:
            self.notify("nothing to copy yet", severity="warning", timeout=2)
            return ""
        try:
            self.copy_to_clipboard(text)
        except Exception:
            self.notify("clipboard unavailable in this terminal", severity="warning", timeout=3)
        return text

    def retry(self) -> None:
        if self._busy or self.agent is None:
            return
        asyncio.create_task(self.agent.retry_last())

    def set_model(self, model: str) -> None:
        assert self.agent
        self.agent.set_model(model)

    def set_provider(self, name: str) -> str:
        try:
            from .agent.providers import get_provider

            provider = get_provider(name)
        except ProviderError as exc:
            return f"`/provider {name}` failed — {exc}"
        assert self.agent
        self.agent.provider = provider
        self.agent.session.provider = provider.name
        self.agent.session.model = provider.model
        self.refresh_chrome()
        return f"Provider → `{name}`, model `{provider.model}`. `esc` mid-turn is unaffected."

    def set_auto_approve(self, value: bool) -> None:
        assert self.agent
        self.agent.config.auto_approve = bool(value)
        self.refresh_chrome()

    def set_theme(self, name: str) -> None:
        if name in THEMES:
            try:
                self.theme = name
            except Exception:
                pass
        self.apply_palette()
        try:
            self.notify(f"theme: {self.theme}", timeout=1.5)
        except Exception:
            pass

    def gauge_text(self, pct: float) -> str:
        return f"{Gauge.plain(pct)} {pct:.0f}%"

    def tool_preview(self, call: ToolCall) -> str:
        return tool_preview(call.tool, call.args)

    def on_unmount(self) -> None:
        try:
            if self.agent:
                self.agent.session.persist()
        except Exception:
            pass


def _shorten(path: str, keep: int = 34) -> str:
    text = path.replace(str(Path.home()), "~", 1)
    return text if len(text) <= keep else "…" + text[-keep:]
