"""Custom widgets. Nothing here talks to a model — they render state.

Design rules (these are what make an agent TUI feel professional):

* one visual language per block kind — user / assistant / tool / error / notice
* assistant text is markdown, streamed and throttled (never a repaint per token)
* tool calls are foldable, show duration + exit status, and a diff for edits
* the prompt grows with the text and never steals scroll from the transcript
"""

from __future__ import annotations

import time
from typing import Iterable

from rich.text import Text as RichText
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Label,
    Markdown,
    Static,
    TextArea,
)

from textual.widgets._option_list import Option, OptionList

from .agent.session import ToolCall
from .agent.tools import preview as tool_preview

_UNSET = object()  # sentinel for "attribute not present"

BRAND = "✳"
CHECK = "✓"
CROSS = "✕"
ARROW = "❯"
DIAMOND = "◆"
FOLD = "▸"

STATE_MARK = {"running": "…", "ok": CHECK, "error": CROSS, "denied": "—", "pending": "?"}

# --------------------------------------------------------------------------- palette

# `$var` tokens are resolved by Textual in CSS only — rich style strings get no
# substitution, so we resolve them here against the live theme (with fallbacks).
_FALLBACK = {
    "primary": "#7c9bff", "secondary": "#a78bfa", "accent": "#22d3ee",
    "warning": "#f5b93c", "error": "#ff6b6b", "success": "#4ade80",
    "diff-add": "#3fb950", "diff-del": "#f85149", "muted": "#8b93a7",
    "tool-bg": "#10131d", "user-bg": "#171b28", "brand": "#7c9bff",
}


def pal(widget, spec: str) -> str:
    """Resolve `$token` colour names in a rich style string against `widget.app`."""
    if not spec or "$" not in spec:
        return spec
    try:
        theme = widget.app.current_theme
    except Exception:
        theme = None
    variables = (getattr(theme, "variables", None) or {}) if theme is not None else {}
    out: list[str] = []
    for token in spec.split():
        if token.startswith("$"):
            key = token[1:]
            value = None
            if theme is not None:
                value = getattr(theme, key.replace("-", "_"), None) or variables.get(key)
            out.append(str(value or _FALLBACK.get(key) or "dim"))
        else:
            out.append(token)
    return " ".join(out)




def fuzzy(needle: str, haystack: str) -> bool:
    """Subsequence match — same behaviour the palette needs, zero deps."""
    if not needle:
        return True
    needle, hay = needle.lower(), haystack.lower()
    pos = 0
    for char in needle:
        pos = hay.find(char, pos)
        if pos < 0:
            return False
        pos += 1
    return True


# --------------------------------------------------------------------------- top bar


class TopBar(Horizontal):
    """`✳ mharo │ provider · model │ ⎇ branch ✱n │ path`  ←→  `context ███░ 34% · $0.42`"""

    backend = reactive("demo")
    model = reactive("demo")
    branch = reactive("no-git")
    dirty = reactive(0)
    path = reactive("")
    gauge = reactive("")
    cost = reactive("")
    state = reactive("")

    def __init__(self, **kwargs) -> None:
        self.left = Static("", classes="top-cell")
        self.right = Static("", classes="top-cell top-right")
        super().__init__(self.left, self.right, **kwargs)

    def _paint(self) -> None:
        from . import __version__

        left = RichText(no_wrap=True)
        left.append(f" {BRAND} MHARO ", style=pal(self, "bold $primary"))
        left.append(f"v{__version__}", style="dim")
        if self.branch and self.branch != "no-git":
            left.append(f"  │ ⎇ {self.branch}", style="bold")
            if self.dirty:
                left.append(f" ✱{self.dirty}", style=pal(self, "$warning"))
        if self.path:
            left.append(f"  │ {self.path}", style="dim")
        right = RichText(no_wrap=True)
        ready = "●" if self.state not in ("thinking", "busy") else "○"
        right.append(f"{ready} {self.backend} · {self.model}", style=pal(self, "$success") if ready == "●" else "dim")
        parts = [p for p in (self.gauge, self.cost) if p]
        for part in parts:
            right.append("  ·  " + part, style="dim")
        right.append("  ", style="dim")
        self.left.update(left)
        self.right.update(right)

    def watch_backend(self, *_): self._paint()
    watch_model = watch_backend
    watch_branch = watch_backend
    watch_dirty = watch_backend
    watch_path = watch_backend
    watch_gauge = watch_backend
    watch_cost = watch_backend
    watch_state = watch_backend

    def on_mount(self) -> None:
        self._paint()


class PromptPlaceholder(Static):
    """Sits inside the empty prompt box, like a real editor's placeholder."""

    def __init__(self, text: str = "ask the agent — or / for commands, ! for a shell command") -> None:
        self.text = text
        super().__init__(classes="prompt-placeholder")
        self.display = True

    def render(self) -> RichText:
        return RichText(f"❯ {self.text}", style="dim")


# --------------------------------------------------------------------------- transcript


class Transcript(VerticalScroll):
    """The scrollback. Blocks are appended, never rebuilt."""

    def follow(self) -> None:
        self.scroll_end(animate=False)


class UserBlock(Static):
    """The user's turn. `body` — not `content`, which Static owns as a property."""

    def __init__(self, body: str = "") -> None:
        self._body = body or ""
        super().__init__()

    @property
    def body(self) -> str:
        return self._body

    def render(self) -> RichText:
        out = RichText()
        lines = self._body.splitlines() or [""]
        for i, line in enumerate(lines):
            out.append(f"{ARROW} " if i == 0 else "  ", style=pal(self, "bold $primary"))
            out.append(line.rstrip() + "\n", style="bold" if i == 0 else "")
        return out


class AssistantBlock(Static):
    """Label line + streaming markdown body. `append()` buffers, timer flushes."""

    def __init__(self, model: str = "") -> None:
        self.buffer = ""
        self._pending = False
        self._started = time.time()
        self.model = model
        self.state = "streaming"
        self.head = Static("", classes="block-head")
        self.body = Markdown("")
        super().__init__()

    def compose(self) -> Iterable[Widget]:
        yield self.head
        yield self.body

    def on_mount(self) -> None:
        self.update_head()

    def update_head(self) -> None:
        secs = time.time() - self._started
        label = f"{DIAMOND} mharo"
        if self.model:
            label += f" · {self.model}"
        if self.state == "streaming":
            label += f"   {secs:.1f}s  (esc to interrupt)"
        elif self.state == "done":
            label += f"   {CHECK} {secs:.1f}s"
        else:
            label += f"   {CROSS} stopped {secs:.1f}s"
        style = "bold $primary" if self.state == "streaming" else "dim"
        self.head.update(RichText(label, style=style))

    def tick(self) -> None:
        if self.state == "streaming":
            self.update_head()

    def append(self, chunk: str) -> None:
        self.buffer += chunk
        if self._pending:
            return
        self._pending = True
        self.set_timer(0.05, self._flush)

    async def _flush(self) -> None:
        self._pending = False
        try:
            await self.body.update(self.buffer)
        except Exception:
            self.body.update(self.buffer)
        try:
            self.query_ancestor(Transcript).follow()
        except Exception:
            pass

    def finish(self, interrupted: bool = False) -> None:
        self.state = "interrupted" if interrupted else "done"
        self.update_head()


class ErrorBlock(Static):
    def __init__(self, text: str) -> None:
        self.detail = text or ""
        super().__init__()

    def render(self) -> RichText:
        out = RichText()
        out.append(f" {CROSS} provider error\n", style=pal(self, "bold $error"))
        out.append("\n".join(f"   {line}" for line in self.detail.splitlines()[:30]), style="dim")
        return out


class NoticeBlock(Static):
    """System/notice markdown (used by slash commands)."""

    def __init__(self, markdown: str = "") -> None:
        self.markdown = markdown or ""
        self.body = Markdown(self.markdown)
        super().__init__()

    def compose(self) -> Iterable[Widget]:
        yield self.body

    def set_markdown(self, text: str) -> None:
        self.markdown = text
        try:
            self.query_one(Markdown).update(text)
        except Exception:
            pass


class Welcome(Static):
    """Shown while the transcript is empty."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        super().__init__()

    def render(self) -> RichText:
        out = RichText(no_wrap=True)
        out.append("\n")
        out.append("  M H A R O   A G E N T\n", style=pal(self, "bold $primary"))
        out.append("\n")
        for line in self.lines:
            out.append(f"  {line}\n", style="dim")
        return out


# --------------------------------------------------------------------------- tools


class ToolView(Collapsible):
    """Foldable tool call. Collapsed shows preview + status + duration."""

    SPINNER = "⠋⠙⠹⠼⠴⠧⠇⠏ "

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.state = "pending"
        self._running_since = 0.0
        # Deliberately flat: one Static per part, both direct children of the
        # Collapsible. Wrapping them in a Horizontal (default height:1fr) made
        # every tool block stretch to fill the transcript.
        self.output_view = Static("", classes="tool-out")
        self.diff_view = Static("", classes="tool-diff")
        super().__init__(
            self.output_view,
            self.diff_view,
            title=self.preview_title(),
            collapsed=True,
        )

    def preview_title(self) -> str:
        call = self.call
        mark = STATE_MARK.get(self.state, "?")
        dur = ""
        if call.duration_ms and call.duration_ms > 800:
            dur = f"  {call.duration_ms / 1000:.1f}s"
        return f"{FOLD} {call.tool} {mark}  {tool_preview(call.tool, call.args)}{dur}"

    def set_title_text(self, title: str) -> None:
        setter = getattr(self, "set_title", None)
        if callable(setter):
            try:
                setter(title)
                return
            except Exception:
                pass
        try:
            self.title = title
        except Exception:
            pass

    def set_state(self, state: str, output: str = "", diff: str | None = None) -> None:
        self.state = state
        if state == "running":
            self._running_since = time.time()
            output = output or f"{self.SPINNER[0]} running…"
        if output:
            self.output_view.update(RichText("\n".join(output.rstrip().splitlines()[:400]), style="dim"))
        if diff:
            self.diff_view.display = True
            self.diff_view.update(self._colour_diff(diff))
        self.set_title_text(self.preview_title())
        self.remove_class("tool-ok", "tool-error", "tool-denied")
        mapping = {"ok": "tool-ok", "error": "tool-error", "denied": "tool-denied"}
        if state in mapping:
            self.add_class(mapping[state])
        body_lines = len((output or "").splitlines())
        if state in {"ok", "error"} and 0 < body_lines <= 4 and not diff:
            self.collapsed = False

    def _colour_diff(self, diff: str) -> RichText:
        out = RichText()
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                out.append(line + "\n", style=pal(self, "$diff-add"))
            elif line.startswith("-") and not line.startswith("---"):
                out.append(line + "\n", style=pal(self, "$diff-del"))
            elif line.startswith("@@"):
                out.append(line + "\n", style=pal(self, "$primary"))
            else:
                out.append(line + "\n", style="dim")
        return out

    def toggle_fold(self) -> None:
        self.collapsed = not self.collapsed


# --------------------------------------------------------------------------- approval


class ApprovalPrompt(Container):
    """Inline modal: shows exactly what will run (and the diff) before approval."""

    class Answered(Message):
        def __init__(self, verdict: str) -> None:
            self.verdict = verdict
            super().__init__()

    def __init__(self) -> None:
        self.call: ToolCall | None = None
        self.title_view = Static("", classes="approve-title")
        self.detail_view = Static("", classes="approve-detail")
        self.buttons = Horizontal(
            Button("Allow  (y)", id="approve-yes", variant="success"),
            Button("Always allow  (a)", id="approve-always", variant="primary"),
            Button("Deny  (n)", id="approve-no", variant="error"),
            classes="approve-buttons",
        )
        super().__init__(
            Label("  ⚠ needs your approval", classes="approve-kicker"),
            self.title_view,
            self.detail_view,
            self.buttons,
            id="approval",
        )
        self.display = False

    def ask(self, call: ToolCall, detail: str = "") -> None:
        self.call = call
        self.title_view.update(RichText(f" {ARROW} {tool_preview(call.tool, call.args)}", style="bold"))
        body = RichText()
        for line in (detail or "").splitlines()[:36]:
            if line.startswith("+"):
                body.append(line + "\n", style=pal(self, "$diff-add"))
            elif line.startswith("-"):
                body.append(line + "\n", style=pal(self, "$diff-del"))
            else:
                body.append(line + "\n", style="dim")
        self.detail_view.update(body or RichText(f" args: {call.args}", style="dim"))
        self.display = True
        try:
            self.query_one("#approve-yes", Button).focus()
        except Exception:
            pass

    def answer(self) -> None:
        self.display = False
        self.call = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        verdict = {
            "approve-yes": "once",
            "approve-always": "always",
            "approve-no": "deny",
        }.get(event.button.id or "", "deny")
        self.display = False
        self.post_message(self.Answered(verdict))


# --------------------------------------------------------------------------- prompt


class PromptInput(TextArea):
    """Multi-line prompt. Enter sends, shift+enter / ctrl+j inserts a newline."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    NEWLINE_KEYS = {"shift+enter", "ctrl+j", "ctrl+enter", "alt+enter"}

    @property
    def content_text(self) -> str:
        """The buffer, read through whichever accessor this Textual exposes.

        `getattr(..., "")` is a trap here: the empty-string default is itself a
        str, so the loop used to return "" and typing looked dead. Use a
        non-str sentinel for "attribute missing".
        """
        for attr in ("text", "value", "content"):
            value = getattr(self, attr, _UNSET)
            if isinstance(value, str):
                return value
        return ""

    @property
    def safe_line_count(self) -> int:
        for attr in ("line_count", "document_height"):
            count = getattr(self, attr, _UNSET)
            if isinstance(count, int) and count > 0:
                return count
        return max(1, len(self.content_text.splitlines()))

    def load_text_safe(self, text: str) -> None:
        try:
            self.load_text(text)
            return
        except Exception:
            pass
        for attr in ("clear", "reset"):
            fn = getattr(self, attr, None)
            if callable(fn):
                fn()
                return

    # Textual's TextArea inserts a newline on enter; a priority binding on the
    # focused widget replaces that without touching the editor's own key logic.
    BINDINGS = [
        Binding("enter", "submit_prompt", "Send", priority=True),
        Binding("shift+enter", "insert_newline", "Newline", priority=True),
        Binding("ctrl+j", "insert_newline", "Newline", priority=True),
    ]

    def action_submit_prompt(self) -> None:
        text = self.content_text
        self.load_text_safe("")
        self.post_message(self.Submitted(text))

    def action_insert_newline(self) -> None:
        try:
            self.insert("\n")
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Re-post a bubbling signal so both the widget and the app can react."""
        event.stop()
        self.post_message(self.Typed(self.content_text))

    class Typed(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()


class CompletionPopup(Static):
    """Slash-command / @file suggestions, rendered above the prompt."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        super().__init__(id="completion")
        self.display = False

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows[:8]
        self.display = bool(rows)
        self.refresh()

    def hide(self) -> None:
        self.rows = []
        self.display = False
        self.refresh()

    def render(self) -> RichText:
        out = RichText(no_wrap=True)
        for left, right in self.rows:
            out.append(f" {left:<24s}", style=pal(self, "bold $primary"))
            out.append(right[:70], style="dim")
            out.append("\n")
        return out


class Palette(Container):
    """ctrl+p command palette: fuzzy list of commands + actions."""

    class Picked(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self) -> None:
        self.results = OptionList(id="palette-list")
        self.entries: list[tuple[str, str]] = []
        self.visible_rows: list[tuple[str, str]] = []
        super().__init__(
            Label("  Commands & actions   —   ↑↓ pick · enter run · esc close", classes="palette-title"),
            self.results,
            id="palette",
        )
        self.display = False

    def populate(self, entries: list[tuple[str, str]]) -> None:
        self.entries = entries
        self.show("")

    def show(self, needle: str = "") -> None:
        self.display = True
        self.filter(needle)

    def filter(self, needle: str) -> None:
        rows = [e for e in self.entries if fuzzy(needle, f"{e[0]} {e[1]}")] if needle else list(self.entries)
        self.visible_rows = rows[:40]
        self.results.remove_children()
        if self.visible_rows:
            self.results.add_options(
                [
                    Option(
                        RichText(f" {name:<26s}", style=pal(self, "bold $primary")) + RichText(help_[:64], style="dim"),
                        id=f"opt{i}",
                    )
                    for i, (name, help_) in enumerate(self.visible_rows)
                ]
            )
        self.results.highlighted = 0 if self.visible_rows else None

    def move(self, delta: int) -> None:
        count = len(self.visible_rows)
        if not count:
            return
        cur = self.results.highlighted or 0
        self.results.highlighted = (cur + delta) % count

    def current(self) -> str | None:
        idx = self.results.highlighted
        if idx is None or idx >= len(self.visible_rows):
            return None
        return self.visible_rows[idx][0]


# --------------------------------------------------------------------------- chrome


class StatusLine(Horizontal):
    """`⠹ working… 4.2s  ·  thinking: …` on the left, mode badges on the right."""

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self.state_view = Static("", id="state-text")
        self.hints = Static("", id="status-hints")
        self._frame = 0
        self._busy = False
        super().__init__(self.state_view, self.hints, id="status")

    def set_state(self, text: str, busy: bool, hints: str = "") -> None:
        self._busy = busy
        mark = self.SPINNER[self._frame % len(self.SPINNER)] if busy else "●"
        self.state_view.update(RichText(f"{mark} {text}", style=pal(self, "bold $primary" if busy else "dim")))
        self.hints.update(RichText(f" {hints}", style="dim"))

    def spin(self) -> None:
        """Called by the app's 8fps tick; no-op when idle so we never repaint."""
        if not self._busy:
            return
        self._frame += 1
        self.state_view.refresh()


class Sidebar(Vertical):
    def __init__(self) -> None:
        self.todos = Static("", classes="side-body")
        self.files = Static("", classes="side-body")
        self.modes = Static("", classes="side-body")
        self.usage = Static("", classes="side-body")
        self.keys = Static("", classes="side-body")
        super().__init__(
            self._head("Todos"), self.todos,
            self._head("Files touched"), self.files,
            self._head("Modes"), self.modes,
            self._head("Context"), self.usage,
            self._head("Keys"), self.keys,
            id="sidebar",
        )
        self.shown = False
        self.display = False

    @staticmethod
    def _head(text: str) -> Static:
        return Static(RichText(text, style="bold dim"), classes="side-head")

    def set_usage(self, stats: dict) -> None:
        out = RichText()
        for label, value in (
            ("prompt", f"{stats.get('prompt_tokens', 0):,} tok"),
            ("window", f"{stats.get('window', 0):,} tok"),
            ("used", f"{stats.get('used_pct', 0)}%"),
            ("calls", str(stats.get("calls", 0))),
            ("cost", f"${stats.get('cost', 0):.4f}"),
            ("messages", str(stats.get("messages", 0))),
        ):
            out.append(f" {label:<8s}", style="dim")
            out.append(f"{value}\n", style="")
        self.usage.update(out)

    def set_todos(self, todos: list[dict]) -> None:
        if not todos:
            self.todos.update(RichText(" none — /todo <text>", style="dim"))
            return
        out = RichText()
        for item in todos:
            out.append(" ☑ " if item.get("done") else " ☐ ", style=pal(self, "$diff-add" if item.get("done") else "dim"))
            out.append(item["text"][:36] + "\n", style="dim")
        self.todos.update(out)

    def set_files(self, files: dict[str, int]) -> None:
        if not files:
            self.files.update(RichText(" none yet", style="dim"))
            return
        out = RichText()
        for name, lines in list(files.items())[:16]:
            out.append(" ● ", style=pal(self, "$diff-add"))
            out.append(name[:34], style="dim")
            out.append(f"  {lines}L\n" if lines else "\n", style="dim")
        self.files.update(out)

    def set_modes(self, model: str, provider: str, auto: bool, expanded: bool) -> None:
        out = RichText()
        out.append(" model    ", style="dim")
        out.append(f"{model}\n", style="")
        out.append(" provider ", style="dim")
        out.append(f"{provider}\n", style="")
        out.append(" approve  ", style="dim")
        out.append("always allow\n" if auto else "ask first\n", style=pal(self, "$warning" if auto else "$success"))
        out.append(" tools    ", style="dim")
        out.append("expanded\n" if expanded else "folded\n", style="dim")
        self.modes.update(out)

    def set_keys(self, rows: list[tuple[str, str]]) -> None:
        out = RichText()
        for key, what in rows:
            out.append(f" {key:<12s}", style=pal(self, "$primary"))
            out.append(what + "\n", style="dim")
        self.keys.update(out)


class Gauge(Static):
    def __init__(self) -> None:
        self.pct = 0.0
        super().__init__(id="gauge")

    def render(self) -> RichText:
        filled = int(round(max(0.0, min(100.0, self.pct)) / 100 * 12))
        out = RichText()
        out.append("█" * filled, style=pal(self, "bold $primary"))
        out.append("░" * (12 - filled), style="dim")
        out.append(f" {self.pct:.0f}%", style="dim")
        return out

    @staticmethod
    def plain(pct: float, width: int = 12) -> str:
        filled = int(round(max(0.0, min(100.0, pct)) / 100 * width))
        return "█" * filled + "░" * (width - filled)
