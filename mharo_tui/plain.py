"""Headless mode: `mharo --print "…"` and the no-TTY REPL.

Shares the exact same Agent, tool and approval code as the TUI, so CI output
matches what you saw in the terminal. Approvals fall back to prompting on
stdin (or pass `--yes` to skip).
"""

from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .agent import Agent, AgentConfig
from .agent.providers import ProviderError
from .agent.session import Session, ToolCall
from .agent.tools import preview as tool_preview

console = Console()

try:  # used by callers that want to degrade gracefully
    import rich  # noqa: F401

    HAS_RICH = True
except Exception:  # pragma: no cover
    HAS_RICH = False


def render_event(agent: Agent, evt: dict) -> None:
    """Mirror transcript events onto stdout, one line per state change."""
    kind = evt.get("type")
    if kind == "user_message":
        console.print()
        console.print(
            Panel(evt["message"].text, title="you", border_style="blue", padding=(0, 1), expand=False)
        )
    elif kind == "assistant_start":
        console.print(f"[bold]◆ mharo[/] [dim]· {evt.get('model', '')}[/]")
    elif kind == "tool_start":
        call = evt["call"]
        console.print(f"  [yellow]▸ {tool_preview(call.tool, call.args)}[/]")
    elif kind == "tool_end":
        call: ToolCall = evt["call"]
        colour = "green" if call.ok else "red"
        mark = "✓" if call.ok else "✕"
        console.print(f"  [{colour}]{mark} {call.tool} ({call.duration_ms or 0} ms)[/]")
        body = (call.result or "").strip()
        if body and console.width >= 60:
            for line in body.splitlines()[:10]:
                console.print(f"    [dim]{line[:160]}[/]")
            if len(body.splitlines()) > 10:
                console.print(f"    [dim]… {len(body.splitlines()) - 10} more lines[/]")
    elif kind == "error":
        console.print(f"[red]✕ {evt.get('text', '')}[/]")
    elif kind == "notice":
        console.print(f"[dim]{evt.get('text', '')}[/]")


def print_reply(text: str, hint: str = "") -> None:
    console.print(Markdown(text or hint or "_(empty response)_"))


def reply_hint(message: object) -> str:
    """When a turn produced no prose, say why instead of shrugging."""
    for block in getattr(message, "blocks", None) or []:
        if type(block).__name__ == "ErrorNote":
            line = str(getattr(block, "text", "")).strip().splitlines()
            if line:
                return "_(no answer — " + line[0][:180] + ")_"
    return ""


def stdin_approver(call: ToolCall) -> bool:
    console.print(f"[yellow]⚠ {tool_preview(call.tool, call.args)}[/]")
    if not sys.stdin.isatty():
        console.print(
            "[red]  denied — stdin is not a terminal, so there is nobody to ask. "
            "Re-run with --yes to allow edits in a script.[/]"
        )
        return False
    try:
        answer = console.input("[yellow]  allow this? [y/N] [/]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]  denied[/]")
        return False
    return answer in {"y", "yes"}


async def one_shot(agent: Agent, prompt: str, approver: Callable[[ToolCall], bool] | None = None) -> str:
    """Programmatic entry point (used by tests and by other tooling)."""
    agent.on_event = lambda evt: render_event(agent, evt)
    if approver is not None:
        agent.request_approval = approver
    await agent.submit(prompt)
    reply = agent.session.last_assistant()
    return reply.text if reply else ""


def _show_usage(agent: Agent) -> None:
    stats = agent.stats()
    console.print(
        Rule(
            f"{stats['calls']} call(s) · {stats['input_tokens']:,} in / {stats['completion_tokens']:,} out · "
            f"${stats['cost']:.4f} · saved {agent.session.path.name}",
            style="dim",
        )
    )


async def run(
    config: AgentConfig | None = None,
    prompt: str | None = None,
    *,
    interactive: bool = True,
    theme: str | None = None,
    session: Session | None = None,
) -> int:
    """Entry used by `mharo --print` and by the no-TTY REPL."""
    config = config or AgentConfig()

    def _emit(evt: dict) -> None:
        render_event(agent, evt)

    try:
        agent = Agent(config, on_event=_emit, session=session)
    except ProviderError as exc:
        console.print(f"[red]✕ provider unavailable:[/] {exc}")
        console.print("[yellow]  continuing on the offline demo provider so the tools stay usable[/]")
        config.provider, config.model = "demo", None
        agent = Agent(config, on_event=_emit, session=session)
    agent.request_approval = None if config.auto_approve else stdin_approver

    from . import __version__

    header = Text()
    header.append(" MHARO ", style="bold #f0997b")
    header.append(f"v{__version__}", style="dim")
    header.append("   ")
    header.append("● ready", style="bold green")
    header.append(f"   {getattr(agent.provider, 'describe', lambda: agent.provider.name)()} · {agent.ctx.cwd}", style="dim")
    console.print(header)
    console.print(Rule(style="dim"))
    if config.auto_approve:
        console.print("[yellow]auto-approve is ON — edits and commands run without asking[/]")

    if prompt:
        await agent.submit(prompt)
        reply = agent.session.last_assistant()
        print_reply(reply.text if reply else "", reply_hint(reply))
        _show_usage(agent)
        return 0

    console.print("[dim]type /help for commands · ctrl+d or /quit to exit[/]")
    while interactive:
        console.print()
        console.print(Rule(style="dim"))
        try:
            text = console.input("[bold green]➜ [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in {"/quit", "/exit", "quit"}:
            break
        if text == "/help":
            from .commands import COMMANDS

            console.print("\n".join(f"  [bold]/{c.name}[/] — {c.help}" for c in COMMANDS.values()))
            continue
        if text in {"/context", "/cost", "/session", "/tools", "/version"}:
            stats = agent.stats()
            console.print(
                f"  prompt {stats['prompt_tokens']:,} tok · window {stats['window']:,} tok · used {stats['used_pct']}% · "
                f"calls {stats['calls']} · ${stats['cost']:.4f} · branch {stats['branch']}"
            )
            continue
        if text.startswith("!"):
            from .agent.tools import run_tool

            result = run_tool("bash", {"command": text[1:]}, agent.ctx)
            console.print(
                Panel(
                    result.output or "(no output)",
                    title=f"$ {text[1:].strip()}",
                    border_style="green" if result.ok else "red",
                )
            )
            continue
        try:
            await agent.submit(text)
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}: {exc}[/]")
            continue
        reply = agent.session.last_assistant()
        print_reply(reply.text if reply else "", reply_hint(reply))
        _show_usage(agent)
    return 0


def stdin_is_pipe() -> bool:
    return not sys.stdin.isatty()
