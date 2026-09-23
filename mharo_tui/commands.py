"""Slash commands. Each returns either a markdown string (printed as a notice
block in the transcript) or None when it performed an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Command:
    name: str
    help: str
    run: Callable[[Any, str], Any]
    args: str = ""
    group: str = "session"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def cmd_help(app, arg: str) -> str:
    rows = "\n".join(
        f"| `/{c.name}`{' ' + c.args if c.args else ''} | {c.help} |"
        for c in sorted(COMMANDS.values(), key=lambda c: (c.group, c.name))
    )
    return (
        "### Commands\n\n| command | what it does |\n|---|---|\n" + rows
        + "\n\nKeys: `enter` send · `shift+enter` newline · `esc` interrupt · "
        "`ctrl+p` palette · `ctrl+b` sidebar · `ctrl+o` fold tools · `ctrl+f` auto-approve · "
        "`ctrl+t` theme · `ctrl+n` new session"
    )


def cmd_context(app, arg: str) -> str:
    s = app.agent.stats()
    bar = app.gauge_text(s["used_pct"])
    return (
        f"**Context** {bar}  {s['prompt_tokens']:,} / {s['window']:,} tokens "
        f"({s['used_pct']}%)\n\n"
        f"- messages: {s['messages']} · provider calls: {s['calls']}\n"
        f"- in {s['input_tokens']:,} / out {s['completion_tokens']:,} tokens · "
        f"est. cost ${s['cost']:.4f}\n"
        f"- model `{s.get('model', app.agent.session.model)}` on `{app.agent.provider.name}`\n"
        + ("" if s["used_pct"] < 80 else "\nAbove 80% — run `/compact` to summarise older turns.")
    )


def cmd_cost(app, arg: str) -> str:
    s = app.agent.stats()
    return (
        f"Session spend: **${s['cost']:.4f}** over {s['calls']} call(s) · "
        f"{s['input_tokens']:,} in / {s['completion_tokens']:,} out tokens."
    )


def cmd_model(app, arg: str) -> Any:
    if not arg:
        models = getattr(app.agent.provider, "models", None) or ["default"]
        return (
            f"Active model: `{app.agent.session.model}` on `{app.agent.provider.name}`.\n\n"
            f"Try: `/model {models[0]}` · `/model gpt-4o` · `/model claude-sonnet-4-5`\n"
            "No key set? `/provider auto` rides free providers first. Change backend with `/provider auto|demo|openai|anthropic`."
        )
    app.set_model(arg)
    return f"Model set to `{app.agent.session.model}`."


def cmd_provider(app, arg: str) -> Any:
    if arg not in {"auto", "free", "demo", "openai", "anthropic"}:
        return "Usage: `/provider auto|demo|openai|anthropic` (`auto` = free ladder, no key needed)."
    return app.set_provider(arg)


def cmd_free(app, arg: str) -> str:
    """Show the free-first ladder the agent is riding (cached probes, no network wait)."""
    provider = getattr(app.agent, "provider", None)
    ladder = getattr(provider, "ladder", None)
    if ladder is None:
        from .agent.ladder import build_default_ladder, quota_hint

        ladder = build_default_ladder()
        head = f"Not using the free ladder right now — provider is `{getattr(provider, 'name', '?')}`.\n\n"
        tail = "\n\nSwitch with `/provider auto`."
        rows = _ladder_table(ladder)
        return head + rows + (f"\n\n{quota_hint(ladder)}" if tail else tail)
    from .agent.ladder import quota_hint

    current = getattr(provider, "current", None)
    notes = "".join(f"\n- {n}" for n in ladder.notices[-3:])
    tried = f"\n\nActive rung: `{current.label()}`" if current is not None else "\n\nNo rung used yet this session."
    return ("### Free-first ladder\n\n" + _ladder_table(ladder)
            + tried
            + (f"\n\nRotation notes:{notes}" if notes else "")
            + (f"\n\n{quota_hint(ladder)}" if not any(r.available() for r in ladder.ordered()) else ""))


def _ladder_table(ladder: Any) -> str:
    rows = ["| rung | kind | model | gate | state |", "|---|---|---|---|---|"]
    for row in ladder.status():
        if row["kind"] == "paid" and not row.get("has_key"):
            continue
        state = "needs a key" if row.get("missing_key") else ("ready" if row["probed_ok"] else
                                                              ("down" if row["probed_ok"] is False else "unprobed"))
        if row["wait_s"]:
            state = f"cooling {int(row['wait_s'])}s"
        gate = f"{row['min_interval_s']:.0f}s" if row["min_interval_s"] else "—"
        what = {"local": "local", "free-anon": "free · no key", "free-key": "free · key"}.get(row["kind"], row["kind"])
        rows.append(f"| `{row['name']}` | {what} | `{row['model'][:30]}` | {gate} | {state} |")
    return "\n".join(rows)


def cmd_theme(app, arg: str) -> Any:
    if arg not in app.mharo_themes:
        return "Themes: " + ", ".join(f"`{t}`" for t in app.mharo_themes)
    app.set_theme(arg)
    return f"Theme → `{arg}`"


def cmd_auto(app, arg: str) -> Any:
    app.set_auto_approve(not app.agent.config.auto_approve)
    state = "ON" if app.agent.config.auto_approve else "OFF"
    return f"Auto-approve edits & commands **{state}**."


def cmd_clear(app, arg: str) -> Any:
    app.clear_transcript()
    return None


def cmd_new(app, arg: str) -> Any:
    app.new_session()
    return "Started a fresh session."


def cmd_resume(app, arg: str) -> Any:
    paths = app.list_sessions()
    if not arg:
        if not paths:
            return "No saved sessions yet."
        rows = "\n".join(
            f"| `{p.stem}` | {app.session_title(p)} |" for p in paths[:12]
        )
        return f"### Recent sessions\n\n| id | first prompt |\n|---|---|\n{rows}\n\nUse `/resume <id>`."
    match = next((p for p in paths if arg in p.stem), None)
    if match is None:
        return f"No session matching `{arg}`."
    app.load_session(match)
    return f"Resumed `{match.stem}`."


def cmd_session(app, arg: str) -> str:
    s = app.agent.session
    files = s.files_touched or {}
    return (
        f"- id: `{s.id}`\n- cwd: `{s.cwd}`\n- model: `{s.model}` on `{s.provider}`\n"
        f"- messages: {len(s.messages)}\n- file: `{s.path}`\n"
        + ("- files touched: " + ", ".join(f"`{k}`" for k in list(files)[:12]) if files else "- no files written yet")
    )


def cmd_files(app, arg: str) -> Any:
    from pathlib import Path

    root = Path(app.agent.ctx.cwd)
    try:
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and all(p not in {".git", "node_modules"} for p in f.parts))
    except OSError:
        size = 0
    return f"`{root}` — {len(list(root.iterdir()))} entries, {_fmt_bytes(size)} tracked-ish (vendored dirs excluded from tools)."


def cmd_tools(app, arg: str) -> str:
    from .agent.tools import TOOLS

    rows = "\n".join(
        f"| `{t.name}` | {t.description.split('.')[0]} | {'approval' if t.approval else 'auto'} |"
        for t in TOOLS.values()
    )
    return (
        "### Tools\n\n| tool | purpose | gate |\n|---|---|---|\n" + rows
        + "\n\n`ctrl+f` toggles auto-approve for everything."
    )


def cmd_todos(app, arg: str) -> Any:
    if not arg:
        items = app.agent.session.todos
        if not items:
            return "No todos. Add one: `/todo fix the test flake`"
        rows = "\n".join(f"- [{'x' if i['done'] else ' '}] {i['text']}" for i in items)
        return f"### Todos\n\n{rows}"
    if arg in {"done", "rm", "clear"}:
        app.toggle_todo(arg)
        return None
    app.add_todo(arg)
    return None


def cmd_copy(app, arg: str) -> Any:
    return app.copy_last_reply()


def cmd_retry(app, arg: str) -> Any:
    app.retry()
    return None


def cmd_init(app, arg: str) -> str:
    return (
        "Scanned this project for context. Commit an `AGENTS.md` / `MHARO.md` with build "
        "commands, layout notes and house rules and I'll pick it up automatically — "
        "it gets prepended to the system prompt."
    )


def cmd_quit(app, arg: str) -> Any:
    app.exit()
    return None


def cmd_version(app, arg: str) -> str:
    from . import __version__

    return f"mharo-tui {__version__} · textual-backed · provider `{app.agent.provider.name}`"


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in [
        Command("help", "list every command and key", cmd_help, group="info"),
        Command("context", "token budget + usage breakdown", cmd_context, group="info"),
        Command("cost", "session spend", cmd_cost, group="info"),
        Command("session", "session id, cwd, files touched", cmd_session, group="info"),
        Command("tools", "tool inventory and approval gates", cmd_tools, group="info"),
        Command("files", "project summary", cmd_files, group="info"),
        Command("version", "print version", cmd_version, group="info"),
        Command("model", "show or set the model", cmd_model, "[name]", group="config"),
        Command("provider", "switch backend (auto = free ladder, no key)", cmd_provider,
                "[auto|demo|openai|anthropic]", group="config"),
        Command("free", "show the free-first provider ladder", cmd_free, group="config"),
        Command("theme", "switch colour theme", cmd_theme, "[name]", group="config"),
        Command("auto", "toggle auto-approve for edits & commands", cmd_auto, group="config"),
        Command("todos", "show todo list", cmd_todos, "[text|done]", group="tasks"),
        Command("clear", "clear the visible transcript", cmd_clear, group="session"),
        Command("new", "start a fresh session", cmd_new, group="session"),
        Command("resume", "list or load saved sessions", cmd_resume, "[id]", group="session"),
        Command("compact", "summarise older turns to free context", lambda a, x: a.agent_compact(), group="session"),
        Command("retry", "re-run the last prompt", cmd_retry, group="session"),
        Command("copy", "copy the last reply to clipboard", cmd_copy, group="session"),
        Command("init", "about project instructions", cmd_init, group="session"),
        Command("quit", "exit", cmd_quit, group="session"),
    ]
}
