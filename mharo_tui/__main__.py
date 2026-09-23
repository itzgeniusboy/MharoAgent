"""`mharo` entry point.

    mharo                       # TUI, provider sniffed from env
    mharo -p openai -m gpt-4o   # TUI with an OpenAI-compatible backend
    mharo "explain @src/app.ts" # one-shot prompt, prints markdown, exits
    mharo --print "run tests"    # headless (CI-friendly, no TTY needed)
    mharo -c                     # resume the most recent session
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from .agent import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mharo",
        description="Terminal UI for AI coding agents (opencode / Claude Code style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt", nargs="*", help="optional one-shot prompt (implies --print)")
    p.add_argument("--provider", "-p", choices=["auto", "free", "demo", "openai", "anthropic"],
                   help="backend (default: your keys if set, else the free ladder — no key needed)")
    p.add_argument("--model", "-m", help="model id")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (Ollama, OpenRouter, vLLM…)")
    p.add_argument("--cwd", default=os.environ.get("MHARO_CWD", "."), help="project directory")
    p.add_argument("--theme", "-t", help=f"one of: {', '.join(_theme_names())}")
    p.add_argument("--yes", "-y", action="store_true", help="auto-approve edits and commands")
    p.add_argument("--max-turns", type=int, default=12, help="tool-loop cap per prompt")
    p.add_argument("--temperature", type=float, default=float(os.environ.get("MHARO_TEMPERATURE", 0.2)))
    p.add_argument("--print", "--print-only", dest="print_mode", action="store_true", help="no TUI; print answer and exit")
    p.add_argument("--plain", action="store_true", help="rich REPL instead of the full TUI")
    p.add_argument("--continue", "-c", dest="continue_last", action="store_true", help="resume latest session")
    p.add_argument("--resume", metavar="ID", help="resume a saved session id")
    p.add_argument("--system", help="replace the system prompt")
    p.add_argument("--version", action="version", version=f"mharo-tui {__version__}")
    return p


def _theme_names() -> list[str]:
    try:
        from .themes import THEMES

        return list(THEMES)
    except Exception:
        return []


def make_config(args: argparse.Namespace) -> AgentConfig:
    opts: dict[str, object] = {"temperature": args.temperature}
    if args.base_url:
        opts["base_url"] = args.base_url
    cwd = Path(args.cwd).expanduser()
    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        cwd=cwd,
        auto_approve=args.yes,
        max_turns=args.max_turns,
        provider_opts=opts,
    )
    if args.system:
        config.system_prompt = args.system
    project_prompt = load_project_instructions(cwd)
    if project_prompt:
        config.system_prompt = f"{config.system_prompt}\n\n# Project instructions (AGENTS.md)\n{project_prompt}"
    return config


def load_project_instructions(cwd: Path) -> str:
    for name in ("AGENTS.md", "Mharo.md", "MHARO.md", "CLAUDE.md", "CURSOR.md"):
        candidate = cwd / name
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")[:12_000]
            except OSError:
                return ""
    return ""


def tty_available() -> bool:
    return sys.stdout.isatty() and sys.stdin.isatty() and os.environ.get("TERM", "") not in ("", "dumb")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = make_config(args)

    session = None
    if args.resume or args.continue_last:
        from .agent.session import Session

        if args.resume:
            matches = [p for p in Session.recent(60) if args.resume in p.stem]
            if not matches:
                print(f"no session matching {args.resume!r}", file=sys.stderr)
                return 2
            session = Session.load(matches[0])
        else:
            recent = Session.recent(1)
            if recent:
                session = Session.load(recent[0])

    prompt = " ".join(args.prompt).strip()

    def _run_plain(reason: str | None = None) -> int:
        from .plain import run

        if reason:
            print(f"[mharo] {reason}", file=sys.stderr)
        try:
            return asyncio.run(
                run(
                    config=config,
                    prompt=prompt or None,
                    interactive=not args.print_mode and not prompt,
                    theme=args.theme,
                    session=session,
                )
            )
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # a CLI must never print a stack for a config problem
            from .agent.providers import ProviderError

            if isinstance(exc, ProviderError):
                print(f"mharo: {exc}", file=sys.stderr)
                return 2
            raise

    # --print / a one-shot prompt / --plain always mean "no full-screen UI".
    if args.print_mode or args.plain or (prompt and not args.plain and args.print_mode):
        return _run_plain()
    if prompt:
        # one-shot prompt with no --plain flag: still headless, just interactive=False
        return _run_plain()

    # Otherwise ALWAYS try the real, boxed Textual UI first — this is what
    # gives the docked bottom input, the top bar, panels, etc. Only fall
    # back to the plain REPL if Textual genuinely cannot start (no tty,
    # unsupported terminal), never on a guess.
    if not tty_available():
        return _run_plain(
            "no interactive terminal detected (stdin/stdout not a TTY, or $TERM unset) "
            "— falling back to plain mode. Run `mharo` directly inside your terminal app "
            "(not via a script/pipe) to get the full boxed UI."
        )

    from .app import MharoApp

    app = MharoApp(config, session=session)
    if args.theme:
        app.mharo_forced_theme = args.theme  # type: ignore[attr-defined]
    try:
        app.run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Textual failed to take over the screen for some environment-specific
        # reason — degrade gracefully instead of crashing.
        return _run_plain(f"full UI failed to start ({type(exc).__name__}: {exc}) — using plain mode.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
