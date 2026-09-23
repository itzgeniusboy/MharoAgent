"""Mharo TUI — a professional terminal interface for AI coding agents.

Layout of the package::

    mharo_tui/
      app.py        the Textual App (composition, keymap, event routing)
      widgets.py    transcript blocks, prompt input, sidebar panels
      commands.py   slash-command registry
      themes.py     colour themes
      app.tcss      stylesheet
      agent/
        providers.py  streaming LLM adapters (demo / openai-compatible / anthropic)
        tools.py      bash, read, write, edit, list, search + approval gate
        session.py    message model, token + cost accounting, JSONL persistence
      plain.py      no-TTY / CI fallback (rich REPL, same agent core)

Public API is deliberately small so you can embed this inside an existing
project (e.g. `MharoAgent`) without dragging the whole app in.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "MharoApp", "Agent", "AgentConfig", "get_provider"]


def __getattr__(name: str):  # lazy imports keep `mharo --version` instant
    if name == "MharoApp":
        from .app import MharoApp

        return MharoApp
    if name in {"Agent", "AgentConfig"}:
        from .agent import Agent, AgentConfig

        return Agent if name == "Agent" else AgentConfig
    if name == "get_provider":
        from .agent.providers import get_provider

        return get_provider
    raise AttributeError(f"module 'mharo_tui' has no attribute {name!r}")
