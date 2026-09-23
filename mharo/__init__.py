"""Mharo Agent — Phase 1 core engine.

Public surface (everything else is internal):

    Config      resolved config (file + env)
    Engine      plan → act → verify loop
    EngineSettings
    Result      what a run returns, incl. proof
    ProviderHub tier × key-pool × protocol transport
    Router      cheap/strong decisions + dynamic upgrade
    Memory      SQLite vector recall + compaction
    Permissions allow/ask/deny policy
    Vault       secret store + output redaction
    SessionDB   sessions, ledger, undo snapshots

`mharo` deliberately reuses `mharo_tui`'s tool registry and provider classes, so
the TUI and the CLI agent are the same engine wearing different faces.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "Config", "Engine", "EngineSettings", "Result", "ProviderHub", "Router",
    "Memory", "Permissions", "Vault", "SessionDB", "Skill", "main", "__version__",
]


def __getattr__(name: str):
    """Lazy exports so `import mharo` stays cheap in a TUI boot path."""
    if name == "Config":
        from .config import Config

        return Config
    if name in {"Engine", "EngineSettings", "Result"}:
        from . import engine

        return getattr(engine, name)
    if name == "ProviderHub":
        from .providers import ProviderHub

        return ProviderHub
    if name == "Router":
        from .router import Router

        return Router
    if name == "Memory":
        from .memory import Memory

        return Memory
    if name == "Permissions":
        from .permissions import Permissions

        return Permissions
    if name == "Vault":
        from .vault import Vault

        return Vault
    if name == "SessionDB":
        from .sessiondb import SessionDB

        return SessionDB
    if name == "Skill":
        from .skills import Skill

        return Skill
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module 'mharo' has no attribute {name!r}")
