"""Dashboard — text/console dashboard from Engine stats.

`render(engine)` -> plain-text panel (providers, key counts, session stats,
errors). TTY pe colors, agla `status_line` reuse karta hai.
"""
from __future__ import annotations

from typing import Optional

from mharo.tui import status_line


def render(engine, stream=None) -> str:
    """Engine + Router state ka readable panel."""
    router = engine.router
    lines = ["MHARO DASHBOARD", "-" * 40]
    lines.append(router.strategy + " strategy")
    for p in router.providers:
        nk = len(getattr(p, "keys", []) or [])
        state = "ok" if p.alive else f"dead({getattr(p, '_alive_reason', '')})"
        lines.append(f"  • {p.name:10s} model={getattr(p, 'model', '?'):18s} "
                     f"keys={nk} state={state}")
    lines.append("-" * 40)
    lines.append(status_line(engine.stats, stream=stream))
    if engine.stats.errors:
        lines.append("errors:")
        for e in engine.stats.errors[:5]:
            lines.append(f"  ! {e}")
    return "\n".join(lines)


__all__ = ["render"]