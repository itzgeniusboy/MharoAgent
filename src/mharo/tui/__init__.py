"""TUI — terminal window helpers (Termux-friendly).

`banner()` Termux window ke liye header, `status_line()` stats row,
`box()` labeled divider. TTY me color/unicode, non-TTY me plain — koi
junk escape kisi ke terminal me nahi.
"""
from __future__ import annotations

import shutil
import sys
from typing import Optional


def _tty(stream=None) -> bool:
    s = stream if stream is not None else sys.stdout
    return bool(getattr(s, "isatty", lambda: False)())


def width(default: int = 60) -> int:
    try:
        return max(20, min(shutil.get_terminal_size((default, 20)).columns, 100))
    except Exception:
        return default


def banner(title: str, sub: str = "", stream=None) -> str:
    w = width() if _tty(stream) else 48
    inner = title if not sub else f"{title} — {sub}"
    if _tty(stream):
        return f"\x1b[1m{'=' * w}\n{inner.center(w)}\n{'=' * w}\x1b[0m"
    return f"{'=' * w}\n{inner.center(w)}\n{'=' * w}"


def status_line(stats, stream=None) -> str:
    """SessionStats ko ek hi line me (turns/in/out/latency/fallbacks)."""
    parts = [
        f"turns={stats.turns}",
        f"in={stats.tokens_in}",
        f"out={stats.tokens_out}",
        f"lat={stats.latency_ms:.0f}ms",
        f"fb={stats.fallbacks}",
    ]
    line = " | ".join(parts)
    if _tty(stream):
        return f"\x1b[36m[{line}]\x1b[0m"
    return f"[{line}]"


def box(text: str, label: str = "", stream=None) -> str:
    w = width() if _tty(stream) else 48
    head = f"--- {label} ".ljust(w - 1, "-") + "-" if label else "-" * w
    return f"{head}\n{text}\n{'-' * w}"


__all__ = ["banner", "status_line", "box", "width"]