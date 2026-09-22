"""AppMode — one-shot app-style runner (headless, non-interactive).

`run(engine, prompt)` -> reply text is single shot (history reuse). CLI
interactive ke liye hi hai, app/dashboard integration ke liye ye entry.
"""
from __future__ import annotations

from typing import Optional

from mharo.core.engine import Engine


async def run(engine: Engine, prompt: str, extra_system: str = "") -> str:
    """Ek hi shot — user ke liye returns final text. Engine REAL path."""
    return await engine.respond(prompt, extra_system=extra_system)


__all__ = ["run"]