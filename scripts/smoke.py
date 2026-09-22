"""smoke — real end-to-end run: CLI builder -> Engine -> Router -> provider.
Bina API key ke bhi full stack verify karta hai (LocalProvider). Exit 0 on ok.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mharo.__main__ as cli
from mharo.core.engine import Engine
from mharo.core.router import Router
from mharo.providers.local import LocalProvider
from mharo.providers.types import Completion, Usage


class _RecordingProvider(LocalProvider):
    """Contract check: Router dispatch + usage tracking + history append."""

    def __init__(self):
        super().__init__("record", "m")
        self.seen_lens = []

    async def complete(self, messages, tools=None, max_tokens=None, temperature=None):
        self.seen_lens.append(len(messages))
        return Completion(provider=self.name, model=self.model,
                          text="smoke-ack", finish_reason="stop",
                          usage=Usage(inp=3, out=1))


async def main() -> int:
    checks = []

    # 1. CLI builder (local mode)
    engine = cli._build_engine(allow_local=True)
    checks.append((engine.router.providers[0].name == "local", "cli builder -> local ok"))

    # 2. Engine + Recording provider: dispatch through Router
    rec = _RecordingProvider()
    e2 = Engine(Router([rec]))
    text = await e2.respond("ping")
    checks.append((text == "smoke-ack", f"engine router dispatch text={text!r}"))
    checks.append((len(e2.history) == 2, "history user+assistant appended"))
    checks.append((e2.stats.turns == 1 and e2.stats.tokens_out == 1, "usage tracked"))
    checks.append((rec.seen_lens == [1], "provider saw exactly 1 message (no system)"))

    bad = [m for ok, m in checks if not ok]
    for ok, m in checks:
        print(("PASS " if ok else "FAIL ") + m)
    if bad:
        print("SMOKE FAIL:", bad)
        return 1
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))