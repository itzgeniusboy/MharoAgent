"""live_smoke — REAL OpenAI/Deepseek public API smoke.

Challenge:  iss script ko ek real API key chahiye (.env ya env me).
- Key nahi -> clear exit 1 (config error, not test failure).
- Key hai  -> OpenAICompatible REAL network ke saath ek turn karta hai
              (SSE streaming, Router stats, Engine history) — asli live verify.

Run:  python scripts/live_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mharo.__main__ as cli


async def main() -> int:
    cli._load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key:
        print(
            "LIVE_SMOKE_SKIP: no API key.\n"
            "Set OPENAI_API_KEY (or write .env: OPENAI_API_KEY=sk-...) "
            "aur dobara chalayein.",
            file=sys.stderr,
        )
        return 1

    engine = cli._build_engine(allow_local=False)
    print("live provider:", engine.router.providers[0].name,
          engine.router.providers[0].model)
    print("user > hi there, here is a test ping")
    text = await engine.respond("Reply with exactly: PONG_OK and nothing else.")
    print("ai   >", text)
    print("stats:", engine.stats.turns, "in", engine.stats.tokens_in,
          "out", engine.stats.tokens_out, "ms", round(engine.stats.latency_ms, 1))
    if text.strip().upper() != "PONG_OK":
        print("LIVE_SMOKE_FAIL: unexpected reply", file=sys.stderr)
        return 2
    await engine.router.close()
    print("LIVE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))