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
from mharo.providers import catalog


async def main() -> int:
    cli._load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    env = catalog.env_with_opencode_auth()
    configured = catalog.configured(env=env)
    if not configured:
        known = ", ".join(c.env[0] for c in catalog.CATALOG)
        print(
            "LIVE_SMOKE_SKIP: no API key.\n"
            f"Set any of: {known} (ya opencode se provider login) aur dobara chalayein.",
            file=sys.stderr,
        )
        return 1

    engine = cli._build_engine(allow_local=False)
    print("live providers:", [(c.name, c.model) for c, _ in configured])
    print("user > hi there, here is a test ping")
    try:
        text = await engine.respond("Reply with exactly: PONG_OK and nothing else.")
    except Exception as exc:
        print("ERROR:", exc)
        for e in engine.router.stats.errors:
            print("  router error:", e)
        await engine.router.close()
        return 2
    print("ai   >", text)
    print("stats:", engine.stats.turns, "in", engine.stats.tokens_in,
          "out", engine.stats.tokens_out, "ms", round(engine.stats.latency_ms, 1))
    if not text.strip():
        print("LIVE_SMOKE_FAIL: empty reply", file=sys.stderr)
        return 2
    await engine.router.close()
    print("LIVE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))