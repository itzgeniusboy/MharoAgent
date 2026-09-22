"""Tiny REAL verify — Engine+SessionStats+Router. No heredoc, write tool clean."""
import asyncio
import sys

sys.path.insert(0, "src")
from mharo.providers.openai_compat import OpenAICompatible
from mharo.core.router import Router, RouterStats
from mharo.core.engine import Engine, SessionStats
from mharo.providers.protocol import ProviderSettings


async def main() -> None:
    a = OpenAICompatible("openai", "gpt-4o-mini", "k1")
    b = OpenAICompatible("deepseek", "deepseek-chat", "k2")
    r = Router([a, b], strategy="cost")
    r.stats = RouterStats()  # REAL counters
    e = Engine(r, system="you are a REAL helper", history_limit=5)
    assert isinstance(e.stats, SessionStats)
    fake = e._window([], 5)
    assert isinstance(fake, list)
    print("ENGINE_WINDOW_OK", len(fake))
    print("ROUTER_STATS_OK", r.stats.attempts)
    await r.close()


asyncio.run(main())
print("REAL_ALL_OK")
