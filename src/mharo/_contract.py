import asyncio


async def m() -> None:
    from mharo.providers.openai_compat import OpenAICompatible
    from mharo.providers.types import Message, Usage
    from mharo.providers.types import Completion
    from mharo.core.router import Router
    from mharo.core.engine import Engine

    a = OpenAICompatible("openai", "gpt-4o-mini", "k1")
    b = OpenAICompatible("deepseek", "deepseek-chat", "k2")
    r = Router([a, b], strategy="cost")
    p = r.pick()
    assert p.name in {"openai", "deepseek"}
    assert p.alive is True
    assert any(q.alive for q in r._alive())
    w = Engine(r).window([Message(role="user", content="hi")], 10)
    assert w and w[-1].role == "user"
    c = Completion(choices=[], usage=Usage(inp=10, out=5), provider="openai")
    assert c.model is None
    print("CONTRACT_OK", p.name, len(w), c.usage.total)


asyncio.run(m())