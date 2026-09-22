"""MharoAgent — REAL pytest suite. Mock HTTP, koi network nahi."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from mharo.core.engine import Engine
from mharo.core.router import Router
from mharo.providers.errors import ProviderError, RateLimitError, TimeoutError2
from mharo.providers.openai_compat import OpenAICompatible
from mharo.providers.protocol import ProviderSettings
from mharo.providers.stream import Event, openai_events
from mharo.providers.types import Completion, Message, Usage

pytest_plugins = []


def run(coro):
    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_message_to_dict():
    m = Message(role="tool", content="42", name="calc", tool_call_id="t1")
    d = m.to_dict()
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "t1"
    m2 = Message(role="user", content="hi")
    assert m2.to_dict() == {"role": "user", "content": "hi"}


def test_usage_total():
    u = Usage(inp=10, out=5)
    assert u.total == 15


def test_completion_usage_cost():
    c = Completion(model="gpt-4o-mini", text="hi", usage=Usage(inp=4, out=6))
    assert c.text == "hi"
    assert c.usage.total == 10
    assert c.cost_usd > 0
    assert c.has_tools is False


def test_router_pick_cost_uses_first_alive():
    a = OpenAICompatible("openai", "gpt-4o-mini", "k1")
    b = OpenAICompatible("deepseek", "deepseek-chat", "k2")
    r = Router([a, b], strategy="cost")
    assert r.pick().name == "openai"
    assert any(p.alive for p in r._alive())


def test_router_pick_skips_dead():
    a = OpenAICompatible("openai", "gpt-4o-mini", "k1")
    b = OpenAICompatible("deepseek", "deepseek-chat", "k2")
    a.alive = False
    r = Router([a, b])
    assert r.pick().name == "deepseek"


def test_router_no_alive_raises():
    r = Router([], strategy="cost")
    with pytest.raises(ProviderError):
        r.pick()


def test_openai_events_text_and_usage_loop():
    consumed = {"text": [], "usage": None}

    async def go():
        buf = []
        raw = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        ]
        for line in raw:
            for ev in openai_events(line, buf):
                if ev.kind == "text":
                    consumed["text"].append(ev.delta)
                elif ev.kind == "usage":
                    consumed["usage"] = ev.usage

    run(go())
    assert consumed["text"] == ["Hel", "lo"]
    assert consumed["usage"].inp == 5
    assert consumed["usage"].out == 3


def test_engine_window_respects_limit():
    eng = Engine(None, system="sys", history_limit=2)
    msgs = [Message(role="user", content=str(i)) for i in range(5)]
    w = eng.window(msgs, limit=2)
    assert [m.content for m in w] == ["sys", "3", "4"]


def test_engine_window_implicit_history_limit():
    eng = Engine(None, system="", history_limit=2)
    w = eng.window([Message(role="user", content=str(i)) for i in range(4)])
    assert len(w) == 2


@pytest.mark.asyncio
async def test_openai_compat_payload_and_close():
    p = OpenAICompatible("openai", "gpt-4o-mini", "k1")
    payload = p._payload([Message(role="user", content="hi")], max_tokens=100)
    assert payload["model"] == "gpt-4o-mini"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 100
    assert p.url.endswith("/chat/completions")
    await p.close()


@pytest.mark.asyncio
async def test_router_rotates_keys_on_429_first_then_success():
    responses = iter([429, 200])

    async def handler(request: httpx.Request) -> httpx.Response:
        code = next(responses)
        if code == 200:
            body = "data: " + '{"choices":[{"delta":{"content":"ok"}}]}' + "\n\n"
            body += "data: " + '{"choices":[{"delta":{}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}' + "\n\n"
            body += "data: [DONE]\n\n"
            return httpx.Response(200, content=body)
        return httpx.Response(429, content="rate limited")

    p = OpenAICompatible("openai", "gpt-4o-mini", "k1", base_url="http://mock")
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p.keys = ["k1", "k2"]

    r = Router([p])
    comp = await r.complete([Message(role="user", content="hi")])
    assert comp.text == "ok"
    assert p.api_key == "k2"
    assert r.stats.rotations >= 1
    await r.close()


@pytest.mark.asyncio
async def test_router_fallback_to_next_provider():
    async def fail(req):
        raise RateLimitError("nope", provider="openai")

    async def ok(req):
        body = "data: " + '{"choices":[{"delta":{"content":"bye"}}]}' + "\n\n"
        body += "data: [DONE]\n\n"
        return httpx.Response(200, content=body)

    a = OpenAICompatible("openai", "gpt-4o-mini", "k1", base_url="http://a")
    b = OpenAICompatible("deepseek", "deepseek-chat", "k2", base_url="http://b")
    a._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(ok))

    r = Router([a, b])
    comp = await r.complete([Message(role="user", content="hi")])
    assert comp.provider == "deepseek"
    assert comp.text == "bye"
    assert r.stats.fallbacks >= 1
    await r.close()


@pytest.mark.asyncio
async def test_engine_respond_runs_router():
    async def ok(req):
        body = "data: " + '{"choices":[{"delta":{"content":"hello there"}}]}' + "\n\n"
        body += "data: [DONE]\n\n"
        return httpx.Response(200, content=body)

    p = OpenAICompatible("openai", "gpt-4o-mini", "k1", base_url="http://m")
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(ok))
    eng = Engine(Router([p]))
    out = await eng.respond("namaste")
    assert out == "hello there"
    assert eng.stats.turns == 1
    assert eng.history[-1].role == "assistant"
    assert eng.stats.latency_ms >= 0
    await eng.router.close()