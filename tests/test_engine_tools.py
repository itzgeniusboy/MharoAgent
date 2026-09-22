"""Engine tool-use tests — registry wiring + tool result append."""
from __future__ import annotations

import pytest

from mharo.core.engine import Engine
from mharo.core.router import Router
from mharo.providers.protocol import Provider, ProviderSettings
from mharo.providers.types import Completion, Usage
from mharo.tools import REGISTRY


class _Fake(Provider):
    def __init__(self):
        super().__init__(ProviderSettings(name="fake", model="m"))

    async def complete(self, messages, tools=None, max_tokens=None, temperature=None, **kw):
        from mharo.providers.errors import ProviderError
        for msg in messages:
            if msg.role == "tool":
                return Completion(provider=self.name, model=self.model,
                                  text=f"got-tool:{msg.content}",
                                  finish_reason="stop",
                                  usage=Usage(inp=10, out=5))
        raise ProviderError("no tool message seen")


@pytest.fixture
def engine():
    return Engine(Router([_Fake()]), model="fake", tools=dict(REGISTRY))


@pytest.mark.asyncio
async def test_call_tool_returns_and_appends(engine):
    out = await engine.call_tool("calculator", {"expression": "2*21"})
    assert out == "42.0"
    assert engine.history[-1].role == "tool"
    assert engine.history[-1].content == "42.0"


@pytest.mark.asyncio
async def test_unknown_tool_raises(engine):
    with pytest.raises(Exception):
        await engine.call_tool("nope", {})


@pytest.mark.asyncio
async def test_tool_result_visible_to_next_completion(engine):
    await engine.call_tool("calculator", {"expression": "6*7"})
    reply = await engine.respond("what did the tool say?")
    assert reply == "got-tool:42.0"


@pytest.mark.asyncio
async def test_schemas_exposed(engine):
    schemas = engine._tool_schemas()
    assert schemas and schemas[0]["function"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_engine_no_tools_no_schemas():
    e = Engine(Router([_Fake()]))
    assert e._tool_schemas() is None
    with pytest.raises(Exception):
        await e.call_tool("calculator", {})