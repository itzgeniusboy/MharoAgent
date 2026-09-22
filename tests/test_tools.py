"""Tools tests — calculator safety + registry + schemas."""
from __future__ import annotations

import pytest

import mharo.tools as tools


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+3*4", 14.0),
        ("(2+3)*4", 20.0),
        ("10/4", 2.5),
        ("-5 + 2", -3.0),
        ("2**10", 1024.0),
    ],
)
def test_calculator(expr, expected):
    t = tools.REGISTRY["calculator"]
    assert t.fn(expression=expr) == expected


def test_calculator_rejects_code_injection():
    t = tools.REGISTRY["calculator"]
    with pytest.raises(Exception):
        t.fn(expression="__import__('os').system('true')")


def test_registry_has_echo_and_schema():
    assert "echo" in tools.registered()
    t = tools.REGISTRY["echo"]
    assert t.fn(text="hello") == "hello"
    s = t.schema()
    assert s["function"]["name"] == "echo"
    assert s["function"]["parameters"]["properties"]["text"]["type"] == "string"


def test_tool_run_async_wrap():
    import asyncio
    t = tools.REGISTRY["calculator"]
    assert asyncio.run(t.run(expression="6*7")) == 42.0


def test_add_tool_custom_registry():
    reg = {}
    t = tools.Tool(name="whoami", description="x", fn=lambda: "mharo")
    tools.add_tool(t, reg)
    assert "whoami" in reg and "whoami" not in tools.REGISTRY