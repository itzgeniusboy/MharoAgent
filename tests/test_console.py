"""TUI/dashboard/appmode tests — Termux-safe console output."""
from __future__ import annotations

import io

import pytest

from mharo.dashboard import render as dashboard_render
from mharo.tui import banner, box, status_line
from mharo.appmode import run as app_run
from mharo.core.engine import Engine
from mharo.core.router import Router
from mharo.providers.local import LocalProvider
from mharo.providers.protocol import ProviderSettings
from mharo.providers.types import Completion, Usage
from mharo.providers.protocol import Provider


def _non_tty(stream_holder):
    return stream_holder


STREAM = io.StringIO()


def test_banner_plain_no_escape_when_not_tty():
    s = banner("Hi", "sub", stream=STREAM)
    assert "\x1b" not in s
    assert "Hi" in s and "sub" in s


def test_status_line_plain():
    class S:
        turns, tokens_in, tokens_out = 3, 10, 5
        latency_ms, fallbacks = 20.5, 1

    assert status_line(S(), stream=STREAM) == "[turns=3 | in=10 | out=5 | lat=20ms | fb=1]"


def test_box_borders():
    out = box("content", label="logs", stream=STREAM)
    assert out.startswith("--- logs ")
    assert "content" in out


def test_dashboard_shows_providers():
    router = Router([LocalProvider("demo", "m1")])
    engine = Engine(router)
    panel = dashboard_render(engine, stream=STREAM)
    assert "demo" in panel and "keys=" in panel and "cost" in panel


class _FakeProvider(Provider):
    def __init__(self):
        super().__init__(ProviderSettings(name="fake", model="m"))

    async def complete(self, messages, tools=None, max_tokens=None,
                       temperature=None, **kw):
        return Completion(provider=self.name, model=self.model, text="hi",
                          finish_reason="stop", usage=Usage(2, 3))


@pytest.mark.asyncio
async def test_appmode_single_shot():
    engine = Engine(Router([_FakeProvider()]))
    text = await app_run(engine, "hello")
    assert text == "hi"
    assert len(engine.history) == 2  # user + assistant