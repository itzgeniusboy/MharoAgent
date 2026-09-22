"""LIVE network SSE test — real httpx over a real local SSE server.

OpenAICompatible -> httpx stream -> sse_lines -> openai_events -> Completion.
Yahi path production OpenAI/Deepseek ke saath chalta hai; yahan 127.0.0.1
self-hosted fake OpenAI server (threaded) use hota hai. No API key needed.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mharo.providers.openai_compat import OpenAICompatible
from mharo.providers.types import Message
from mharo.core.router import Router
from mharo.core.engine import Engine
from mharo.providers.local import LocalProvider
from mharo.providers.errors import RateLimitError


class _SSEHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        # Authorization required — regression: kabhi header chhut jaata tha
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"missing auth"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        # rate-limit-me model bhejne par ek 429 milta hai (fallback test ke liye)
        if b"rate-limit-me" in body:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"rate limit"}')
            return
        chunks = [
            {"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "c2", "choices": [{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}]},
            {"id": "c3", "choices": [{"index": 0, "delta": {"content": "there"}, "finish_reason": None}]},
            {"id": "c4", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "c5", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()
        self.wfile.close()

    def do_GET(self):
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def sse_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    thread.join(timeout=3)


@pytest.mark.asyncio
async def test_live_sse_completion_bytes_over_http(sse_server):
    provider = OpenAICompatible("fake-openai", "gpt-4o-mini", "sk-test",
                                base_url=sse_server, timeout_s=10)
    comp = await provider.complete(
        [Message(role="user", content="hi")], temperature=0.2
    )
    assert comp.text == "hello there"
    assert comp.finish_reason == "stop"
    assert comp.usage is not None and comp.usage.inp == 8 and comp.usage.out == 3
    assert comp.model == "gpt-4o-mini"
    await provider.close()


@pytest.mark.asyncio
async def test_live_sse_requires_auth_header(sse_server):
    """Sanity: bina Authorization ke local server bhi 401 manta hai."""
    provider = OpenAICompatible("noauth", "gpt-4o-mini", "",
                                base_url=sse_server, timeout_s=10)
    from mharo.providers.errors import ProviderError
    with pytest.raises(ProviderError):
        await provider.complete([Message(role="user", content="hi")])
    await provider.close()


@pytest.mark.asyncio
async def test_live_router_fallback_on_429(sse_server):
    """Live 429 -> Router local fallback -> reply milta hai."""
    broken = OpenAICompatible("upstream", "rate-limit-me", "sk-test",
                              base_url=sse_server, timeout_s=10)
    local = LocalProvider("local")
    router = Router([broken, local])
    engine = Engine(router)
    text = await engine.respond("ping")
    assert text.startswith("(local demo")
    assert router.stats.fallbacks == 1
    assert router.stats.last_provider == "local"
    assert engine.stats.fallbacks == 1  # engine counter router se sync
    await broken.close()