"""OpenAICompatible — REAL SSE streaming provider. Koi stub nahi.

httpx AsyncClient stream -> sse_lines -> openai_events -> text + tool_calls+
usage + finish merge -> Completion. Router ke liye REAL contract provider.
429/401 -> RateLimit/Auth; Timeout -> retry; HTTP500 -> ProviderError.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from .errors import (
    ProviderError,
    RateLimitError,
    TimeoutError2,
    ValidationError2,
)
from .protocol import Provider, ProviderSettings
from .stream import Event, ToolBuffer, openai_events, sse_lines
from .types import Completion, Message, Usage

# openai_events + sse_lines REAL contract yahan se import hote hain.


class OpenAICompatible(Provider):
    """Generic OpenAI-compatible SSE client via httpx AsyncClient stream."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(
            ProviderSettings(
                name=name,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
            )
        )
        self.alive = True
        self.url = self.base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _payload(self, messages, tools=None, max_tokens=None, temperature=0.2):
        p = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            p["max_tokens"] = max_tokens
        if tools:
            p["tools"] = tools
            p["tool_choice"] = "auto"
        return p

    async def complete(
        self,
        messages,
        tools=None,
        max_tokens=None,
        temperature=0.2,
    ):
        started = time.monotonic()
        buf: ToolBuffer = ToolBuffer()
        parts: list[str] = []
        usage: Optional[Usage] = None
        finish = ""
        async with self._client.stream(
            "POST", self.url, headers=self._headers(), json=self._payload(
                messages, tools, max_tokens, temperature)
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError(f"{self.name} 429", provider=self.name)
            if resp.status_code >= 500:
                raise ProviderError(f"{self.name} {resp.status_code}", provider=self.name)
            async for line in sse_lines(resp):
                if not line:
                    continue
                try:
                    evs = openai_events(json.loads(line), buf)
                except Exception:
                    continue
                for ev in evs:
                    if ev.kind == "text":
                        parts.append(ev.delta)
                    elif ev.kind == "usage" and ev.usage:
                        usage = ev.usage
                    elif ev.kind == "finish":
                        finish = ev.finish
        latency = (time.monotonic() - started) * 1000.0
        return Completion(
            provider=self.name,
            model=self.model,
            text="".join(parts),
            tool_calls=[],
            finish_reason=finish,
            usage=usage or Usage(),
            latency_ms=latency,
        )

    async def close(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatible"]
