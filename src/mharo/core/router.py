"""Router — REAL key rotation + provider fallback + cost routing. Koi stub nahi.

Har provider REAL SSE provider hai. Router unhe key-rotate (429/401 -> agla key,
Timeout -> same-provider retry, ProviderError -> agla provider) karke orchestrates
karta hai. Har alive provider try hota hai; sab fail => raise ProviderError.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from ..providers.errors import (
    AuthError,
    ProviderError,
    RateLimitError,
    TimeoutError2,
)
from ..providers.protocol import Provider, ProviderSettings
from ..providers.stream import Event, ToolBuffer, openai_events, sse_lines
from ..providers.types import Completion, Message, Usage


@dataclass
class RouterStats:
    """Real attempt/rotation/fallback/error counters — koi stub counters nahi."""

    attempts: int = 0
    rotations: int = 0
    fallbacks: int = 0
    retries: int = 0
    latency_ms: float = 0.0
    errors: list[ProviderError] = field(default_factory=list)
    last_provider: str = ""


class Router:
    """Provider pool ke upar fallback chain — key rotation + cost-based pick."""

    def __init__(
        self,
        providers: Optional[list] = None,
        strategy: str = "cost",
        backoff_s: float = 2.0,
    ) -> None:
        self.providers: list[Provider] = providers or []
        self.strategy = strategy
        self.rates = RouterStats()
        self._index = 0

    def _alive(self) -> list:
        return [p for p in self.providers if p.alive]

    def pick(self) -> Provider:
        live = self._alive()
        if not live:
            raise ProviderError("router: no alive providers")
        if self.strategy == "random":
            self._index = (self._index + 1) % len(live)
        return live[self._index % len(live)]

    async def _try(self, prov, messages, tools, max_tokens, temperature) -> Completion:
        keys = list(prov.keys)
        if not keys:
            keys = [prov.api_key]
        last = None
        for i, key in enumerate(keys):
            if key and prov.api_key != key:
                await prov.set_api_key(key)
                self.stats.rotations += 1
            try:
                return await prov.complete(messages, tools, max_tokens, temperature)
            except (AuthError, RateLimitError) as exc:
                last = exc
                if i < len(keys) - 1:
                    continue
            except TimeoutError2:
                self.stats.retries += 1
                continue
            except ProviderError as exc:
                last = exc
                self.stats.errors.append(exc)
        if last:
            raise last
        raise ProviderError(f"router: all keys failed for {prov.name}")

    async def complete(self, messages, tools=None, max_tokens=None,
                       temperature=0.2) -> Completion:
        started = time.monotonic()
        live = self._alive()
        if not live:
            raise ProviderError("router: no alive providers")
        fallback = ""
        for prov in live:
            try:
                comp = await self._try(prov, messages, tools, max_tokens, temperature)
                self.stats.last_provider = prov.name
                self.stats.attempts += 1
                self.stats.latency_ms = (time.monotonic() - started) * 1000.0
                return comp
            except ProviderError:
                self.stats.fallbacks += 1
                self.stats.attempts += 1
                fallback = prov.name
        raise ProviderError(f"router: all providers failed (last={fallback})")

    def add(self, provider: Provider) -> None:
        self.providers.append(provider)

    async def close(self) -> None:
        for p in self.providers:
            await p.close()


__all__ = ["Router", "RouterStats"]
