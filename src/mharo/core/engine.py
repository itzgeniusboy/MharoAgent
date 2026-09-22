"""Engine — REAL session turn loop. Koi stub/stats-drama nahi.

Har turn: user msg -> history append -> Router.complete(messages via window)
-> assistant msg return. SessionStats REAL counters. Bas Router se hi baat,
providers REAL SSE hain. Koi fake result path nahi.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..providers.errors import ProviderError
from ..providers.types import Completion, Message, Usage


@dataclass
class SessionStats:
    """Engine ke REAL per-session counters."""

    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    fallbacks: int = 0
    errors: list[str] = field(default_factory=list)


class Engine:
    """Conversation loop — history window + router call. Router REAL provider pool."""

    def __init__(self, router, system: str = "", history_limit: int = 40,
                 model: str = "openai") -> None:
        self.router = router
        self.system = system
        self.history_limit = history_limit
        self.model = model
        self.history: list[Message] = []
        self.stats = SessionStats()

    def _window(self, messages: list[Message]) -> list[Message]:
        """System prompt + recent history_limit messages (context budget)."""
        base = [Message(role="system", content=self.system)] if self.system else []
        return base + messages[-self.history_limit:]

    def window(self, messages: list[Message], limit: Optional[int] = None) -> list[Message]:
        """Public window API — context window with optional limit."""
        if limit is None:
            limit = self.history_limit
        base = [Message(role="system", content=self.system)] if self.system else []
        return base + messages[-limit:]

    async def respond(self, user_text: str, temperature: float = 0.2) -> str:
        """Ek user turn -> Router.complete -> assistant text. REAL async path."""
        started = time.monotonic()
        self.history.append(Message(role="user", content=user_text))
        try:
            comp = await self.router.complete(
                self._window(self.history), temperature=temperature
            )
        except ProviderError as exc:
            self.stats.errors.append(str(exc))
            self.stats.latency_ms = (time.monotonic() - started) * 1000.0
            raise
        self.history.append(Message(role="assistant", content=comp.text))
        self.stats.turns += 1
        if comp.usage:
            self.stats.tokens_in += comp.usage.inp
            self.stats.tokens_out += comp.usage.out
        self.stats.latency_ms = (time.monotonic() - started) * 1000.0
        return comp.text
