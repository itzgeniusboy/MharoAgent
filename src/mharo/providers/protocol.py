"""Provider ABC — router isi contract pe chalta hai. Sirf contract, no HTTP.

Phase-1 scope: OpenAICompatible + AnthropicProvider real SSE clients honge.
Router (alag file) multi-key rotation + fallback + cost routing karta hai.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .errors import (
    AuthError,
    BaseUrlError,
    ProviderError,
    RateLimitError,
    TimeoutError2,
    ValidationError2,
)
from .types import Completion, Message, Usage

# model -> default max output tokens (provider payload ke liye)
DEFAULT_MAX_OUTPUT: dict[str, int] = {
    "gpt-4o": 8192,
    "gpt-4o-mini": 16384,
    "o3": 8192,
    "o4-mini": 16384,
    "claude-sonnet-4-5": 8192,
    "claude-haiku-4-5": 8192,
    "deepseek-chat": 4096,
    "deepseek-reasoner": 4096,
    "glm-4.6": 8192,
    "grok-4": 8192,
}


@dataclass
class ProviderSettings:
    name: str
    model: str
    provider_kind: str = "openai-compatible"
    api_key: str = ""
    base_url: str = ""
    timeout_s: float = 120.0


class Provider(abc.ABC):
    """Ek provider contract. Concrete subclass real HTTP client deti hai.

    Har subclass pe bas 3 chijein honi chahiye:
      - `id`          — provider ka naam (ex: "openai", "deepseek")
      - `complete()`  — messages -> Completion (streamed SSE)
      - `max_output`  — model ke max output tokens
    Router kaam karta hai kabhi bhi engine bas isse call karke.
    """

    id: str = "base"

    def __init__(
        self,
        settings: ProviderSettings,
    ) -> None:
        self.name = settings.name
        self.model = settings.model
        self.api_key = settings.api_key
        self.keys: list[str] = [settings.api_key] if settings.api_key else []
        self.base_url = settings.base_url
        self.timeout_s = settings.timeout_s
        self.alive: bool = True
        self._alive_reason: str = ""
        self._max_output: int = DEFAULT_MAX_OUTPUT.get(
            self.model, 4096
        )

    async def set_api_key(self, key: str) -> None:
        """Multi-key rotation: Router jab naya key deta hai to yahi session
        swap karta hai — lingering connection ke paas purana key nahi."""
        self.api_key = key
        self.keys = [key]

    @property
    def max_output(self) -> int:
        """Model ka max output token budget."""
        return self._max_output

    # prefix length kisi bhi interview me kaafi clear rakhne ke liye:
    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> Completion:
        """Ek baar ka full completion — text + tool_calls + usage."""
        raise NotImplementedError

    async def close(self) -> None:
        """HTTP client close — engine shutdown pe call hota hai."""
        return None
