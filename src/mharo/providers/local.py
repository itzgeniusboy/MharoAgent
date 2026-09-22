"""LocalProvider — zero-key demo provider.

Har user ke liye CLI chalu rahe (kahin se bhi, bina API key ke). Deterministic
echo-ish reply with prompt frame. Router/Engine contract REAL — isliye CI me
bhi use hota hai (probing/fallback). SSE nahi, ek static completion.
"""
from __future__ import annotations

from typing import Optional

from .protocol import Provider, ProviderSettings
from .types import Completion, Usage


class LocalProvider(Provider):
    """Bina network. Kadam: messages ko simple reply banate hain."""

    def __init__(self, name: str = "local", model: str = "mharo-local") -> None:
        super().__init__(ProviderSettings(name=name, model=model, api_key="local"))

    async def complete(
        self,
        messages,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        last = next(
            (m for m in reversed(messages) if m.role in {"user", "tool"}),
            None,
        )
        text = ""
        if last:
            payload = str(last.content).strip()
            text = (
                f"(local demo — set OPENAI_API_KEY for real AI)\n"
                f"You said: {payload}\n"
                f"[router sees you: messages={len(messages)}]"
            )
        if not text:
            text = "Namaste from mharo-local. Add an API key to chat for real."
        return Completion(
            provider=self.name,
            model=self.model,
            text=text,
            finish_reason="stop",
            usage=Usage(inp=sum(len(str(m.content or "")) for m in messages),
                        out=len(text)),
        )


__all__ = ["LocalProvider"]