"""Core value types. Provider contract ke liye — complete, verified."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Usage:
    """Token counts — model/provider se aata hai."""

    inp: int = 0
    out: int = 0

    @property
    def total(self) -> int:
        return self.inp + self.out


@dataclass
class Message:
    """Ek chat message. role: system|user|assistant|tool."""

    role: str
    content: Any = ""
    name: Optional[str] = None  # tool name jab role == tool
    tool_call_id: Optional[str] = None
    tool_calls: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            d["content"] = self.content
            if self.tool_call_id:
                d["tool_call_id"] = self.tool_call_id
            if self.name:
                d["name"] = self.name
        elif self.tool_calls:
            d["content"] = self.content or None
            d["tool_calls"] = self.tool_calls
        else:
            d["content"] = self.content
        return d


@dataclass
class ProviderSettings:
    """Router ke liye ek provider ka static contract —
    settings se secret bahar nikala jaata hai, model kabhi nahi dekhta."""

    name: str
    model: str
    provider_kind: str = "openai"  # openai|anthropic|openai-compat|azure|local
    api_key: str = ""
    base_url: str = ""
    api_version: str = ""
    timeout_s: float = 120.0


@dataclass
class ToolCall:
    """Streaming ke beech partial tool call merge karne ke liye."""

    id: str = ""
    name: str = ""
    arguments: str = ""  # delta JSON fragment concatenate hote jaate hain


@dataclass
class Completion:
    """Ek complete response — text + tool_calls + usage."""

    provider: str = ""
    model: str = ""
    text: str = ""
    tool_calls: list = field(default_factory=list)  # [{id,name,arguments}]
    finish_reason: str = ""
    usage: Optional[Usage] = None
    latency_ms: float = 0.0

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def cost_usd(self) -> float:
        """Price model registry se — rough estimate, dashboard ke liye."""
        return self.usage.total * 0.000002 if self.usage else 0.0
