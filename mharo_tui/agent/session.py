"""Message model, token/cost accounting and on-disk session persistence.

Kept UI-free so the same core drives both the TUI and `mharo --print`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- content


@dataclass
class Text:
    text: str = ""

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class Thinking:
    text: str = ""

    def to_dict(self) -> dict:
        return {"type": "thinking", "text": self.text}


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    ok: bool | None = None
    duration_ms: int | None = None
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "type": "tool_use",
            "tool": self.tool,
            "args": self.args,
            "call_id": self.call_id,
            "result": self.result,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ErrorNote:
    text: str

    def to_dict(self) -> dict:
        return {"type": "error", "text": self.text}


Block = Text | Thinking | ToolCall | ErrorNote


@dataclass
class Message:
    role: str  # user | assistant | system | tool
    blocks: list[Block] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------
    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, Text))

    def append_text(self, chunk: str) -> None:
        if self.blocks and isinstance(self.blocks[-1], Text):
            self.blocks[-1].text += chunk
        else:
            self.blocks.append(Text(chunk))

    def append_thinking(self, chunk: str) -> None:
        if self.blocks and isinstance(self.blocks[-1], Thinking):
            self.blocks[-1].text += chunk
        else:
            self.blocks.append(Thinking(chunk))

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "created_at": self.created_at,
            "model": self.model,
            "usage": self.usage,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        blocks: list[Block] = []
        for raw in data.get("blocks", []):
            kind = raw.get("type")
            if kind == "text":
                blocks.append(Text(raw.get("text", "")))
            elif kind == "thinking":
                blocks.append(Thinking(raw.get("text", "")))
            elif kind == "error":
                blocks.append(ErrorNote(raw.get("text", "")))
            elif kind == "tool_use":
                blocks.append(
                    ToolCall(
                        tool=raw.get("tool", "unknown"),
                        args=raw.get("args", {}) or {},
                        result=raw.get("result"),
                        ok=raw.get("ok"),
                        duration_ms=raw.get("duration_ms"),
                        call_id=raw.get("call_id") or uuid.uuid4().hex[:12],
                    )
                )
        msg = cls(role=data.get("role", "assistant"), blocks=blocks)
        msg.created_at = data.get("created_at", msg.created_at)
        msg.model = data.get("model")
        msg.usage = data.get("usage", {}) or {}
        return msg


# --------------------------------------------------------------------------- usage


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token + word tax).

    Deliberately not a real BPE tokenizer: a TUI must never block on a
    tokenizer import just to paint a status bar.
    """
    if not text:
        return 0
    chars = len(text)
    words = len(text.split())
    return max(1, int(chars / 4 + words * 0.35))


MODEL_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_000_000,
    "o4-mini": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-3-5-haiku": 200_000,
    "qwen2.5-coder:32b": 32_768,
    "llama3.1:8b": 128_000,
    "deepseek-chat": 64_000,
    "demo": 32_000,
}

# USD per million tokens: (input, output). Only used for the cost readout.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "deepseek-chat": (0.27, 1.10),
    "demo": (0.0, 0.0),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, usage: dict[str, Any]) -> None:
        self.input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        self.output_tokens += int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        self.cache_read += int(usage.get("cache_read_input_tokens", 0) or 0)
        self.calls += 1

    def cost(self, model: str) -> float:
        pin, pout = PRICE_TABLE.get(model, (0.0, 0.0))
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


def context_window(model: str) -> int:
    for key, window in MODEL_WINDOWS.items():
        if model.startswith(key.split(":")[0]):
            return window
    return 128_000


# --------------------------------------------------------------------------- session


def sessions_dir() -> Path:
    base = os.environ.get("MHARO_HOME") or str(Path.home() / ".mharo")
    path = Path(base) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Session:
    cwd: str
    model: str
    provider: str
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    title: str = ""
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    todos: list[dict[str, Any]] = field(default_factory=list)
    files_touched: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    path_override: Path | None = None

    # -- transcript ---------------------------------------------------------
    def add(self, message: Message) -> Message:
        self.messages.append(message)
        if message.role == "user" and not self.title:
            self.title = _first_line(message.text)[:70]
        return message

    def last_assistant(self) -> Message | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    @property
    def prompt_tokens(self) -> int:
        return estimate_tokens(self.serialise_prompt())

    def serialise_prompt(self) -> str:
        out: list[str] = []
        for msg in self.messages:
            if isinstance(msg.blocks[0], ErrorNote) if msg.blocks else False:
                continue
            for block in msg.blocks:
                if isinstance(block, Text):
                    out.append(block.text)
                elif isinstance(block, ToolCall):
                    out.append(json.dumps(block.args, ensure_ascii=False))
                    if block.result:
                        out.append(block.result)
        return "\n".join(out)

    def to_openai(self) -> list[dict[str, Any]]:
        """Flatten the transcript into OpenAI chat messages (+ tool results)."""
        out: list[dict[str, Any]] = []
        for msg in self.messages:
            if msg.role == "user":
                out.append({"role": "user", "content": msg.text or " "})
            elif msg.role == "system":
                out.append({"role": "system", "content": msg.text})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
                calls = [b for b in msg.blocks if isinstance(b, ToolCall)]
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.call_id,
                            "type": "function",
                            "function": {
                                "name": c.tool,
                                "arguments": json.dumps(c.args, ensure_ascii=False),
                            },
                        }
                        for c in calls
                    ]
                out.append(entry)
                for call in calls:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.call_id,
                            "content": (call.result or "")[:40_000],
                        }
                    )
        return out

    def to_anthropic(self) -> list[dict[str, Any]]:
        """Flatten into Anthropic `messages` content-block form."""
        out: list[dict[str, Any]] = []
        for msg in self.messages:
            if msg.role in {"user", "tool"}:
                content: list[dict[str, Any]] = []
                for block in msg.blocks:
                    if isinstance(block, Text) and block.text:
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolCall):
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.call_id,
                                "content": (block.result or "")[:40_000],
                                "is_error": block.ok is False,
                            }
                        )
                if content:
                    out.append({"role": "user", "content": content})
            elif msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                for block in msg.blocks:
                    if isinstance(block, Text) and block.text:
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolCall):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.call_id,
                                "name": block.tool,
                                "input": block.args,
                            }
                        )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        return out

    # -- persistence --------------------------------------------------------
    @property
    def path(self) -> Path:
        """Where `persist()` writes. A resumed session keeps its original file."""
        return self.path_override or sessions_dir() / f"{self.id}.jsonl"

    def persist(self) -> Path:
        payload = {
            "id": self.id,
            "cwd": self.cwd,
            "model": self.model,
            "provider": self.provider,
            "title": self.title,
            "created_at": self.created_at,
            "todos": self.todos,
            "files_touched": self.files_touched,
            "usage": self.usage.__dict__,
        }
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": payload}, ensure_ascii=False) + "\n")
            for msg in self.messages:
                fh.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        path = Path(path)
        session = cls(cwd=".", model="?", provider="demo")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "meta" in data:
                    meta = data["meta"]
                    session = cls(
                        cwd=meta.get("cwd", "."),
                        model=meta.get("model", "?"),
                        provider=meta.get("provider", "demo"),
                        title=meta.get("title", ""),
                        todos=meta.get("todos", []),
                        files_touched=meta.get("files_touched", {}),
                    )
                    session.id = meta.get("id", session.id)
                    usage = meta.get("usage", {})
                    session.usage = Usage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_read=usage.get("cache_read", 0),
                        calls=usage.get("calls", 0),
                    )
                else:
                    session.messages.append(Message.from_dict(data))
        session.path_override = path
        return session

    @staticmethod
    def recent(limit: int = 20) -> list[Path]:
        try:
            files = sorted(
                sessions_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            return []
        return files[:limit]


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return "new session"


def summarise(messages: Iterable[Message], max_chars: int = 2000) -> str:
    """Naive compaction used by /compact when no provider is configured."""
    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"- user asked: {_first_line(msg.text)[:160]}")
        elif msg.role == "assistant":
            for block in msg.blocks:
                if isinstance(block, Text) and block.text.strip():
                    lines.append(f"- assistant: {_first_line(block.text)[:160]}")
                elif isinstance(block, ToolCall):
                    lines.append(f"- ran {block.tool} -> {'ok' if block.ok else 'fail'}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text
