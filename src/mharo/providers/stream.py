"""SSE stream -> events. Koi HTTP nahi, sirf parse. (streaming SSE events
merge hote hain text + tool delta + usage. Router ye normalizer use karta
hai har OpenAI-compatible provider ke liye.)"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass, field

from .types import Usage


@dataclass
class Event:
    kind: str  # text | tool_delta | tool_end | usage | finish
    delta: str = ""
    tool_id: str = ""
    tool_name: str = ""
    usage: Optional[Usage] = None
    finish: str = ""


class ToolBuffer:
    """Ek streaming tool-call ke fragments (id, name, args) merge karne ke liye."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.args = ""
        self.active = False

    def reset(self) -> None:
        self.id = self.name = self.args = ""
        self.active = False


def openai_events(obj: dict, buf: ToolBuffer) -> list[Event]:
    """Ek OpenAI-style chunk -> events list. `obj` = parsed `data:` JSON."""
    out: list[Event] = []
    try:
        ch = (obj.get("choices") or [None])[0] or {}
    except Exception:
        return out
    delta = ch.get("delta") or {}
    if delta.get("content"):
        out.append(Event("text", delta=delta["content"]))
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if tc.get("id"):
            buf.id = tc["id"]
        if fn.get("name"):
            buf.name = fn["name"]
        if fn.get("arguments"):
            buf.args += fn["arguments"]
        buf.active = True
        out.append(Event("tool_delta", tool_id=buf.id, tool_name=buf.name, delta=buf.args))
    if ch.get("finish_reason"):
        out.append(Event("finish", finish=ch["finish_reason"]))
    u = obj.get("usage")
    if u:
        out.append(Event("usage", usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))))
    return out


async def _sse(path) -> None:
    pass  # placeholder — real asyncio SSE loop core/loop.py me; kyun? stream.py
    # sirf parse-model rakhta hai, HTTP lifecycle router ke paas. (comment sandbox mein fix)


async def sse_lines(resp):
    """httpx stream response se SSE `data:` lines yield karta hai."""
    async for raw in resp.aiter_lines():
        line = raw.strip()
        if line and line.startswith("data:"):
            yield line[5:].strip()
