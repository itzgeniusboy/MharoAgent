"""Streaming provider adapters.

All providers expose one async method::

    async for event in provider.stream(messages, tools):
        ...

Event shapes (plain dicts, so the TUI and the CLI share them):

    {"type": "text",         "text": "..."}         content delta
    {"type": "thinking",     "text": "..."}         reasoning delta
    {"type": "tool_call",    "tool": "bash", "args": {...}, "call_id": "..."}
    {"type": "usage",        "input_tokens": 0, "output_tokens": 0}
    {"type": "done",         "stop_reason": "end_turn"}
    {"type": "error",        "text": "..."}

Anything network-ish raises ProviderError with the response body excerpt, so a
typo in $MHARO_BASE_URL shows up as a readable block in the transcript instead
of a traceback.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator

import httpx

from .session import Message, Text, ToolCall, estimate_tokens


class ProviderError(Exception):
    pass


def _sse_lines(response: httpx.Response) -> Any:
    async def iter_():
        async for raw in response.aiter_lines():
            yield raw
    return iter_()


class Provider:
    """Base class: name + model + stream()."""

    name = "base"
    supports_tools = True

    def __init__(self, model: str, **opts: Any) -> None:
        self.model = model
        self.opts = opts
        self.temperature = float(opts.get("temperature", 0.2))
        self.max_tokens = int(opts.get("max_tokens", 4096))

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        raise NotImplementedError

    async def complete(self, prompt: str) -> str:
        """One-shot completion (used by /compact)."""
        chunks: list[str] = []
        msgs = [Message(role="user", blocks=[Text(prompt)])]
        async for evt in self.stream(msgs, tools=None):
            if evt.get("type") == "text":
                chunks.append(evt["text"])
            elif evt.get("type") == "error":
                raise ProviderError(evt["text"])
        return "".join(chunks).strip()

    def describe(self) -> str:
        return f"{self.name} · {self.model}"


# --------------------------------------------------------------------------- demo


SYSTEM_PERSONA = (
    "You are Mharo, a coding agent working inside the user's repository.\n"
    "Be concise. Prefer tools over guessing. Never invent file contents.\n"
    "Use markdown: short paragraphs, fenced code blocks for snippets.\n"
)

DEMO_MODELS = ["demo", "demo-gpt-4o-mini", "demo-fast"]


class DemoProvider(Provider):
    """Fully offline provider so the TUI is demoable with zero API keys.

    It is not a toy: it inspects the real transcript, issues real tool calls
    (list / read / search / bash), and then reports what came back.
    """

    name = "demo"

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        prompt = (last_user.text if last_user else "").strip()
        turn = [b for b in (messages[-1].blocks if messages else []) if isinstance(b, ToolCall)]
        already_ran = any(c.result for m in messages if m.role == "assistant" for c in m.blocks if isinstance(c, ToolCall))

        if turn and all(c.result is not None for c in turn):
            async for evt in self._report(turn, prompt):
                yield evt
            yield {"type": "done", "stop_reason": "end_turn"}
            return

        intent = self._intent(prompt)
        if intent and not already_ran:
            tool, args = intent
            await asyncio.sleep(0.05)
            yield {"type": "thinking", "text": f"inspecting the repo with `{tool}` first"}
            yield {"type": "tool_call", "tool": tool, "args": args, "call_id": f"demo_{id(messages):x}"}
            yield {"type": "done", "stop_reason": "tool_use"}
            return

        async for evt in self._answer(prompt):
            yield evt
        chars = sum(len(p) for p in _canned(prompt))
        yield {"type": "usage", "input_tokens": estimate_tokens("\n".join(m.text for m in messages)),
               "output_tokens": max(1, chars // 4)}
        yield {"type": "done", "stop_reason": "end_turn"}

    @staticmethod
    def _intent(prompt: str) -> tuple[str, dict[str, Any]] | None:
        low = (prompt or "").lower()
        if re.search(r"\b(ls|tree|structure|layout|files?|folder|repo)\b", low):
            return "list_dir", {"path": ".", "depth": 2}
        if re.search(r"\b(test|pytest|npm|build|typecheck|lint|run)\b", low):
            cmd = "npm test" if "npm" in low or "node" in low else "pytest -q"
            return "bash", {"command": cmd}
        if re.search(r"\b(where|find|grep|search|used|called)\b", low):
            word = next((w for w in re.findall(r"[A-Za-z_][\w.]{3,}", prompt) if w.lower() not in {"find", "where", "search", "used"}), "TODO")
            return "search", {"pattern": word}
        if re.search(r"\b(read|show|explain|open)\b", low):
            path = next((w for w in re.findall(r"[\w./-]+\.[A-Za-z]{1,5}", prompt)), "")
            if path:
                return "read_file", {"path": path}
        return None

    async def _report(self, calls: list[ToolCall], prompt: str) -> AsyncIterator[dict]:
        lines = []
        for call in calls:
            status = "ok" if call.ok else "failed"
            body = (call.result or "").strip()
            snippet = "\n".join(body.splitlines()[:14])
            lines.append(f"**`{call.tool}` → {status}**\n\n```\n{snippet or '(no output)'}\n```")
        text = "\n\n".join(lines) or "No tool output came back."
        followup = (
            "\n\n---\n\n*demo provider — set `MHARO_PROVIDER=openai` or "
            "`anthropic` (with a key in your env) and this becomes a real model loop; "
            "everything above was produced by the same pipeline the TUI uses.*"
        )
        for piece in _chunks(text + followup):
            await asyncio.sleep(0.012)
            yield {"type": "text", "text": piece}

    async def _answer(self, prompt: str) -> AsyncIterator[dict]:
        await asyncio.sleep(0.06)
        yield {"type": "thinking", "text": "planning: repo state is unknown to me until I read it"}
        for piece in _chunks("\n".join(_canned(prompt))):
            await asyncio.sleep(0.012)
            yield {"type": "text", "text": piece}


def _canned(prompt: str) -> list[str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return ["Type a request, or `/` for commands."]
    if prompt.startswith("!"):
        return [f"Run `{prompt[1:].strip()}` through the `bash` tool to see live output here."]
    if any(k in prompt.lower() for k in ("hello", "hi ", "who are you", "kaun")):
        return [
            "I'm the **Mharo agent** running in your terminal.",
            "",
            "- `/help` — commands, `/model` — switch provider, `/theme` — colours",
            "- `ctrl+p` — command palette, `ctrl+b` — sidebar, `ctrl+o` — fold/unfold tool output",
            "- Prefix a prompt with `!` to run a shell command straight away",
        ]
    if any(k in prompt.lower() for k in ("plan", "how should", "approach", "refactor")):
        return [
            f"Here's how I'd take `{prompt[:56]}{'…' if len(prompt) > 56 else ''}`:",
            "",
            "1. Read the code that owns this behaviour before touching it.",
            "2. Reproduce it — a failing test or a `bash` command beats guessing.",
            "3. Smallest change that fixes the cause, not the symptom.",
            "4. Re-run the check, then commit with the reason.",
            "",
            "Say *go* and I'll start at step 1 with real tool calls.",
        ]
    return [
        "Give me a moment with the repo before I answer that properly.",
        "",
        f"You asked: “{prompt[:180]}”",
        "",
        "Two things I can do right now, offline:",
        "",
        "| ask | what happens |",
        "|---|---|",
        "| `list the files` | `list_dir` tool call renders in the transcript |",
        "| `run pytest -q` | `bash` tool call, with an approval prompt |",
        "",
        "No key was needed to get here: `ma free` shows the free providers this agent runs on.",
        "For your own key: `ma keys add openai sk-…` (or `mharo --provider openai --model gpt-4o-mini`).",
    ]


def _chunks(text: str, size: int = 22) -> list[str]:
    out, buf = [], ""
    for word in text.split(" "):
        if len(buf) + len(word) + 1 > size and buf:
            out.append(buf + "\n" if word.endswith("\n") else buf + " ")
            buf = word
        else:
            buf = f"{buf} {word}".strip() if buf else word
        buf += " "
    if buf.strip():
        out.append(buf)
    return out or [""]


# --------------------------------------------------------------------------- openai


class OpenAICompatProvider(Provider):
    """Anything speaking the OpenAI /chat/completions SSE dialect.

    Covers OpenAI, Azure-compatible gateways, OpenRouter, Together, Groq,
    DeepSeek, vLLM, LM Studio and Ollama's OpenAI endpoint.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", **opts: Any) -> None:
        super().__init__(model, **opts)
        self.base_url = (opts.get("base_url") or os.environ.get("MHARO_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = opts.get("api_key") or os.environ.get("MHARO_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        # Public free tiers (OVH anonymous, Pollinations) need no key at all; `keyless`
        # says so instead of inventing an `Authorization: Bearer ` header.
        self.keyless = bool(opts.get("keyless")) or "localhost" in self.base_url or "127.0.0.1" in self.base_url
        self.transport = opts.get("transport")  # tests inject an httpx.MockTransport
        from .ladder import PROFILE_NAMES

        self.profile = str(opts.get("payload_profile") or "full")
        if self.profile not in PROFILE_NAMES:
            self.profile = "full"
        if not self.api_key and not self.keyless:
            raise ProviderError(
                "no API key. Set OPENAI_API_KEY (or MHARO_API_KEY), use "
                "--base-url http://localhost:11434/v1 for a local server, or "
                "--provider auto to ride the free ladder."
            )

    @property
    def drops(self) -> tuple[str, ...]:
        """Which payload keys this host must not receive."""
        from .ladder import PROFILE_STEPS

        return dict(PROFILE_STEPS).get(self.profile, ())

    def headers(self) -> dict[str, str]:
        accept = "application/json" if "stream" in self.drops else "text/event-stream"
        head: dict[str, str] = {"Content-Type": "application/json", "Accept": accept}
        if self.api_key:
            head["Authorization"] = f"Bearer {self.api_key}"
        return head

    def _payload(self, messages: list[Message], tools: list[dict] | None) -> dict[str, Any]:
        """Request body, shaped by what this host accepted last time.

        Public free tiers are usually a plain vLLM deployment: a normal completion
        works, while `stream_options` or `tools` draws a 400/422. Leaning the body
        out is what makes those rungs usable at all — so we learn it per host.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PERSONA}, *messages_to_chat(messages)],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        drops = set(self.drops)
        if "stream" not in drops:
            payload["stream"] = True
        if "stream_options" not in drops:
            payload["stream_options"] = {"include_usage": True}
        if tools and self.supports_tools and "tools" not in drops:
            payload["tools"] = [self._tool_spec(t) for t in tools]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
        """The OpenAI tool envelope: {"type":"function","function":{...}}.

        Our registry hands out the inner object; a flat `{"type":"function","name":…}`
        is what vLLM/OpenAI-compatible servers 422 on — and the 422 reads like "this
        free tier has no tool calling", which is a wrong conclusion to draw about
        somebody else's server.
        """
        inner = tool.get("function")
        if isinstance(inner, dict) and inner:
            return {"type": "function", "function": dict(inner)}
        body = {k: v for k, v in tool.items() if k in {"name", "description", "parameters", "strict"}}
        return {"type": "function", "function": body}

    async def _one_shot(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        """A non-streaming completion, emitted as the same event stream."""
        timeout = httpx.Timeout(120.0, read=float(self.opts.get("read_timeout", 600)))
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            res = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers())
        if res.status_code >= 400:
            raise ProviderError(f"HTTP {res.status_code} from {self.base_url}\n{res.text[:900]}")
        try:
            body = res.json()
        except ValueError:
            raise ProviderError(f"{self.name}: {self.base_url} replied {res.content[:180]!r}, not JSON") from None
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or msg.get("reasoning") or ""
        if text:
            yield {"type": "text", "text": str(text)}
        for index, call in enumerate(msg.get("tool_calls") or []):
            fn = call.get("function") or {}
            args = _safe_json(fn.get("arguments") or "{}")
            if args is None:
                yield {"type": "error", "text": f"malformed tool arguments for {fn.get('name', 'tool')}"}
                continue
            yield {"type": "tool_call", "tool": fn.get("name") or "", "args": args,
                   "call_id": call.get("id") or f"c{index}"}
        usage = body.get("usage") or {}
        if usage:
            yield {"type": "usage", "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                   "output_tokens": int(usage.get("completion_tokens", 0) or 0)}
        yield {"type": "done", "stop_reason": choice.get("finish_reason") or "end_turn"}

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        payload = self._payload(messages, tools)
        if "stream" not in payload:
            async for event in self._one_shot(payload):
                yield event
            return
        acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        produced = False  # a 200 carrying no events is a silent failure, not an answer
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, read=float(self.opts.get("read_timeout", 600))),
                transport=self.transport,
            ) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self.headers()) as res:
                    if res.status_code >= 400:
                        body = (await res.aread()).decode("utf-8", "replace")
                        raise ProviderError(f"HTTP {res.status_code} from {self.base_url}\n{body[:900]}")
                    async for line in res.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        produced = True
                        if chunk.get("usage"):
                            usage = {
                                "input_tokens": chunk["usage"].get("prompt_tokens", 0),
                                "output_tokens": chunk["usage"].get("completion_tokens", 0),
                            }
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                yield {"type": "thinking", "text": delta["reasoning_content"]}
                            if delta.get("content"):
                                yield {"type": "text", "text": delta["content"]}
                            for tc in delta.get("tool_calls") or []:
                                slot = acc.setdefault(int(tc.get("index", 0)), {"name": "", "args": "", "id": ""})
                                fn = tc.get("function") or {}
                                slot["name"] += fn.get("name") or ""
                                slot["args"] += fn.get("arguments") or ""
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                            if choice.get("finish_reason") == "tool_calls":
                                for slot in acc.values():
                                    args = _safe_json(slot["args"])
                                    if args is None:
                                        yield {"type": "error", "text": f"malformed tool arguments for {slot['name']}"}
                                        continue
                                    yield {"type": "tool_call", "tool": slot["name"], "args": args, "call_id": slot["id"] or f"c{len(acc)}"}
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc} — endpoint {self.base_url}") from exc
        if not produced:
            raise ProviderError(
                f"{self.name}: empty response from {self.base_url} "
                "(rate-limited free tier, or the model refused the request)"
            )
        if usage:
            yield {"type": "usage", **usage}
        yield {"type": "done", "stop_reason": "end_turn"}


def messages_to_chat(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            out.append({"role": "system", "content": msg.text})
        elif msg.role == "user":
            out.append({"role": "user", "content": msg.text or " "})
        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
            calls = [b for b in msg.blocks if isinstance(b, ToolCall) and b.result is not None]
            if calls:
                entry["tool_calls"] = [
                    {"id": c.call_id, "type": "function", "function": {"name": c.tool, "arguments": json.dumps(c.args)}}
                    for c in calls
                ]
            out.append(entry)
            for c in calls:
                out.append({"role": "tool", "tool_call_id": c.call_id, "content": (c.result or "")[:40_000]})
    return out


def _safe_json(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        try:  # models sometimes emit trailing commas
            val = json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))
        except json.JSONDecodeError:
            return None
    return val if isinstance(val, dict) else {"value": val}


# --------------------------------------------------------------------------- anthropic


class AnthropicProvider(Provider):
    """Anthropic Messages API with streaming blocks + native tool use."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", **opts: Any) -> None:
        super().__init__(model, **opts)
        self.base_url = (opts.get("base_url") or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.api_key = opts.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MHARO_API_KEY")
        if not self.api_key:
            raise ProviderError("no API key. Set ANTHROPIC_API_KEY, or use --provider openai.")

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "system": [{"type": "text", "text": SYSTEM_PERSONA}],
            "messages": _anthropic_msgs(messages),
        }
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        blocks: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=float(self.opts.get("read_timeout", 600)))) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/messages", json=payload,
                    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                ) as res:
                    if res.status_code >= 400:
                        raise ProviderError(f"HTTP {res.status_code}\n{(await res.aread()).decode('utf-8', 'replace')[:900]}")
                    async for line in res.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            evt = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        kind = evt.get("type")
                        if kind == "content_block_start":
                            block = evt.get("content_block") or {}
                            blocks[evt.get("index", 0)] = {"type": block.get("type"), "name": block.get("name"), "id": block.get("id"), "json": ""}
                        elif kind == "content_block_delta":
                            delta = evt.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                yield {"type": "text", "text": delta.get("text", "")}
                            elif delta.get("type") == "thinking_delta":
                                yield {"type": "thinking", "text": delta.get("thinking", "")}
                            elif delta.get("type") == "input_json_delta":
                                slot = blocks.setdefault(evt.get("index", 0), {"type": "tool_use", "json": ""})
                                slot["json"] += delta.get("partial_json", "")
                        elif kind == "content_block_stop":
                            slot = blocks.get(evt.get("index", 0)) or {}
                            if slot.get("type") == "tool_use":
                                yield {"type": "tool_call", "tool": slot.get("name", ""), "args": _safe_json(slot.get("json", "")) or {}, "call_id": slot.get("id", "")}
                        elif kind == "message_delta":
                            usage.update({k: v for k, v in (evt.get("usage") or {}).items() if isinstance(v, int)})
                        elif kind == "message_start":
                            u = (evt.get("message") or {}).get("usage") or {}
                            usage.update({"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)})
                        elif kind == "error":
                            raise ProviderError((evt.get("error") or {}).get("message", "stream error"))
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        if usage:
            yield {"type": "usage", "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
        yield {"type": "done", "stop_reason": "end_turn"}


def _anthropic_msgs(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "assistant":
            content = []
            for block in msg.blocks:
                if isinstance(block, Text) and block.text:
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCall):
                    content.append({"type": "tool_use", "id": block.call_id, "name": block.tool, "input": block.args})
            if content:
                out.append({"role": "assistant", "content": content})
        else:
            content = []
            for block in msg.blocks:
                if isinstance(block, Text) and block.text:
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCall):
                    content.append({"type": "tool_result", "tool_use_id": block.call_id, "content": block.result or "", "is_error": block.ok is False})
            if not content and msg.text:
                content = [{"type": "text", "text": msg.text}]
            if content:
                out.append({"role": "user", "content": content})
    if not out or out[-1]["role"] != "user":
        out.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
    return out


# --------------------------------------------------------------------------- factory


class AutoProvider(Provider):
    """Free-first transport: a ladder rung is picked per call, no key required.

    The TUI wants it to rotate on its own (one 429 must not kill a whole turn);
    the engine hub owns rotation through the key pool, so it passes `rotate=False`
    and binds rungs with :meth:`use`.
    """

    name = "auto"

    def __init__(self, model: str = "auto", **opts: Any) -> None:
        super().__init__(model, **opts)
        from .ladder import Ladder

        self.ladder = opts.get("ladder") if isinstance(opts.get("ladder"), Ladder) else Ladder()
        self.rotate = bool(opts.get("rotate", True))
        self.max_rungs = int(opts.get("max_rungs", 4))
        self.current = None
        self.last_rung = None      # what answered last, kept for describe()/status
        self.notices: list[str] = []
        self.extra_opts = {k: opts[k] for k in ("max_tokens", "temperature", "read_timeout", "transport")
                          if opts.get(k) is not None}

    @property
    def supports_tools(self) -> bool:  # type: ignore[override]
        rung = self.current or self.last_rung
        return bool(rung.tools) if rung is not None else False

    def describe(self) -> str:
        rung = self.current or self.last_rung
        return f"auto · {rung.label()}" if rung is not None else "auto · no rung picked yet"

    def use(self, rung: Any) -> "AutoProvider":
        """Bind a rung that something else (the engine's pool) selected."""
        self.current = rung
        if rung is not None:
            self.model = rung.model
            self.base_url = rung.base_url
        return self

    def _build(self, rung: Any) -> OpenAICompatProvider:
        return OpenAICompatProvider(model=rung.model, base_url=rung.base_url, api_key=rung.key(),
                                    keyless=rung.keyless, payload_profile=rung.profile,
                                    **self.extra_opts)

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        from .ladder import PROFILE_NAMES, quota_hint

        tried: set[str] = set()
        last = "no attempt made"
        waited = False
        rung = self.current
        # rung attempts, plus room to lean the payload out a few times
        for _ in range((self.max_rungs if self.rotate else 1) + len(PROFILE_NAMES)):
            if rung is None:
                rung = self.ladder.pick(skip=tried)
            if rung is None and self.rotate and not waited:
                # a free tier's gate is something to wait out, not to fail against
                patience = float(self.opts.get("patience_s", 20.0) or 0.0)
                soonest = self.ladder.pick(wait=True, skip=tried)
                gap = soonest.wait_s() if soonest is not None else 0.0
                if 0 < gap <= patience:
                    await asyncio.sleep(gap)
                    waited = True
                    rung = self.ladder.pick(skip=tried)
            if rung is None:
                raise ProviderError(
                    f"free providers exhausted — {self.ladder.soothe()}. {quota_hint(self.ladder)}"
                )
            self.current = rung
            provider = self._build(rung)
            carried = 0
            try:
                async for event in provider.stream(messages, tools if provider.supports_tools else None):
                    if event.get("type") in {"text", "tool_call", "thinking"}:
                        carried += 1
                    yield event
            except Exception as exc:  # noqa: BLE001 - classify, then fix / rotate / surface
                from .ladder import classify as _classify

                kind = _classify(exc)
                last = (str(exc).strip().splitlines() or [kind])[0][:140]
                if not carried and kind == "shape" and self.ladder.downgrade_profile(rung):
                    continue  # healthy host, wrong body: same rung, leaner payload
                self.ladder.mark_failure(rung, kind, error=str(exc)[:200])
                note = f"{rung.label()} → {kind}"
                if note not in self.notices:
                    self.notices.append(note)
                tried.add(rung.label())
                if kind in {"quota", "auth", "unreachable"}:
                    tried.add(rung.name)  # host-level trouble: skip the whole host
                self.current = None
                rung = None
                if carried or not self.rotate:
                    raise
                if self.ladder.pick(skip=tried) is None:
                    raise ProviderError(f"free providers exhausted — {last}. {quota_hint(self.ladder)}") from exc
                continue
            self.ladder.mark_used(rung)
            self.last_rung = rung
            if self.rotate:
                self.current = None  # the gate is per pairing: re-picking must respect it
            return


PROVIDERS = {"auto": AutoProvider, "demo": DemoProvider, "openai": OpenAICompatProvider,
             "anthropic": AnthropicProvider}


def get_provider(name: str | None = None, model: str | None = None, **opts: Any) -> Provider:
    """Resolve a provider by name, or sniff the environment when name is None."""
    name = (name or os.environ.get("MHARO_PROVIDER") or "").strip().lower()
    if not name:
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY") or os.environ.get("MHARO_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            name = "openai"
        else:
            # No key anywhere: ride the free ladder instead of a canned demo.
            name = "auto"
        if name == "demo" and not os.environ.get("MHARO_ALLOW_DEMO"):
            name = "auto"
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r} — pick from {', '.join(PROVIDERS)}")
    if name in {"auto", "free"} and not opts.get("ladder"):
        from .ladder import build_default_ladder

        opts["ladder"] = build_default_ladder()
    default_model = {"demo": "demo", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5",
                     "auto": "auto"}[name]
    return PROVIDERS[name](model or os.environ.get("MHARO_MODEL") or default_model, **opts)
