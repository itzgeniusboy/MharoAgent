"""The agent loop — provider streaming, tool execution, approvals, cancellation.

`Agent` knows nothing about Textual. The UI supplies two async-capable
callbacks (`on_event`, `request_approval`), which is also how the headless
`--print` mode and the tests drive it.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .providers import SYSTEM_PERSONA, Provider, ProviderError, get_provider
from .session import ErrorNote, Message, Session, Text, ToolCall, estimate_tokens
from .tools import TOOLS, ToolContext, ToolSpec, repo_info, run_tool, tool_schemas

__all__ = ["Agent", "AgentConfig"]

EventHandler = Callable[[dict], Any]
Approver = Callable[[ToolCall], Awaitable[bool]]


@dataclass
class AgentConfig:
    provider: str | None = None
    model: str | None = None
    cwd: str | Path = "."
    auto_approve: bool = False
    max_turns: int = 12
    max_output_tokens: int = 4096
    temperature: float = 0.2
    system_prompt: str = SYSTEM_PERSONA
    tools_enabled: bool = True
    extra_tools: dict[str, ToolSpec] = field(default_factory=dict)
    provider_opts: dict[str, Any] = field(default_factory=dict)

    def resolved_provider(self) -> Provider:
        return get_provider(self.provider, self.model, **self.provider_opts)


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        on_event: EventHandler | None = None,
        request_approval: Approver | None = None,
        session: Session | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.on_event = on_event
        self.request_approval = request_approval
        self.provider = provider or self.config.resolved_provider()
        self.ctx = ToolContext(
            cwd=Path(self.config.cwd).resolve(),
            allow_outside=bool(Path(self.config.cwd).resolve().name != "/"),
        )
        self.session = session or Session(
            cwd=str(self.ctx.cwd), model=self.provider.model, provider=self.provider.name
        )
        self.session.model = self.provider.model
        self.session.provider = self.provider.name
        self._cancel = asyncio.Event()
        self.busy = False
        self.repo = repo_info(self.ctx.cwd)

    # ------------------------------------------------------------------ plumbing

    @property
    def system_message(self) -> Message:
        return Message(role="system", blocks=[Text(self.config.system_prompt)])

    def set_model(self, model: str) -> None:
        self.provider.model = model
        self.session.model = model

    def uses(self) -> dict[str, Any]:
        prompt_tokens = estimate_tokens(self.session.serialise_prompt())
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": self.session.usage.output_tokens,
            "calls": self.session.usage.calls,
            "cost": self.session.usage.cost(self.session.model),
        }

    def cancel(self) -> None:
        self._cancel.set()

    async def _emit(self, evt: dict) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(evt)
            if inspect.isawaitable(result):
                await result
        except Exception:  # a UI bug must not break the transcript
            pass

    # ------------------------------------------------------------------ turns

    async def submit(self, text: str) -> Message:
        """Run one user turn: append, stream, execute tools, repeat."""
        text = (text or "").strip()
        if not text:
            raise ValueError("empty prompt")
        if self.busy:
            raise RuntimeError("a turn is already running")
        self._cancel.clear()
        self.busy = True
        user = Message(role="user", blocks=[Text(text)])
        self.session.add(user)
        await self._emit({"type": "user_message", "message": user})
        try:
            await self._loop()
        finally:
            self.busy = False
            self.session.persist()
            await self._emit({"type": "turn_end", "usage": self.uses()})
        return user

    async def retry_last(self) -> None:
        """Drop the final assistant reply and re-run the last user prompt."""
        while self.session.messages and self.session.messages[-1].role != "user":
            self.session.messages.pop()
        if not self.session.messages:
            raise ValueError("nothing to retry")
        prompt = self.session.messages[-1].text
        self.session.messages.pop()
        await self.submit(prompt)

    async def _loop(self) -> None:
        tools = tool_schemas() if self.config.tools_enabled else []
        for turn in range(self.config.max_turns):
            if self._cancel.is_set():
                await self._emit({"type": "notice", "text": "interrupted"})
                return
            await self._emit({"type": "turn_start", "turn": turn + 1})
            assistant = await self._stream_once(tools)
            calls = [b for b in assistant.blocks if isinstance(b, ToolCall) and b.result is None]
            if not calls:
                return
            for call in calls:
                if self._cancel.is_set():
                    call.result, call.ok = "interrupted before running", False
                    await self._emit({"type": "tool_end", "call": call})
                    continue
                await self._run_call(call)
            self.session.persist()
        await self._emit({"type": "notice", "text": f"stopped after {self.config.max_turns} turns"})

    async def _stream_once(self, tools: list[dict]) -> Message:
        assistant = Message(role="assistant", model=self.provider.model)
        self.session.add(assistant)
        await self._emit({"type": "assistant_start", "message": assistant, "model": self.provider.model})
        try:
            async for evt in self.provider.stream([self.system_message, *self.session.messages], tools):
                if self._cancel.is_set():
                    break
                kind = evt.get("type")
                if kind == "text":
                    assistant.append_text(evt.get("text", ""))
                elif kind == "thinking":
                    assistant.append_thinking(evt.get("text", ""))
                elif kind == "tool_call":
                    call = ToolCall(
                        tool=evt.get("tool", "bash"),
                        args=evt.get("args") or {},
                        call_id=evt.get("call_id") or None or assistant._next_id(),  # type: ignore[attr-defined]
                    )
                    assistant.blocks.append(call)
                    await self._emit({"type": "tool_start", "call": call})
                    continue
                elif kind == "usage":
                    usage = {k: v for k, v in evt.items() if k != "type"}
                    assistant.usage = usage
                    self.session.usage.add(usage)
                    await self._emit({"type": "usage", "usage": usage})
                    continue
                elif kind == "error":
                    assistant.blocks.append(ErrorNote(evt.get("text", "provider error")))
                    await self._emit({"type": "error", "text": evt.get("text", "")})
                    continue
                await self._emit(evt)
        except ProviderError as exc:
            assistant.blocks.append(ErrorNote(str(exc)))
            await self._emit({"type": "error", "text": str(exc)})
        except asyncio.CancelledError:
            await self._emit({"type": "notice", "text": "cancelled mid-stream"})
            raise
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            assistant.blocks.append(ErrorNote(detail))
            await self._emit({"type": "error", "text": detail})
        return assistant

    def requires_approval(self, call: ToolCall) -> bool:
        """Only mutating tools gate on approval — `read_file` should never prompt."""
        table = dict(TOOLS)
        table.update(self.config.extra_tools)
        spec = table.get(call.tool)
        if spec is None:
            return True  # unknown tool: assume it can write
        return bool(spec.approval)

    async def _run_call(self, call: ToolCall) -> None:
        gated = self.requires_approval(call)
        approved = not gated or self.config.auto_approve
        if gated and not approved and self.request_approval is not None:
            await self._emit({"type": "approval_request", "call": call})
            try:
                verdict = self.request_approval(call)
                approved = await verdict if inspect.isawaitable(verdict) else bool(verdict)
            except Exception as exc:
                await self._emit({"type": "error", "text": f"approval failed: {exc}"})
                approved = False
        if not approved:
            call.result, call.ok = "User denied this action. Ask for a different approach.", False
            call.duration_ms = 0
            await self._emit({"type": "approval_resolved", "call": call, "approved": False})
            await self._emit({"type": "tool_end", "call": call})
            return
        await self._emit({"type": "approval_resolved", "call": call, "approved": True})
        await self._emit({"type": "tool_running", "call": call})
        result = await asyncio.to_thread(run_tool, call.tool, call.args, self.ctx, self.config.extra_tools)
        call.result, call.ok, call.duration_ms = result.output, result.ok, result.duration_ms
        if call.tool in {"write_file", "edit_file"}:
            # the tool layer records paths + line counts on the context; mirror
            # them onto the session so the sidebar and /session stay truthful.
            self.session.files_touched = dict(self.ctx.files_touched)
        await self._emit({"type": "tool_end", "call": call, "result": result})

    # ------------------------------------------------------------------ context

    async def compact(self, keep_last: int = 4) -> str:
        """Summarise older turns into one system note (keeps the thread alive)."""
        from .session import summarise

        msgs = self.session.messages
        if len(msgs) <= keep_last + 1:
            return "Nothing worth compacting yet."
        old, recent = msgs[:-keep_last], msgs[-keep_last:]
        summary = ""
        try:
            summary = await self.provider.complete(
                "Summarise this agent transcript into <=12 terse bullets that preserve "
                "file paths, commands and decisions:\n\n" + summarise(old, 6000)
            )
        except Exception:
            summary = summarise(old, 1600)
        note = Message(role="system", blocks=[Text(f"[compacted {len(old)} earlier messages]\n{summary}")])
        self.session.messages = [note, *recent]
        self.session.persist()
        return f"Compacted {len(old)} messages → {estimate_tokens(summary):,} tokens of context."

    def stats(self) -> dict[str, Any]:
        from .session import context_window

        prompt_tokens = estimate_tokens(self.session.serialise_prompt())
        window = context_window(self.session.model)
        return {
            "prompt_tokens": prompt_tokens,
            "window": window,
            "used_pct": min(100.0, round(prompt_tokens / window * 100, 1)),
            "completion_tokens": self.session.usage.output_tokens,
            "input_tokens": self.session.usage.input_tokens,
            "calls": self.session.usage.calls,
            "cost": self.session.usage.cost(self.session.model),
            "messages": len(self.session.messages),
            "branch": self.repo.get("branch", "no-git"),
            "dirty": self.repo.get("dirty", 0),
        }


# Small helper used above; attached here to keep Message untouched elsewhere.
def _next_id(self: Message) -> str:  # pragma: no cover - trivial
    import uuid

    return uuid.uuid4().hex[:12]


Message._next_id = _next_id  # type: ignore[attr-defined]
