"""P1-4 · SubAgent army: specialists with *isolated* tool sets.

Each subagent is an `Engine`-driven mini-run: own message list, own tool
allowlist (a subset of the registry), own tier, own budget share, and it can
only report back through a structured result — it cannot touch the parent's
transcript or approve its own permissions.

Isolation is enforced, not decorative: `run_subagent` builds a fresh
`ToolContext` and filters the registry, so a `reader` literally cannot call
`bash`, and `tester` cannot write files. Parallel dispatch is bounded so a
phone CPU is not murdered by six headless compilers.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from mharo_tui.agent.tools import TOOLS, ToolContext, ToolSpec, run_tool

JSON_BLOCK = re.compile(r"\{.*\}", re.S)

ROLE_PROMPTS: dict[str, str] = {
    "reader": (
        "You are a READER subagent. Your job: locate and quote the exact code that matters. "
        "Never edit anything. Answer with a JSON object: "
        '{"summary": str, "files": [{"path": str, "why": str, "lines": [int, int]}], "open_questions": [str]}'
    ),
    "tester": (
        "You are a TESTER subagent. Run the project's checks, read the failure output, and report "
        "the root cause precisely. You may run commands but never modify files. Answer with JSON: "
        '{"passed": bool, "command": str, "root_cause": str, "evidence": str, "suspect_files": [str]}'
    ),
    "patcher": (
        "You are a PATCHER subagent. Apply the smallest correct change with edit_file/write_file, "
        "then re-read what you changed to confirm. Answer with JSON: "
        '{"changed": [str], "rationale": str, "residual_risk": str}'
    ),
    "reviewer": (
        "You are a REVIEWER subagent. Be adversarial: find what breaks, what is untested, what is "
        "leftover. Do not edit. Answer with JSON: "
        '{"issues": [{"file": str, "issue": str, "severity": "blocker|major|minor"}], "verdict": "ship|fix-first"}'
    ),
}


@dataclass
class SubAgentResult:
    role: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    tool_calls: int = 0
    error: str = ""

    def render(self) -> str:
        if self.error:
            return f"[{self.role}] error: {self.error}"
        head = f"[{self.role}]"
        if self.data:
            return head + " " + json.dumps(self.data, ensure_ascii=False)[:900]
        return f"{head} {self.text[:900]}"


@dataclass
class SubAgent:
    role: str
    tools: list[str]
    tier: str = "cheap"
    max_turns: int = 4
    prompt: str = ""
    writes: bool = False

    def registry(self) -> dict[str, ToolSpec]:
        return {name: TOOLS[name] for name in self.tools if name in TOOLS}


def build_roots(config: dict[str, Any] | None = None) -> dict[str, SubAgent]:
    cfg = config or {}
    out: dict[str, SubAgent] = {}
    for role, spec in cfg.items():
        tools = list(spec.get("tools") or [])
        out[role] = SubAgent(
            role=role,
            tools=tools,
            tier=str(spec.get("tier", "cheap")),
            max_turns=int(spec.get("max_turns", 4)),
            prompt=ROLE_PROMPTS.get(role, f"You are the {role} subagent."),
            writes=any(t in {"write_file", "edit_file"} for t in tools),
        )
    if not out:
        out = {role: SubAgent(role=role, tools=TOOLS_ORDER[role], tier="cheap",
                              prompt=ROLE_PROMPTS[role], writes=role == "patcher")
               for role in TOOLS_ORDER}
    return out


TOOLS_ORDER: dict[str, list[str]] = {
    "reader": ["read_file", "list_dir", "search"],
    "tester": ["bash", "read_file"],
    "patcher": ["read_file", "edit_file", "write_file", "search"],
    "reviewer": ["read_file", "search", "list_dir"],
}


class SubAgentRunner:
    """Runs roles through the same provider hub + tool layer, with isolation."""

    def __init__(
        self,
        hub: Any,
        *,
        cwd: Any,
        permissions: Any,
        vault: Any = None,
        roles: dict[str, SubAgent] | None = None,
        max_parallel: int = 3,
        on_event: Callable[[dict], Any] | None = None,
        session_db: Any = None,
        session_id: int | None = None,
    ) -> None:
        self.hub = hub
        self.cwd = cwd
        self.permissions = permissions
        self.vault = vault
        self.roles = roles or build_roots()
        self.sem = asyncio.Semaphore(max(1, max_parallel))
        self.on_event = on_event
        self.db = session_db
        self.session_id = session_id

    def available(self) -> list[str]:
        return sorted(self.roles)

    async def run(self, role: str, task: str, *, context: str = "") -> SubAgentResult:
        agent = self.roles.get(role)
        if agent is None:
            return SubAgentResult(role, False, error=f"unknown subagent {role!r} (have: {', '.join(self.available())})")
        async with self.sem:
            return await self._run(agent, task, context)

    async def run_many(self, jobs: list[tuple[str, str]], *, context: str = "") -> list[SubAgentResult]:
        """Parallel dispatch, results in submission order."""
        tasks = [asyncio.create_task(self.run(role, task, context=context)) for role, task in jobs]
        return list(await asyncio.gather(*tasks))

    async def _run(self, agent: SubAgent, task: str, context: str) -> SubAgentResult:
        from mharo_tui.agent.session import Message, Text as TextBlock, ToolCall
        from mharo_tui.agent.providers import SYSTEM_PERSONA

        registry = agent.registry()
        if not registry:
            return SubAgentResult(agent.role, False, error="no tools in allowlist")
        ctx = ToolContext(cwd=self.cwd, allow_outside=False, timeout=90.0)
        if self.vault is not None:
            ctx.env_extra = self.vault.env_for([])
        messages = [
            Message(role="system", blocks=[TextBlock(f"{SYSTEM_PERSONA}\n\n{agent.prompt}")]),
            Message(role="user", blocks=[TextBlock((context + "\n\n" if context else "") + task)]),
        ]
        tools = [spec.schema() for spec in registry.values()]
        calls = 0
        final_text = ""
        errors: list[str] = []
        for _turn in range(agent.max_turns):
            assistant = Message(role="assistant", blocks=[])
            events_text: list[str] = []
            pending: list[ToolCall] = []
            async for evt in self.hub.stream(agent.tier, messages, tools):
                kind = evt.get("type")
                if kind == "text":
                    events_text.append(evt.get("text", ""))
                    assistant.append_text(evt.get("text", ""))
                elif kind == "tool_call":
                    call = ToolCall(tool=evt.get("tool", ""), args=evt.get("args") or {})
                    if call.tool not in registry:
                        call.result, call.ok = f"refused: {call.tool} is not allowed for subagent {agent.role}", False
                        errors.append(f"isolated-tools violation: {agent.role} tried {call.tool}")
                    else:
                        pending.append(call)
                    assistant.blocks.append(call)
                elif kind == "done":
                    break
            final_text = "".join(events_text).strip() or final_text
            messages.append(assistant)
            if not pending:
                break
            for call in pending:
                calls += 1
                decision = self.permissions.decide(call.tool, call.args)
                if decision.blocked or decision.action == "ask":
                    # subagents never pop interactive prompts; anything needing
                    # approval is denied and reported to the parent instead.
                    call.result = f"denied for subagent ({decision.action}: {decision.rule or decision.reason})"
                    call.ok = False
                else:
                    result = await asyncio.to_thread(run_tool, call.tool, call.args, ctx)
                    call.result, call.ok, call.duration_ms = result.output, result.ok, result.duration_ms
                    if self.db is not None and self.session_id:
                        self.db.add_tool_call(
                            self.session_id, call.tool, call.args, ok=call.ok,
                            duration_ms=call.duration_ms, output=call.result or "",
                            permission=decision.action, step=-1,
                        )
                if call.result:
                    messages.append(Message(role="user", blocks=[TextBlock(f"{call.tool} result:\n{(call.result or '')[:3000]}")]))
        data = _extract_json(final_text)
        if data is None and final_text:
            data = {}
        ok = bool(final_text) and not errors and any_tool_ok(messages)
        if self.db is not None and self.session_id:
            self.db.add_event(self.session_id, f"subagent:{agent.role}", f"calls={calls} ok={ok} errors={errors[:2]}")
        return SubAgentResult(agent.role, ok, data or {}, final_text, calls, "; ".join(errors[:3]))


def any_tool_ok(messages: list[Any]) -> bool:
    """True unless the subagent ran a tool and every single call failed.

    A purely analytic answer (no tool calls) is still a valid answer, so the only
    hard failure mode is "it did try, and nothing worked".
    """
    calls = [b for msg in messages for b in getattr(msg, "blocks", []) if b.__class__.__name__ == "ToolCall"]
    return (not calls) or any(b.ok for b in calls)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        match = JSON_BLOCK.search(text)
        candidate = match.group(0) if match else None
    if not candidate:
        return None
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            data = json.loads(attempt)
            return data if isinstance(data, dict) else {"value": data}
        except ValueError:
            continue
    return None
