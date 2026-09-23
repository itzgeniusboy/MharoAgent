"""P1-2 · The engine: task → plan → act(tools) → verify → upgrade → proof.

Design rules that make this an agent and not a chat client:

* the plan is data (JSON steps) and every step is executed, not narrated
* files are snapshotted **before** mutation, so `/undo` and failure recovery work
* tool output is redacted at the boundary, then truncated, then stored in SQLite
* verification is a gate: `Result.ok` is False when checks failed or a completion
  claim has no proof, even if the model sounded confident
* the router can promote the *fix round* to the strong tier mid-run (R4/G6)
* memory: recall before planning, remember after a successful run
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

from mharo_tui.agent.session import Message, Text as TextBlock, ToolCall
from mharo_tui.agent.tools import TOOLS, ToolContext, ToolSpec, repo_info, run_tool

from . import verify as V
from .config import Config
from .memory import Memory
from .permissions import Permissions
from .providers import ProviderHub
from .router import Router
from .sessiondb import SessionDB
from .vault import Vault

SYSTEM = """You are Mharo, an autonomous coding agent working inside the user's repository.

Rules you are graded on:
- Look before you assert. Read/search the real files; never invent paths, APIs or line numbers.
- One step at a time: make the smallest change that fixes the cause, then re-read it.
- Verify with the project's own commands. If a check fails, fix it; do not declare done.
- Never claim "done/fixed/passing" unless a check you ran proves it.
- Use markdown. Code in fences. No preamble, no "certainly", no restating the task.
"""

PLAN_PROMPT = """Produce a plan for this task as STRICT JSON only:
{"steps": [{"goal": str, "how": str, "files": [str], "tier": "cheap|strong"}], "risks": [str], "verify": str}
Rules: 1-6 steps, each independently checkable, ordered by dependency. If the task is a
single obvious action, return one step. No prose outside the JSON.

Task: @TASK@

Memory of prior work on this repo:
@MEMORY@"""

FIX_PROMPT = """The change did not verify. Fix it.

Task: {task}

Failing evidence:
{evidence}

Previous answer:
{previous}

Requirements for this round: address each failing item, then state exactly what you changed."""


@dataclass
class Step:
    goal: str
    how: str = ""
    files: list[str] = field(default_factory=list)
    tier_hint: str = ""
    done: bool = False
    answer: str = ""
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "done": self.done, "tool_calls": self.tool_calls, "errors": self.errors[:2]}


@dataclass
class Result:
    task: str
    ok: bool
    answer: str
    steps: list[Step] = field(default_factory=list)
    verification: V.Verification | None = None
    files: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    session_id: int | None = None
    upgrades: list[str] = field(default_factory=list)
    tier: str = ""
    turns: int = 0
    errors: list[str] = field(default_factory=list)
    memory_ids: list[int] = field(default_factory=list)
    elapsed_ms: int = 0

    def report(self) -> str:
        lines = [f"{'✓' if self.ok else '✗'} {self.task[:90]}"]
        if self.steps:
            lines.append("  steps: " + " → ".join(("✓" if s.done else "…") if len(s.goal) < 3 else s.goal[:28] for s in self.steps))
            lines.append("  " + "\n  ".join(f"{i+1}. [{'x' if s.done else ' '}] {s.goal[:70]} ({s.tool_calls} tool calls)" for i, s in enumerate(self.steps)))
        if self.verification:
            lines.append("  proof:\n    " + self.verification.proof().replace("\n", "\n    "))
        if self.upgrades:
            lines.append("  routing: " + "; ".join(self.upgrades))
        lines.append(f"  cost: ${self.cost_usd:.4f} · {self.tokens_in} in / {self.tokens_out} out · {self.turns} turns · {self.elapsed_ms / 1000:.1f}s")
        if self.errors:
            lines.append("  errors: " + "; ".join(e[:120] for e in self.errors[:3]))
        return "\n".join(lines)


@dataclass
class EngineSettings:
    tier: str = "auto"            # auto | cheap | strong
    max_turns: int = 12
    max_steps: int = 6
    plan: bool = True
    verify: bool = True
    peer_review: bool = True
    use_memory: bool = True
    subagents: bool = True
    max_fix_rounds: int = 2
    turn_timeout: float = 300.0
    resume_session: int | None = None


class Engine:
    def __init__(
        self,
        config: Config | None = None,
        *,
        cwd: str | Path = ".",
        db: SessionDB | None = None,
        vault: Vault | None = None,
        hub: ProviderHub | None = None,
        settings: EngineSettings | None = None,
        approver: Callable[[str, str, dict], Any] | None = None,
        on_event: Callable[[dict], Any] | None = None,
        extra_tools: dict[str, ToolSpec] | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.cwd = Path(cwd).expanduser().resolve()
        self.db = db
        self.vault = vault if vault is not None else Vault.load(self.config.vault_path())
        self.hub = hub or ProviderHub(
            self.config, db=db, redact=lambda text: self.vault.redact(text)[0] if self.vault else text
        )
        self.settings = settings or EngineSettings()
        self.permissions = Permissions.from_config(self.config.permissions)
        self.approver = approver
        self.on_event = on_event
        self.router = Router(tiers=self.config.tiers, budget_usd=float(self.config.budget.get("max_session_usd", 2.0)))
        self.memory = Memory.build(db, self.config.memory) if (db is not None and self.config.memory.get("enabled", True)) else None
        self.tools: dict[str, ToolSpec] = dict(TOOLS)
        self.tools.update(extra_tools or {})
        self.ctx = ToolContext(cwd=self.cwd, allow_outside=False, timeout=float(self.config.budget.get("turn_timeout_s", 300)))
        self.messages: list[Message] = []
        self.session_id: int | None = None
        self.spent_usd = 0.0
        self._mutated_after_check = False
        self._cancelled = asyncio.Event()
        self.repo = repo_info(self.cwd)
        self._turns = 0
        self._upgrades: list[str] = []
        self.plan_risks: list[str] = []
        self.plan_verify = ""
        self._load_loops()

    # -- setup -----------------------------------------------------------
    def _load_loops(self) -> None:
        """AGENTS.md / LOOPS.md become part of the system prompt (project rules)."""
        extra: list[str] = []
        for name in ("AGENTS.md", "MHARO.md", "Mharo.md", "LOOPS.md", "CLAUDE.md"):
            path = self.cwd / name
            if path.is_file():
                try:
                    extra.append(f"### {name}\n" + path.read_text(encoding="utf-8", errors="replace")[:8000])
                except OSError:
                    pass
        self.system = SYSTEM + ("\n\n# Project rules\n" + "\n\n".join(extra) if extra else "")

    def cancel(self) -> None:
        self._cancelled.set()

    async def _emit(self, evt: dict) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(evt)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    # -- main entry ------------------------------------------------------
    async def run(self, task: str) -> Result:
        task = (task or "").strip()
        if not task:
            raise ValueError("empty task")
        started = time.monotonic()
        routing = self.router.pick(task, forced=self.settings.tier)
        tier = routing.tier
        if self.db is not None:
            self.session_id = self.db.start_session(
                cwd=str(self.cwd), task=task, model=self.config.tier(tier).model, provider=self.config.tier(tier).provider
            )
            self.hub.session_id = self.session_id
            if self.settings.resume_session:
                for row in self.db.resume_context(self.settings.resume_session):
                    self.messages.append(Message(role=row["role"], blocks=[TextBlock(row["content"])]))
        await self._emit({"type": "start", "task": task, "tier": tier, "route": routing.explain()})

        memory_block = ""
        if self.memory is not None and self.settings.use_memory:
            memory_block = self.memory.context_block(task, cwd=str(self.cwd), k=int(self.config.memory.get("recall_k", 4)))
            if memory_block:
                await self._emit({"type": "memory", "block": memory_block})

        if not self.messages:
            self.messages.append(Message(role="system", blocks=[TextBlock(self.system)]))

        steps = await self._plan(task, memory_block) if self.settings.plan else [Step(goal=task)]
        await self._emit({"type": "plan", "steps": [s.goal for s in steps]})

        answer = ""
        for index, step in enumerate(steps):
            if self._cancelled.is_set():
                break
            decision = self.router.pick(
                step.goal,
                step={"tier": step.tier_hint} if step.tier_hint else None,
                forced=None if self.settings.tier == "auto" else self.settings.tier,
            )
            answer = await self._act(step, decision.tier, index, context=f"Overall task: {task}")
            await self._emit({"type": "step_done", "step": index + 1, "goal": step.goal, "ok": step.done})

        verification = V.Verification()
        if self.settings.verify:
            verification = await self._verify(task, answer)
            tier = await self._fix_rounds(task, answer, verification, tier)
            verification = await self._verify(task, answer)

        cost = await self._cost_total()
        files = sorted(self.ctx.files_touched)
        ok = verification.ok and all(s.done for s in steps) and not self._cancelled.is_set()
        result = Result(
            task=task, ok=ok, answer=answer, steps=steps, verification=verification, files=files,
            cost_usd=cost, tokens_in=self._sum("tokens_in"), tokens_out=self._sum("tokens_out"),
            session_id=self.session_id, tier=tier, turns=self._turns,
            upgrades=list(getattr(self, "_upgrades", [])), elapsed_ms=int((time.monotonic() - started) * 1000),
            errors=[e for s in steps for e in s.errors][:6],
        )
        if self.memory is not None and self.settings.use_memory:
            result.memory_ids = [self.memory.remember(
                title=task[:70], body=_digest(task, answer, verification, files), kind="task",
                cwd=str(self.cwd), files=files, cost_usd=cost,
                tags="ok" if ok else "unverified",
            )]
        if self.db is not None and self.session_id:
            for s in steps:
                self.db.add_message(self.session_id, "assistant", s.answer or s.goal, tier=tier)
            self.db.add_message(self.session_id, "user", task)
            self.db.end_session(self.session_id, status="done" if ok else "unverified", proof=verification.proof())
        await self._emit({"type": "end", "ok": ok, "result": result})
        return result

    # -- planning --------------------------------------------------------
    async def _plan(self, task: str, memory_block: str) -> list[Step]:
        # token substitution, not str.format: the template contains literal JSON braces
        prompt = (PLAN_PROMPT.replace("@TASK@", task[:2000])
                  .replace("@MEMORY@", memory_block or "(no prior work in memory)"))
        tier = self.router.pick(task, forced=None if self.settings.tier == "auto" else self.settings.tier).tier
        try:
            raw = await self.hub.complete(tier, prompt)
        except Exception as exc:
            await self._emit({"type": "notice", "text": f"planning skipped: {type(exc).__name__}"})
            return [Step(goal=task)]
        data = _extract_json(raw)
        if not data:
            return [Step(goal=task, how="no structured plan returned")]
        steps: list[Step] = []
        for item in (data.get("steps") or [])[: self.settings.max_steps]:
            if isinstance(item, str):
                steps.append(Step(goal=item))
            elif isinstance(item, dict) and item.get("goal"):
                steps.append(Step(
                    goal=str(item["goal"])[:400], how=str(item.get("how", ""))[:600],
                    files=[str(f) for f in (item.get("files") or [])][:12], tier_hint=str(item.get("tier", "")),
                ))
        if not steps:
            steps = [Step(goal=task)]
        self.plan_risks = [str(r)[:200] for r in (data.get("risks") or [])][:6]
        self.plan_verify = str(data.get("verify") or "")[:300]
        return steps

    # -- acting ----------------------------------------------------------
    async def _act(self, step: Step, tier: str, index: int, *, context: str = "", extra_prompt: str = "") -> str:
        tools = [spec.schema() for spec in self.tools.values()]
        collected: list[str] = []
        failures = 0
        for turn in range(self.settings.max_turns):
            if self._cancelled.is_set():
                break
            self._turns += 1
            user_tail = Message(role="user", blocks=[TextBlock(context + (("\n\n" + extra_prompt) if extra_prompt else "") + f"\n\nCurrent step {index + 1}: {step.goal}" + (f"\nHow: {step.how}" if step.how else ""))]) if turn == 0 else None
            if user_tail is not None:
                self.messages.append(user_tail)
            assistant = Message(role="assistant", model=self.config.tier(tier).model)
            self.messages.append(assistant)
            pending: list[ToolCall] = []
            try:
                async with asyncio.timeout(self.settings.turn_timeout):
                    async for evt in self.hub.stream(tier, self.messages, tools):
                        kind = evt.get("type")
                        if kind == "text":
                            assistant.append_text(evt.get("text", ""))
                            collected.append(evt.get("text", ""))
                            await self._emit({"type": "text", "text": evt.get("text", "")})
                        elif kind == "thinking":
                            assistant.append_thinking(evt.get("text", ""))
                        elif kind == "tool_call":
                            call = ToolCall(tool=evt.get("tool", ""), args=evt.get("args") or {})
                            assistant.blocks.append(call)
                            pending.append(call)
                            await self._emit({"type": "tool_start", "call": call})
                        elif kind == "error":
                            step.errors.append(evt.get("text", "provider error"))
                            failures += 1
                        elif kind == "done":
                            break
            except (TimeoutError, asyncio.TimeoutError):
                step.errors.append(f"turn timed out after {self.settings.turn_timeout:.0f}s")
                failures += 1
            except Exception as exc:
                step.errors.append(f"{type(exc).__name__}: {exc}")
                failures += 1
            if not pending:
                step.done = True
                break
            for call in pending:
                step.tool_calls += 1
                ok = await self._execute(call, tier, index)
                failures = failures + 0 if ok else failures + 1
            self.spent_usd = await self._cost_total()
            decision = self.router.should_upgrade(
                tier=tier, failures=failures, files_touched=len(self.ctx.files_touched),
                diff_lines=_diff_len("".join(collected)), verify_failed=False, spent_usd=self.spent_usd,
            )
            if decision.tier != tier and self.settings.tier == "auto":
                await self._emit({"type": "upgrade", "from": tier, "to": decision.tier, "why": decision.reasons})
                self._upgrades.append(f"{tier}→{decision.tier}: {'; '.join(decision.reasons)[:120]}")
                tier = decision.tier
        step.answer = "".join(collected).strip()
        return step.answer

    async def _execute(self, call: ToolCall, tier: str, index: int) -> bool:
        spec = self.tools.get(call.tool)
        if spec is None:
            call.result, call.ok = f"unknown tool {call.tool!r}; available: {', '.join(sorted(self.tools))}", False
            await self._emit({"type": "tool_end", "call": call})
            return False
        decision = self.permissions.decide(call.tool, call.args)
        self.permissions.record(call.tool, call.args, decision)
        if decision.blocked:
            call.result, call.ok, call.duration_ms = f"blocked by permission rule {decision.rule!r}: {decision.reason}", False, 0
            self._log_tool(call, decision.action, index)
            await self._emit({"type": "tool_end", "call": call})
            return False
        if spec.approval and decision.action == "ask":
            verdict = await self._ask(call)
            if verdict not in {"once", "always"}:
                call.result, call.ok, call.duration_ms = "User denied this action. Ask for a different approach.", False, 0
                self._log_tool(call, "deny", index)
                await self._emit({"type": "tool_end", "call": call})
                return False
            if verdict == "always":
                self.permissions.remember_always(call.tool)
        if call.tool in {"write_file", "edit_file"}:
            target = str(call.args.get("path") or "")
            if target and self.db is not None and self.session_id:
                try:
                    self.db.snapshot_file(self.session_id, self.cwd / target)
                except OSError:
                    pass
        if self.vault is not None:
            self.ctx.env_extra = self.vault.env_for([])
        result = await asyncio.to_thread(run_tool, call.tool, call.args, self.ctx, self.tools)
        redacted, hits = (self.vault.redact(result.output) if self.vault else (result.output, 0))
        call.result, call.ok, call.duration_ms = redacted, result.ok, result.duration_ms
        self._record_savings(result, redacted, hits, index)
        if hits:
            call.result += f"\n\n[{hits} secret(s) redacted before they could reach the model]"
        if result.ok and call.tool in {"write_file", "edit_file"}:
            self._mutated_after_check = True
            target = str(call.args.get("path") or "")
            if target and self.db is not None and self.session_id:
                try:
                    self.db.record_after(self.session_id, self.cwd / target)
                except OSError:
                    pass
        self._log_tool(call, decision.action, index)
        await self._emit({"type": "tool_end", "call": call})
        self.messages.append(Message(role="user", blocks=[TextBlock(f"{call.tool} → {'ok' if call.ok else 'failed'}\n{(call.result or '')[:6000]}")]))
        return bool(call.ok)

    def _log_tool(self, call: ToolCall, permission: str, index: int) -> None:
        if self.db is not None and self.session_id:
            self.db.add_tool_call(
                self.session_id, call.tool, call.args, ok=call.ok, duration_ms=call.duration_ms or 0,
                output=call.result or "", permission=permission, step=index,
            )

    async def _ask(self, call: ToolCall) -> str:
        """Ask a human. `approver(tool, args, registry)` may return str or bool."""
        if self.approver is None:
            if not sys.stdin.isatty():
                return "deny"
            from mharo_tui.agent.tools import preview as tool_preview

            print(f"  ⚠ {tool_preview(call.tool, call.args)}", flush=True)
            answer = input("    [y] once  ·  [a] always  ·  [N] deny  ").strip().lower()
            return {"y": "once", "yes": "once", "a": "always"}.get(answer, "deny")
        verdict = self.approver(call.tool, call.args, self.tools)
        if asyncio.iscoroutine(verdict):
            verdict = await verdict
        if isinstance(verdict, bool):
            return "once" if verdict else "deny"
        return str(verdict or "deny")

    # -- verification ----------------------------------------------------
    async def _verify(self, task: str, answer: str) -> V.Verification:
        verification = V.Verification()
        found = V.detect_checks(self.cwd, str(self.config.verify.get("auto_checks", "auto")))
        changed = sorted(self.ctx.files_touched) or V.changed_files(self.cwd)
        verification.sanity = V.sanity_scan(self.cwd, changed)
        verification.diff_stat = V.diff_stat(self.cwd) or "; ".join(
            f"{f} ({self.ctx.files_touched.get(f, 0)} lines)" for f in changed[:8]
        )

        async def run_all_checks() -> None:
            verification.checks = []
            for name, argv in found:
                check = await asyncio.to_thread(V.run_check, self.cwd, name, argv)
                verification.checks.append(check)
                if self.db is not None and self.session_id:
                    self.db.add_check(self.session_id, check.name, check.ok, command=check.command, detail=check.detail[:1500])

        await run_all_checks()
        if self._mutated_after_check and found:
            # a check that ran before the last edit proves nothing: re-run after
            self._mutated_after_check = False
            await run_all_checks()
        verification.claim = V.audit_claim(
            answer,
            checks=verification.checks,
            mutated_after_check=self._mutated_after_check and bool(verification.checks),
        )
        if self.settings.peer_review and self.config.verify.get("peer_review", True) and changed:
            diff = (verification.diff_stat or "") + "\n" + _changed_diff(self.cwd, changed)
            verification.review = await V.peer_review(self.hub, self.router.strong_tier, task, diff, limit=6)
            verification.review = [r for r in verification.review if "unavailable" not in r]
        return verification

    async def _fix_rounds(self, task: str, answer: str, verification: V.Verification, tier: str) -> str:
        rounds = int(self.config.verify.get("max_fix_rounds", 2))
        for attempt in range(rounds):
            if verification.ok or self._cancelled.is_set():
                break
            evidence = verification.proof()
            decision = self.router.should_upgrade(
                tier=tier, failures=0, files_touched=len(self.ctx.files_touched),
                diff_lines=len((verification.diff_stat or "").splitlines()), verify_failed=True, spent_usd=self.spent_usd,
            )
            use_tier = decision.tier
            if use_tier != tier and self.settings.tier == "auto":
                self._upgrades.append(f"fix round {attempt + 1}: {decision.explain()}")
                await self._emit({"type": "upgrade", "from": tier, "to": use_tier, "why": decision.reasons})
                tier = use_tier
            step = Step(goal=f"fix verification failures (round {attempt + 1})")
            answer = await self._act(
                step, tier, 900 + attempt, context=f"Overall task: {task}",
                extra_prompt=FIX_PROMPT.format(task=task[:800], evidence=evidence[:2500], previous=answer[:1500]),
            )
            self._mutated_after_check = bool(self.ctx.files_touched)
            verification = await self._verify(task, answer)
        return tier

    # -- bookkeeping -----------------------------------------------------
    def _record_savings(self, result: Any, redacted: str, hits: int, index: int) -> None:
        """Measurable savings: what truncation and redaction did NOT send up.

        `run_tool` truncates its output and records the raw size in `meta`, so the
        difference is characters that never reached a model — a real number, not a
        marketing estimate.
        """
        raw = int((getattr(result, "meta", None) or {}).get("raw_chars", 0) or 0)
        saved = max(0, raw - len(redacted or ""))
        if (saved or hits) and self.db is not None and self.session_id:
            self.db.add_event(
                self.session_id, "savings",
                json.dumps({"chars": saved, "redactions": hits, "step": index}),
            )

    def _sum(self, field_name: str) -> int:
        if self.db is None or not self.session_id:
            return 0
        try:
            rows = self.db.ledger(self.session_id)["usage"]
            key = "ti" if field_name == "tokens_in" else "to_"
            return int(sum(int(r.get(key) or 0) for r in rows))
        except Exception:
            return 0

    async def _cost_total(self) -> float:
        if self.db is None or not self.session_id:
            return self.spent_usd
        try:
            rows = self.db.ledger(self.session_id)["usage"]
            return round(sum(float(r.get("cost") or 0.0) for r in rows), 6)
        except Exception:
            return self.spent_usd


# --------------------------------------------------------------------- helpers


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                data = json.loads(attempt)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _diff_len(text: str) -> int:
    return len([ln for ln in (text or "").splitlines() if ln.startswith(("+", "-"))])


def _changed_diff(cwd: Path, files: list[str]) -> str:
    out: list[str] = []
    for rel in files[:6]:
        path = cwd / rel
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        out.append(f"--- {rel} (current contents)")
        out.extend("    " + ln for ln in body[:120])
    return "\n".join(out)[:6000]


def _digest(task: str, answer: str, verification: V.Verification | None, files: list[str]) -> str:
    parts = [f"task: {task[:300]}", f"files: {', '.join(files) if files else 'none'}"]
    if verification:
        parts.append(f"proof: {verification.proof()[:400]}")
    parts.append(f"result: {' '.join((answer or '').split())[:500]}")
    return "\n".join(parts)
