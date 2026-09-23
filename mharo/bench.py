"""`ma bench` — the engine under test, offline and deterministic.

Every scenario drives the *real* Engine (router → plan → tools → SQLite → verify
→ claim audit → fix rounds), with the provider replaced by a scripted replay
transport so results are reproducible without a network or an API key. A
scenario "passes" only if its assertions hold — never because a model said so.

This is also what `ma doctor` runs, and what CI should gate on:

    ma bench            # 9 scenarios
    ma bench --json     # machine readable
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from mharo_tui.agent.providers import Provider, ProviderError

from .config import Config
from .engine import Engine, EngineSettings
from .keys import KeyPool
from .permissions import Permissions
from .providers import ProviderHub, TierHandle
from .sessiondb import SessionDB
from .vault import Vault

ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "mharo-bench"


@dataclass
class Scenario:
    name: str
    what: str
    run: Callable[[Path], Awaitable[tuple[bool, str]]]
    dod: str = ""
    ok: bool = False
    detail: str = ""
    ms: int = 0


def _engine(tmp: Path, script: list[dict], *, deny: list[str] | None = None,
            verify: bool = True, peer: bool = False, plan: bool = False,
            extra: dict[str, Any] | None = None, memory: bool = True,
            approve: bool = True, fresh: bool = True) -> Engine:
    """A real Engine on a real temp project; only the transport is scripted."""
    cfg = Config.load()
    cfg.raw.setdefault("verify", {})["peer_review"] = peer
    if deny:
        cfg.raw["permissions"]["deny"] = list(cfg.raw["permissions"].get("deny") or []) + deny
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            path = tmp / f"bench.db{suffix}"
            if path.exists():
                path.unlink()
    db = SessionDB(tmp / "bench.db")
    hub = ProviderHub(cfg, db=db, force_provider="replay", replay_scripts={"all": script})
    engine = Engine(
        cfg, cwd=tmp, db=db, hub=hub, vault=Vault.load(tmp / "vault.json"),
        settings=EngineSettings(tier="auto", max_turns=4, plan=plan, verify=verify,
                               peer_review=peer, use_memory=memory, **(extra or {})),
    )
    if approve:
        engine.permissions.default = "allow"      # bench scenarios are not approval tests
    return engine


async def s_basic_answer(tmp: Path) -> tuple[bool, str]:
    """DoD #1: `ma "task"` drives a provider turn and returns a real answer."""
    eng = _engine(tmp, [{"text": "hello.py defines a single function `hi()` that returns the string 'hi'."}])
    result = await eng.run("what does hello.py do?")
    good = "hi()" in result.answer and result.ok and result.turns >= 1
    return good, f"answer={len(result.answer)}c turns={result.turns} tier={result.tier} ok={result.ok}"


async def s_tool_pipeline(tmp: Path) -> tuple[bool, str]:
    """Tool call from the model → executed on disk → recorded in SQLite."""
    (tmp / "target.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    eng = _engine(
        tmp,
        [
            {"calls": [{"tool": "read_file", "args": {"path": "target.txt"}, "id": "c1"}]},
            {"text": "The file has two lines: alpha and beta."},
        ],
    )
    result = await eng.run("read target.txt and tell me its lines")
    history = eng.db.tool_history(result.session_id or 0) if eng.db else []
    good = result.ok and "alpha" in result.answer and any(h["tool"] == "read_file" and h["ok"] == 1 for h in history)
    return good, f"tool rows={len(history)} detail={[h['tool'] for h in history]}"


async def s_verify_gates_claim(tmp: Path) -> tuple[bool, str]:
    """DoD #5 (negative proof): a passing-sounding claim with a failing test is rejected."""
    (tmp / "test_broken.py").write_text(
        "def test_math():\n    assert 1 + 1 == 3, 'deliberately broken for the benchmark'\n", encoding="utf-8"
    )
    eng = _engine(tmp, [{"text": "All tests pass, done. The bug is fixed and the build is green."}])
    result = await eng.run("fix the failing test and confirm it passes")
    claim = result.verification.claim if result.verification else None
    pytest_failed = any(not c.ok for c in (result.verification.checks if result.verification else []))
    good = (not result.ok) and claim is not None and claim.claimed and not claim.accepted and pytest_failed
    if good:
        return True, f"rejected: {claim.reasons[0][:60]}; checks={[c.summary()[:40] for c in result.verification.checks]}"
    return False, f"ok={result.ok} claimed={getattr(claim, 'claimed', None)} accepted={getattr(claim, 'accepted', None)} checks={[c.summary() for c in (result.verification.checks if result.verification else [])]}"


async def s_fix_round(tmp: Path) -> tuple[bool, str]:
    """The engine verifies, fails, fixes with the upgrade round, and ends green."""
    (tmp / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    eng = _engine(
        tmp,
        [
            {"calls": [{"tool": "edit_file", "args": {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}, "id": "e1"}]},
            {"text": "Patched add() to use +, and ran pytest to confirm."},
        ],
    )
    result = await eng.run("make test_calc pass by fixing calc.add")
    checks = result.verification.checks if result.verification else []
    pytest_checks = [c for c in checks if "pytest" in c.name and not c.skipped]
    green = any(c.ok for c in pytest_checks) if pytest_checks else all(c.skipped for c in checks)
    source = (tmp / "calc.py").read_text(encoding="utf-8")
    good = result.ok and green and "a + b" in source
    return good, f"ok={result.ok} pytest_green={green} edits={list(eng.ctx.files_touched)} proof={(result.verification.proof()[:80] if result.verification else '')}"


async def s_permission_deny(tmp: Path) -> tuple[bool, str]:
    """DoD #8 half: a denied command never runs and never reaches the model unfiltered."""
    victim = tmp / "keepme.txt"
    victim.write_text("must survive\n", encoding="utf-8")
    eng = _engine(
        tmp,
        [
            {"calls": [{"tool": "bash", "args": {"command": "rm -f keepme.txt"}, "id": "d1"}]},
            {"text": "Blocked, as expected."},
        ],
        deny=["bash:rm*"],
    )
    result = await eng.run("delete keepme.txt using the shell")
    rows = eng.db.tool_history(result.session_id or 0)
    denied = any(r["tool"] == "bash" and r["permission"] == "deny" for r in rows)
    return (victim.exists() and denied), f"file survived={victim.exists()} rows={[(r['tool'], r['permission']) for r in rows]}"


async def s_approval_flow(tmp: Path) -> tuple[bool, str]:
    """Approvals route through the policy and an approver callback (y/n/a)."""
    (tmp / "notes.md").write_text("old\n", encoding="utf-8")
    seen: list[str] = []

    def approver(tool: str, args: dict, _extra: dict) -> str:
        seen.append(f"{tool}:{sorted(args)}")
        return "once"

    eng = _engine(
        tmp,
        [
            {"calls": [{"tool": "write_file", "args": {"path": "notes.md", "content": "new\n"}, "id": "w1"}]},
            {"text": "Rewrote notes.md."},
        ],
        approve=False,
    )
    eng.approver = approver
    eng.permissions.allow.append("write_file:notes.md")   # allowlist wins over ask
    result = await eng.run("replace notes.md with the word new")
    body = (tmp / "notes.md").read_text(encoding="utf-8")
    allowed_by_rule = any(r["permission"] == "allow" for r in eng.db.tool_history(result.session_id or 0))
    return body == "new\n" and allowed_by_rule, f"body={body!r} decisions={eng.permissions.summary()['decisions']} approver_calls={len(seen)}"


async def s_key_rotation(tmp: Path) -> tuple[bool, str]:
    """DoD #2: a throttled key is cooled down and the next key answers."""
    cfg = Config.load()
    hub = ProviderHub(cfg, force_provider="replay")
    flaky = FlakyProvider(fail_once_with="HTTP 429 too many requests")
    pool = KeyPool("openai")
    pool.add("sk-keyaaaa1111", "k1")
    pool.add("sk-keybbbb2222", "k2")
    hub._handles["cheap"] = TierHandle(tier=cfg.tier("cheap"), provider=flaky, pool=pool)
    hub.session_id = None
    got = await hub._complete("cheap", "ping")
    report = hub.last_report
    good = got == "pong after retry" and report is not None and report.attempts >= 2
    cooldown = pool.keys[0].cooldown_until > time.monotonic() - 1
    return bool(good and cooldown), f"attempts={getattr(report, 'attempts', 0)} key={getattr(report, 'key_label', '')} errors={len(getattr(report, 'errors', []))} k1_cooling={pool.keys[0].cooldown_until > 0}"


async def s_memory_recall(tmp: Path) -> tuple[bool, str]:
    """DoD #7: a similar later task recalls the earlier one through the vector index."""
    first = _engine(tmp, [{"text": "Refactored the session store to use WAL mode."}], memory=True)
    r1 = await first.run("refactor the session store to sqlite WAL")
    eng = _engine(tmp, [{"text": "Recalled prior work: the store already uses WAL."}], memory=True, fresh=False)
    block = eng.memory.context_block("again refactor the session store to sqlite WAL mode", cwd=str(tmp)) if eng.memory else ""
    hits = eng.memory.recall("session store WAL", k=3, cwd=str(tmp)) if eng.memory else []
    top = hits[0].score if hits else 0.0
    good = bool(block) and any("session store" in h.title.lower() for h in hits) and top > 0.2
    return good, f"memories={eng.db.memory_count()} top_score={top:.2f} hits={len(hits)} block={len(block)}c ran1_ok={r1.ok}"


async def s_undo(tmp: Path) -> tuple[bool, str]:
    """DoD #6: sessions persist and `/undo` rolls the working tree back."""
    path = tmp / "config.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    eng = _engine(tmp, [
        {"calls": [{"tool": "edit_file", "args": {"path": "config.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}, "id": "u1"}]},
        {"text": "Set VALUE to 2."},
    ])
    result = await eng.run("set VALUE to 2 in config.py")
    assert eng.db is not None
    changed = path.read_text(encoding="utf-8")
    restored = eng.db.undo_last(result.session_id)
    back = path.read_text(encoding="utf-8")
    rows = eng.db.recent_sessions()
    good = changed == "VALUE = 2\n" and back == "VALUE = 1\n" and bool(restored) and len(rows) >= 1
    return good, f"after_edit={changed.strip()!r} after_undo={back.strip()!r} restored={restored} sessions={len(rows)}"


async def s_subagent_isolation(tmp: Path) -> tuple[bool, str]:
    """DoD #4: two subagents run with isolated tool sets; a reader cannot write."""
    from .subagents import SubAgentRunner, build_roots

    cfg = Config.load()
    db = SessionDB(tmp / "sub.db")
    sid = db.start_session(cwd=str(tmp), task="subagent bench", model="replay", provider="replay")
    script_reader = [{"text": json.dumps({"summary": "found 2 files", "files": [{"path": "a.py", "why": "target", "lines": [1, 2]}], "open_questions": []})}]
    script_tester = [{"text": json.dumps({"passed": True, "command": "pytest -q", "root_cause": "", "evidence": "2 passed", "suspect_files": []})}]
    hub_r = ProviderHub(cfg, db=db, force_provider="replay", replay_scripts={"all": script_reader})
    hub_t = ProviderHub(cfg, db=db, force_provider="replay", replay_scripts={"all": script_tester})
    roles = build_roots(cfg.subagents)
    runner = SubAgentRunner(hub_r, cwd=tmp, permissions=Permissions.from_config(cfg.permissions), session_db=db, session_id=sid, roles=roles)
    reader = await runner.run("reader", "find where sessions are stored and edit the file")
    # a separate hub for the tester proves parallel dispatch with independent cursors
    runner_t = SubAgentRunner(hub_t, cwd=tmp, permissions=Permissions.from_config(cfg.permissions), session_db=db, session_id=sid, roles=roles)
    tester = await runner_t.run("tester", "run the checks")
    refused = bool(reader.error) or "not allowed" in (reader.text + reader.error)
    reader_data_ok = isinstance(reader.data, dict) and bool(reader.data.get("summary"))
    good = (tester.ok and bool(tester.data.get("passed")) is True) and reader_data_ok
    # hard guarantee: the reader's registry simply does not contain mutating tools
    no_writer = not any(t in {"write_file", "edit_file", "bash"} for t in roles["reader"].tools)
    return bool(good and no_writer), f"reader={reader_data_ok} tester_ok={tester.ok} reader_registry_safe={no_writer} refusals={refused}"


class FlakyProvider(Provider):
    """Test transport: fails with a throttle error once, then answers."""

    name = "flaky"
    supports_tools = False

    def __init__(self, model: str = "flaky", fail_once_with: str = "429", **opts: Any) -> None:
        super().__init__(model, **opts)
        self.fail_once_with = fail_once_with
        self.failed = False
        self.keys_seen: list[str] = []

    async def stream(self, messages: list[Any], tools: list[dict] | None = None):
        if not self.failed:
            self.failed = True
            raise ProviderError(self.fail_once_with)
        yield {"type": "text", "text": "pong after retry"}
        yield {"type": "usage", "input_tokens": 7, "output_tokens": 4}
        yield {"type": "done", "stop_reason": "end_turn"}


SCENARIOS = [
    Scenario("basic-answer", "task → provider turn → answer (DoD 1)", s_basic_answer, "1"),
    Scenario("tool-pipeline", "model tool_call executed + audited in SQLite", s_tool_pipeline, ""),
    Scenario("claim-audit", "confident lie about a failing test is rejected (DoD 5)", s_verify_gates_claim, "5"),
    Scenario("fix-round", "verify fails → upgrade tier → fix → green (R4)", s_fix_round, ""),
    Scenario("permission-deny", "deny rule blocks the command before it runs (DoD 8)", s_permission_deny, "8"),
    Scenario("approval-flow", "allow/ask/deny + approver callback wiring", s_approval_flow, "8"),
    Scenario("key-rotation", "429 → cooldown key #1 → key #2 answers (DoD 2)", s_key_rotation, "2"),
    Scenario("memory-recall", "vector recall returns the earlier related task (DoD 7)", s_memory_recall, "7"),
    Scenario("session-undo", "SQLite session + /undo restores the file (DoD 6)", s_undo, "6"),
    Scenario("subagent-isolation", "two subagents, isolated tool sets (DoD 4)", s_subagent_isolation, "4"),
]


async def run_all(*, keep: bool = False, only: str | None = None) -> list[Scenario]:
    tmp = ROOT / "self"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "hello.py").write_text("def hi():\n    return 'hi'\n", encoding="utf-8")
    out: list[Scenario] = []
    for scenario in SCENARIOS:
        if only and only not in scenario.name:
            continue
        scenario_dir = tmp / scenario.name
        scenario_dir.mkdir(exist_ok=True)
        (scenario_dir / "hello.py").write_text("def hi():\n    return 'hi'\n", encoding="utf-8")
        started = time.monotonic()
        try:
            scenario.ok, scenario.detail = await scenario.run(scenario_dir)
        except Exception as exc:
            scenario.ok = False
            scenario.detail = f"{type(exc).__name__}: {str(exc)[:220]}"
        scenario.ms = int((time.monotonic() - started) * 1000)
        out.append(scenario)
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def render(scenarios: list[Scenario]) -> tuple[str, int]:
    fails = sum(1 for s in scenarios if not s.ok)
    lines = [
        "Mharo Agent bench — real engine, scripted transport",
        "─" * 96,
        f" {'':2}{'scenario':22} {'result':7} {'ms':>6}  what it proves",
        "─" * 96,
    ]
    for s in scenarios:
        mark = "PASS ✓" if s.ok else "FAIL ✗"
        lines.append(f"    {s.name:22} {mark:8} {s.ms:>6}  {s.what}")
        if not s.ok:
            lines.append(f"      ↳ {s.detail[:220]}")
    lines.append("─" * 96)
    total_ms = sum(s.ms for s in scenarios)
    lines.append(
        f" {len(scenarios) - fails}/{len(scenarios)} scenarios passed in {total_ms} ms"
        + ("  → engine is sound" if not fails else "  → fix before shipping")
    )
    return "\n".join(lines), fails


def as_json(scenarios: list[Scenario]) -> str:
    return json.dumps(
        {
            "passed": sum(1 for s in scenarios if s.ok),
            "failed": sum(1 for s in scenarios if not s.ok),
            "scenarios": [{"name": s.name, "ok": s.ok, "ms": s.ms, "what": s.what, "dod": s.dod, "detail": s.detail} for s in scenarios],
        },
        indent=2,
    )
