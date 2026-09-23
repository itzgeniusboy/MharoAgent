"""`ma` — the Phase 1 command line (stdlib argparse only, so it runs before any optional dep).

    ma "fix the flaky test"              plan → act with tools → verify → proof
    ma --replay-file t.json "…"          drive the same engine from a script (offline)
    ma doctor [--strict]                 self-check; every probe really executes
    ma bench [--only NAME]               10 engine scenarios, deterministic
    ma repl                                session with /review /debt /gain /undo /mem …
    ma cost | sessions | undo | memory | skill | vault | init | tui

Global flags are also accepted after the subcommand name where they make sense
(each subparser re-declares --cwd/--config), so `ma doctor --cwd /repo` works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from mharo_tui.agent.tools import TOOLS, preview as tool_preview

from . import __version__, cost as cost_mod, skills as skills_mod, verify as V
from .bench import as_json as bench_json
from .bench import render as bench_render
from .bench import run_all as bench_run_all
from .config import Config, home_dir
from .doctor import Doctor
from .engine import Engine, EngineSettings, Result
from .keys import load_env_files
from .memory import Memory
from .providers import ProviderHub
from .sessiondb import SessionDB
from .subagents import SubAgentRunner, build_roots
from .vault import Vault

BANNER = "◆ mharo agent"


class Console:
    """Minimal markdown-ish writer so headless `ma` output stays readable."""

    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet or os.environ.get("NO_COLOR") == "1"
        self.streamed = False

    @staticmethod
    def _c(code: str, text: str) -> str:
        return text if os.environ.get("NO_COLOR") == "1" else f"\x1b[{code}m{text}\x1b[0m"

    def head(self, text: str) -> None:
        self.streamed = False
        if not self.quiet:
            print("\n" + self._c("1;36", text), flush=True)

    def write(self, text: str) -> None:
        if self.quiet or not text:
            return
        self.streamed = True
        print(text, end="", flush=True)

    def newline(self) -> None:
        if self.streamed:
            print("", flush=True)
            self.streamed = False

    def note(self, text: str) -> None:
        self.newline()
        if not self.quiet:
            print(text, flush=True)

    def tool(self, text: str, ok: bool) -> None:
        self.newline()
        mark = self._c("32", "✓") if ok else self._c("31", "✗")
        print(f"  {mark} {self._c('1', text)}", flush=True)


# --------------------------------------------------------------------- shared


def add_shared(parser: argparse.ArgumentParser) -> None:
    """`--cwd` / `--config`, before OR after the subcommand.

    Sub-parsers use SUPPRESS so a value already parsed on the root parser is not
    overwritten by the sub-parser's default (argparse reapplies defaults).
    """
    is_root = parser is not None and getattr(parser, "prog", "") == "ma"
    defaults: dict[str, Any] = {"default": "."} if is_root else {"default": argparse.SUPPRESS}
    parser.add_argument("--cwd", help="working directory (default: current)", **defaults)
    parser.add_argument("--config", help="config.json (default ~/.mharo/config.json)",
                        **({"default": None} if is_root else {"default": argparse.SUPPRESS}))


def load_config(args: argparse.Namespace) -> Config:
    cwd = Path(getattr(args, "cwd", ".") or ".").expanduser().resolve()
    load_env_files(cwd)
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "model", None):
        tier = "cheap"
        cfg.raw["tiers"][tier] = {**cfg.raw["tiers"].get(tier, {}), "model": args.model}
    if getattr(args, "base_url", None):
        provider = cfg.raw["tiers"]["cheap"].get("provider", "openai")
        cfg.raw.setdefault("providers", {}).setdefault(provider, {})["base_url"] = args.base_url
    if getattr(args, "provider", None):
        cfg.raw["tiers"]["cheap"]["provider"] = args.provider
    return cfg


def open_db(cfg: Config) -> SessionDB | None:
    try:
        return SessionDB(cfg.db_path())
    except Exception as exc:  # corrupt or locked db must not hide the task
        print(f"warning: could not open {cfg.db_path()}: {exc}", file=sys.stderr)
        return None


def build_hub(cfg: Config, db: SessionDB | None, args: argparse.Namespace) -> ProviderHub:
    scripts: dict[str, Any] | None = None
    replay_file = getattr(args, "replay_file", None)
    if replay_file:
        raw = json.loads(Path(replay_file).read_text(encoding="utf-8"))
        scripts = {"all": raw} if isinstance(raw, list) else raw
    return ProviderHub(cfg, db=db, force_provider=getattr(args, "provider", None), replay_scripts=scripts)


def interactive_approver(console: Console) -> Callable[[str, dict, dict], str]:
    def approver(tool: str, args: dict, _registry: dict) -> str:
        if not sys.stdin.isatty():
            console.note("  (non-interactive: denied — re-run with --yes to auto-approve)")
            return "deny"
        console.newline()
        print(f"  ⚠ {tool_preview(tool, args)}", flush=True)
        try:
            answer = input("    [y] once  ·  [a] always  ·  [N] deny  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        return {"y": "once", "yes": "once", "a": "always"}.get(answer, "deny")

    return approver


def make_engine(args: argparse.Namespace, cfg: Config, console: Console, *, db: SessionDB | None = None) -> Engine:
    cwd = Path(args.cwd or ".").expanduser().resolve()
    db = db if db is not None else open_db(cfg)
    hub = build_hub(cfg, db, args)
    vault = Vault.load(cfg.vault_path())
    hub.redact = (lambda text: vault.redact(text)[0]) if getattr(args, "redact", True) else (lambda text: text)
    settings = EngineSettings(
        tier=getattr(args, "tier", "auto"),
        max_turns=getattr(args, "max_turns", 12),
        max_steps=getattr(args, "max_steps", 6),
        plan=not getattr(args, "no_plan", False),
        verify=not getattr(args, "no_verify", False),
        peer_review=not getattr(args, "no_review", False),
        use_memory=not getattr(args, "no_memory", False),
        subagents=not getattr(args, "no_subagents", False),
    )
    engine = Engine(
        cfg, cwd=cwd, db=db, hub=hub, vault=vault, settings=settings,
        approver=None if getattr(args, "yes", False) else interactive_approver(console),
        on_event=(lambda evt: handle_event(evt, console)) if not getattr(args, "quiet", False) else None,
    )
    if getattr(args, "yes", False):
        engine.permissions.default = "allow"
    for rule in getattr(args, "allow", None) or []:
        engine.permissions.allow.append(rule)
    for rule in getattr(args, "deny", None) or []:
        engine.permissions.deny.insert(0, rule)
    resume = getattr(args, "resume", None)
    if resume:
        engine.settings.resume_session = int(resume)
    return engine


def handle_event(evt: dict, console: Console) -> None:
    kind = evt.get("type")
    if kind == "start":
        console.note(f"  route · {evt.get('route') or evt.get('tier')}")
    elif kind == "plan":
        steps = evt.get("steps") or []
        console.note("  plan · " + " → ".join(f"{i + 1}. {str(s)[:60]}" for i, s in enumerate(steps)))
    elif kind == "text":
        console.write(evt.get("text", ""))
    elif kind == "tool_start":
        call = evt["call"]
        console.note(f"  … {tool_preview(call.tool, call.args)}")
    elif kind == "tool_end":
        call = evt["call"]
        console.tool(f"{call.tool} {'ok' if call.ok else 'failed'} ({(call.duration_ms or 0) / 1000:.1f}s)", bool(call.ok))
    elif kind == "upgrade":
        console.note(f"  ⤴ tier {evt.get('from')}→{evt.get('to')} · {'; '.join(evt.get('why') or [])}")
    elif kind == "memory":
        console.note(f"  ◆ recalled {len(evt.get('block') or '')} chars of prior context")
    elif kind == "notice":
        console.note(f"  ! {evt.get('text')}")


def result_payload(result: Result) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "task": result.task,
        "answer": result.answer,
        "steps": [s.to_dict() for s in result.steps],
        "files": result.files,
        "tier": result.tier,
        "turns": result.turns,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cost_usd": result.cost_usd,
        "elapsed_ms": result.elapsed_ms,
        "session_id": result.session_id,
        "upgrades": result.upgrades,
        "memory_ids": result.memory_ids,
        "errors": result.errors,
        "verification": result.verification.to_dict() if result.verification else None,
    }


# ------------------------------------------------------------------- commands


def provider_label(engine: Any, cfg: Config) -> str:
    """What will answer this run — a free rung, or the configured key.

    The header used to print the *tier's* model even when no key existed, which
    read like a promise. This says what is actually wired up.
    """
    from .ladder import FreePool

    spec = cfg.tier("cheap")
    try:
        handle = engine.hub.handle("cheap")
    except Exception:  # noqa: BLE001 - a header must never break a run
        return f"{spec.provider}:{spec.model}"
    pool = getattr(handle, "pool", None)
    if isinstance(pool, FreePool):
        rung = pool.rung_for(pool.keys[0]) if pool.keys else None
        ready = sum(1 for k in pool.keys if k.available)
        where = f"{rung.name}/{rung.model}" if rung is not None else "ladder"
        keys = "no key" if not any(k.token for k in pool.keys) else "key + free rungs"
        return f"free · {where} ({ready}/{len(pool.keys)} pairings ready, {keys})"
    if getattr(handle.provider, "name", "") == "demo":
        return "offline demo · no provider reachable (see `ma free`)"
    return f"{handle.provider.name}:{handle.provider.model}"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    task = " ".join(args.task).strip() if args.task else ""
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()
    if not task:
        print('ma: nothing to do. Usage: ma "<task>"   (or `ma repl` for a session)', file=sys.stderr)
        return 2
    console = Console(quiet=args.quiet or args.json)
    engine = make_engine(args, cfg, console)
    if not args.json:
        console.head(f"{BANNER} · {__version__} · {provider_label(engine, cfg)}")
        engine_dir = Path(args.cwd or ".").expanduser().resolve()
        console.note(f"  {engine_dir}  ·  router={'off' if args.tier != 'auto' else 'auto'}  ·  verify={'on' if not args.no_verify else 'off'}")
    started = time.monotonic()
    try:
        result = asyncio.run(engine.run(task))
    finally:
        if engine.db is not None:
            engine.db.close()
    took = int((time.monotonic() - started) * 1000)
    if args.json:
        payload = result_payload(result)
        payload["wall_ms"] = took
        notices = engine.hub.drain_notices()
        if notices:
            payload["notices"] = notices
        print(json.dumps(payload, indent=2))
        return 0 if result.ok else 1
    for note in engine.hub.drain_notices():
        console.note("  ! " + note)
    console.newline()
    print("\n" + result.report())
    if not result.ok:
        print("  ⚠ unverified — read the proof lines above before trusting this answer", file=sys.stderr)
    return 0 if result.ok else 1


def cmd_repl(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    console = Console()
    print(f"{BANNER} · repl · /help for commands\n")
    engine = make_engine(args, cfg, console)
    memory = engine.memory
    print(f"  {engine.cwd}  ·  {provider_label(engine, cfg)}  ·  "
          f"subagents: {', '.join(sorted(build_roots(cfg.subagents)))}")
    try:
        while True:
            try:
                line = input("\n\x1b[36m❯\x1b[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if not repl_command(line, engine, console, memory, args):
                    break
                continue
            result = asyncio.run(engine.run(line))
            console.newline()
            print(result.report())
    finally:
        if engine.db is not None:
            engine.db.close()
    return 0


def repl_command(line: str, engine: Engine, console: Console, memory: Memory | None, args: argparse.Namespace) -> bool:
    """Returns False when the repl should exit."""
    parts = shlex.split(line)
    cmd, text = parts[0].lstrip("/"), " ".join(parts[1:])
    db = engine.db
    if cmd in {"help", "?"}:
        print(
            "  /review [note]   adversarial review of the working tree (strong tier)\n"
            "  /debt            unverified sessions + sanity scan of the tree\n"
            "  /gain            measured tokens not sent: truncation + redaction + compaction\n"
            "  /cost            spend by model / session\n"
            "  /undo [id]       roll back the last recorded file edits\n"
            "  /mem [query]     recall memory      /mem note <text>   store one\n"
            "  /skill [name]    list or run a SKILL.md\n"
            "  /sub <role> <t>  dispatch one subagent (" + ", ".join(sorted(build_roots(engine.config.subagents))) + ")\n"
            "  /keys            vault + keypool status\n"
            "  /doctor          full self-check\n"
            "  /quit            leave"
        )
    elif cmd == "review":
        files = sorted(engine.ctx.files_touched)
        payload = (V.diff_stat(engine.cwd) or "(no git diff — reviewing the files this session touched)") + "\n" + _changed_payload(engine.cwd, files)
        if not payload.strip():
            print("  nothing to review yet")
            return True
        verdict = asyncio.run(engine.hub.complete("strong", V.REVIEW_PROMPT.format(task=text or "current change set", diff=payload[:7000])))
        print("\n" + (verdict or "(provider returned nothing)").strip()[:4000])
    elif cmd == "debt":
        if db is None:
            print("  no database")
            return True
        unverified = [r for r in db.recent_sessions(limit=100) if r.status != "done"]
        print(f"  sessions not verified: {len(unverified)}")
        for row in unverified[:8]:
            print(f"    #{row.id} {row.age:>9}  {row.status:<10} {row.name[:52]}")
        tree = [str(p.relative_to(engine.cwd)) for p in engine.cwd.rglob("*")
                if p.suffix in {".py", ".ts", ".js"} and p.is_file() and p.stat().st_size < 200_000][:60]
        findings = V.sanity_scan(engine.cwd, tree)
        print(f"  sanity findings: {len(findings)}")
        for finding in findings[:10]:
            print(f"    {finding[:110]}")
    elif cmd == "gain":
        if db is None:
            print("  no database")
            return True
        rows = db.conn.execute("SELECT kind, detail FROM events WHERE kind IN ('savings','compaction','undo')").fetchall()
        savings = sum(int(json.loads(r["detail"]).get("chars", 0)) for r in rows if r["kind"] == "savings")
        redactions = sum(int(json.loads(r["detail"]).get("redactions", 0)) for r in rows if r["kind"] == "savings")
        print(f"  {savings:,} chars never reached a model ≈ {savings // 4:,} tokens avoided")
        print(f"  {redactions} secret(s) redacted · {len(rows)} tracked event(s) across {db.stats()['sessions']} session(s)")
    elif cmd == "cost":
        if db is None:
            print("  no database")
            return True
        print(cost_mod.render(db, engine.config, group=text.split()[0] if text else "model"))
    elif cmd == "undo":
        if db is None:
            print("  no database")
            return True
        sid = int(text) if text.isdigit() else engine.session_id
        restored = db.undo_last(sid)
        print("  " + (f"restored {len(restored)} file(s): " + ", ".join(restored) if restored else "nothing to undo"))
    elif cmd == "mem":
        if memory is None:
            print("  memory disabled (config: memory.enabled)")
            return True
        if text.lower().startswith("note "):
            body = text[5:].strip()
            mid = memory.remember(title=body[:70], body=body, kind="note", cwd=str(engine.cwd), tags="note")
            print(f"  stored memory #{mid}")
            return True
        hits = memory.recall(text or None, k=8, cwd=str(engine.cwd)) if text else memory.recent(8, cwd=str(engine.cwd))
        if not hits:
            print("  (memory is empty — it fills as tasks complete)")
        for hit in hits:
            print(f"  #{hit.id} [{hit.kind}] {hit.title} · sim={hit.score:.2f} · {hit.age()} ago")
            print(f"      {hit.snippet(220)}")
    elif cmd == "skill":
        skills, problems = skills_mod.discover(engine.config.skills_dirs(engine.cwd), known_tools=list(TOOLS))
        if not text:
            for skill in skills:
                print(f"  {skill.name:22} {skill.description[:56]} [{skill.tier}]")
            for problem in problems:
                print(f"  ! {problem}")
            return True
        name, _, rest = text.partition(" ")
        skill = next((s for s in skills if s.name == name), None)
        if skill is None:
            print(f"  unknown skill {name!r} · /skill lists what exists")
            return True
        saved = (dict(engine.tools), engine.settings.tier, engine.settings.max_turns, engine.settings.verify)
        engine.tools = skill.tool_subset(dict(TOOLS))
        engine.settings.tier = skill.tier
        engine.settings.max_turns = skill.max_turns
        engine.settings.verify = skill.verify != "none"
        try:
            memory_block = engine.memory.context_block(rest or skill.name, cwd=str(engine.cwd)) if engine.memory else ""
            asyncio.run(engine.run(skill.render(rest or f"run the {skill.name} skill", diff=V.diff_stat(engine.cwd), memory=memory_block)))
        finally:
            engine.tools, engine.settings.tier, engine.settings.max_turns, engine.settings.verify = saved
    elif cmd == "sub":
        if " " not in text:
            print("  usage: /sub <role> <task>")
            return True
        role, task = text.split(" ", 1)
        runner = SubAgentRunner(
            engine.hub, cwd=engine.cwd, permissions=engine.permissions, vault=engine.vault,
            roles=build_roots(engine.config.subagents), session_db=engine.db, session_id=engine.session_id,
        )
        result = asyncio.run(runner.run(role, task))
        print("\n" + result.render())
    elif cmd == "keys":
        status = engine.vault.status()
        print(f"  vault {status['path']} mode {status['mode']} · services: {', '.join(status['services']) or 'none'}")
        for tier, info in engine.hub.status().items():
            pool = info.get("pool") or {}
            print(f"  {tier}: {info.get('provider', '?')}:{info.get('model', '?')} keys {pool.get('available', 0)}/{pool.get('keys', 0)}"
                  + (f" (errors: {info['error']})" if "error" in info else ""))
    elif cmd == "doctor":
        doctor = Doctor(engine.config, cwd=engine.cwd)
        asyncio.run(doctor.run_all())
        print(doctor.report()[0])
    elif cmd in {"clear", "cls"}:
        os.system("cls" if os.name == "nt" else "clear")
    elif cmd in {"quit", "exit", "q"}:
        return False
    else:
        print(f"  unknown command /{cmd} — /help")
    return True


def _changed_payload(cwd: Path, files: list[str], limit: int = 5) -> str:
    out: list[str] = []
    for rel in files[:limit]:
        path = cwd / rel
        if path.is_file():
            out.append(f"--- {rel}\n" + path.read_text(encoding="utf-8", errors="replace")[:4000])
    return "\n".join(out)


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    doctor = Doctor(cfg, cwd=Path(args.cwd or ".").expanduser().resolve(), strict=args.strict)
    asyncio.run(doctor.run_all())
    report, failures = doctor.report()
    if args.json:
        print(json.dumps({"results": [{"name": r.name, "status": r.status, "detail": r.detail, "hint": r.hint} for r in doctor.results],
                          "failures": failures}, indent=2))
    else:
        print(report)
    return 1 if failures else 0


def cmd_bench(args: argparse.Namespace) -> int:
    scenarios = asyncio.run(bench_run_all(keep=args.keep, only=args.only))
    if args.json:
        print(bench_json(scenarios))
        return 0 if all(s.ok for s in scenarios) else 1
    report, fails = bench_render(scenarios)
    print(report)
    return 1 if fails else 0


def cmd_cost(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    db = open_db(cfg)
    if db is None:
        return 1
    try:
        print(cost_mod.as_json(db, cfg) if args.json else cost_mod.render(db, cfg, group=args.by, limit=args.limit))
    finally:
        db.close()
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    db = open_db(cfg)
    if db is None:
        return 1
    try:
        if args.show:
            row = db.session(args.show)
            if row is None:
                print(f"  no session #{args.show}", file=sys.stderr)
                return 1
            data = {
                "session": {k: getattr(row, k) for k in ("id", "name", "cwd", "model", "provider", "task", "status", "cost_usd", "tokens_in", "tokens_out", "proof", "started_at")},
                "messages": db.transcript(args.show),
                "tools": db.tool_history(args.show),
                "ledger": db.ledger(args.show),
                "undo": db.pending_undo(args.show),
            }
            print(json.dumps(data, indent=2, default=str)[: args.max * 200])
            return 0
        rows = db.recent_sessions(limit=args.limit)
        if not rows:
            print("  no sessions yet — run a task first")
            return 0
        print(f"  {'id':>4}  {'started':16}  {'status':10} {'usd':>8}  task")
        for row in rows:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row.started_at))
            print(f"  {row.id:>4}  {stamp:16}  {row.status:10} ${row.cost_usd:>7.4f}  {row.name[:56]}")
        print("\n  ma sessions --show ID · ma undo ID · ma --resume ID \"next step\"")
    finally:
        db.close()
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    db = open_db(cfg)
    if db is None:
        return 1
    try:
        if args.list:
            pending = db.pending_undo()
            if not pending:
                print("  no pending undo snapshots")
                return 0
            for row in pending:
                print(f"  #{row['id']} session {row['session_id']} {row['kind']:<5} {row['path']}")
            return 0
        restored = db.undo_last(args.session)
        if restored:
            print("  " + ", ".join(restored))
            return 0
        print("  nothing to undo")
        return 1
    finally:
        db.close()


def cmd_memory(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    db = open_db(cfg)
    if db is None:
        return 1
    memory = Memory.build(db, cfg.memory)
    try:
        if args.note:
            body = " ".join(args.note)
            mid = memory.remember(title=(args.title or body[:70]), body=body, kind=args.kind,
                                   cwd=str(Path(args.cwd).expanduser().resolve()), files=args.file or [], tags=",".join(args.tag))
            print(f"  stored memory #{mid}")
            return 0
        if args.search is not None:
            query = " ".join(args.search) or None
            hits = memory.recall(query, k=args.limit, cwd=str(Path(args.cwd).expanduser().resolve())) if query else memory.recent(args.limit)
            if not hits:
                print("  no matches")
                return 0
            for hit in hits:
                print(f"  #{hit.id} [{hit.kind}] {hit.title} · sim={hit.score:.2f} · {hit.age()} ago · cost ${hit.cost_usd:.4f}")
                if not args.brief:
                    print("      " + hit.body[:600].replace("\n", "\n      "))
            return 0
        stats = db.stats()
        if args.json:
            print(json.dumps({"stats": stats, "memories": [dict(r) for r in db.all_memories(limit=args.limit)]}, indent=2, default=str))
            return 0
        print(f"  memories={stats['memories']} · sessions={stats['sessions']} · db={stats['db']} ({stats['size_kb']}KB)")
        for row in db.all_memories(limit=args.limit):
            print(f"    #{row['id']} [{row['kind']:<7}] {row['title'][:64]}")
    finally:
        db.close()
    return 0


def cmd_skill(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    cwd = Path(args.cwd or ".").expanduser().resolve()
    dirs = cfg.skills_dirs(cwd)
    skills, problems = skills_mod.discover(dirs, known_tools=list(TOOLS))
    action = args.action or ("show" if args.name and not args.task else "list")
    if args.action == "new":
        target = skills_mod.create_skill(dirs or [cwd / "skills"], args.name, args.description or "custom skill", tools=args.tools)
        print(f"  created {target}")
        return 0
    if action == "list" or not args.name:
        if not skills:
            print(f"  no skills found in {', '.join(str(d) for d in dirs) or '(none configured)'}")
            print("  create skills/<name>/SKILL.md  ·  or: ma skill new NAME --description \"…\"")
        for skill in skills:
            mark = "✗" if skill.problems else "✓"
            print(f"  {mark} {skill.name:22} {skill.description[:56]}")
            print(f"      tools={','.join(skill.tools) or '(all)'} tier={skill.tier} max_turns={skill.max_turns} verify={skill.verify}")
            print(f"      {skill.path}")
            for problem in skill.problems:
                print(f"      ! {problem}")
        for problem in problems:
            if not any(problem in (s.problems or []) for s in skills):
                print(f"  ! {problem}")
        return 1 if (problems and args.strict) else 0
    skill = next((s for s in skills if s.name == args.name), None)
    if skill is None:
        print(f"  no skill named {args.name!r} (ma skill list)", file=sys.stderr)
        return 1
    if action == "show":
        print(skill.path.read_text(encoding="utf-8") if skill.path else skill.prompt)
        return 0
    task = " ".join(args.task) if args.task else f"run the {skill.name} skill"
    ns = argparse.Namespace(
        cwd=str(cwd), config=args.config, json=False, quiet=False, provider=None, base_url=None, model=None,
        tier=skill.tier, max_turns=skill.max_turns, max_steps=1, no_plan=True, no_verify=skill.verify == "none",
        no_review=True, no_memory=False, no_subagents=True, yes=args.yes, allow=args.allow or [], deny=args.deny or [],
        redact=True, replay_file=None, resume=None, task=[skill.render(task, diff=V.diff_stat(cwd))],
    )
    return cmd_run(ns)


def cmd_vault(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    vault = Vault.load(cfg.vault_path())
    if args.action == "set":
        if not args.value:
            print("  ma vault set <service> <value>", file=sys.stderr)
            return 2
        vault.set(args.service, args.value)
        print(f"  stored {args.service!r} in {vault.path} (mode 0600)")
        return 0
    if args.action == "rm":
        print("  removed" if vault.unset(args.service) else "  not found")
        return 0 if vault.has(args.service) is False else 1
    if args.action == "get":
        value = vault.get(args.service) or ""
        if not value:
            print(f"  no secret named {args.service!r}", file=sys.stderr)
            return 1
        print(value if args.raw else mask(value))
        return 0
    if args.action == "exec":
        if not args.command:
            print("  usage: ma vault exec <cmd> [args…]  (optionally --service name to inject only that one)", file=sys.stderr)
            return 2
        env = Vault.scrub_env() | vault.env_for([args.service] if args.service else [])
        proc = subprocess.run(list(args.command), env=env, cwd=str(Path(args.cwd or ".").expanduser().resolve()))
        return proc.returncode
    status = vault.status()
    print(f"  {status['path']}  mode {status['mode']}  {status['count']} service(s)")
    for name in status["services"]:
        print(f"    {name:22} → env {name.upper()}_API_KEY   (ma vault get {name} --raw to reveal)")
    for problem in status["problems"]:
        print(f"    ! {problem}")
    return 1 if status["problems"] else 0


def mask(value: str) -> str:
    return f"{value[:3]}…{value[-2:]} ({len(value)} chars)" if len(value) > 8 else "(short value)"


def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    target = Path(args.config).expanduser() if args.config else (cfg.path or home_dir() / "config.json")
    if target.exists() and not args.force:
        print(f"  {target} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    cfg.path = target
    written = cfg.write_template()
    home = home_dir()
    (home / "skills").mkdir(parents=True, exist_ok=True)
    print(f"  wrote {written}")
    print(f"  data dir {home} (agent.db, vault.json, skills/)")
    print("  next: ma free            (the agent already runs on free providers — no key needed)")
    print("        ma keys add openai sk-…    (only if you want your own key in front)")
    print("        ma doctor && ma bench && ma \"first task\"")
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    """Open the Textual TUI on the same config — one engine, two faces."""
    cfg = load_config(args)
    spec = cfg.tier("cheap")
    os.environ.setdefault("MHARO_MODEL", spec.model)
    os.environ.setdefault("MHARO_PROVIDER", "ollama" if "local" in spec.provider else spec.provider)
    os.environ.setdefault("MHARO_HOME", str(home_dir()))
    from mharo_tui.__main__ import main as tui_main

    forwarded = ["--cwd", str(Path(args.cwd or ".").expanduser().resolve())]
    if getattr(args, "yes", False):
        forwarded.append("-y")
    return tui_main(forwarded)



def config_target(cfg: Config) -> Path:
    return Path(cfg.path) if getattr(cfg, "path", None) else (home_dir() / "config.json")


def save_config(cfg: Config) -> Path:
    """Persist config.json with mode 0600 — after `ma keys add` it holds API keys."""
    target = config_target(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg.raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def cmd_free(args: argparse.Namespace) -> int:
    """The free-first provider ladder: what runs with no key, and what to do next."""
    from .ladder import ask_rung, build_ladder, quota_hint

    cfg = load_config(args)
    section = dict(cfg.free)
    ladder = build_ladder(cfg)
    ladder.sync_models()

    if getattr(args, "pick", None):
        name = str(args.pick).strip()
        known = {r.name for r in ladder.rungs}
        if name != "none" and name not in known:
            print(f"ma free: unknown rung {name!r} — pick from {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        free = cfg.raw.setdefault("free", {})
        free["prefer"] = [name] if name != "none" else []
        free["enabled"] = name != "none"
        if name != "none":
            cfg.raw.setdefault("tiers", {}).setdefault("cheap", {})["provider"] = "auto"
        target = save_config(cfg)
        print(f"  prefer → {name or 'off'} · free.enabled={free['enabled']} · wrote {target}")
        print('  next: ma "summarise this repo"')
        return 0

    if getattr(args, "test", None):
        forced = getattr(args, "rung", None)
        if forced:
            pairings = [r for r in ladder.candidates() if r.name == forced]
            if not pairings:
                matches = [r for r in ladder.ordered() if r.name == forced]
                if not matches:
                    print(f"ma free: no rung named {forced!r} (see `ma free`)", file=sys.stderr)
                    return 2
                pairings = [matches[0]]
            rung = pairings[0]
        else:
            rung = ladder.pick()
        if rung is None:
            print(f"  no free rung available — {ladder.soothe()}", file=sys.stderr)
            print(f"  {quota_hint(ladder)}", file=sys.stderr)
            return 1
        replies = []
        # one rung answering 429 is not the end of the story: walk the ladder
        for candidate in ([rung] + [r for r in ladder.candidates() if r.label() != rung.label()])[:4]:
            if args.json and replies:
                break
            print(f"  asking {candidate.label()} … (no key sent)")
            sys.stdout.flush()
            reply = ask_rung(candidate, args.test)
            reply["rung"] = reply.get("rung") or candidate.label()
            replies.append(reply)
            if reply.get("ok"):
                rung = candidate
                break
            ladder.mark_failure(candidate, reply.get("kind") or "throttle", error=str(reply.get("error"))[:200])
            print(f"     ↳ {str(reply.get('error'))[:110]} — trying the next rung")
            sys.stdout.flush()
        if args.json:
            print(json.dumps(replies if len(replies) > 1 else replies[0], indent=2))
            return 0 if replies[0].get("ok") or any(r.get("ok") for r in replies) else 1
        reply = next((r for r in replies if r.get("ok")), replies[-1])
        if not reply.get("ok"):
            print("\n  ✗ no free rung answered:")
            for item in replies:
                print(f"      {item.get('rung')}: {str(item.get('error'))[:120]}")
            print(f"  {quota_hint(ladder)}")
            return 1
        usage = reply.get("usage") or {}
        print("\n" + "\n".join("  " + str(line) for line in str(reply.get("text", "")).splitlines()))
        print(f"\n  ✓ {rung.label()} · {reply.get('ms')} ms · "
              f"{usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out · $0.00")
        return 0

    results = ladder.probe(completions=bool(getattr(args, "completions", False)), timeout=args.timeout) \
        if getattr(args, "probe", False) else {}
    rows = ladder.status()
    if args.json:
        print(json.dumps({
            "enabled": bool(section.get("enabled")), "prefer": section.get("prefer"), "rungs": rows,
            "probe": {name: {"ok": v.ok, "detail": v.detail, "models": list(v.models)[:12]}
                      for name, v in results.items()},
            "hint": quota_hint(ladder) if not any(r["available"] for r in rows) else "",
        }, indent=2))
        return 0
    prefer = ", ".join(section.get("prefer") or []) or "—"
    print(f"  free-first ladder · enabled={bool(section.get('enabled'))} · prefer={prefer}")
    print("  keys are optional: the rungs above answer first, your keys are the backup\n")
    print(f"  {'rung':<13}{'what':<13}{'model':<32}{'tools':<7}{'gate':<6}{'state':<18}endpoint")
    for row in rows:
        what = {"local": "local", "free-anon": "free·no key", "free-key": "free·key", "paid": "paid"}.get(row["kind"], row["kind"])
        if row.get("missing_key"):
            state = "needs a key"
        elif row.get("has_key"):
            state = "key set"
        else:
            state = "ready" if row["probed_ok"] else ("down" if row["probed_ok"] is False else "unprobed")
        if row["wait_s"]:
            state = f"cooling {int(row['wait_s'])}s"
        gate = f"{row['min_interval_s']:.0f}s" if row["min_interval_s"] else "—"
        if row["kind"] == "local" and row["probed_ok"] is False:
            state = "offline"
        host = str(row["base_url"]).split("://")[-1][:30]
        print(f"  {row['name']:<13}{what:<13}{row['model'][:31]:<32}"
              f"{'yes' if row['tools'] else 'no':<7}{gate:<6}{state:<18}{host}")
        if row["note"]:
            print(f"  {'':<13}↳ {row['note'][:100]}")
        verdict = results.get(row["name"])
        if verdict is not None:
            print(f"  {'':<13}{'✓ ' if verdict.ok else '✗ '}{verdict.detail[:110]}")
    for note in ladder.notices[-4:]:
        print(f"  ! {note}")
    ready = [r for r in rows if r["available"] and r["kind"] != "paid"]
    if not ready:
        print(f"\n  {quota_hint(ladder)}")
        return 1
    first = ready[0]
    print(f"\n  next call uses {first['name']}:{first['model']}")
    print('  try it: ma free --test "explain what this repo does"   ·   ma free --probe --json')
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    """List / add / remove provider API keys — the step after the free rungs run out."""
    from .keys import KeyPool

    cfg = load_config(args)
    action = getattr(args, "action", None) or "list"
    names = list(cfg.raw.get("providers", {}).keys())

    if action == "add":
        service = str(args.service or "").strip()
        if not service:
            print("  usage: ma keys add <service> [key]      e.g. ma keys add openai sk-…", file=sys.stderr)
            return 2
        key = str(args.key or "") or (os.environ.get(args.env or "", "") if getattr(args, "env", None) else "")
        if not key and not sys.stdin.isatty():
            key = sys.stdin.read().strip()
        if not key:
            import getpass

            key = getpass.getpass(f"  key for {service} (hidden): ").strip()
        if not key:
            print("  no key given — nothing written", file=sys.stderr)
            return 2
        entry = cfg.raw.setdefault("providers", {}).setdefault(service, {})
        keys = [str(k) for k in (entry.get("keys") or [])]
        if key in keys:
            print(f"  {service}: that key is already stored ({len(keys)} total)")
            return 0
        keys.append(key)
        entry["keys"] = keys
        entry.setdefault("keys_env", [f"MHARO_{service.upper().replace('-', '_')}_KEYS",
                                     f"{service.upper().replace('-', '_')}_API_KEY"])
        target = save_config(cfg)
        print(f"  stored a key for {service!r} in {target} (mode 0600) · {mask(key)}")
        if cfg.free.get("enabled"):
            print("  order of use: free rungs first, this key when they run out · `ma free` to see them")
        return 0

    if action == "rm":
        service = str(args.service or "").strip()
        entry = cfg.raw.get("providers", {}).get(service) or {}
        keys = [str(k) for k in (entry.get("keys") or [])]
        if not keys and not args.key:
            print(f"  {service}: no stored key", file=sys.stderr)
            return 1
        keys = [k for k in keys if k != args.key] if args.key else []
        entry["keys"] = keys
        save_config(cfg)
        print(f"  {service}: {len(keys)} key(s) left")
        return 0

    if getattr(args, "json", False):
        print(json.dumps({name: KeyPool.from_env(name, cfg.key_envs(name),
                                                  cfg.provider(name).get("keys") or []).status()
                          for name in names}, indent=2))
        return 0
    print("  provider keys (env vars + config) — with no key at all, the free ladder answers")
    seen = 0
    for name in names:
        pool = KeyPool.from_env(name, cfg.key_envs(name), cfg.provider(name).get("keys") or [])
        status = pool.status()
        states = status["detail"]
        seen += len(states)
        envs = ", ".join(cfg.key_envs(name)) or "—"
        print(f"  {name:<10}{len(states)} key(s) · env: {envs}")
        for state in states:
            if state["disabled"]:
                note = "disabled (auth)"
            elif state["cooling_down_s"] > 0:
                note = f"cooling {int(state['cooling_down_s'])}s"
            else:
                note = f"ready · {state['successes']} ok / {state['failures']} fail"
            print(f"      {state['key']:<26}{note:<24}{str(state.get('last_error') or '')[:44]}")
    if not seen:
        free = ProviderHub(cfg).free_status()
        ready = [r for r in free.get("rungs", []) if r["ready"]]
        line = ", ".join(f"{r['name']} ({r['ready']}/{r['pairings']} model(s))" for r in ready) or "none reachable"
        print(f"\n  free rungs answering right now: {line}")
    if not seen:
        print("\n  none yet — and none needed: `ma` runs on free providers first.")
        print("  when the free rungs are spent:  ma keys add openai sk-…   (see `ma free`)")
    return 0


# --------------------------------------------------------------------- parser


SUPPRESS = argparse.SUPPRESS
COMMANDS = ("doctor", "bench", "cost", "sessions", "undo", "memory", "mem", "skill", "vault",
            "free", "keys", "init", "repl", "tui")
FLAGS_WITH_VALUE = {
    "--cwd", "--config", "--provider", "--model", "--base-url", "--replay-file", "--tier",
    "--max-turns", "--max-steps", "--resume", "--allow", "--deny", "--only", "--by", "--limit",
    "--rung", "--timeout", "--test", "--pick", "--env",
    "--show", "--max", "--kind", "--title", "--tag", "--file", "--search", "--note", "--description",
    "--tools", "--action", "--session",
}


def _add_run_flags(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """The task-running flags, shared by the root parser and `ma run`.

    `suppress=True` keeps the sub-parser from clobbering values already parsed on
    the root parser (`ma --json run "task"`), while still allowing them after the
    subcommand (`ma run "task" --json`).
    """
    d: dict[str, Any] = {"default": SUPPRESS} if suppress else {}
    parser.add_argument("--provider", choices=["auto", "free", "openai", "anthropic", "local", "demo", "replay"],
                        help="auto = free ladder first, keys only when free runs out", **d)
    parser.add_argument("--model", help="override the active tier's model", **d)
    parser.add_argument("--base-url", help="override the provider endpoint", **d)
    parser.add_argument("--replay-file", help="JSON list of scripted provider events — offline, deterministic", **d)
    parser.add_argument("--tier", choices=["auto", "cheap", "strong"], **{**d, "default": d.get("default", "auto")})
    parser.add_argument("-y", "--yes", action="store_true", help="auto-approve mutating tools", **d)
    parser.add_argument("--allow", action="append", help="extra allow rule, e.g. --allow 'write_file:*.md'", **d)
    parser.add_argument("--deny", action="append", help="extra deny rule, e.g. --deny 'bash:npm publish*'", **d)
    parser.add_argument("--max-turns", type=int, **{**d, "default": d.get("default", 12)})
    parser.add_argument("--max-steps", type=int, **{**d, "default": d.get("default", 6)})
    parser.add_argument("--no-plan", action="store_true", **d)
    parser.add_argument("--no-verify", action="store_true", **d)
    parser.add_argument("--no-review", action="store_true", help="skip the peer-review pass", **d)
    parser.add_argument("--no-memory", action="store_true", **d)
    parser.add_argument("--no-subagents", action="store_true", **d)
    parser.add_argument("--no-redact", dest="redact", action="store_false", help="stop redacting secrets out of tool output",
                        **{**d, "default": d.get("default", True)})
    parser.add_argument("--resume", type=int, help="continue a stored session id", **{**d, "default": d.get("default", None)})
    parser.add_argument("--json", action="store_true", help="machine-readable result", **d)
    parser.add_argument("-q", "--quiet", action="store_true", help="do not stream", **d)


def insert_run(argv: list[str]) -> list[str]:
    """`ma "fix the tests"` is the same as `ma run "fix the tests"`."""
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token.startswith("-"):
            if token in FLAGS_WITH_VALUE and "=" not in token:
                idx += 2
            else:
                idx += 1
            continue
        if token in COMMANDS or token == "run":
            return argv
        return argv[:idx] + ["run"] + argv[idx:]
    return argv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ma",
        description="Mharo Agent — plan · act with tools · verify with proof.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  ma "make the tests pass"          one-shot task (same as: ma run ...)\n'
            '  ma --tier strong --json "..."     pinned tier, machine-readable\n'
            "  ma repl                           interactive: /review /debt /gain /undo\n"
            "  ma doctor --strict && ma bench    self-check gates"
        ),
    )
    add_shared(p)
    p.add_argument("--version", action="version", version=f"ma {__version__}")
    _add_run_flags(p)

    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run a task (the default when no subcommand is given)",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared(r)
    r.add_argument("task", nargs="*", help="the task, in plain language")
    _add_run_flags(r, suppress=True)
    r.set_defaults(func=cmd_run)

    def shared(name: str, help_text: str, **kw: Any) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_text, **kw)
        add_shared(s)
        return s

    d = shared("doctor", "self-check: python, deps, config, db, keys, tools, router, memory, skills")
    d.add_argument("--strict", action="store_true", help="warnings also count as failures")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    b = shared("bench", "run the engine benchmark (10 scenarios, offline, deterministic)")
    b.add_argument("--only", help="substring filter on scenario name")
    b.add_argument("--keep", action="store_true", help="keep /tmp/mharo-bench to inspect")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_bench)

    c = shared("cost", "cost dashboard from the SQLite ledger")
    c.add_argument("--by", choices=["model", "session", "day"], default="model")
    c.add_argument("--limit", type=int, default=12)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_cost)

    s = shared("sessions", "list stored sessions")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--show", type=int, default=None, help="dump one session as JSON")
    s.add_argument("--max", type=int, default=200, help="cap how much of --show prints")
    s.set_defaults(func=cmd_sessions)

    u = shared("undo", "roll back file edits the agent recorded")
    u.add_argument("session", nargs="?", type=int, default=None)
    u.add_argument("--list", action="store_true", help="show pending snapshots")
    u.set_defaults(func=cmd_undo)

    m = shared("memory", "inspect / write the memory store (alias: mem)")
    m.add_argument("--note", nargs="*", default=None, help="store a memory")
    m.add_argument("--search", nargs="*", default=None, help="semantic recall")
    m.add_argument("--kind", default="note")
    m.add_argument("--title", default="")
    m.add_argument("--tag", action="append", default=[])
    m.add_argument("--file", action="append", default=[])
    m.add_argument("--limit", type=int, default=10)
    m.add_argument("--brief", action="store_true", help="titles only")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_memory)

    sk = shared("skill", "list / show / run / create SKILL.md skills")
    sk.add_argument("action", nargs="?", default=None, choices=["list", "show", "run", "new"])
    sk.add_argument("name", nargs="?", default=None)
    sk.add_argument("task", nargs="*", default=[])
    sk.add_argument("--description", default="")
    sk.add_argument("--tools", nargs="*", default=None)
    sk.add_argument("--strict", action="store_true", help="invalid skills exit non-zero")
    sk.add_argument("-y", "--yes", action="store_true")
    sk.set_defaults(func=cmd_skill)

    f = shared("free", "probe and drive the free-first provider ladder (no API key needed)")
    f.add_argument("--probe", action="store_true", help="actually ask each endpoint whether it is alive")
    f.add_argument("--completions", action="store_true",
                   help="with --probe: also run one tiny completion per rung (uses real quota, slow)")
    f.add_argument("--timeout", type=float, default=8.0, help="per-endpoint probe timeout in seconds")
    f.add_argument("--test", metavar="PROMPT", help="send a real prompt to the best rung and print the answer")
    f.add_argument("--rung", help="force one rung for --test/--probe (e.g. --rung ovh)")
    f.add_argument("--pick", metavar="NAME", help="write free.prefer + tiers.cheap.provider=auto")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_free)

    k = shared("keys", "show / add / remove provider API keys")
    k.add_argument("action", nargs="?", choices=["list", "add", "rm"], default="list")
    k.add_argument("service", nargs="?", help="provider name, e.g. openai")
    k.add_argument("key", nargs="?", help="the key (omit to be prompted, or pipe it in)")
    k.add_argument("--env", help="read the key from this environment variable instead")
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_keys)

    v = shared("vault", "local secret store + redaction (0600, never echoed)")
    v.add_argument("action", nargs="?", default="list", choices=["list", "set", "get", "rm", "exec"])
    v.add_argument("service", nargs="?", default=None)
    v.add_argument("value", nargs="?", default=None)
    v.add_argument("command", nargs="*", default=None)
    v.add_argument("--comment", default="")
    v.add_argument("--raw", action="store_true", help="print the real value")
    v.set_defaults(func=cmd_vault)

    i = shared("init", "write a config template + data directories")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)

    rr = shared("repl", "interactive session with /review /debt /gain /undo …")
    _add_run_flags(rr, suppress=True)
    rr.set_defaults(func=cmd_repl)

    t = shared("tui", "open the Textual TUI on this same config")
    t.add_argument("-y", "--yes", action="store_true")
    t.set_defaults(func=cmd_tui)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(insert_run(list(sys.argv[1:] if argv is None else argv)))
    if getattr(args, "cmd", None) == "mem":                       # alias
        args.cmd = "memory"
    for name, value in (("cwd", "."), ("config", None), ("task", None)):
        if not hasattr(args, name):
            setattr(args, name, value)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    if func is not None:
        try:
            return int(func(args))
        except KeyboardInterrupt:
            print("\n  interrupted", file=sys.stderr)
            return 130
        except FileNotFoundError as exc:
            print(f"ma: missing file: {exc.filename}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"ma: {exc}", file=sys.stderr)
            return 2
    if not getattr(args, "task", None):
        parser.print_help()
        return 0
    try:
        return cmd_run(args)
    except KeyboardInterrupt:
        print("\n  interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if os.environ.get("MHARO_DEBUG"):
            raise
        print(f"\nma: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  re-run with MHARO_DEBUG=1 for the traceback; `ma doctor` covers the usual causes", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
