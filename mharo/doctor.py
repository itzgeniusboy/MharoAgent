"""`ma doctor` — self-diagnosis where every check actually runs.

PASS / WARN / FAIL with the evidence that produced the verdict (version strings,
file modes, timings, DB row counts). `--strict` makes WARN a failure, which is
what CI should use. Exit code 1 on any FAIL.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .memory import HashingEmbedder
from .permissions import Permissions
from . import verify as V
from .providers import ProviderHub
from .router import Router
from .sessiondb import SessionDB
from .skills import discover
from .vault import Vault


@dataclass
class CheckResult:
    name: str
    status: str          # PASS | WARN | FAIL
    detail: str
    hint: str = ""

    def line(self) -> str:
        mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(self.status, "?")
        out = f" {mark} {self.status:<4} {self.name:<26} {self.detail[:96]}"
        if self.hint and self.status != "PASS":
            out += f"\n       → {self.hint}"
        return out


def _schema_of(db) -> str:
    try:
        row = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return str(row[0]) if row else "?"
    except sqlite3.Error:
        return "?"


def _run(argv: list[str], timeout: float = 6.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr).strip() else ""
    except FileNotFoundError:
        return 127, "not installed"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


class Doctor:
    def __init__(self, config: Config | None = None, *, cwd: Path | None = None, strict: bool = False) -> None:
        self.config = config or Config.load()
        self.cwd = (cwd or Path.cwd()).expanduser().resolve()
        self.strict = strict
        self.results: list[CheckResult] = []

    def add(self, *args: Any, **kw: Any) -> None:
        self.results.append(CheckResult(*args, **kw))

    async def run_all(self) -> list[CheckResult]:
        self.results.clear()
        for name, fn in self.checks():
            try:
                outcome = fn()
                if asyncio.iscoroutine(outcome):
                    outcome = await outcome
                items = [outcome] if isinstance(outcome, CheckResult) else list(outcome or [])
                self.results.extend(items)
            except Exception as exc:
                self.add(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}")
        return self.results

    def checks(self) -> list[tuple[str, Any]]:
        return [
            ("python", self.check_python),
            ("deps", self.check_deps),
            ("config", self.check_config),
            ("cwd", self.check_cwd),
            ("git", self.check_git),
            ("tools", self.check_tools),
            ("database", self.check_database),
            ("vault", self.check_vault),
            ("keys", self.check_keys),
            ("providers", self.check_providers),
            ("router", self.check_router),
            ("permissions", self.check_permissions),
            ("embedder", self.check_embedder),
            ("verify", self.check_verify),
            ("skills", self.check_skills),
            ("subagents", self.check_subagents),
            ("replay", self.check_replay_engine),
            ("cost", self.check_cost),
            ("free", self.check_free),
            ("network", self.check_network),
        ]

    # -- individual checks ------------------------------------------------
    def check_python(self) -> list[CheckResult]:
        out: list[CheckResult] = []
        out.append(CheckResult(
            "python", "PASS" if sys.version_info >= (3, 10) else "FAIL",
            f"{platform.python_version()} on {platform.system().lower()}-{platform.machine().lower()}",
            "" if sys.version_info >= (3, 10) else "Mharo needs Python 3.10+",
        ))
        if sys.version_info < (3, 11):
            out.append(CheckResult("asyncio.timeout", "WARN", "3.11+ required for timeout guards; running 3.10",
                                   "upgrade python or accept unbounded turns"))
        return out

    def check_deps(self) -> list[CheckResult]:
        out: list[CheckResult] = []
        for module, needed in (("textual", "TUI"), ("rich", "headless output"), ("httpx", "providers")):
            try:
                mod = __import__(module)
                version = getattr(mod, "__version__", "?")
                try:
                    from importlib.metadata import version as V
                    version = V(module)
                except Exception:
                    pass
                out.append(CheckResult(f"dep:{module}", "PASS", f"{version} (for {needed})"))
            except ImportError:
                out.append(CheckResult(f"dep:{module}", "FAIL", "not importable", f"pip install {module}"))
        return out

    def check_config(self) -> list[CheckResult]:
        path = self.config.path or Path("defaults")
        exists = path.exists() if path != Path("defaults") else False
        res = [CheckResult("config.file", "PASS" if exists else "WARN",
                           f"{path}" + ("" if exists else " (defaults in use — `ma init` to write one)"),
                           "" if exists else "run: ma init")]
        for problem in self.config.problems:
            res.append(CheckResult("config.parse", "WARN", problem, "fix the JSON"))
        tiers = self.config.tiers
        res.append(CheckResult("config.tiers", "PASS" if "cheap" in tiers else "WARN",
                               " · ".join(f"{t}={self.config.tier(t).provider}:{self.config.tier(t).model}" for t in tiers)))
        return res

    def check_cwd(self) -> list[CheckResult]:
        writable = os.access(self.cwd, os.W_OK)
        entries = len(list(self.cwd.iterdir())) if self.cwd.is_dir() else 0
        project = any((self.cwd / f).exists() for f in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"))
        return [
            CheckResult("cwd", "PASS" if self.cwd.is_dir() else "FAIL", str(self.cwd)),
            CheckResult("cwd.writable", "PASS" if writable else "FAIL", f"{entries} entries, {'writable' if writable else 'read-only'}",
                        "" if writable else "run inside a writable directory"),
            CheckResult("project.manifest", "PASS" if project else "WARN",
                        "manifest found" if project else "no pyproject/package.json/Cargo.toml/go.mod",
                        "verify checks are auto-detected from these files"),
        ]

    def check_git(self) -> list[CheckResult]:
        code, line = _run(["git", "--version"])
        if code != 0:
            return [CheckResult("git", "WARN", line, "install git for branch/diff features (they degrade cleanly)")]
        inside = subprocess.run(["git", "-C", str(self.cwd), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True).stdout.strip()
        if inside != "true":
            return [CheckResult("git", "WARN", "git present, cwd not a repo", "`git init` to enable diff + undo against VCS")]
        branch = _run(["git", "-C", str(self.cwd), "rev-parse", "--abbrev-ref", "HEAD"])[1]
        dirty = len(subprocess.run(["git", "-C", str(self.cwd), "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines())
        return [CheckResult("git", "PASS", f"{line.split()[-1]} · branch {branch or '?'} · {dirty} dirty")]

    def check_tools(self) -> list[CheckResult]:
        out: list[CheckResult] = []
        for tool, why in (("rg", "fast search"), ("pytest", "python checks"), ("node", "js checks"),
                          ("npm", "js checks"), ("cargo", "rust checks"), ("go", "go checks"), ("make", "make targets")):
            found = shutil.which(tool)
            out.append(CheckResult(f"tool:{tool}", "PASS" if found else "WARN",
                                   found or f"missing ({why} will be skipped)"))
        return out

    def check_database(self) -> list[CheckResult]:
        path = self.config.db_path()
        try:
            db = SessionDB(path)
        except sqlite3.Error as exc:
            return [CheckResult("database", "FAIL", f"cannot open {path}: {exc}", "check disk / permissions")]
        stats = db.stats()
        row = db.conn.execute("PRAGMA integrity_check").fetchone()
        integrity = row[0] if row else "unknown"
        journal = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        out = [
            CheckResult("database", "PASS" if integrity == "ok" else "FAIL",
                        f"{stats['db']} · {stats['size_kb']}KB · sessions={stats['sessions']} memories={stats['memories']} undo={stats['undoable']}"),
            CheckResult("database.integrity", "PASS" if integrity == "ok" else "FAIL", f"integrity_check={integrity}, journal={journal}"),
            CheckResult("database.schema", "PASS" if stats["schema"] else "WARN", f"schema={_schema_of(db)}"),
        ]
        db.close()
        return out

    def check_vault(self) -> list[CheckResult]:
        path = self.config.vault_path()
        vault = Vault.load(path)
        info = vault.status()
        mode = info["mode"]
        out: list[CheckResult] = []
        if not path.exists():
            out.append(CheckResult("vault", "WARN", f"{path} not created yet", "ma vault set openai sk-… (writes 0600)"))
        else:
            ok = mode.endswith("600")
            out.append(CheckResult("vault", "PASS" if ok else "FAIL", f"{path} mode {mode}, {info['count']} service(s): {', '.join(info['services']) or 'none'}",
                                   "" if ok else f"chmod 600 {path}"))
        for problem in info["problems"]:
            out.append(CheckResult("vault.perm", "FAIL", problem, f"chmod 600 {path}"))
        probe, hits = vault.redact("token sk-abcdefghij1234 and ghp_" + "x" * 24)
        out.append(CheckResult("vault.redaction", "PASS" if hits >= 2 and "sk-abcdefghij1234" not in probe else "FAIL",
                               f"probe redacted {hits} secret(s)"))
        env_probe = Vault.scrub_env({"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "nope", "HOME": "/root"})
        out.append(CheckResult("vault.env-scrub", "PASS" if "AWS_SECRET_ACCESS_KEY" not in env_probe and "PATH" in env_probe else "FAIL",
                               f"{len(env_probe)} of 3 parent env vars survive the scrub"))
        return out

    def check_keys(self) -> list[CheckResult]:
        hub = ProviderHub(self.config)
        out: list[CheckResult] = []
        for tier, info in hub.status().items():
            pool = info.get("pool") or {}
            if "error" in info:
                out.append(CheckResult(f"keys.{tier}", "WARN", info["error"], "set a key or use --replay"))
                continue
            n, avail = pool.get("keys", 0), pool.get("available", 0)
            if n == 0:
                hint = "add a key: `ma keys add openai sk-…` — or run on free providers: `ma free`"
                out.append(CheckResult(f"keys.{tier}", "WARN",
                                       f"{info['provider']} has 0 keys in {self.config.key_envs(info['provider']) or 'env'}"
                                       + (" · free ladder covers this tier" if pool.get("kind") == "ladder" else ""),
                                       hint))
                continue
            status = "PASS" if avail else "FAIL"
            detail = f"{info['provider']}:{info['model']} · {avail}/{n} keys ready"
            if pool.get("disabled"):
                detail += f", {pool['disabled']} disabled(auth)"
            if pool.get("cooling"):
                detail += f", {pool['cooling']} cooling ({pool['retry_in_s']}s)"
            out.append(CheckResult(f"keys.{tier}", status, detail, "" if avail else "all keys throttled/invalid"))
            for k in pool.get("detail", [])[:4]:
                if k["disabled"]:
                    out.append(CheckResult(f"keys.{tier}.{k['key']}", "FAIL", f"disabled: {k['last_error'][:70]}", "rotate or fix this key"))
        return out or [CheckResult("keys", "WARN", "no tiers configured")]

    def check_free(self) -> list[CheckResult]:
        """Free-first ladder: can this box run the agent with no API key at all?"""
        from .ladder import build_ladder, quota_hint

        section = dict(self.config.free)
        out: list[CheckResult] = []
        if not section.get("enabled", True):
            return [CheckResult("free.enabled", "WARN", "free ladder disabled — a key is required to run",
                                "set free.enabled=true (or drop the key) to run free-first")]
        out.append(CheckResult("free.enabled", "PASS",
                               f"on · prefer={', '.join(section.get('prefer') or []) or '—'} · "
                               f"paid-keys-at-end={bool(section.get('allow_paid'))} · "
                               f"demo-fallback={bool(section.get('fallback_to_demo'))}"))
        try:
            ladder = build_ladder(self.config)
        except Exception as exc:  # noqa: BLE001 - a broken rung spec must be reported, not crash the doctor
            return out + [CheckResult("free.ladder", "FAIL", f"could not build ladder: {type(exc).__name__}: {exc}",
                                      "fix free.extra_rungs / free.rungs in config")]
        ladder.sync_models()
        rungs = ladder.ordered()
        keyless = sorted({r.name for r in rungs if r.keyless})
        with_key = [r for r in rungs if not r.keyless and r.key()]
        pairings = ladder.candidates()
        live = [r for r in pairings if r.available()]
        out.append(CheckResult(
            "free.ladder", "PASS" if live else "WARN",
            f"{len(pairings)} pairing(s) on {len(ladder.hosts())} host(s) · {len(live)} ready now · "
            + (f"{len(keyless)} host(s) need no key · " if keyless else "")
            + (ladder.summary()[:120] if pairings else ladder.soothe()),
            "" if live else quota_hint(ladder)))
        if with_key:
            out.append(CheckResult("free.keys-ahead", "PASS",
                                   f"{len(with_key)} free-with-key rung(s) ready when anonymous ones run dry: "
                                   + ", ".join(sorted({r.name for r in with_key}))))
        # a rate gate that is *below* the documented floor turns a free tier into a 429 loop
        from mharo_tui.agent.ladder import RATE_FLOORS

        hosts = {r.name: r for r in ladder.ordered()}
        thin = [
            f"{name}={rung.min_interval_s:.0f}s<[{RATE_FLOORS[name]:g}]"
            for name, rung in hosts.items()
            if name in RATE_FLOORS and 0 < rung.min_interval_s < RATE_FLOORS[name]
        ]
        unguarded = [name for name, rung in hosts.items()
                     if rung.kind in {"free-anon", "free-key"} and rung.min_interval_s <= 0]
        if thin or unguarded:
            out.append(CheckResult("free.rate-gate", "FAIL",
                                   "gates below the tier's own limit: " + ", ".join(thin + unguarded),
                                   "raise free.min_interval_s so we do not hammer a free tier"))
        else:
            gated = {}
            for r in rungs:
                if r.min_interval_s > 0:
                    gated.setdefault(r.name, r.min_interval_s)
            out.append(CheckResult("free.rate-gate", "PASS",
                                   f"{len(gated)} host(s) rate-gated across {len(rungs)} pairing(s) · "
                                   + ", ".join(f"{name}≥{sec:.0f}s" for name, sec in list(gated.items())[:4])
                                   + " · per-model rotation on shared hosts"))
        from .keys import KeyPool

        any_key = any(KeyPool.from_env(name, self.config.key_envs(name),
                                       self.config.provider(name).get("keys") or []).keys
                      for name in self.config.raw.get("providers", {}))
        if not any_key and live:
            out.append(CheckResult("free.keyless", "PASS",
                                   "no API key configured anywhere and the agent still runs (free rungs first)",
                                   "when free quota runs out: `ma keys add openai sk-…`"))
        elif not any_key:
            out.append(CheckResult("free.keyless", "WARN",
                                   "no key and no reachable free rung — tasks will fall back to the offline demo",
                                   quota_hint(ladder)))
        else:
            out.append(CheckResult("free.keyless", "PASS", "keys configured; free rungs still go first when present"))
        for note in ladder.notices[-3:]:
            out.append(CheckResult("free.notices", "WARN", note, "transient: the ladder already backed that rung off"))
        if any(r.wait_s() > 0 for r in rungs):
            cooling = [r for r in rungs if r.wait_s() > 0]
            out.append(CheckResult("free.cooling", "WARN",
                                   ", ".join(f"{r.label()} in {int(r.wait_s())}s" for r in cooling[:4]),
                                   "wait, add a key, or start a local model (`ollama serve`)"))
        return out

    async def check_providers(self) -> list[CheckResult]:
        """Live reachability probe — only when a key exists, so it never hangs offline."""
        out: list[CheckResult] = []
        for tier, info in ProviderHub(self.config).status().items():
            pool = (info.get("pool") or {})
            if not pool.get("available"):
                out.append(CheckResult(f"net.{tier}", "WARN", "skipped (no usable key)", "add a key to test the live endpoint"))
                continue
            provider_name = info.get("provider", "")
            base = self.config.provider(provider_name).get("base_url") or ""
            if pool.get("kind") == "ladder":
                # the ladder has no single endpoint: probe the rung we would actually use
                from .ladder import build_ladder

                rung = build_ladder(self.config).pick()
                base = rung.base_url if rung is not None else ""
                if not base:
                    out.append(CheckResult(f"net.{tier}", "WARN", "no free rung reachable to probe",
                                           "ma free · or start Ollama / add a key"))
                    continue
            code, line = await asyncio.get_running_loop().run_in_executor(None, _run, ["curl", "-s", "-o", "/dev/null",
                                                                                        "-w", "%{http_code} %{time_total}s",
                                                                                        "--max-time", "6", base], 8.0)
            ok = code == 0 and line.split(" ")[0] in {"200", "401", "403", "404", "301", "302"}
            out.append(CheckResult(f"net.{tier}", "PASS" if ok else "WARN",
                                   f"{base} → {line or 'no response'}" if line else f"{base} unreachable"))
        return out or [CheckResult("net", "WARN", "no endpoints configured")]

    def check_router(self) -> list[CheckResult]:
        router = Router(tiers=self.config.tiers, budget_usd=float(self.config.budget.get("max_session_usd", 2.0)))
        cheap = router.pick("fix typo in readme")
        strong = router.pick("refactor the auth middleware across the repo")
        up = router.should_upgrade(tier="cheap", failures=3, files_touched=1, diff_lines=10, verify_failed=True, spent_usd=0.1)
        guard = router.should_upgrade(tier="cheap", failures=3, files_touched=1, diff_lines=10, verify_failed=True, spent_usd=float(router.budget_usd))
        ok = cheap.tier == "cheap" and strong.tier == "strong" and up.tier == "strong" and guard.tier == "cheap"
        return [CheckResult("router", "PASS" if ok else "FAIL",
                            f"routine→{cheap.tier}, hard→{strong.tier}, upgrade→{up.tier}, budget-guard→{guard.tier}")]

    def check_permissions(self) -> list[CheckResult]:
        perms = Permissions.from_config(self.config.permissions)
        allow = perms.decide("read_file", {"path": "a.py"})
        deny = perms.decide("bash", {"command": "sudo rm -rf /"})
        ask = perms.decide("write_file", {"path": "b.py"})
        ok = allow.allowed and deny.blocked and ask.action == "ask"
        return [CheckResult("permissions", "PASS" if ok else "FAIL",
                            f"read_file→{allow.action}, sudo→{deny.action}({deny.rule}), write_file→{ask.action}; default={perms.default}",
                            "" if ok else "check permissions.allow/deny rules in config")]

    def check_embedder(self) -> list[CheckResult]:
        emb = HashingEmbedder(int(self.config.memory.get("dims", 256)))
        a = emb.vector("refactor the sqlite session store and add undo")
        b = emb.vector("refactor the sqlite session store to support undo hooks")
        c = emb.vector("bake a chocolate cake with ganache layers")
        sim_rel, sim_unrel = HashingEmbedder.cosine(a, b), HashingEmbedder.cosine(a, c)
        packed = emb.unpack(emb.pack(a))
        ok = sim_rel > sim_unrel and sim_rel > 0.4 and abs(sum(x * x for x in a) - 1.0) < 1e-3 and len(packed) == len(a)
        return [CheckResult("memory.embed", "PASS" if ok else "FAIL",
                             f"dims={emb.dims} | related-sim={sim_rel:.3f} vs unrelated={sim_unrel:.3f} | unit-norm ✓",
                             "" if ok else "embedder scoring looks off")]

    def check_verify(self) -> list[CheckResult]:
        found = V.detect_checks(self.cwd, "auto")
        names = ", ".join(n for n, _ in found) or "none detected"
        claim = V.audit_claim("all tests pass, done", checks=[], mutated_after_check=False)
        out = [CheckResult("verify.checks", "PASS" if found else "WARN", f"auto-detected: {names}",
                           "add a test script or pytest config to enable gating")]
        out.append(CheckResult("verify.claim-audit", "PASS" if claim.claimed else "FAIL",
                               f"unproven claim flagged: {claim.verdict()[:70]}"))
        return out

    def check_skills(self) -> list[CheckResult]:
        from mharo_tui.agent.tools import TOOLS

        dirs = self.config.skills_dirs(self.cwd)
        skills, problems = discover(dirs, known_tools=list(TOOLS))
        out = [CheckResult("skills", "PASS" if skills else "WARN",
                           f"{len(skills)} skill(s) from {len(dirs)} dir(s): " + (", ".join(s.name for s in skills[:6]) or "none"),
                           "create skills/<name>/SKILL.md or `ma skill new <name>`")]
        for problem in problems[:6]:
            out.append(CheckResult("skills.validate", "FAIL" if "not in the registry" in problem else "WARN", problem))
        return out

    def check_subagents(self) -> list[CheckResult]:
        from mharo.subagents import build_roots
        from mharo_tui.agent.tools import TOOLS

        roots = build_roots(self.config.subagents)
        out: list[CheckResult] = []
        for role, agent in sorted(roots.items()):
            bad = [t for t in agent.tools if t not in TOOLS]
            writer = any(t in {"write_file", "edit_file"} for t in agent.tools)
            if bad:
                out.append(CheckResult(f"subagent.{role}", "FAIL", f"unknown tools: {', '.join(bad)}"))
            else:
                out.append(CheckResult(f"subagent.{role}", "PASS",
                                       f"{len(agent.tools)} tools ({', '.join(agent.tools)}) tier={agent.tier} writes={'yes' if writer else 'no'}"))
        reader = roots.get("reader")
        if reader is not None:
            isolated = not any(t in {"bash", "write_file", "edit_file"} for t in reader.tools)
            out.append(CheckResult("subagent.isolation", "PASS" if isolated else "FAIL",
                                   "reader cannot execute or write" if isolated else "reader has mutating tools!"))
        return out or [CheckResult("subagent", "WARN", "no subagents configured")]

    async def check_replay_engine(self) -> list[CheckResult]:
        """End-to-end: engine + router + tools + verify + db, offline and deterministic."""
        from .engine import Engine, EngineSettings
        from .providers import ProviderHub

        tmp = Path("/tmp/mharo-doctor")
        tmp.mkdir(exist_ok=True)
        (tmp / "hello.py").write_text("def hi():\n    return 'hi'\n", encoding="utf-8")
        cfg = Config.load()
        db = SessionDB(tmp / "doctor.db")
        hub = ProviderHub(cfg, db=db, replay_scripts={"all": [{"text": "Read the file."}, {"text": "Verified: function returns 'hi'."}]})
        hub.session_id = None
        engine = Engine(cfg, cwd=tmp, db=db, hub=hub, vault=Vault.load(tmp / "vault.json"),
                        settings=EngineSettings(tier="auto", max_turns=2, plan=False, peer_review=False, use_memory=True))
        started = time.monotonic()
        result = await engine.run("explain what hello.py does")
        ms = int((time.monotonic() - started) * 1000)
        out = [CheckResult("engine.replay", "PASS" if result.answer else "FAIL",
                           f"turn ran in {ms}ms · answer {len(result.answer)} chars · steps={len(result.steps)} · cost ${result.cost_usd:.4f}")]
        try:
            stats = db.stats()
            out.append(CheckResult("engine.persistence", "PASS" if stats["sessions"] else "FAIL",
                                   f"db wrote {stats['sessions']} session(s), {stats['messages']} message(s), {stats['tool_calls']} tool call(s)"))
        finally:
            db.close()
        return out

    def check_cost(self) -> list[CheckResult]:
        prices = self.config.prices()
        unknown = [m for m in (self.config.tier(t).model for t in self.config.tiers) if m not in prices and not any(m.startswith(k) for k in prices)]
        return [CheckResult("cost.prices", "PASS" if not unknown else "WARN",
                           f"{len(prices)} priced model(s)" + (f"; unmapped tiers: {', '.join(unknown)} (billed $0)" if unknown else ""),
                           "set cost.prices in config for your models")]

    def check_network(self) -> list[CheckResult]:
        proxy = self.config.proxy
        active = {k: v for k, v in proxy.items() if v}
        env_proxy = {k: v for k, v in os.environ.items() if k.lower().endswith("_proxy")}
        return [CheckResult("net.proxy", "PASS", f"config {active or 'none'} · env {sorted(env_proxy) or 'none'}")]

    # -- rendering --------------------------------------------------------
    def report(self) -> tuple[str, int]:
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for res in self.results:
            counts[res.status] = counts.get(res.status, 0) + 1
        lines = [
            f"Mharo Agent doctor · python {platform.python_version()} · {self.cwd}",
            "─" * 96,
        ]
        lines += [r.line() for r in self.results]
        lines.append("─" * 96)
        verdict = "healthy" if not counts["FAIL"] else "needs attention"
        lines.append(f" {counts['PASS']} pass · {counts['WARN']} warn · {counts['FAIL']} fail → {verdict}")
        fail = counts["FAIL"] + (counts["WARN"] if self.strict else 0)
        return "\n".join(lines), fail
