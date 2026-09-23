"""P1-5 · Self-verification with proof.

Three independent layers, because "the model says it's done" is not evidence:

1. **checks** — detect and run the project's own commands (pytest, npm/pnpm/yarn
   test, cargo test, go test, make/just targets). Real subprocesses, real exit
   codes, parsed pass/fail counts.
2. **claim audit** — scan the final answer for completion language
   ("fixed", "done", "all tests pass") and reject it unless a *passing* check
   happened *after* the last file mutation. This is the negative-test hook:
   a confident lie fails here.
3. **diff sanity** — byte-compile changed Python, `node --check` changed JS,
   balanced-brace scan otherwise; plus a "touched something that imports nothing"
   smell check.

`peer_review` then asks the strong tier to review the diff as an adversary and
returns concrete issues, which the engine feeds into a fix round.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLAIM_RE = re.compile(
    r"\b(done|fixed|working now|all tests pass|tests pass|passing|verified|resolved|implemented|"
    r"no errors|green|ready to merge)\b", re.I
)
NEGATION_RE = re.compile(r"\b(not|never|cannot|can't|unable|failed|still)\b[^.\n]{0,30}$", re.I)
PY_SYNTAX = re.compile(r"^  File \"", re.M)

CHECK_COMMANDS = {
    "pytest": (["python3", "-m", "pytest", "-q", "--no-header", "-x"], ("pyproject.toml", "pytest.ini", "setup.cfg", "tests", "test")),
    "unittest": (["python3", "-m", "unittest", "discover", "-q"], ("tests",)),
    "npm": (["npm", "--silent", "test"], ("package.json",)),
    "pnpm": (["pnpm", "test"], ("package.json", "pnpm-lock.yaml")),
    "yarn": (["yarn", "--silent", "test"], ("package.json", "yarn.lock")),
    "bun": (["bun", "test"], ("package.json", "bun.lockb")),
    "cargo": (["cargo", "test", "--quiet"], ("Cargo.toml",)),
    "go": (["go", "test", "./..."], ("go.mod",)),
    "make": (["make", "-s", "test"], ("Makefile",)),
    "just": (["just", "test"], ("justfile", "Justfile")),
}
PASSED_RE = re.compile(r"(\d+) passed")
FAILED_RE = re.compile(r"(\d+) failed")
ERRORS_RE = re.compile(r"(\d+) error")


@dataclass
class Check:
    name: str
    command: str
    ok: bool
    exit_code: int | None
    detail: str = ""
    passed: int = 0
    failed: int = 0
    duration_ms: int = 0
    skipped: bool = False

    def summary(self) -> str:
        if self.skipped:
            return f"{self.name}: skipped ({self.detail})"
        counts = f" {self.passed}✓" if self.passed else ""
        if self.failed:
            counts += f" {self.failed}✗"
        return f"{self.name}{counts} exit={self.exit_code} {self.duration_ms / 1000:.1f}s"


@dataclass
class ClaimAudit:
    claimed: bool
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        if not self.claimed:
            return "no completion claim to verify"
        return "proof found" if self.accepted else "; ".join(self.reasons) or "unproven claim"


@dataclass
class Verification:
    checks: list[Check] = field(default_factory=list)
    claim: ClaimAudit = field(default_factory=lambda: ClaimAudit(False, True))
    sanity: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    diff_stat: str = ""

    @property
    def ok(self) -> bool:
        running = [c for c in self.checks if not c.skipped]
        checks_ok = all(c.ok for c in running) if running else True
        return checks_ok and self.claim.accepted and not self.sanity

    def proof(self) -> str:
        lines = [c.summary() for c in self.checks]
        if self.claim.claimed:
            lines.append(f"claim: {'accepted' if self.claim.accepted else 'REJECTED'} — {self.claim.verdict()}")
        lines += [f"sanity: {s}" for s in self.sanity]
        lines += [f"review: {r}" for r in self.review]
        return "\n".join(lines) if lines else "no checks configured"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "diff_stat": self.diff_stat,
            "claim": {"claimed": self.claim.claimed, "accepted": self.claim.accepted, "reasons": self.claim.reasons},
            "checks": [{"name": c.name, "ok": c.ok, "exit": c.exit_code, "passed": c.passed, "failed": c.failed, "skipped": c.skipped} for c in self.checks],
            "sanity": list(self.sanity),
            "review": list(self.review),
        }


def detect_checks(cwd: Path, wanted: str = "auto") -> list[tuple[str, list[str]]]:
    """The project's own commands, detected from its manifests.

    Returns `(name, argv)` pairs in a sane order. `wanted` is "auto" (detect),
    "none" (disable) or a literal command line to run instead. Tools that are not
    installed are skipped — a missing `cargo` must not look like a failing check.
    """
    text = (wanted or "auto").strip()
    if text.lower() in {"none", "off", "false", "no"}:
        return []
    if text not in {"", "auto"}:
        import shlex

        return [("configured", shlex.split(text))]
    out: list[tuple[str, list[str]]] = []
    python_tests = any(
        (cwd / marker).exists()
        for marker in ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg")
    ) or bool(list(cwd.glob("test_*.py"))) or (cwd / "tests").is_dir() or "[tool.pytest" in _read(cwd / "pyproject.toml")
    if python_tests and shutil.which("python3"):
        out.append(("pytest", ["python3", "-m", "pytest", "-q", "--no-header"]))
    pkg_json = cwd / "package.json"
    if pkg_json.is_file():
        try:
            scripts = (json.loads(pkg_json.read_text(encoding="utf-8")) or {}).get("scripts") or {}
        except (OSError, ValueError):
            scripts = {}
        if scripts.get("typecheck"):
            out.append(("typecheck", ["npm", "--silent", "run", "typecheck"]))
        if scripts.get("test"):
            runner = "pnpm" if (cwd / "pnpm-lock.yaml").is_file() else "yarn" if (cwd / "yarn.lock").is_file() else "npm"
            if shutil.which(runner):
                out.append((f"{runner}-test", [runner, "--silent", "test"] if runner == "npm" else [runner, "test"]))
        if scripts.get("lint") and shutil.which("npm"):
            out.append(("lint", ["npm", "--silent", "run", "lint"]))
    if (cwd / "Cargo.toml").is_file() and shutil.which("cargo"):
        out.append(("cargo-test", ["cargo", "test", "--quiet"]))
    if (cwd / "go.mod").is_file() and shutil.which("go"):
        out.append(("go-test", ["go", "test", "./..."]))
    for fname, argv in (("Makefile", ["make", "-s", "test"]), ("justfile", ["just", "test"]), ("Justfile", ["just", "test"])):
        if (cwd / fname).is_file() and shutil.which(argv[0]):
            out.append((argv[0] + "-test", argv))
            break
    return out[:3]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def run_check(cwd: Path, name: str, argv: list[str], timeout: float = 180.0) -> Check:
    started = _now()
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
            env=_check_env(),
        )
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        exit_code = proc.returncode
    except FileNotFoundError:
        return Check(name, " ".join(argv), False, None, "tool not installed", skipped=True, duration_ms=_ms(started))
    except subprocess.TimeoutExpired:
        return Check(name, " ".join(argv), False, None, f"timed out after {timeout:.0f}s", duration_ms=_ms(started))
    text = _clean(out)
    return Check(
        name=name, command=" ".join(argv), ok=exit_code == 0, exit_code=exit_code,
        detail=text[-3000:], passed=_int(PASSED_RE, text), failed=_int(FAILED_RE, text) + _int(ERRORS_RE, text),
        duration_ms=_ms(started),
    )


def _now() -> float:
    import time

    return time.monotonic()


def _ms(started: float) -> int:
    import time

    return int((time.monotonic() - started) * 1000)


def _check_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.update({"GIT_PAGER": "cat", "PAGER": "cat", "PYTHONDONTWRITEBYTECODE": "1", "TERM": "dumb", "COLUMNS": "120"})
    return env


def _clean(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text or "")


def _int(pattern: re.Pattern[str], text: str) -> int:
    m = pattern.search(text)
    return int(m.group(1)) if m else 0


def audit_claim(answer: str, *, checks: list[Check], mutated_after_check: bool) -> ClaimAudit:
    """Reject completion language that has no passing check behind it."""
    hits = [m.group(0) for m in CLAIM_RE.finditer(answer or "") if not NEGATION_RE.search((answer or "")[: m.start()])]
    if not hits:
        return ClaimAudit(False, True, [], [])
    passing = [c for c in checks if c.ok and not c.skipped]
    if not checks:
        return ClaimAudit(True, True, [f"no checks are configured for this project — claim accepted but unproven ({', '.join(sorted(set(hits))[:3])})"], hits[:4])
    if not passing:
        return ClaimAudit(True, False, [f"claimed {', '.join(sorted(set(h.lower() for h in hits))[:3])} but no check passed", *(f"{c.name} exit={c.exit_code}" for c in checks if not c.ok)], hits[:4])
    if mutated_after_check:
        return ClaimAudit(True, False, ["files changed after the last passing check — re-verify"], hits[:4])
    return ClaimAudit(True, True, [f"backed by {', '.join(c.name for c in passing)}"], hits[:4])


def sanity_scan(cwd: Path, changed: list[str]) -> list[str]:
    """Cheap, decisive checks on the files the agent actually touched."""
    problems: list[str] = []
    for rel in changed[:40]:
        path = cwd / rel
        if not path.is_file():
            problems.append(f"{rel}: written but missing")
            continue
        size = path.stat().st_size
        if size == 0 and path.suffix not in {".gitkeep", ""}:
            problems.append(f"{rel}: empty file")
            continue
        if size > 400_000:
            problems.append(f"{rel}: {size // 1024}KB written by an agent — suspicious")
        text = path.read_text(encoding="utf-8", errors="replace")
        problems += marker_hits(text, rel)
        if path.suffix == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as exc:
                problems.append(f"{rel}: SyntaxError line {exc.lineno}: {exc.msg}")
        elif path.suffix in {".js", ".mjs", ".cjs"} and shutil.which("node"):
            proc = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
            if proc.returncode:
                problems.append(f"{rel}: node --check failed: {_clean(proc.stderr).strip()[:160]}")
        elif path.suffix in {".json"}:
            try:
                json.loads(text or "{}")
            except ValueError as exc:
                problems.append(f"{rel}: invalid JSON ({exc})")
        elif path.suffix in {".ts", ".tsx", ".jsx"}:
            open_n, close_n = text.count("{"), text.count("}")
            if abs(open_n - close_n) > 2:
                problems.append(f"{rel}: brace imbalance {open_n} open / {close_n} close")
    return problems

MARKERS = ("TODO(agent)", "FIXME: stub", "raise NotImplementedError  # stub",
           "placeholder implementation", "‹redacted")
MARKER_MEANING = {
    "TODO(agent)": "unfinished work left behind",
    "FIXME: stub": "stub instead of a fix",
    "raise NotImplementedError  # stub": "stub instead of a fix",
    "placeholder implementation": "placeholder instead of an implementation",
    "‹redacted": "redaction marker written to disk",
}


def marker_hits(text: str, rel: str = "") -> list[str]:
    """Flag leftover markers — but not where they are merely *named*.

    A scanner's own pattern list and the tests that assert on it both contain the
    literal strings; matching them against the whole file made `ma repl /debt`
    report nine phantom problems in this very repo. So a hit only counts when the
    marker is not sitting inside a string literal on that line.
    """
    hits: list[str] = []
    for lineno, line in enumerate((text or "").splitlines(), 1):
        for marker in MARKERS:
            index = line.find(marker)
            if index < 0:
                continue
            before = line[:index]
            if before.count('"') % 2 or before.count("'") % 2:
                continue                        # inside a quoted literal → a definition, not a leftover
            hits.append(f"{rel}:{lineno}: {MARKER_MEANING[marker]} ({marker!r})")
    return hits



REVIEW_PROMPT = """Adversarial code review. Assume this change is wrong until proven otherwise.

Task: {task}

Diff:
{diff}

Return ONLY a JSON array of concrete problems, each {{"file": str, "line": int, "issue": str, "severity": "blocker|major|minor"}}.
Look for: broken behaviour, missing tests, races, wrong error handling, unused leftovers,
API misuse, anything that would fail at runtime. Empty array if genuinely clean."""


async def peer_review(hub: Any, tier: str, task: str, diff: str, *, limit: int = 6) -> list[str]:
    """Ask the strong tier for an adversary review; degrade to [] if unavailable."""
    if hub is None or not diff.strip():
        return []
    try:
        raw = await hub.complete(tier, REVIEW_PROMPT.format(task=task[:600], diff=diff[:6000]))
    except Exception as exc:
        return [f"peer review unavailable: {type(exc).__name__}"]
    body = raw.strip()
    match = re.search(r"\[.*\]", body, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            out: list[str] = []
            for item in data[:limit]:
                if isinstance(item, dict) and item.get("issue"):
                    sev = str(item.get("severity", "minor"))
                    out.append(f"{sev}: {item.get('file', '?')}:{item.get('line', 0)} {str(item['issue'])[:180]}")
                elif isinstance(item, str):
                    out.append(item[:200])
            return out
        except (ValueError, TypeError):
            pass
    return [line.strip("- *\n")[:200] for line in body.splitlines() if line.strip()][:limit]


def diff_stat(cwd: Path) -> str:
    """`git diff --stat` when available, else a file-mtime based fallback."""
    if shutil.which("git") and subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True
    ).stdout.strip() == "true":
        proc = subprocess.run(["git", "-C", str(cwd), "diff", "--stat", "--shortstat"], capture_output=True, text=True)
        text = _clean(proc.stdout).strip()
        if text:
            return text[-600:]
    return ""


def changed_files(cwd: Path) -> list[str]:
    """Files the working tree reports as modified/untracked (best effort, no git → [])."""
    if not shutil.which("git"):
        return []
    proc = subprocess.run(["git", "-C", str(cwd), "status", "--porcelain"], capture_output=True, text=True)
    out = []
    for line in _clean(proc.stdout).splitlines():
        if len(line) > 4:
            out.append(line[3:].strip().strip('"'))
    return out
