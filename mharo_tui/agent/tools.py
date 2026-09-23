"""Tool layer: what the agent can actually *do* in your repo.

Every tool is declared once, with a JSON schema (for providers that support
native function calling) plus a human-readable preview (for the approval
prompt and the transcript). Execution is synchronous and runs inside a worker
thread, so the TUI never blocks on a build or a test run.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", ".nuxt", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target",
    ".cache", "coverage", ".turbo", ".idea", ".vscode",
}
MAX_OUTPUT = 24_000


class ToolError(Exception):
    """Raised by a tool for an expected, model-visible failure."""


@dataclass
class ToolResult:
    ok: bool
    output: str
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def diff(self) -> str | None:
        return self.meta.get("diff")


@dataclass
class ToolContext:
    cwd: Path
    allow_outside: bool = False
    timeout: float = 120.0
    files_touched: dict[str, int] = field(default_factory=dict)
    env_extra: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    title: str
    icon: str
    description: str
    params: dict[str, Any]
    run: Callable[..., str]
    approval: bool = False
    read_only: bool = True
    danger: str | None = None

    def schema(self) -> dict[str, Any]:
        required = [k for k in self.params if not k.startswith("_")]
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {k: v for k, v in self.params.items() if not k.startswith("_")},
                "required": required[:1],
            },
        }


# --------------------------------------------------------------------------- helpers


def _resolve(ctx: ToolContext, path: str) -> tuple[Path, bool]:
    """Return (absolute path, outside_of_cwd)."""
    raw = (path or ".").strip().strip("'\"")
    candidate = Path(os.path.expanduser(raw))
    full = (candidate if candidate.is_absolute() else ctx.cwd / candidate).resolve()
    try:
        full.relative_to(ctx.cwd.resolve())
        inside = True
    except ValueError:
        inside = False
    if not inside and not ctx.allow_outside:
        raise ToolError(
            f"refused: {path} resolves outside the working dir ({ctx.cwd}). "
            "Set MHARO_ALLOW_OUTSIDE_CWD=1 to allow."
        )
    return full, not inside


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2):]
    missing = len(text) - len(head) - len(tail)
    return f"{head}\n\n… {missing:,} chars omitted …\n\n{tail}"


def _rel(ctx: ToolContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.cwd.resolve()))
    except ValueError:
        return str(path)


def _diff(before: str, after: str, name: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=name, tofile=name, lineterm="", n=1,
        )
    )


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def _skip_name(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith((".pyc", ".pyo", ".lock")) or name == ".DS_Store"


# --------------------------------------------------------------------------- tools


def tool_bash(ctx: ToolContext, command: str = "", timeout: float | None = None, **_: Any) -> str:
    command = (command or "").strip()
    if not command:
        raise ToolError("bash: empty command")
    blocked = _looks_destructive(command)
    if blocked:
        raise ToolError(f"bash: refused — {blocked}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MHARO_SECRET")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "GIT_PAGER": "cat", "PAGER": "cat", "TERM": "dumb"})
    env.update(ctx.env_extra)
    started = time.time()
    proc = subprocess.run(
        [os.environ.get("SHELL", "/bin/sh"), "-lc", command],
        cwd=str(ctx.cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout or ctx.timeout,
    )
    out = proc.stdout or ""
    if proc.stderr:
        out += (("\n" if out else "") + "stderr:\n" + proc.stderr)
    tail = f"\n\n[exit {proc.returncode} · {int((time.time() - started) * 1000)} ms]"
    # a hard cap keeps a megabyte of stdout out of memory; `run_tool` does the
    # display truncation so callers can measure how much was trimmed
    body = out.strip()[: MAX_OUTPUT * 4] or "(no output)"
    return body + tail


def _looks_destructive(command: str) -> str | None:
    patterns = {
        r"rm\s+-[a-z]*r[a-z]*f\s+/(\s|$)": "recursive force-delete of /",
        r":\(\)\s*\{.*\};\s*:": "fork bomb",
        r"\bmkfs\b": "filesystem format",
        r"\bdd\b.*\bof=/dev/": "raw device write",
        r">\s*/dev/sd[a-z]\b": "raw device write",
    }
    for pattern, why in patterns.items():
        if re.search(pattern, command):
            return why
    return None


def tool_read(ctx: ToolContext, path: str = "", offset: int = 1, limit: int = 400, **_: Any) -> str:
    full, _ = _resolve(ctx, path)
    if not full.exists():
        raise ToolError(f"read: no such file: {_rel(ctx, full)}")
    if full.is_dir():
        raise ToolError(f"read: {_rel(ctx, full)} is a directory — use list_dir")
    if full.stat().st_size > 3_000_000:
        raise ToolError(f"read: {_rel(ctx, full)} is >3 MB, refusing to dump it")
    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(offset)) - 1
    count = max(1, int(limit))
    chunk = lines[start: start + count]
    width = len(str(start + len(chunk)))
    numbered = "\n".join(f"{start + i + 1:>{width}} | {ln}" for i, ln in enumerate(chunk))
    more = ""
    if start + count < len(lines):
        more = f"\n\n[{len(lines) - start - count:,} more lines — pass offset={start + count + 1}]"
    return f"{_rel(ctx, full)}  ({len(lines):,} lines)\n{numbered}{more}"


def tool_list(ctx: ToolContext, path: str = ".", depth: int = 2, **_: Any) -> str:
    full, _ = _resolve(ctx, path or ".")
    if not full.is_dir():
        raise ToolError(f"list_dir: not a directory: {path}")
    max_depth = max(1, min(int(depth), 5))
    rows: list[str] = []
    for root, dirs, files in os.walk(full):
        rel_root = Path(root)
        level = 0 if rel_root == full else len(rel_root.relative_to(full).parts)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not _skip_name(d))
        if level >= max_depth:
            dirs[:] = []
        indent = "  " * level
        if level:
            rows.append(f"{indent}{rel_root.name}/")
        pad = "  " * (level + 1)
        for f in sorted(files)[:60]:
            if _skip_name(f):
                continue
            size = (rel_root / f).stat().st_size
            rows.append(f"{pad}{f}  {_human(size)}")
        if len(files) > 60:
            rows.append(f"{pad}… {len(files) - 60} more files")
        if len(rows) > 500:
            rows.append("… (truncated — narrow with `path`)")
            break
    return "\n".join(rows) or "(empty)"


def tool_write(ctx: ToolContext, path: str = "", content: str = "", **_: Any) -> str:
    if not path:
        raise ToolError("write_file: `path` is required")
    full, _ = _resolve(ctx, path)
    before = full.read_text(encoding="utf-8", errors="replace") if full.exists() else ""
    body = content if (content.endswith("\n") or not content) else content + "\n"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body, encoding="utf-8")
    rel = _rel(ctx, full)
    ctx.files_touched[rel] = len(body.splitlines())
    diff = _diff(before, body, rel) or "(no textual change)"
    action = "updated" if before else "created"
    return f"{action} {rel} ({len(body.splitlines()):,} lines)\n\n{diff}"


def tool_edit(
    ctx: ToolContext,
    path: str = "",
    old_text: str = "",
    new_text: str = "",
    replace_all: bool = False,
    **_: Any,
) -> str:
    full, _ = _resolve(ctx, path)
    if not full.exists():
        raise ToolError(f"edit_file: no such file: {_rel(ctx, full)}")
    if not old_text:
        raise ToolError("edit_file: `old_text` must not be empty")
    before = full.read_text(encoding="utf-8", errors="replace")
    hits = before.count(old_text)
    if hits == 0:
        sample = "\n".join(before.splitlines()[:12])
        raise ToolError(
            f"edit_file: `old_text` not found in {_rel(ctx, full)}. Re-read the file "
            f"(whitespace matters). First lines:\n{sample}"
        )
    if hits > 1 and not replace_all:
        raise ToolError(
            f"edit_file: `old_text` matches {hits} places in {_rel(ctx, full)}. "
            "Add surrounding context or pass replace_all=true."
        )
    after = (
        before.replace(old_text, new_text)
        if replace_all
        else before.replace(old_text, new_text, 1)
    )
    full.write_text(after, encoding="utf-8")
    rel = _rel(ctx, full)
    ctx.files_touched[rel] = len(after.splitlines())
    diff = _diff(before, after, rel)
    return f"patched {rel} ({hits} site{'s' if hits > 1 else ''})\n\n{diff}"


def tool_search(
    ctx: ToolContext,
    pattern: str = "",
    path: str = ".",
    glob: str = "",
    max_hits: int = 60,
    **_: Any,
) -> str:
    if not pattern:
        raise ToolError("search: `pattern` is required")
    try:
        rx = re.compile(pattern)
        literal = None
    except re.error:
        literal = pattern.lower()
        rx = None
    root, _ = _resolve(ctx, path or ".")
    exts = _globs(glob)
    rows: list[str] = []
    scanned = 0
    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir():
            continue
        rel = _rel(ctx, candidate)
        if any(part in SKIP_DIRS for part in Path(rel).parts):
            continue
        if exts and candidate.suffix.lower() not in exts:
            continue
        try:
            if candidate.stat().st_size > 1_500_000:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):
            continue
        scanned += 1
        for no, line in enumerate(text.splitlines(), 1):
            hit = rx.search(line) if rx else literal in line.lower()
            if hit:
                rows.append(f"{rel}:{no}: {line.strip()[:180]}")
                if len(rows) >= max(1, int(max_hits)):
                    return "\n".join(rows) + f"\n\n[stopped at {max_hits} hits]"
    head = f"{len(rows)} match{'es' if rows != 1 else ''} for {pattern!r} in {scanned} file(s)"
    return "\n".join([head, *rows]) if rows else head


def _globs(glob: str) -> set[str]:
    if not glob:
        return set()
    return {g.strip() if g.strip().startswith(".") else f".{g.strip()}" for g in glob.split(",")}


# --------------------------------------------------------------------------- registry


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "bash", "Run shell command", "",
            "Execute a shell command in the project working directory. Returns stdout+stderr "
            "and the exit code. Use for builds, tests, git, package managers.",
            {
                "command": {"type": "string", "description": "Shell command, e.g. 'npm test'"},
                "timeout": {"type": "number", "description": "Seconds before the command is killed"},
            },
            tool_bash,
            approval=True,
            read_only=False,
            danger="Runs arbitrary commands",
        ),
        ToolSpec(
            "read_file", "Read file", "",
            "Read a text file with 1-based line numbers. Supports offset/limit paging.",
            {
                "path": {"type": "string", "description": "Path relative to the working directory"},
                "offset": {"type": "integer", "description": "First line to read (default 1)"},
                "limit": {"type": "integer", "description": "How many lines (default 400)"},
            },
            tool_read,
            read_only=True,
        ),
        ToolSpec(
            "list_dir", "List directory", "",
            "Print a compact recursive tree with file sizes, skipping vendored dirs.",
            {
                "path": {"type": "string", "description": "Directory (default '.')"},
                "depth": {"type": "integer", "description": "How deep (default 2)"},
            },
            tool_list,
            read_only=True,
        ),
        ToolSpec(
            "search", "Search code", "",
            "Regex (or literal) search across the project with path:line output.",
            {
                "pattern": {"type": "string", "description": "Regex or literal text"},
                "path": {"type": "string", "description": "Subdirectory to search"},
                "glob": {"type": "string", "description": "Comma separated extensions, e.g. py,ts"},
            },
            tool_search,
            read_only=True,
        ),
        ToolSpec(
            "write_file", "Write file", "",
            "Create or overwrite a file with the exact content provided. Always shows a diff.",
            {
                "path": {"type": "string", "description": "Target path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            tool_write,
            approval=True,
            read_only=False,
        ),
        ToolSpec(
            "edit_file", "Edit file", "",
            "Surgical find/replace. old_text must match exactly once unless replace_all is set.",
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            tool_edit,
            approval=True,
            read_only=False,
        ),
    ]
}


def preview(name: str, args: dict[str, Any]) -> str:
    """One-line human summary for the transcript and the approval prompt."""
    args = args or {}
    if name == "bash":
        return f"$ {str(args.get('command', '')).strip()[:300]}"
    if name == "read_file":
        extra = ""
        if args.get("offset") or args.get("limit"):
            extra = f"  @L{args.get('offset', 1)}+{args.get('limit', 400)}"
        return f"read {args.get('path', '?')}{extra}"
    if name == "list_dir":
        return f"tree {args.get('path', '.')}"
    if name == "search":
        return f"grep /{args.get('pattern', '')}/ {args.get('path', '.')}"
    if name == "write_file":
        lines = len(str(args.get("content", "")).splitlines())
        return f"write {args.get('path', '?')} ({lines} lines)"
    if name == "edit_file":
        return f"edit {args.get('path', '?')}"
    spec = TOOLS.get(name)
    if spec is None:
        return f"{name} {json.dumps(args, ensure_ascii=False)[:120]}"
    return spec.title


def run_tool(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    extra: dict[str, ToolSpec] | None = None,
) -> ToolResult:
    table = dict(TOOLS)
    table.update(extra or {})
    spec = table.get(name)
    if spec is None:
        return ToolResult(False, f"unknown tool {name!r}. Available: {', '.join(sorted(table))}")
    clean = {k: v for k, v in (args or {}).items() if not k.startswith("_")}
    started = time.time()
    try:
        out = spec.run(ctx, **clean)
    except ToolError as exc:
        return ToolResult(False, str(exc), _ms(started))
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"{name}: timed out after {ctx.timeout:.0f}s", _ms(started))
    except Exception as exc:  # a tool blowing up must not kill the session
        return ToolResult(False, f"{name} crashed: {type(exc).__name__}: {exc}", _ms(started))
    full = out or "(no output)"
    shown = _truncate(full)
    # raw_chars lets callers report how much output truncation actually saved
    return ToolResult(True, shown, _ms(started), {"tool": name, "raw_chars": len(full)})


def _ms(started: float) -> int:
    return int((time.time() - started) * 1000)


def tool_schemas() -> list[dict[str, Any]]:
    return [spec.schema() for spec in TOOLS.values()]


def repo_info(cwd: Path) -> dict[str, Any]:
    """Branch + dirty state for the status bar. Never raises, never blocks long."""
    base = {"git": False, "branch": "no-git", "dirty": 0, "upstream": ""}
    if shutil.which("git") is None:
        return base

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""

    if git("rev-parse", "--is-inside-work-tree") != "true":
        return base
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "detached"
    dirty = len([ln for ln in git("status", "--porcelain").splitlines() if ln.strip()])
    return {"git": True, "branch": branch, "dirty": dirty, "upstream": branch.split("/")[-1]}
