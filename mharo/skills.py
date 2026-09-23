"""P1-10 · Skills: `SKILL.md` files that are executable, not documentation.

A skill is a prompt template + hard constraints (tool allowlist, model tier,
turn budget, required verification). Parsing is dependency-free (frontmatter is
`key: value` lines) and *validated*: an unknown tool or a bad tier is a
`ma doctor` failure, not a silent ignore.

    ---
    name: code-review
    description: Adversarial review of the working tree
    tools: read_file, search, list_dir        # must exist in the registry
    tier: strong                              # cheap | strong
    max_turns: 3
    verify: checks                            # checks | none
    tags: quality, review
    ---
    Body is the prompt; {task}, {diff}, {memory} are substituted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
KNOWN_KEYS = {"name", "description", "tools", "tier", "max_turns", "verify", "tags", "timeout"}


@dataclass
class Skill:
    name: str
    description: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    tier: str = "cheap"
    max_turns: int = 4
    verify: str = "checks"
    tags: list[str] = field(default_factory=list)
    path: Path | None = None
    problems: list[str] = field(default_factory=list)

    def render(self, task: str, *, diff: str = "", memory: str = "") -> str:
        out = self.prompt or f"Run the {self.name} skill."
        for key, value in (("task", task), ("diff", diff), ("memory", memory)):
            out = out.replace("{" + key + "}", value)
        return out.strip()

    def tool_subset(self, registry: dict[str, Any]) -> dict[str, Any]:
        if not self.tools:
            return dict(registry)
        return {name: registry[name] for name in self.tools if name in registry}


def parse(text: str, path: Path | None = None, *, known_tools: list[str] | None = None) -> Skill:
    meta: dict[str, str] = {}
    body = text
    match = FRONT.match(text or "")
    if match:
        raw_meta, body = match.group(1), match.group(2)
        for line in raw_meta.splitlines():
            if ":" not in line or line.strip().startswith("#"):
                continue
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    problems: list[str] = []
    for key in meta:
        if key not in KNOWN_KEYS:
            problems.append(f"unknown frontmatter key {key!r}")
    tools = [t.strip() for t in re.split(r"[,\s]+", meta.get("tools", "")) if t.strip()]
    if known_tools is not None:
        for tool in tools:
            if tool not in known_tools:
                problems.append(f"tool {tool!r} is not in the registry")
    tier = meta.get("tier", "cheap")
    if tier not in {"cheap", "strong"}:
        problems.append(f"tier must be cheap|strong, got {tier!r}")
    try:
        max_turns = int(meta.get("max_turns", "4"))
    except ValueError:
        problems.append(f"max_turns is not an integer: {meta.get('max_turns')!r}")
        max_turns = 4
    if not (body or "").strip():
        problems.append("skill has no instructions in the body")
    name = meta.get("name") or (path.parent.name if path else "unnamed")
    return Skill(
        name=name,
        description=meta.get("description", "").strip("\"' ") or "(no description)",
        prompt=body.strip(),
        tools=tools,
        tier=tier,
        max_turns=max(1, min(24, max_turns)),
        verify=meta.get("verify", "checks"),
        tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
        path=path,
        problems=problems,
    )


def discover(dirs: list[Path], *, known_tools: list[str] | None = None) -> tuple[list[Skill], list[str]]:
    skills: list[Skill] = []
    problems: list[str] = []
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/SKILL.md")) + sorted(root.glob("*.skill.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                problems.append(f"{path}: unreadable ({exc})")
                continue
            skill = parse(text, path, known_tools=known_tools)
            for problem in skill.problems:
                problems.append(f"{skill.name}: {problem}")
            skills.append(skill)
    seen: set[str] = set()
    unique: list[Skill] = []
    for skill in skills:
        if skill.name in seen:
            problems.append(f"{skill.name}: duplicate skill name (later one wins: {skill.path})")
        seen.add(skill.name)
        unique.append(skill)
    return unique, problems


def create_skill(dirs: list[Path], name: str, description: str, *, tools: list[str] | None = None) -> Path:
    root = next((d for d in dirs if d.is_dir()), dirs[0] if dirs else Path("skills"))
    target = root / name / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    tool_line = ", ".join(tools or ["read_file", "search", "list_dir"])
    target.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"tools: {tool_line}\n"
        "tier: cheap\n"
        "max_turns: 4\n"
        "verify: checks\n"
        f"tags: {name}\n"
        "---\n"
        f"You are running the **{name}** skill on this task:\n\n"
        "{task}\n\n"
        "Project instructions and current rules:\n{memory}\n\n"
        "Steps:\n"
        "1. Read the files involved before deciding anything.\n"
        "2. Apply the smallest change or produce the shortest useful report.\n"
        "3. State what you verified and what you did not.\n",
        encoding="utf-8",
    )
    return target
