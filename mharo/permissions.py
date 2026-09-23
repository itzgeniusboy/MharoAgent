"""P1-8 · Permissions: allow / ask / deny per tool, matched on the argument too.

Rules are simple globs in config — `bash:git status*` matches the tool `bash`
whose `command` starts with `git status`. First match in
`deny → allow → default` order decides. `always` answers are remembered for the
rest of the process (and recorded in the DB so `ma cost`/audit can show them).

The engine never executes a tool without a decision from here.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable

ALLOW, ASK, DENY = "allow", "ask", "deny"


def describe(tool: str, args: dict[str, Any]) -> str:
    """The string rules match against: `bash:git status -s`."""
    args = args or {}
    if tool == "bash":
        return f"bash:{' '.join(str(args.get('command', '')).split())}"
    if tool in {"read_file", "write_file", "edit_file", "list_dir", "search"}:
        target = args.get("path") or args.get("pattern") or ""
        return f"{tool}:{target}"
    first = next(iter(args.values()), "") if args else ""
    return f"{tool}:{first}"


def matches(rule: str, subject: str) -> bool:
    rule = (rule or "").strip()
    if not rule:
        return False
    if ":" not in rule:                       # "read_file" → any arg
        return rule == subject.split(":", 1)[0]
    return fnmatch.fnmatchcase(subject, rule) or fnmatch.fnmatchcase(subject, f"{rule}*")


@dataclass
class Decision:
    action: str                 # allow | ask | deny
    rule: str = ""
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == ALLOW

    @property
    def blocked(self) -> bool:
        return self.action == DENY


@dataclass
class Permissions:
    default: str = ASK
    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    _always: set[str] = field(default_factory=set)
    history: list[tuple[str, str, str]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Permissions":
        return cls(
            default=str(cfg.get("default", ASK)),
            allow=[str(r) for r in (cfg.get("allow") or [])],
            ask=[str(r) for r in (cfg.get("ask") or [])],
            deny=[str(r) for r in (cfg.get("deny") or [])],
        )

    def decide(self, tool: str, args: dict[str, Any]) -> Decision:
        subject = describe(tool, args)
        for rule in self.deny:
            if matches(rule, subject):
                return Decision(DENY, rule, f"denied by rule {rule!r}")
        if subject.split(":", 1)[0] in self._always:
            return Decision(ALLOW, "always", "user allowed this tool for the session")
        for rule in self.allow:
            if matches(rule, subject):
                return Decision(ALLOW, rule, f"allowed by rule {rule!r}")
        for rule in self.ask:
            if matches(rule, subject):
                return Decision(ASK, rule, f"rule {rule!r} requires approval")
        if self.default not in {ALLOW, ASK, DENY}:
            return Decision(ASK, "default", f"unknown default {self.default!r}, asking instead")
        return Decision(self.default, "default", f"default policy is {self.default!r}")

    def remember_always(self, tool: str) -> None:
        self._always.add(tool)

    @property
    def always_allowed(self) -> list[str]:
        return sorted(self._always)

    def record(self, tool: str, args: dict[str, Any], decision: Decision) -> None:
        self.history.append((describe(tool, args), decision.action, decision.rule))

    def summary(self) -> dict[str, Any]:
        return {
            "default": self.default,
            "allow": list(self.allow),
            "ask": list(self.ask),
            "deny": list(self.deny),
            "always": self.always_allowed,
            "decisions": len(self.history),
            "blocked": sum(1 for _, a, _ in self.history if a == DENY),
        }


async def prompt_approver(ask: Callable[[str, dict], Any]) -> Callable[[str, dict], Any]:
    """Helper for CLI wiring: `ask` may be sync or async."""
    import inspect

    async def decide(tool: str, args: dict[str, Any]) -> bool:
        verdict = ask(tool, args)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        return bool(verdict)

    return decide
