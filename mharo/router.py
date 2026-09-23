"""P1-3 · Router: start cheap, upgrade on evidence — never on vibes.

Signals that move a step from `cheap` to `strong`:
  1. task keywords (architecture, migration, race, concurrency, debugging…)
  2. repeated tool failures in the same step (model is flailing)
  3. blast radius: many files touched, or a large diff
  4. verification failed once already → the fix pass gets the strong model
  5. the step is explicitly marked by a skill/subagent as needing `strong`
Signals that pull it back to `cheap`:
  6. session spend above the configured budget (fail cheap, not broke)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

HARD_WORDS = {
    "refactor", "architecture", "migrat", "concurrency", "race", "deadlock", "optimize",
    "performance", "security", "auth", "crypto", "protocol", "schema", "dead code",
    "debug", "segfault", "leak", "async", "rewrite", "design", "why", "root cause",
}
EASY_WORDS = {"typo", "rename", "readme", "comment", "print", "log line", "list", "show", "what", "explain"}
COMPLEX_TASK = re.compile(
    r"across (the )?(repo|codebase|files)|every (file|module)|whole (repo|codebase)|"
    r"\d{2,} files|all tests|full (migration|rewrite)", re.I
)


@dataclass
class RoutingDecision:
    tier: str
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def explain(self) -> str:
        return f"{self.tier} ← " + "; ".join(self.reasons) if self.reasons else self.tier


@dataclass
class Router:
    tiers: list[str] = field(default_factory=lambda: ["cheap", "strong"])
    budget_usd: float = 2.0
    fail_upgrade_after: int = 2

    def pick(self, task: str, *, step: dict[str, Any] | None = None, forced: str | None = None) -> RoutingDecision:
        if forced and forced in ("cheap", "strong", "auto"):
            if forced != "auto":
                return RoutingDecision(forced, [f"forced via --tier {forced}"], 1.0)
        low = (task or "").lower()
        hard = sorted({w for w in HARD_WORDS if w in low})
        easy = sorted({w for w in EASY_WORDS if w in low})
        reasons: list[str] = []
        tier = self.tiers[0]

        if hard:
            tier = self.strong_tier
            reasons.append(f"hard-signal words: {', '.join(hard[:4])}")
        if COMPLEX_TASK.search(task or ""):
            tier = self.strong_tier
            reasons.append("scope looks repo-wide")
        if not hard and easy:
            reasons.append(f"routine words: {', '.join(easy[:3])}")
        if not reasons:
            words = len((task or "").split())
            if words <= 8:
                reasons.append(f"short ask ({words} words)")
            else:
                tier = self.tiers[-1] if len(self.tiers) > 1 else tier
                reasons.append(f"medium complexity ({words} words)")
        if step and step.get("tier") in self.tiers:
            tier = step["tier"]
            reasons.append(f"step asked for {tier}")
        return RoutingDecision(tier, reasons, 0.9 if forced else 0.55)

    # -- dynamic upgrade -------------------------------------------------
    @property
    def strong_tier(self) -> str:
        return "strong" if "strong" in self.tiers else self.tiers[-1]

    def should_upgrade(self, *, tier: str, failures: int, files_touched: int, diff_lines: int,
                       verify_failed: bool, spent_usd: float) -> RoutingDecision:
        reasons: list[str] = []
        if tier == self.strong_tier:
            return RoutingDecision(tier, ["already on strong"], 1.0)
        if failures >= self.fail_upgrade_after:
            reasons.append(f"{failures} consecutive tool failures")
        if verify_failed:
            reasons.append("verification failed once — fix pass deserves the stronger model")
        if diff_lines >= 250 or files_touched >= 6:
            reasons.append(f"blast radius ({files_touched} files, {diff_lines} diff lines)")
        if spent_usd > self.budget_usd * 0.85:
            return RoutingDecision(
                self.tiers[0],
                [f"budget guard: ${spent_usd:.2f} of ${self.budget_usd:.2f} — staying cheap"],
                1.0,
            )
        if not reasons:
            return RoutingDecision(tier, ["no upgrade signal"], 0.6)
        return RoutingDecision(self.strong_tier, reasons, 0.9)

    def summary(self) -> dict[str, Any]:
        return {
            "tiers": list(self.tiers),
            "budget_usd": self.budget_usd,
            "hard_signals": len(HARD_WORDS),
            "fail_upgrade_after": self.fail_upgrade_after,
        }
