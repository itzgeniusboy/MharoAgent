"""P1-1 · Multi-key rotation with cooldowns and per-provider fallback.

Real behaviour, no stubs:
  * keys come from env vars (comma / newline separated) so nothing is committed
  * round-robin across keys; a key that fails is put on exponential cooldown
  * 401/403 (invalid key) is treated as *broken*, not throttled — it is marked
    disabled and skipped for the rest of the process
  * 429 / 5xx / transport errors → cooldown (2s, 5s, 15s, 60s cap), then retried
  * `ma doctor` renders this state so rotation is observable, not folklore
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

AUTH_RE = re.compile(r"\b(401|403|invalid api key|invalid api-key|unauthorized|authentication)", re.I)
THROTTLE_RE = re.compile(r"\b(429|rate.?limit|overloaded|capacity|too many requests)", re.I)
TRANSIENT_RE = re.compile(r"\b(500|502|503|504|timeout|timed out|connection|network|remotedisconnected)", re.I)
COOLDOWN_STEPS = (2.0, 5.0, 15.0, 60.0)


@dataclass
class KeyState:
    label: str                 # never the secret itself
    token: str = field(repr=False, default="")
    failures: int = 0
    successes: int = 0
    disabled: bool = False
    cooldown_until: float = 0.0
    last_error: str = ""

    @property
    def available(self) -> bool:
        return not self.disabled and time.monotonic() >= self.cooldown_until

    def to_dict(self) -> dict:
        wait = max(0.0, self.cooldown_until - time.monotonic())
        return {
            "key": self.label,
            "failures": self.failures,
            "successes": self.successes,
            "disabled": self.disabled,
            "cooling_down_s": round(wait, 1),
            "last_error": self.last_error[:140],
        }


@dataclass
class KeyPool:
    provider: str
    keys: list[KeyState] = field(default_factory=list)
    _cursor: int = 0

    # -- construction ----------------------------------------------------
    @classmethod
    def from_env(cls, provider: str, env_names: Iterable[str], extra: Iterable[str] = ()) -> "KeyPool":
        pool = cls(provider=provider)
        seen: set[str] = set()
        for name in list(env_names) + [""]:
            raw = os.environ.get(name, "") if name else ""
            chunks = [c.strip() for c in re.split(r"[,\n;\s]+", raw) if c.strip()]
            for token in chunks:
                if token in seen:
                    continue
                seen.add(token)
                pool.keys.append(KeyState(label=_label(name or "inline", token), token=token))
        for token in extra:
            if token and token not in seen:
                seen.add(token)
                pool.keys.append(KeyState(label=_label("config", token), token=token))
        return pool

    def add(self, token: str, label: str | None = None) -> None:
        if not token or any(k.token == token for k in self.keys):
            return
        self.keys.append(KeyState(label=label or _label("cfg", token), token=token))

    # -- selection -------------------------------------------------------
    def next(self) -> KeyState | None:
        """Round-robin over keys that are not disabled / cooling down."""
        n = len(self.keys)
        for offset in range(n):
            state = self.keys[(self._cursor + offset) % n]
            if state.available:
                self._cursor = (self._cursor + offset + 1) % n
                return state
        # everyone is cooling: hand back the soonest-available key so the caller
        # can decide whether to wait instead of failing outright.
        waiting = [k for k in self.keys if not k.disabled]
        if not waiting:
            return None
        return min(waiting, key=lambda k: k.cooldown_until)

    def usable_now(self) -> list[KeyState]:
        return [k for k in self.keys if k.available]

    # -- bookkeeping -----------------------------------------------------
    def report_success(self, state: KeyState) -> None:
        state.successes += 1
        state.failures = 0
        state.cooldown_until = 0.0
        state.last_error = ""

    def report_failure(self, state: KeyState, error: object) -> str:
        text = str(error)
        state.failures += 1
        state.last_error = text[:300]
        if AUTH_RE.search(text):
            state.disabled = True
            kind = "auth"
        elif THROTTLE_RE.search(text) or TRANSIENT_RE.search(text) or state.failures >= 3:
            step = COOLDOWN_STEPS[min(state.failures - 1, len(COOLDOWN_STEPS) - 1)]
            state.cooldown_until = time.monotonic() + step
            kind = "throttle" if THROTTLE_RE.search(text) else "transient"
        else:
            kind = "error"
        return kind

    def retry_in(self) -> float:
        waiting = [k for k in self.keys if not k.disabled]
        if not waiting:
            return 0.0
        soonest = min(max(0.0, k.cooldown_until - time.monotonic()) for k in waiting)
        return round(soonest, 1)

    def status(self) -> dict:
        return {
            "provider": self.provider,
            "keys": len(self.keys),
            "available": len(self.usable_now()),
            "disabled": sum(1 for k in self.keys if k.disabled),
            "cooling": sum(1 for k in self.keys if k.cooldown_until > time.monotonic()),
            "retry_in_s": self.retry_in(),
            "detail": [k.to_dict() for k in self.keys],
        }

    def __bool__(self) -> bool:
        return bool(self.keys)


def load_env_files(cwd: "os.PathLike | str | None" = None, *, override: bool = False) -> list[str]:
    """Import KEY=VALUE pairs from `.env`, `.env.local`, `~/.mharo/env`.

    Keys must live somewhere that is not committed; a `.env` in the repo is the
    habit most people already have, so `ma` reads it. Values already present in
    the real environment win unless `override=True`. Returns the *names* loaded —
    never the values.
    """
    import os
    from pathlib import Path

    loaded: list[str] = []
    base = Path(cwd or Path.cwd())
    candidates = [base / ".env", base / ".env.local", Path.home() / ".mharo" / "env", Path.home() / ".mharo.env"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def _label(source: str, token: str) -> str:
    """`OPENAI_API_KEY:…abc123` — enough to identify a key, never the secret."""
    tail = token[-6:] if len(token) > 10 else "short"
    return f"{source}:…{tail}"


def looks_like_secret(token: str) -> bool:
    return bool(re.match(r"^(sk-|sk_|xox|ghp_|gho_|AKIA|AIza|ya29\.|glpat-)", token or ""))
