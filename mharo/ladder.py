"""Engine-side glue for the free-first ladder (see `mharo_tui.agent.ladder`).

The point of this module: make *free providers* look exactly like *multiple API
keys* to the provider hub, so there is one rotation mechanism instead of two.
A "key" is a `(rung, model)` pairing; its cooldown is that pairing's rate gate; a
quota/billing error backs the whole rung off for 15 minutes; and when nothing is
left, the message tells the user which command adds a key.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from mharo.keys import KeyPool, KeyState
from mharo_tui.agent.ladder import (
    DEFAULT_RUNGS,
    KINDS,
    Ladder,
    ProbeResult,
    Rung,
    completion_probe,
    probe_rung,
    rung_from_spec,
)

LOCAL_PARK_S = 20.0   # a closed local port is re-checked after this, not per call

__all__ = ["DEFAULT_RUNGS", "KINDS", "Ladder", "ProbeResult", "Rung", "FreePool",
           "build_ladder", "classify", "quota_hint", "completion_probe", "probe_rung",
           "rung_from_spec", "ask_rung"]

# The classifier lives with the ladder (mharo_tui.agent.ladder) so the TUI and the
# engine agree on what "quota" means; re-exported here for callers of `ma`-side code.
from mharo_tui.agent.ladder import UNREACHABLE_RE as _UNREACHABLE_RE  # noqa: F401
from mharo_tui.agent.ladder import (  # noqa: F401 - single source of truth for classification
    AUTH_RE,
    QUOTA_RE,
    THROTTLE_RE,
    TRANSIENT_RE,
    classify,
    quota_hint,
)

def build_ladder(config: Any = None, *, state_path: Any = None, env: dict[str, str] | None = None) -> Ladder:
    """Ladder from config: honours `free.enabled/prefer/disable/min_interval_s/extra_rungs/rungs`."""
    section: dict[str, Any] = {}
    if config is not None:
        section = dict(getattr(config, "raw", {}).get("free") or {})
    disable = {str(d) for d in (section.get("disable") or [])}
    rungs = None
    if section.get("rungs"):                                   # full override
        rungs = [rung_from_spec(spec) for spec in section["rungs"]]
    ladder = Ladder(
        rungs=rungs,
        enabled=bool(section.get("enabled", True)),
        prefer=tuple(section.get("prefer") or ()),
        intervals=dict(section.get("min_interval_s") or {}),
        extra=list(section.get("extra_rungs") or []),
        allow_paid=bool(section.get("allow_paid", False)),
        state_path=state_path if state_path is not None else (
            config.db_path().parent / "ladder.json" if config is not None and hasattr(config, "db_path") else None),
        env=env,
    )
    if disable:
        ladder.rungs = [r for r in ladder.rungs if r.name not in disable]
    return ladder


def ask_rung(rung: Rung, prompt: str, *, max_tokens: int = 400, timeout: float = 60.0) -> dict[str, Any]:
    """One real, non-streaming completion from a rung — what `ma free --test` runs.

    Shares the TUI's request path so "the probe worked" and "the agent then worked"
    are the same code, and always reports `kind` (quota/throttle/…) on failure.
    """
    from mharo_tui.agent.ladder import _post_completion

    reply = _post_completion(rung, prompt, max_tokens=max_tokens, timeout=timeout)
    reply.setdefault("kind", classify(reply.get("error", "")) if not reply.get("ok") else "ok")
    return reply


class FreePool(KeyPool):
    """A `KeyPool` whose keys are ladder pairings."""

    def __init__(self, provider: str = "free", *, ladder: Ladder | None = None) -> None:
        super().__init__(provider=provider)
        self.ladder = ladder or Ladder()
        self.by_label: dict[str, Rung] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def from_ladder(cls, ladder: Ladder, *, extra: Iterable[tuple[str, str]] = ()) -> "FreePool":
        """One key per `(rung, model)` pairing, plus optional `(label, token)` paid keys.

        `extra` lets a configured paid provider (openai/anthropic) sit at the *end*
        of the same rotation: free rungs first, your key only when free is spent.
        """
        pool = cls(ladder=ladder)
        for rung in ladder.candidates():
            label = rung.label()
            if label in pool.by_label:
                continue
            pool.by_label[label] = rung
            pool.keys.append(KeyState(label=label, token=rung.key(ladder.env)))
        for label, token in extra or ():
            if label and token and label not in pool.by_label:
                pool.keys.append(KeyState(label=label, token=token))
        return pool

    # -- rotation --------------------------------------------------------
    def _sync(self) -> None:
        """Mirror the ladder's view of each pairing onto the pool.

        `wait_s()` only covers *timers*. A rung that is unavailable for another
        reason — a local port we already probed as dead — has no timer at all, so
        it must be parked explicitly; otherwise `next()` hands it out, the call
        dies in a connect error, and the run pays a backoff for a server that was
        never going to answer.
        """
        now = time.monotonic()
        for state in self.keys:
            rung = self.by_label.get(state.label)
            if rung is None:
                continue
            if not rung.available(now):
                gap = rung.wait_s(now) or (LOCAL_PARK_S if rung.local else 1.0)
                state.cooldown_until = max(state.cooldown_until, now + gap)
            elif state.cooldown_until and state.cooldown_until <= now:
                state.cooldown_until = 0.0

    def next(self) -> KeyState | None:
        """Next *usable* pairing, or None.

        `KeyPool.next` hands back the soonest-available key when all are cooling —
        right for a paid key you plan to retry, wrong for a free tier you were just
        rate-limited by: calling it again is the hammering the gate exists to stop.
        """
        self._sync()
        ready = [k for k in self.keys if k.available]
        if not ready:
            return None
        state = ready[0]
        self._cursor = (self.keys.index(state) + 1) % len(self.keys)
        return state

    def usable_now(self) -> list[KeyState]:
        self._sync()
        return super().usable_now()

    def rung_for(self, state: KeyState | None) -> Rung | None:
        return self.by_label.get(state.label) if state else None

    # -- bookkeeping -----------------------------------------------------
    def report_success(self, state: KeyState) -> None:
        super().report_success(state)
        rung = self.by_label.get(state.label)
        if rung is not None:
            self.ladder.mark_used(rung)  # remembers the hit, saves probe state
            wait = rung.wait_s()
            if wait > 0:
                # apply the gate now, not lazily: a pool that looks free right after
                # a call is how a free tier gets hammered into a 429 spiral
                state.cooldown_until = time.monotonic() + wait

    def report_failure(self, state: KeyState | None, error: object) -> str:
        kind = classify(error) if state is not None else super().report_failure(state, error)
        if kind == "shape":
            # the host works, the body did not — no cooldown, no disabled key
            if state is not None:
                state.last_error = str(error)[:200]
            return "shape"
        if kind in {"quota", "auth", "unreachable", "throttle", "error"} and state is not None:
            rung = self.by_label.get(state.label)
            if rung is not None:
                # the rung decides the shape of the backoff; the pool mirrors it
                seconds = self.ladder.mark_failure(rung, kind, error=str(error)[:200])
                if kind in {"quota", "auth"}:
                    state.disabled = False          # not broken forever: another model on the host may work
                    state.failures += 1
                    state.cooldown_until = time.monotonic() + seconds
                else:
                    state.cooldown_until = time.monotonic() + max(seconds, 2.0)
                state.last_error = str(error)[:200]
                return kind
        if state is not None:
            super().report_failure(state, error)
        return kind

    # -- reporting -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        base = super().status()
        base["kind"] = "ladder"
        base["summary"] = self.ladder.summary()
        base["notices"] = list(self.ladder.notices[-4:])
        base["hint"] = quota_hint(self.ladder) if not self.usable_now() else ""
        return base
