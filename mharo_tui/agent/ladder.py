"""Free-first provider ladder: run the agent with no API key at all.

Preference order (each rung is *probed*, never assumed):

    local      Ollama / LM Studio / llama.cpp / vLLM   — unlimited, private
    free-anon  OVHcloud AI Endpoints, Pollinations      — no key, no signup
    free-key   OpenRouter :free, LLM7 turbo, Gemini, Groq — a key you get for free
    paid       openai / anthropic, per tier config

Why a ladder and not a "free" flag: every free tier has a *shape*. OVH's
anonymous tier is ~2 requests/minute **per model per IP**, so a rung here is a
`(host, model)` pairing with its own clock — after one call the next call moves to
another model on the same host instead of waiting 30 seconds. Billing-shaped
failures (402/quota/401) back the whole host off for 15 minutes; a 429 backs off
only that pairing. Nothing here claims to be unlimited: rate gates, TTL'd probes
and `~/.mharo/ladder.json` are the design.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

KINDS = ("local", "free-anon", "free-key", "paid")

#: defaults, in preference order. `min_interval_s` = client-side gate between two
#: calls to this *pairing* (0 = no gate).
DEFAULT_RUNGS: list[dict[str, Any]] = [
    {
        "name": "ollama", "kind": "local", "base_url": "http://127.0.0.1:11434/v1",
        "models": ["qwen2.5-coder:32b", "qwen2.5-coder:14b", "deepseek-r1:14b", "llama3.1:8b"],
        "keyless": True, "tools": True,
        "note": "local + unlimited — `ollama pull qwen2.5-coder:32b`",
    },
    {
        "name": "lmstudio", "kind": "local", "base_url": "http://127.0.0.1:1234/v1",
        "models": ["local-model"], "keyless": True, "tools": True,
        "note": "LM Studio local server (turn on 'Serve on localhost')",
    },
    {"name": "llamacpp", "kind": "local", "base_url": "http://127.0.0.1:8080/v1",
     "models": ["local-model"], "keyless": True, "tools": True, "note": "llama.cpp server"},
    {"name": "vllm", "kind": "local", "base_url": "http://127.0.0.1:8000/v1",
     "models": ["local-model"], "keyless": True, "tools": True, "note": "vLLM / SGLang server"},
    {
        "name": "ovh", "kind": "free-anon", "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        "models": ["gpt-oss-120b", "Qwen3-Coder-30B-A3B-Instruct", "gpt-oss-20b",
                   "Meta-Llama-3_3-70B-Instruct", "Mistral-Small-3.2-24B-Instruct-2506", "Qwen3.6-27B"],
        "keyless": True, "tools": True, "min_interval_s": 31.0,
        "note": "anonymous free tier, ~2 req/min per model per IP (EU); rotates models for you",
    },
    {
        "name": "pollinations", "kind": "free-anon", "base_url": "https://text.pollinations.ai/openai",
        "models": ["openai-fast"], "keyless": True, "tools": False, "min_interval_s": 2.0,
        "note": "no key; its upstream credits dry up sometimes, so it sits last among anonymous",
    },
    {
        "name": "openrouter", "kind": "free-key", "base_url": "https://openrouter.ai/api/v1",
        "env_keys": ["OPENROUTER_API_KEY", "MHARO_OPENROUTER_KEYS"],
        "models": ["nvidia/nemotron-3-super-120b-a12b:free", "openai/gpt-oss-20b:free",
                   "google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free",
                   "cohere/north-mini-code:free", "nvidia/nemotron-nano-9b-v2:free"],
        "tools": True, "min_interval_s": 3.5,
        "note": "$0/token `:free` models; 20 req/min + 50/day free, 1000/day after $10 credit",
    },
    {
        "name": "llm7", "kind": "free-key", "base_url": "https://api.llm7.io/v1",
        "env_keys": ["LLM7_API_KEY", "MHARO_LLM7_KEYS"],
        "models": ["DeepSeek-V4-Flash-0731", "GLM-5.3-Flash", "L3-8B-Lunaris-v1-Turbo"],
        "tools": True, "min_interval_s": 3.0, "note": "free token from token.llm7.io; 400K contexts",
    },
    {
        "name": "gemini", "kind": "free-key", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "models": ["gemini-2.5-flash", "gemini-2.0-flash"], "tools": True, "min_interval_s": 1.0,
        "note": "free tier with a Google AI Studio key",
    },
    {
        "name": "groq", "kind": "free-key", "base_url": "https://api.groq.com/openai/v1",
        "env_keys": ["GROQ_API_KEY"], "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        "tools": True, "min_interval_s": 1.0, "note": "fast, small free quota",
    },
]

LOCAL_TTL = 20.0
REMOTE_TTL = 300.0
QUOTA_BACKOFF_S = 900.0
HOST_DOWN_S = 60.0
#: how many 429s on one *host* (any model) inside a minute count as saturation
THROTTLE_STORM = 3

#: Payload shapes, richest first. Some free OpenAI-compatible hosts answer a plain
#: completion but 422 on `stream_options` or `tools` — so each host *learns* the
#: shape it accepts, and the answer is cached instead of re-learned every call.
PROFILE_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full", ()),
    ("no_stream_options", ("stream_options",)),
    ("no_tools", ("stream_options", "tools")),
    ("minimal", ("stream", "stream_options", "tools")),
)
PROFILE_NAMES = tuple(name for name, _ in PROFILE_STEPS)

#: the smallest gate we consider polite for each public rung; config below the
#: floor is a bug, not a preference (the doctor fails on it).
RATE_FLOORS: dict[str, float] = {"ovh": 30.0, "pollinations": 1.0, "openrouter": 2.5,
                                 "llm7": 2.0, "gemini": 0.5, "groq": 0.5}

# --- error taxonomy shared by the ladder, the TUI and `mharo.keys` -----------
AUTH_RE = re.compile(r"\b(401|403|unauthorized|forbidden|invalid [a-z ]*api key|"
                     r"incorrect api key|invalid_api_key|authentication)\b", re.I)
SHAPE_RE = re.compile(r"\b(422|400|json_invalid|invalid schema|validation error|field required|"
                      r"extra inputs are not permitted|unnecessary key|unsupported parameter|"
                      r"unknown parameter|body failed validation|invalid_request_error)\b", re.I)
THROTTLE_RE = re.compile(r"\b(429|rate.?limit\w*|too many requests|overloaded|overloaded_error|"
                         r"temporarily unavailable|503|502|529)\b", re.I)
TRANSIENT_RE = re.compile(r"\b(timeout|timed out|connection|connecterror|remotedisconnected|"
                          r"reset by peer|broken pipe|5\d\d)\b", re.I)
#: billing-shaped failures only. Deliberately *not* matching the bare phrase
#: "limit exceeded": OVH answers its 2-req/min gate with "API rate limit exceeded",
#: and reading that as a quota would back a healthy host off for 15 minutes instead
#: of the 31 seconds (or one model rotation) it actually needs.
QUOTA_RE = re.compile(
    r"\b(402|quota exceeded|exceeded your current quota|quota \(\w+\) has been exceeded|"
    r"out of credits|insufficient (credits|funds|balance)|credit balance|billing|"
    r"payment required|free (tier|quota) (limit|allowance)|usage limit reached|"
    r"daily (limit|cap)|resource_exhausted)\b", re.I)
UNREACHABLE_RE = re.compile(r"\b(connecterror|connect error|connect to host|couldn't connect|"
                            r"unable to connect|name resolution|name or service|network|"
                            r"unreachable|refused|remotedisconnected|sslerror)\b", re.I)


def _now() -> float:
    return time.monotonic()


def classify(error: object) -> str:
    """quota | auth | unreachable | throttle | error — decides how long to back off.

    Order matters: a 429 whose body says "quota" is a *billing* problem (move to
    another host), not a transient blip (stay and retry).
    """
    text = str(error or "")
    if QUOTA_RE.search(text):
        return "quota"
    if SHAPE_RE.search(text) and not THROTTLE_RE.search(text):
        # a healthy endpoint that is not out of credit — it rejected *our body*
        return "shape"
    if AUTH_RE.search(text):
        return "auth"
    if UNREACHABLE_RE.search(text) and not THROTTLE_RE.search(text):
        return "unreachable"
    if THROTTLE_RE.search(text) or TRANSIENT_RE.search(text):
        return "throttle"
    return "error"


@dataclass
class Rung:
    """One way to get tokens: host + model + that pairing's own limits and clock."""

    name: str
    base_url: str
    model: str
    kind: str = "free-anon"
    models: tuple[str, ...] = ()
    catalog: tuple[str, ...] = ()
    env_keys: tuple[str, ...] = ()
    keyless: bool = True
    tools: bool = True
    min_interval_s: float = 0.0
    note: str = ""
    last_used: float = 0.0
    cooling_until: float = 0.0
    quota_until: float = 0.0
    failures: int = 0
    probed_ok: bool | None = None
    latency_ms: int = 0
    throttles: list[float] = field(default_factory=list)  # monotonic stamps of recent 429s
    profile: str = "full"                          # payload shape this host accepted

    # -- identity --------------------------------------------------------
    @property
    def local(self) -> bool:
        return "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def label(self) -> str:
        return f"{self.name}:{self.model}"

    def host_port(self) -> tuple[str, int]:
        raw = self.base_url.split("://", 1)[-1]
        host = raw.split("/", 1)[0]
        port = 443 if self.base_url.startswith("https") else 80
        if ":" in host:
            host, _, text = host.rpartition(":")
            if text.isdigit():
                port = int(text)
        return host, port

    def key(self, env: dict[str, str] | None = None) -> str:
        if self.keyless:
            return ""
        source = os.environ if env is None else env
        for name in self.env_keys:
            value = source.get(name, "")
            if value:
                return value.replace(" ", "").split(",")[0].splitlines()[0]
        return ""

    def needs_key(self) -> bool:
        """True when this rung wants a key and has nowhere to read one from."""
        return not self.keyless and not self.env_keys

    @property
    def missing_key(self) -> bool:
        """A free-with-key rung the user has not given a key to yet."""
        return not self.keyless and not self.key()

    # -- availability ----------------------------------------------------
    def available(self, now: float | None = None) -> bool:
        now = _now() if now is None else now
        if self.probed_ok is False and self.local:
            return False  # a local server that is not listening will not come back mid-run
        if now < self.quota_until or now < self.cooling_until:
            return False
        if self.min_interval_s <= 0:
            return True
        return (now - self.last_used) >= self.min_interval_s

    def wait_s(self, now: float | None = None) -> float:
        now = _now() if now is None else now
        gates = [self.quota_until, self.cooling_until]
        if self.min_interval_s > 0:
            gates.append(self.last_used + self.min_interval_s)
        return max(0.0, round(max(gates) - now, 1))

    def mark_used(self) -> None:
        self.last_used = _now()
        self.cooling_until = self.last_used + self.min_interval_s

    def throttle_storm(self, window: float = 60.0) -> bool:
        """Three 429s on one host inside a minute: the shared pool is saturated."""
        now = _now()
        self.throttles = [t for t in self.throttles if now - t <= window]
        return len(self.throttles) >= 3

    @property
    def drops(self) -> tuple[str, ...]:
        """Which payload keys this host must not receive."""
        return dict(PROFILE_STEPS).get(self.profile, ())

    def downgrade(self) -> bool:
        """Move one step leaner. False when already at the floor."""
        try:
            index = PROFILE_NAMES.index(self.profile)
        except ValueError:
            index = 0
        if index + 1 >= len(PROFILE_NAMES):
            return False
        self.profile = PROFILE_NAMES[index + 1]
        return True

    def mark_failure(self, kind: str) -> float:
        """Back this pairing (or its whole host) off. Returns the seconds."""
        if kind == "shape":
            return 0.0  # nothing to wait for: fix the request, not the schedule
        self.failures += 1
        now = _now()
        if kind in {"quota", "auth"}:
            self.quota_until = now + QUOTA_BACKOFF_S
            return QUOTA_BACKOFF_S
        if kind == "unreachable":
            self.cooling_until = now + HOST_DOWN_S
            return HOST_DOWN_S
        self.throttles.append(now)
        gap = max(self.min_interval_s, 5.0 * min(4, self.failures))
        if self.throttle_storm():
            gap = max(gap, self.min_interval_s, 30.0)
        self.cooling_until = now + gap
        return gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "model": self.model, "kind": self.kind, "base_url": self.base_url,
            "tools": self.tools, "keyless": self.keyless, "min_interval_s": self.min_interval_s,
            "available": self.available(), "wait_s": self.wait_s(), "failures": self.failures,
            "probed_ok": self.probed_ok, "latency_ms": self.latency_ms, "note": self.note,
            "profile": self.profile,
            "missing_key": self.missing_key, "has_key": bool(self.key()),
        }


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    models: tuple[str, ...] = ()
    latency_ms: int = 0


def merge_models(rung: Rung, live: Iterable[str]) -> tuple[str, ...]:
    """Reconcile what a host reports with what we deliberately chose.

    A public `/models` list is the whole catalogue, paid models included, so
    trusting it would let a free rung silently default to a paid model. Local
    servers are the opposite case: whatever the user installed *is* the list.
    """
    found = [str(m) for m in live if m]
    if not found:
        return tuple(m for m in rung.models if m) or (rung.model,)
    if rung.local:
        return tuple(dict.fromkeys(found))
    curated = list(rung.catalog or rung.models)
    keep = [m for m in curated if m in found]
    return tuple(keep or curated)


def probe_rung(rung: Rung, *, timeout: float = 6.0, with_completion: bool = False) -> ProbeResult:
    """Ask the host if it is really there (and, optionally, really answering)."""
    started = _now()
    if rung.local:  # cheap TCP check first — with no server there is no HTTP to wait for
        host, port = rung.host_port()
        try:
            with socket.create_connection((host, port), timeout=min(1.5, timeout)):
                pass
        except OSError as exc:
            return ProbeResult(False, f"{host}:{port} not listening ({exc.strerror or type(exc).__name__})")
    import httpx

    headers = {"Accept": "application/json"}
    key = rung.key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        res = httpx.get(rung.base_url.rstrip("/") + "/models", headers=headers,
                        timeout=timeout, follow_redirects=True)
    except Exception as exc:  # noqa: BLE001 - a probe never raises
        return ProbeResult(False, f"{type(exc).__name__}: {str(exc)[:120]}", latency_ms=int((_now() - started) * 1000))
    ms = int((_now() - started) * 1000)
    if res.status_code >= 400:
        return ProbeResult(False, f"HTTP {res.status_code} on /models", latency_ms=ms)
    models: tuple[str, ...] = ()
    try:
        body = res.json()
        data = body.get("data") or body.get("models") or []
        models = tuple(str(m.get("id") or m.get("name")) for m in data if isinstance(m, dict))[:60]
    except (ValueError, AttributeError, TypeError):
        pass
    detail = f"HTTP {res.status_code} · {len(models)} model(s)"
    if with_completion:
        verdict = completion_probe(rung, timeout=max(timeout, 25.0))
        return ProbeResult(verdict.ok, f"{detail} · completion: {verdict.detail}",
                           models or rung.models, verdict.latency_ms)
    return ProbeResult(True, detail, models or rung.models, ms)


def completion_probe(rung: Rung, *, timeout: float = 25.0) -> ProbeResult:
    """One tiny real completion — the only probe that proves a free tier works."""
    result = _post_completion(rung, "Reply with exactly: OK", max_tokens=8, timeout=timeout)
    if not result.get("ok"):
        return ProbeResult(False, str(result.get("error", "no answer"))[:160], latency_ms=int(result.get("ms", 0)))
    return ProbeResult(True, f"{len(result.get('text', ''))} chars in {result.get('ms')} ms",
                       (rung.model,), int(result.get("ms", 0)))


def _post_completion(rung: Rung, prompt: str, *, max_tokens: int = 400,
                     timeout: float = 60.0, temperature: float = 0.2) -> dict[str, Any]:
    """A single non-streaming chat completion against one rung (no key games)."""
    import httpx

    started = _now()
    payload = {"model": rung.model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Content-Type": "application/json"}
    key = rung.key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        res = httpx.post(rung.base_url.rstrip("/") + "/chat/completions", json=payload,
                         headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - surfaced structurally
        return {"ok": False, "rung": rung.label(),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "kind": classify(exc)}
    ms = int((_now() - started) * 1000)
    out: dict[str, Any] = {"ok": res.status_code < 400, "status": res.status_code,
                           "rung": rung.label(), "ms": ms}
    if res.status_code >= 400:
        out["error"] = f"HTTP {res.status_code}: {res.text[:300].strip()}"
        out["kind"] = classify(out["error"])
        return out
    try:
        body = res.json()
        choice = body["choices"][0]["message"]
        out["text"] = (choice.get("content") or choice.get("reasoning") or "").strip()
        out["tool_calls"] = len(choice.get("tool_calls") or [])
        out["usage"] = body.get("usage") or {}
    except (ValueError, KeyError, IndexError, AttributeError, TypeError):
        out.update(ok=False, error="response was not a chat completion")
    if out.get("ok") and not out.get("text"):
        out.update(ok=False, error="empty completion")
    return out


class Ladder:
    """Ordered, probed, rate-gated set of `(host, model)` pairings."""

    def __init__(
        self,
        rungs: Iterable[Rung] | None = None,
        *,
        enabled: bool = True,
        prefer: tuple[str, ...] = (),
        intervals: dict[str, float] | None = None,
        extra: list[dict[str, Any]] | None = None,
        state_path: str | Path | None = None,
        env: dict[str, str] | None = None,
        allow_paid: bool = False,
    ) -> None:
        self.env = env
        base = [r for r in rungs] if rungs is not None else [rung_from_spec(spec) for spec in DEFAULT_RUNGS]
        for spec in extra or []:
            base.append(rung_from_spec(spec))
        self.intervals = intervals or {}
        prepared: list[Rung] = []
        for rung in base:
            if rung.name in self.intervals:
                rung = replace(rung, min_interval_s=float(self.intervals[rung.name]))
            prepared.append(rung)
        self.rungs = _expand(prepared)
        self.enabled = enabled
        self.prefer = tuple(prefer or ())
        self.allow_paid = allow_paid
        self.state_path = Path(state_path).expanduser() if state_path else None
        self.last_error = ""
        self.notices: list[str] = []
        self.load_state()
        self.check_locals()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_config(cls, config: Any = None, *, state_path: str | Path | None = None) -> "Ladder":
        """Build from a `mharo.config.Config` (duck-typed: no import cycle)."""
        section: dict[str, Any] = {}
        home: Path | None = None
        if config is not None:
            section = dict(getattr(config, "free", {}) or {})
            if state_path is None:
                try:
                    home = Path(config.db_path()).parent
                except Exception:
                    home = None
        disable = {str(d) for d in (section.get("disable") or [])}
        ladder = cls(
            rungs=[rung_from_spec(spec) for spec in section["rungs"]] if section.get("rungs") else None,
            enabled=bool(section.get("enabled", True)),
            prefer=tuple(section.get("prefer") or ()),
            intervals=dict(section.get("min_interval_s") or {}),
            extra=list(section.get("extra_rungs") or []),
            allow_paid=bool(section.get("allow_paid", True)),
            state_path=state_path or (home / "ladder.json" if home else None),
        )
        if disable:
            ladder.rungs = [r for r in ladder.rungs if r.name not in disable]
        ladder.sync_models()
        return ladder

    # -- selection -------------------------------------------------------
    def ordered(self) -> list[Rung]:
        ranks = {name: i for i, name in enumerate(self.prefer)}

        def sort_key(item: tuple[int, Rung]) -> tuple[int, int, int, int]:
            index, rung = item
            in_order = ranks.get(rung.name, len(ranks) + 1)
            kind_rank = {"local": 0, "free-anon": 1, "free-key": 2, "paid": 3}.get(rung.kind, 2)
            if not self.allow_paid and rung.kind == "paid":
                kind_rank = 99
            return (in_order, kind_rank, index, 0)

        return [rung for _, rung in sorted(enumerate(self.rungs), key=sort_key)]

    def hosts(self) -> list[str]:
        seen: dict[str, None] = {}
        for rung in self.ordered():
            seen.setdefault(rung.name, None)
        return list(seen)

    def usable_rungs(self) -> list[Rung]:
        """Pairings we can actually call: anonymous, or with a key that exists."""
        return [rung for rung in self.ordered() if rung.keyless or rung.key(self.env)]

    def candidates(self, *, skip: set[str] | None = None) -> list[Rung]:
        """Every pairing worth trying, best first. Real objects: use them, don't copy."""
        skip = skip or set()
        out = [r for r in self.usable_rungs() if r.name not in skip and r.label() not in skip]
        if not self.allow_paid:
            out = [r for r in out if r.kind != "paid"]
        now = _now()
        out.sort(key=lambda r: (not r.available(now), r.probed_ok is False, r.failures, r.wait_s(now)))
        return out

    def pick(self, *, skip: set[str] | None = None, wait: bool = False) -> Rung | None:
        """The pairing to use now. `wait=True` returns the soonest-free one instead."""
        candidates = self.candidates(skip=skip)
        for rung in candidates:
            if rung.available():
                return rung
        if wait and candidates:
            return min(candidates, key=lambda r: r.wait_s())
        self.last_error = self._exhausted_reason(candidates)
        return None

    def _exhausted_reason(self, candidates: list[Rung]) -> str:
        if not candidates:
            return "no free rung is configured with an available key"
        soonest = min(candidates, key=lambda r: r.wait_s())
        wait = soonest.wait_s()
        if soonest.quota_until > _now():
            return f"{soonest.name} quota exhausted, retrying in {int(wait)}s"
        if wait > 0:
            return f"all free rungs rate-limited, next free in {int(wait)}s ({soonest.label()})"
        if any(r.probed_ok is False for r in candidates):
            return "every configured endpoint answered down"
        return "all free rungs unavailable"

    def soothe(self) -> str:
        return self.last_error

    # -- bookkeeping -----------------------------------------------------
    def _find(self, rung: Rung) -> list[Rung]:
        """The ladder's own objects for this pairing (or the whole host)."""
        exact = [r for r in self.rungs if r.name == rung.name and r.model == rung.model]
        return exact

    def mark_used(self, rung: Rung) -> None:
        for mine in self._find(rung) or [rung]:
            mine.mark_used()
            mine.probed_ok = True
            mine.failures = max(0, mine.failures - 1)
        self.save_state()

    def downgrade_profile(self, rung: Rung) -> bool:
        """Tell a host once that it dislikes `tools`/`stream_options`; all pairings follow.

        Returns True when the payload got leaner, i.e. the caller should retry now.
        """
        changed = False
        profile = rung.profile
        for mine in self.rungs:
            if mine.name == rung.name and mine.downgrade():
                changed = True
                profile = mine.profile
        if changed:
            note = f"{rung.name}: endpoint rejected the request shape — retrying as {profile!r}"
            if note not in self.notices:
                self.notices.append(note)
            self.save_state()
        return changed

    def mark_failure(self, rung: Rung, kind: str, *, error: str = "") -> float:
        """Rate limits cool one pairing; billing and auth cool the whole host."""
        targets = self._find(rung) or [rung]
        if kind in {"quota", "auth", "unreachable"}:
            targets = [r for r in self.rungs if r.name == rung.name] or targets
        elif kind == "throttle":
            siblings = [r for r in self.rungs if r.name == rung.name]
            recent = [t for r in siblings for t in r.throttles if _now() - t <= 60.0]
            if len(recent) + 1 >= THROTTLE_STORM:
                # one 429 = wait out that pairing; a storm = the whole host is
                # saturated by other users, so stop spending calls on it for a bit
                targets = siblings or targets
        seconds = 0.0
        for mine in targets:
            seconds = max(seconds, mine.mark_failure(kind))
            if kind in {"unreachable", "error"}:
                # a 429/quota reply proves the endpoint is alive; only a transport
                # failure marks it down, else one blip would hide a host for a TTL
                mine.probed_ok = False
        if kind == "shape":
            self.last_error = error or "endpoint rejected the request shape"
            return 0.0
        note = f"{rung.label()} → {kind}, backing off {int(seconds)}s"
        if note not in self.notices:
            self.notices.append(note)
        self.last_error = error or kind
        self.save_state()
        return seconds

    def status(self) -> list[dict[str, Any]]:
        """One row per host, summarising its pairings."""
        rows: list[dict[str, Any]] = []
        for name in self.hosts():
            pairings = [r for r in self.ordered() if r.name == name]
            first = pairings[0]
            ready = [r for r in pairings if r.available() and not r.missing_key]
            row = (ready or pairings)[0].to_dict()
            row.update({
                "pairings": len(pairings),
                "ready": len(ready),
            "pairings_ready": len([r for r in pairings if r.available()]),
                "wait_s": round(min((r.wait_s() for r in pairings), default=0.0), 1),
                "models": [r.model for r in pairings],
                "probed_ok": next((r.probed_ok for r in pairings if r.probed_ok is not None), None),
                "note": first.note,
                "base_url": first.base_url,
                "kind": first.kind,
                "tools": first.tools,
                "keyless": first.keyless,
                "min_interval_s": first.min_interval_s,
                "profile": first.profile,
                "missing_key": first.missing_key,
                "has_key": bool(first.key(self.env)),
            })
            rows.append(row)
        return rows

    def summary(self) -> str:
        rows = self.status()
        if not rows:
            return "no rungs configured"
        parts = []
        for row in rows[:6]:
            mark = "✓" if row["ready"] else ("✗" if row["probed_ok"] is False else "…")
            parts.append(f"{mark} {row['name']}/{row['model'][:22]}")
        return " · ".join(parts)

    # -- probe -----------------------------------------------------------
    def probe(self, *, rungs: Iterable[str] | None = None, completions: bool = False,
              timeout: float = 6.0) -> dict[str, ProbeResult]:
        """One probe per host; the verdict is applied to all of its pairings."""
        wanted = set(rungs) if rungs is not None else None
        out: dict[str, ProbeResult] = {}
        for name in self.hosts():
            if wanted is not None and name not in wanted:
                continue
            pairings = [r for r in self.ordered() if r.name == name]
            result = probe_rung(pairings[0], timeout=timeout, with_completion=completions)
            out[name] = result
            for mine in pairings:
                mine.probed_ok = result.ok
                mine.latency_ms = result.latency_ms
            if result.ok and result.models:
                self.apply_models(name, result.models)
        self.save_state()
        return out

    def apply_models(self, host: str, live: Iterable[str]) -> None:
        """Adopt a host's real model list (local: whatever is installed; remote: our curated subset)."""
        pairings = [r for r in self.rungs if r.name == host]
        if not pairings:
            return
        merged = merge_models(pairings[0], live)
        base = pairings[0]
        template = replace(base, models=merged, catalog=base.catalog or merged)
        keep = [r for r in self.rungs if r.name != host]
        self.rungs = _expand(keep + [template])

    def sync_models(self) -> None:
        """Keep one pairing per advertised model, and no stale ones."""
        self.rungs = _expand([r for r in self.rungs])

    # -- persistence (best-effort; a missing cache is never an error) ----
    def load_state(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        stamp = float(raw.get("checked_at", 0))
        age = time.time() - stamp
        if age > REMOTE_TTL * 4:
            return
        for entry in raw.get("rungs", []):
            name = str(entry.get("name", ""))
            pairings = [r for r in self.rungs if r.name == name]
            if not pairings:
                continue
            ok = entry.get("ok")
            if ok is False and age > REMOTE_TTL:
                ok = None  # a dead-looking endpoint gets a fresh chance after 5 min
            cached_profile = str(entry.get("profile") or "full")
            for rung in pairings:
                rung.probed_ok = ok
                rung.latency_ms = int(entry.get("latency_ms", 0) or 0)
                if cached_profile in PROFILE_NAMES:
                    rung.profile = cached_profile
            models = merge_models(pairings[0], tuple(str(m) for m in (entry.get("models") or ())))
            if models:
                kept = [r for r in self.rungs if r.name != name]
                template = replace(pairings[0], models=models, catalog=pairings[0].catalog or pairings[0].models)
                self.rungs = _expand(kept + [template])

    def check_locals(self) -> None:
        """Local servers are cheap to check — one ~1 ms socket call per port."""
        probed: dict[tuple[str, int], bool] = {}
        for rung in self.rungs:
            if not rung.local:
                continue
            host, port = rung.host_port()
            if (host, port) not in probed:
                try:
                    with socket.create_connection((host, port), timeout=0.35):
                        probed[(host, port)] = True
                except OSError:
                    probed[(host, port)] = False
            rung.probed_ok = probed[(host, port)]

    def save_state(self) -> None:
        if self.state_path is None:
            return
        hosts: dict[str, dict[str, Any]] = {}
        for rung in self.ordered():
            entry = hosts.setdefault(rung.name, {"name": rung.name, "ok": None, "latency_ms": 0,
                                                  "model": rung.model, "models": []})
            entry["models"].append(rung.model)
            entry["latency_ms"] = max(entry["latency_ms"], rung.latency_ms)
            entry["ok"] = rung.probed_ok if entry["ok"] is None else (entry["ok"] or rung.probed_ok)
            # keep the leanest shape seen on that host: it is the one that worked
            if rung.profile in PROFILE_NAMES and PROFILE_NAMES.index(rung.profile) > PROFILE_NAMES.index(
                    entry.get("profile") or "full"):
                entry["profile"] = rung.profile
        payload = {"checked_at": time.time(),
                   "rungs": [{**v, "models": list(dict.fromkeys(v["models"]))[:20]} for v in hosts.values()]}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass


def _expand(rungs: list[Rung]) -> list[Rung]:
    """One Rung object per (host, model): each pairing gets its own rate-gate clock."""
    out: list[Rung] = []
    seen: set[str] = set()
    for rung in rungs:
        models = [m for m in dict.fromkeys([*(str(x) for x in rung.models if x), rung.model]) if m] or ["default"]
        catalog = tuple(dict.fromkeys(rung.catalog or tuple(models)))
        for model in models:
            label = f"{rung.name}|{model}"
            if label in seen:
                continue
            seen.add(label)
            out.append(replace(rung, model=model, models=tuple(models), catalog=catalog or (model,)))
    return out


def rung_from_spec(spec: dict[str, Any]) -> Rung:
    models = tuple(str(m) for m in (spec.get("models") or ())) or (str(spec.get("model") or "default"),)
    env_keys = tuple(str(e) for e in (spec.get("env_keys") or spec.get("env") or ()))
    return Rung(
        name=str(spec.get("name", "custom")),
        base_url=str(spec.get("base_url", "http://127.0.0.1:11434/v1")).rstrip("/"),
        model=str(spec.get("model") or models[0]),
        kind=str(spec.get("kind", "free-anon")),
        models=models,
        catalog=models,
        env_keys=env_keys,
        keyless=bool(spec.get("keyless", not env_keys)),
        tools=bool(spec.get("tools", True)),
        min_interval_s=float(spec.get("min_interval_s", 0.0) or 0.0),
        note=str(spec.get("note", "")),
    )


def build_default_ladder(state_dir: Path | None = None) -> Ladder:
    """The ladder a surface gets when nobody hands it one.

    Reads the same `free` block `ma` uses (`$MHARO_HOME/config.json`), so
    `ma free --pick ovh` also changes what the TUI does. A missing or broken
    config is not an error: the built-in rungs still apply.
    """
    home = Path(state_dir) if state_dir else Path(os.environ.get("MHARO_HOME", "~/.mharo")).expanduser()
    ladder = Ladder(state_path=home / "ladder.json")
    section: dict[str, Any] = {}
    for candidate in (home / "config.json", Path.cwd() / ".mharo" / "config.json"):
        try:
            if candidate.is_file():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("free"), dict):
                    section = dict(raw["free"])
                break
        except (ValueError, OSError):
            continue
    if not section:
        return ladder
    disable = {str(d) for d in (section.get("disable") or [])}
    rebuilt = Ladder(
        rungs=[rung_from_spec(spec) for spec in section["rungs"]] if section.get("rungs") else None,
        enabled=bool(section.get("enabled", True)),
        prefer=tuple(section.get("prefer") or ()),
        intervals=dict(section.get("min_interval_s") or {}),
        extra=list(section.get("extra_rungs") or []),
        allow_paid=bool(section.get("allow_paid", True)),
        state_path=home / "ladder.json",
    )
    if disable:
        rebuilt.rungs = [r for r in rebuilt.rungs if r.name not in disable]
    rebuilt.sync_models()
    return rebuilt


def quota_hint(ladder: Ladder) -> str:
    """What to do when every free rung is spent — concrete, not poetic."""
    waiting = sorted((r for r in ladder.ordered() if r.kind != "paid"), key=lambda r: r.wait_s())
    soon = [r for r in waiting if r.wait_s() > 0][:2]
    if soon:
        return ("free rungs cooling: " + ", ".join(f"{r.name} back in {int(r.wait_s())}s" for r in soon)
                + " · add your own key: `ma keys add openai sk-…` (or start Ollama for unlimited local)")
    live = [r for r in ladder.ordered() if r.available() and r.kind != "paid"]
    if live:
        names = ", ".join(sorted({r.name for r in live})[:3])
        return (f"that rung is spent, but {names} still answers — the ladder rotates to it automatically, "
                "or add your own key: `ma keys add openai sk-…`")
    return ("no free rung available · start a local server (`ollama serve`), or add a key: "
            "`ma keys add openrouter sk-or-…` for $0/token `:free` models, or `ma free` to probe")
