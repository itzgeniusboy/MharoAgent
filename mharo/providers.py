"""P1-1 · Provider hub: tiers × keys × protocols, with rotation and fallback.

Protocols implemented for real (streaming SSE, tool calls in both dialects):
  openai     OpenAI /chat/completions  → also OpenRouter, Groq, Together,
                                          DeepSeek, vLLM, LM Studio, Ollama
  anthropic  /v1/messages with content blocks
  replay     deterministic scripted turns — what `ma bench` and the tests drive
                                          the *engine* through without a network

Every call goes through `call()`, which:
  1. picks the tier, 2. rotates keys via KeyPool, 3. retries on 429/5xx with the
  pool's cooldown, 4. falls back through `config.fallbacks`, 5. records usage
  (tokens, latency, which key) into the SQLite ledger, 6. redacts before anything
  reaches the transcript.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from mharo_tui.agent.providers import (
    AnthropicProvider,
    AutoProvider,
    DemoProvider,
    OpenAICompatProvider,
    Provider,
    ProviderError,
)
from mharo_tui.agent.session import Message, estimate_tokens

from .keys import KeyPool
from .ladder import FreePool, build_ladder, classify, quota_hint
from .config import Config, TierSpec


@dataclass
class CallReport:
    tier: str
    provider: str
    model: str
    key_label: str
    attempts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    delivered: bool = False
    fell_back_to: str = ""


class ReplayProvider(Provider):
    """A real provider object that replays scripted events.

    Used by `ma bench`/tests to exercise the whole engine offline, and by
    `ma --replay` to demo a session. No network, no pretending: it streams the
    same event shapes a live model would, including tool calls.
    """

    name = "replay"
    supports_tools = True

    def __init__(self, model: str = "replay", script: list[dict] | None = None, **opts: Any) -> None:
        super().__init__(model, **opts)
        self.script = list(script or [])
        self.cursor = 0
        self.calls: list[list[Message]] = []

    def push(self, *turns: dict) -> None:
        self.script.extend(turns)

    async def stream(self, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        self.calls.append(list(messages))
        turn = self.script[self.cursor] if self.cursor < len(self.script) else {"text": "(replay script exhausted)"}
        self.cursor += 1
        for event in turn.get("events", []):
            yield event
            await asyncio.sleep(0)
        for chunk in _split(turn.get("text", "")):
            yield {"type": "text", "text": chunk}
        for call in turn.get("calls", []):
            yield {"type": "tool_call", "tool": call["tool"], "args": call.get("args", {}), "call_id": call.get("id", f"r{self.cursor}")}
        usage = turn.get("usage")
        if usage:
            yield {"type": "usage", **usage}
        elif turn.get("text") or turn.get("calls"):
            yield {
                "type": "usage",
                "input_tokens": estimate_tokens("\n".join(m.text for m in messages)),
                "output_tokens": max(1, len(turn.get("text", "")) // 4),
            }
        if turn.get("error"):
            yield {"type": "error", "text": turn["error"]}
        yield {"type": "done", "stop_reason": turn.get("stop", "tool_use" if turn.get("calls") else "end_turn")}


def _split(text: str, size: int = 26) -> list[str]:
    """Chunk text for streaming *without altering it*.

    Joining the result must reproduce the input exactly — replayed answers are
    parsed as JSON by callers, so a stray double space would corrupt them.
    """
    if not text:
        return []
    out: list[str] = []
    buf = ""
    for word in text.split(" "):
        candidate = f"{buf} {word}" if buf else word
        if len(candidate) > size and buf:
            out.append(buf + " ")
            buf = word
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


@dataclass
class TierHandle:
    tier: TierSpec
    provider: Provider
    pool: KeyPool
    replay_script: list[dict] | None = None
    rescue: "TierHandle | None" = None  # free-ladder handle, tried when keys are spent


class ProviderHub:
    def __init__(
        self,
        config: Config,
        *,
        db: Any = None,
        redact: Callable[[str], str] | None = None,
        replay_scripts: dict[str, list[dict]] | None = None,
        force_provider: str | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self._redact = redact or (lambda s: s)
        self.replay_scripts = replay_scripts or {}
        self.force_provider = force_provider
        self._handles: dict[str, TierHandle] = {}
        self._replay: dict[int, Provider] = {}
        self.session_id: int | None = None
        self.last_report: CallReport | None = None
        self._ladder: Any = None
        self.notices: list[str] = []
        # extra kwargs handed to every provider class (tests inject an httpx
        # MockTransport here; nothing else needs to know about it)
        self.provider_opts: dict[str, Any] = {}

    # -- construction ----------------------------------------------------
    def handle(self, tier: str) -> TierHandle:  # noqa: C901 - one table of "what answers this tier"
        if tier in self._handles:
            return self._handles[tier]
        spec = self.config.tier(tier)
        provider_name = self.force_provider or spec.provider
        opts: dict[str, Any] = {
            "max_tokens": spec.max_output_tokens,
            "temperature": spec.temperature,
            **self.provider_opts,
        }
        base_url = None
        pool = KeyPool(provider_name)
        if provider_name != "replay":
            conf = self.config.provider(provider_name)
            base_url = conf.get("base_url")
            from .config import DEFAULTS as _DEF
            _default_base = (_DEF.get("providers", {}).get(provider_name) or {}).get("base_url")
            explicit_endpoint = bool(base_url and base_url != _default_base)
            env_names = self.config.key_envs(provider_name)
            pool = KeyPool.from_env(provider_name, env_names)
            extra = conf.get("keys") or conf.get("api_key") or []
            if isinstance(extra, str):
                extra = [extra]
            for token in extra:
                pool.add(str(token))
            if base_url:
                opts["base_url"] = base_url
        script = self.replay_scripts.get(tier) or self.replay_scripts.get("all")
        # A replay script (bench, tests, --replay-file) is deterministic by contract:
        # the free ladder must never get in front of it.
        use_replay = provider_name == "replay" or script is not None
        free_rungs: FreePool | None = None
        free_cfg = getattr(self.config, "free", {}) or {}
        if not use_replay and self.free_enabled():
            ladder = self.ladder()
            want_auto = provider_name in {"auto", "free", "local"}
            # No key at all -> the free ladder. A user-set base_url (a local server,
            # a gateway) is honoured exactly as written.
            if want_auto or (not pool.keys and not explicit_endpoint):
                candidates = ladder.candidates()
                if candidates or want_auto:
                    opts["ladder"] = ladder
                    opts["rotate"] = False
                    opts["max_rungs"] = int(free_cfg.get("max_rungs", 4))
                    provider = AutoProvider(spec.model, **opts)
                    extra = [(k.label, k.token) for k in pool.keys] if free_cfg.get("allow_paid", True) else []
                    free_rungs = FreePool.from_ladder(ladder, extra=extra)
                    if free_rungs.keys:
                        provider.use(free_rungs.rung_for(free_rungs.keys[0]) or candidates[0])
                        pool = free_rungs
                        base_url = provider.base_url if hasattr(provider, "base_url") else base_url
                    else:
                        free_rungs = None
                elif free_cfg.get("fallback_to_demo", True):
                    provider = DemoProvider("demo")
                    provider.name = "demo"
                    self.notices.append(
                        f"tier {tier!r}: no free rung reachable ({ladder.soothe() or 'offline'}) — "
                        "answers come from the offline demo provider; " + quota_hint(ladder)
                    )
                    self._handles[tier] = TierHandle(tier=spec, provider=provider, pool=KeyPool("demo"), replay_script=None)
                    return self._handles[tier]
        if use_replay:
            # One ReplayProvider per *script object*: tiers sharing a script share its
            # cursor, so a cheap->strong upgrade continues the same replay instead of
            # re-serving turn 0 to the stronger tier.
            key = id(script)
            provider = self._replay.get(key)
            if provider is None:
                provider = ReplayProvider(spec.model, script=script or [])
                provider.name = "replay"
                self._replay[key] = provider
        elif free_rungs is None:
            cls = AnthropicProvider if provider_name == "anthropic" else OpenAICompatProvider
            key = pool.keys[0].token if pool.keys else None
            try:
                provider = cls(spec.model, api_key=key, **opts)
            except ProviderError as exc:
                raise ProviderError(f"tier {tier!r} ({provider_name}): {exc}") from exc

        handle = TierHandle(tier=spec, provider=provider, pool=pool, replay_script=script)
        if not use_replay and not isinstance(pool, FreePool) and self.free_enabled() \
                and free_cfg.get("rescue_on_exhaustion", True):
            # The user has a key, so the key answers — but a throttled or out-of-credit
            # key must not fail the task while a free rung is sitting right there.
            rescue_opts = dict(opts)
            rescue_opts["ladder"] = self.ladder()
            rescue_opts["rotate"] = True
            rescue_opts["max_rungs"] = int(free_cfg.get("max_rungs", 4))
            rescue = AutoProvider(spec.model, **rescue_opts)
            handle.rescue = TierHandle(tier=spec, provider=rescue, pool=KeyPool("free"))
        self._handles[tier] = handle
        return handle

    def ladder(self):
        """The free-first ladder (probed rungs), cached per hub."""
        if self._ladder is None:
            self._ladder = build_ladder(self.config)
        return self._ladder

    def free_enabled(self) -> bool:
        section = getattr(self.config, "free", {}) or {}
        return bool(section.get("enabled", True)) and self.force_provider != "replay"

    def warm(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in self.config.tiers:
            try:
                h = self.handle(name)
                out[name] = f"{h.provider.name}:{h.provider.model}"
            except ProviderError as exc:
                out[name] = f"unavailable: {exc}"
        return out

    # -- the one entry point --------------------------------------------
    async def _respect_gate(self, handle: TierHandle, free_cfg: dict[str, Any]) -> float:
        """Sleep until the soonest free pairing is polite to use again.

        A 2-req/min tier means the right move is usually to wait ~20 seconds, not to
        fail the task or hammer the endpoint. Only the *next* call is delayed, and
        only up to `free.patience_s`.
        """
        pool = handle.pool
        if not isinstance(pool, FreePool) or pool.usable_now():
            return 0.0
        patience = float(free_cfg.get("patience_s", 20.0) or 0.0)
        if patience <= 0:
            return 0.0
        # only *timed* blocks can be waited out; a closed local port has no timer,
        # so it must not win the "soonest free" race and cancel the wait
        gated = [r for r in pool.by_label.values() if not r.available() and r.wait_s() > 0]
        if not gated:
            return 0.0
        soonest = min(gated, key=lambda r: r.wait_s())
        wait = soonest.wait_s()
        if wait > patience:
            return 0.0
        self.notices.append(f"waiting {wait:.0f}s for the free-tier rate gate on {soonest.label()}")
        await asyncio.sleep(wait)
        return wait

    async def stream(self, tier: str, messages: list[Message], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        """Stream from `tier`, rotating keys and falling back across tiers."""
        order = [tier] + [t for t in self.config.raw.get("fallbacks", []) if t != tier and t in self.config.tiers]
        report = CallReport(tier=tier, provider="", model="", key_label="")
        free_cfg = getattr(self.config, "free", {}) or {}
        started = time.monotonic()
        last_error: str | None = None
        used_free = False
        for candidate in order:
            try:
                handle = self.handle(candidate)
            except ProviderError as exc:
                report.errors.append(f"{candidate}: {exc}")
                last_error = str(exc)
                continue
            report.provider = handle.provider.name
            report.model = handle.provider.model
            used_free = used_free or isinstance(handle.pool, FreePool)
            attempts_before = report.attempts
            while True:
                waited = await self._respect_gate(handle, free_cfg)
                if waited:
                    report.errors.append(f"{candidate}: waited {waited:.0f}s for a free tier gate")
                state = handle.pool.next() if handle.pool.keys else None
                if handle.pool.keys and state is None:
                    if isinstance(handle.pool, FreePool):
                        report.errors.append(f"{candidate}: {handle.pool.ladder.soothe() or 'all rungs cooling'} "
                                             f"[{quota_hint(handle.pool.ladder)}]")
                    else:
                        report.errors.append(f"{candidate}: all {len(handle.pool.keys)} keys cooling/disabled")
                    break
                report.attempts += 1
                if state:
                    report.key_label = state.label
                    if isinstance(handle.pool, FreePool) and hasattr(handle.provider, "use"):
                        rung = handle.pool.rung_for(state)
                        if rung is None:
                            continue
                        handle.provider.use(rung)
                        report.model = rung.model
                        report.provider = f"free:{rung.name}"
                    else:
                        _apply_key(handle.provider, state.token)
                try:
                    async for event in self._pump(handle, messages, tools, report):
                        yield event
                    if state:
                        handle.pool.report_success(state)
                    break
                except ProviderError as exc:
                    text = self._redact(str(exc))
                    last_error = text
                    if state is not None and isinstance(handle.pool, FreePool) and not report.delivered \
                            and classify(exc) == "shape":
                        rung = handle.pool.rung_for(state)
                        if rung is not None and handle.pool.ladder.downgrade_profile(rung):
                            # the endpoint is fine, our body was not — retry leaner, same rung
                            report.errors.append(f"{candidate}[shape]: {rung.name} → {rung.profile}")
                            continue
                    kind = handle.pool.report_failure(state, exc) if state else "no-keys"
                    report.errors.append(f"{candidate}[{kind}]: {text[:200]}")
                    if kind == "auth" and handle.pool.usable_now():
                        continue                     # try the next key immediately
                    if not handle.pool.usable_now():
                        break                        # nothing left in this tier
            if report.delivered:
                break
            if handle.rescue is not None and not report.delivered:
                # your keys are throttled / out of credit — finish on a free rung instead of failing
                rung = handle.rescue.provider.ladder.pick()
                if rung is not None:
                    handle.rescue.provider.use(rung)
                    before = report.tokens_out
                    async for event in self._pump(handle.rescue, messages, tools, report):
                        yield event
                    if report.delivered or report.tokens_out > before:
                        handle.rescue.provider.ladder.mark_used(rung)
                        report.provider = f"free:{rung.name}"
                        report.model = rung.model
                        report.fell_back_to = f"free:{rung.name}"
                        self.notices.append(
                            f"{candidate}: {provider_key_reason(handle.pool)} — finished on a free rung "
                            f"({rung.label()}) so the task could continue"
                        )
            if report.delivered:
                break            # a rescue answered: no other tier gets to speak
            if report.attempts > attempts_before:
                report.fell_back_to = report.fell_back_to or candidate
        if not report.delivered and used_free and (getattr(self.config, "free", {}) or {}).get("fallback_to_demo", True):
            # The ladder had nothing to give (offline box, no keys, no local server).
            # Rather than fail the task in silence, answer from the offline demo and say so.
            demo = DemoProvider("demo")
            demo.name = "demo"
            self.notices.append(
                "free ladder could not deliver — answering from the offline demo provider (no real model call). "
                + quota_hint(self.ladder())
            )
            report.fell_back_to = "demo"
            report.provider, report.model = "demo", "demo"
            demo_handle = TierHandle(tier=self.config.tier(report.tier), provider=demo, pool=KeyPool("demo"), replay_script=None)
            async for event in self._pump(demo_handle, messages, tools, report):
                yield event
            self._handles[f"{report.tier}+demo"] = demo_handle
        report.latency_ms = int((time.monotonic() - started) * 1000)
        report.cost_usd = self._cost(report)
        self.last_report = report
        if self.db is not None and getattr(self, "session_id", None):
            try:
                self.db.add_usage(
                    self.session_id, provider=report.provider, model=report.model, tier=report.tier,
                    tokens_in=report.tokens_in, tokens_out=report.tokens_out, cost_usd=report.cost_usd,
                    key_label=report.key_label, latency_ms=report.latency_ms,
                )
            except Exception:
                pass
        if report.errors and report.tokens_out == 0:
            yield {
                "type": "error",
                "text": "every provider/key failed — " + (last_error or "; ".join(report.errors[-3:])),
            }

    def drain_notices(self) -> list[str]:
        """Free-ladder notices collected since the last drain (rate gates, backoffs).

        The ladder keeps its own log (payload-shape repairs, per-host backoffs); it is
        pulled in here so `ma` shows *why* a rung changed rather than silently rerouting.
        """
        if self._ladder is not None and self._ladder.notices:
            self.notices.extend(f"ladder: {note}" for note in self._ladder.notices)
            self._ladder.notices.clear()
        out, self.notices = list(self.notices), []
        return out

    async def _pump(self, handle: TierHandle, messages: list[Message], tools: list[dict] | None, report: CallReport) -> AsyncIterator[dict]:
        async for event in handle.provider.stream(messages, tools):
            kind = event.get("type")
            if kind == "usage":
                report.tokens_in += int(event.get("input_tokens", 0) or 0)
                report.tokens_out += int(event.get("output_tokens", 0) or 0)
            elif kind == "text":
                report.delivered = True
                event["text"] = self._redact(event.get("text", ""))
            elif kind == "tool_call":
                report.delivered = True
            elif kind == "error":
                raise ProviderError(event.get("text", "provider error"))
            yield event

    def _cost(self, report: CallReport) -> float:
        prices = self.config.prices()
        pin, pout = prices.get(report.model, (0.0, 0.0))
        return round(report.tokens_in / 1e6 * pin + report.tokens_out / 1e6 * pout, 6)

    def complete(self, tier: str, prompt: str) -> Any:
        """One-shot completion coroutine (summaries, peer review)."""
        return self._complete(tier, prompt)

    async def _complete(self, tier: str, prompt: str) -> str:
        chunks: list[str] = []
        msgs = [Message(role="user", blocks=[_text(prompt)])]
        async for evt in self.stream(tier, msgs, tools=None):
            if evt.get("type") == "text":
                chunks.append(evt["text"])
        return "".join(chunks).strip()

    @property
    def redact(self) -> Callable[[str], str]:
        return self._redact

    @redact.setter
    def redact(self, fn: Callable[[str], str] | None) -> None:
        """Late wiring (`ma --no-redact`): always a callable, never None."""
        self._redact = fn or (lambda s: s)

    def free_status(self) -> dict[str, Any]:
        """What `ma free --json` and `ma doctor` read."""
        ladder = self.ladder()
        pool = next((h.pool for h in self._handles.values() if isinstance(h.pool, FreePool)), None)
        return {
            "enabled": self.free_enabled(),
            "summary": ladder.summary(),
            "soothe": ladder.soothe(),
            "notices": list(ladder.notices[-6:]),
            "hint": quota_hint(ladder) if not any(r.available() for r in ladder.ordered()) else "",
            "rungs": ladder.status(),
            "pool": pool.status() if pool is not None else None,
        }

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.config.tiers:
            try:
                handle = self.handle(name)
                out[name] = {
                    "provider": handle.provider.name,
                    "model": handle.provider.model,
                    "pool": handle.pool.status(),
                }
            except ProviderError as exc:
                out[name] = {"error": str(exc)}
        return out


def provider_key_reason(pool: KeyPool) -> str:
    """Why the paid keys stopped working, in one clause."""
    status = pool.status()
    if status.get("disabled"):
        return f"{status['disabled']} key(s) disabled by auth errors"
    if status.get("cooling"):
        return f"all {status['keys']} key(s) throttled (retry in {status['retry_in_s']:.0f}s)"
    return "no usable key"


def _apply_key(provider: Provider, token: str) -> None:
    if not token:
        return
    if isinstance(provider, AnthropicProvider):
        provider.api_key = token
    elif isinstance(provider, OpenAICompatProvider):
        provider.api_key = token


def _text(value: str):
    from mharo_tui.agent.session import Text

    return Text(value)
