"""Free-first ladder: no key, real rungs, honest rate gates.

Everything here is offline: provider traffic is intercepted by `httpx.MockTransport`,
so the assertions cover the real request/response path (headers, SSE parsing,
429 rotation) without touching the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from mharo_tui.agent.ladder import Ladder, Rung, build_default_ladder, classify, merge_models, quota_hint
from mharo_tui.agent.providers import AutoProvider, OpenAICompatProvider, ProviderError, get_provider
from mharo_tui.agent.session import Message, Text
from mharo.ladder import FreePool, ask_rung, build_ladder


def sse(*chunks: dict) -> str:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def chat_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4}}


def run(coro):
    return asyncio.run(coro)


def ladder_with(*specs: dict) -> Ladder:
    """A ladder holding exactly these rungs (config `free.rungs` is a full override)."""
    return Ladder(
        rungs=None,
        prefer=(),
        extra=list(specs),
        state_path=None,
    )


LADDER_ONLY = Ladder  # readability


# ---------------------------------------------------------------- providers
def test_keyless_rung_sends_no_authorization_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("hello from a free model")))

    provider = OpenAICompatProvider("gpt-oss-20b", base_url="https://free.example/v1",
                                    keyless=True, transport=httpx.MockTransport(handler))
    messages = [Message(role="user", blocks=[Text("hi")])]
    events = run(_collect(provider.stream(messages)))
    assert "authorization" not in {k.lower() for k in seen}
    assert "".join(e["text"] for e in events if e["type"] == "text") == "hello from a free model"


def test_paid_endpoint_without_a_key_is_refused_with_a_ladder_hint() -> None:
    with pytest.raises(ProviderError, match="auto"):
        OpenAICompatProvider("gpt-4o-mini", base_url="https://api.openai.com/v1")


def test_empty_stream_is_an_error_not_a_silent_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="")

    provider = OpenAICompatProvider("openai-fast", base_url="https://free.example/v1", keyless=True,
                                    transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="empty response"):
        run(_collect(provider.stream([Message(role="user", blocks=[Text("hi")])])))


async def _collect(stream) -> list[dict]:
    return [event async for event in stream]


# ---------------------------------------------------------------- rung rules
def test_rate_gate_blocks_the_next_call_on_the_same_pairing() -> None:
    rung = Rung(name="ovh", base_url="https://free.example/v1", model="gpt-oss-120b",
                models=("gpt-oss-120b", "gpt-oss-20b"), min_interval_s=31.0)
    assert rung.available()
    rung.mark_used()
    assert not rung.available()
    assert 25 < rung.wait_s() <= 31


def test_quota_backoff_is_long_and_names_the_escape_hatch() -> None:
    ladder = Ladder(rungs=[Rung(name="ovh", base_url="https://free.example/v1", model="m1",
                               models=("m1",), min_interval_s=31.0)])
    seconds = ladder.mark_failure(ladder.rungs[0], "quota", error="402 quota exceeded")
    assert seconds >= 600
    assert "ma keys add" in quota_hint(ladder)
    assert ladder.pick() is None
    assert "quota exhausted" in ladder.soothe()


def test_models_on_one_host_rotate_instead_of_hammering() -> None:
    ladder = Ladder(rungs=[Rung(name="ovh", base_url="https://free.example/v1", model="a",
                                 models=("a", "b", "c"), min_interval_s=31.0)])
    first = ladder.pick()
    assert first is not None and first.model == "a"
    ladder.mark_used(first)
    second = ladder.pick()
    assert second is not None and second.model == "b"
    assert second.base_url == first.base_url


def test_dead_local_server_is_skipped_and_throttle_keeps_a_rung_alive() -> None:
    dead = Rung(name="ollama", base_url="http://127.0.0.1:1/v1", model="m", kind="local", probed_ok=False)
    free = Rung(name="ovh", base_url="https://free.example/v1", model="gpt-oss-120b", min_interval_s=31.0)
    ladder = Ladder(rungs=[dead, free])
    assert ladder.pick().name == "ovh"
    ladder.mark_failure(free, "throttle", error="HTTP 429 Too Many Requests")
    assert free.probed_ok is not False  # a 429 proves the endpoint is up
    assert classify("HTTP 429 Too Many Requests") == "throttle"
    assert classify("402 payment required: quota exceeded") == "quota"
    assert classify("Couldn't connect to host") == "unreachable"


def test_public_model_list_cannot_promote_a_paid_model() -> None:
    rung = Rung(name="openrouter", base_url="https://openrouter.ai/api/v1", model="x:free",
                models=("x:free",), catalog=("x:free",), keyless=False, env_keys=("OPENROUTER_API_KEY",))
    merged = merge_models(rung, ("paid/model-pro", "x:free", "another"))
    assert merged == ("x:free",)
    local = Rung(name="ollama", base_url="http://127.0.0.1:11434/v1", model="old",
                 models=("old",), kind="local")
    assert merge_models(local, ("newone:1", "other:2")) == ("newone:1", "other:2")


# ---------------------------------------------------------------- auto provider
def test_auto_provider_skips_a_429_rung_and_answers_from_the_next() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "angry":
            return httpx.Response(429, json={"message": "API rate limit exceeded"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("free answer"), {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    rungs = [
        Rung(name="one", base_url="https://free.example/v1", model="angry", models=("angry",), min_interval_s=31.0),
        Rung(name="two", base_url="https://free.example/v1", model="calm", models=("calm",), min_interval_s=31.0),
    ]
    provider = AutoProvider(ladder=Ladder(rungs=rungs), transport=httpx.MockTransport(handler))
    events = run(_collect(provider.stream([Message(role="user", blocks=[Text("hi")])])))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "free answer"
    assert calls == ["angry", "calm"]
    assert provider.notices and "one:angry → throttle" in provider.notices[0]
    assert provider.describe().endswith("two:calm")


def test_auto_provider_gives_up_with_instructions_not_a_stack_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "rate limit exceeded"})

    rungs = [Rung(name=f"r{i}", base_url="https://free.example/v1", model=f"m{i}", models=(f"m{i}",))
             for i in range(5)]
    provider = AutoProvider(ladder=Ladder(rungs=rungs), transport=httpx.MockTransport(handler), max_rungs=3)
    with pytest.raises(ProviderError) as exc:
        run(_collect(provider.stream([Message(role="user", blocks=[Text("hi")])])))
    assert "free providers exhausted" in str(exc.value)
    assert "ma keys add" in str(exc.value)


def test_default_provider_with_no_key_is_the_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MHARO_PROVIDER", "OPENAI_API_KEY", "MHARO_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MHARO_HOME", "/nonexistent-mharo-home")
    assert isinstance(get_provider(None), AutoProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert get_provider(None).name == "openai"


def test_build_default_ladder_reads_the_shared_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(
        {"free": {"prefer": ["mine"], "extra_rungs": [{"name": "mine", "base_url": "http://127.0.0.1:9/v1",
                                                       "models": ["m"], "min_interval_s": 7}]}}), encoding="utf-8")
    ladder = build_default_ladder(tmp_path)
    assert ladder.prefer == ("mine",)
    assert any(r.name == "mine" and r.min_interval_s == 7.0 for r in ladder.rungs)


# ---------------------------------------------------------------- free pool
def test_free_pool_looks_like_a_key_pool() -> None:
    ladder = Ladder(rungs=[Rung(name="ovh", base_url="https://free.example/v1", model="a",
                                 models=("a", "b"), min_interval_s=31.0)])
    pool = FreePool.from_ladder(ladder)
    assert len(pool.keys) == 2
    status = pool.status()
    assert status["kind"] == "ladder" and status["summary"].count("ovh") >= 1
    state = pool.next()
    assert state is not None and state.label == "ovh:a"
    pool.report_success(state)
    cooled = [k for k in pool.keys if not k.available]
    assert [k.label for k in cooled] == ["ovh:a"], "only the used pairing is gated"
    assert [k.label for k in pool.usable_now()] == ["ovh:b"], "the sibling model keeps answering"
    assert pool.status()["cooling"] == 1


def test_free_pool_backs_the_whole_rung_off_on_quota() -> None:
    rung = Rung(name="ovh", base_url="https://free.example/v1", model="a", models=("a", "b"), min_interval_s=31.0)
    ladder = Ladder(rungs=[rung])
    pool = FreePool.from_ladder(ladder)
    state = pool.next()
    kind = pool.report_failure(state, ProviderError("HTTP 402 quota exceeded for this key"))
    assert kind == "quota"
    assert not pool.usable_now()
    assert "quota exhausted" in pool.status()["hint"] or "ma keys add" in pool.status()["hint"]


def test_ask_rung_reports_failures_structurally() -> None:
    rung = Rung(name="x", base_url="https://free.example/v1", model="m")
    result = ask_rung(rung, "hi")
    assert result["ok"] is False
    assert "free.example" in result["error"] or "Name" in result["error"] or "error" in result


# ---------------------------------------------------------------- engine wiring
def test_hub_with_no_keys_at_all_runs_on_the_ladder(tmp_path: Path) -> None:
    from mharo.config import Config
    from mharo.providers import ProviderHub

    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "rungs": [
        {"name": "fake", "base_url": "https://free.example/v1", "models": ["gpt-oss-120b", "gpt-oss-20b"],
         "min_interval_s": 31}]}
    cfg.raw["tiers"]["cheap"] = {"provider": "openai", "model": "gpt-4o-mini"}
    hub = ProviderHub(cfg)
    handle = hub.handle("cheap")
    assert isinstance(handle.provider, AutoProvider)
    assert isinstance(handle.pool, FreePool) and len(handle.pool.keys) == 2
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("done: nothing to change"),
                                       {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    hub._handles.clear()
    handle = hub.handle("cheap")
    events = run(_collect(hub.stream("cheap", [Message(role="user", blocks=[Text("hi")])])))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "done: nothing to change"
    assert handle.provider.name == "auto"
    report = hub.last_report
    assert report.provider.startswith("free:") and report.cost_usd == 0.0
    assert report.model in {"gpt-oss-120b", "gpt-oss-20b"}
    assert len(calls) == 1


def test_hub_falls_back_to_the_demo_instead_of_failing_the_task(tmp_path: Path) -> None:
    from mharo.config import Config
    from mharo.providers import ProviderHub

    cfg = Config.load()
    # nothing reachable: no keys, and every rung points at a closed local port
    cfg.raw["free"] = {"enabled": True, "rungs": [
        {"name": "gonedown", "base_url": "http://127.0.0.1:1/v1", "models": ["m"]}]}
    cfg.raw["tiers"]["cheap"] = {"provider": "auto", "model": "auto"}
    hub = ProviderHub(cfg)
    events = run(_collect(hub.stream("cheap", [Message(role="user", blocks=[Text("hello")])])))
    assert hub.last_report.delivered or any(e["type"] == "text" for e in events)
    assert hub.last_report.fell_back_to == "demo"
    assert any("offline demo" in note for note in hub.drain_notices())


def test_replay_scripts_are_never_stolen_by_the_ladder(tmp_path: Path) -> None:
    from mharo.bench import _engine

    engine = _engine(tmp_path, [{"text": "planned"}, {"text": "all good"}])
    handle = engine.hub.handle("cheap")
    assert not isinstance(handle.pool, FreePool), "bench/tests must stay deterministic"
    assert handle.provider.name == "replay"


# ---------------------------------------------------------------- CLI + doctor
def _argv(*args: str):
    from mharo.cli import main

    return main(list(args))


def test_ma_free_reports_the_ladder_as_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    assert _argv("free", "--json", "--cwd", str(tmp_path)) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {r["name"] for r in payload["rungs"]}
    assert {"ovh", "ollama", "openrouter"} <= names
    assert payload["enabled"] is True
    assert any(r["kind"] == "free-anon" and r["keyless"] for r in payload["rungs"])


def test_ma_free_pick_writes_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    assert _argv("free", "--pick", "ovh", "--cwd", str(tmp_path)) == 0
    capsys.readouterr()
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["free"]["prefer"] == ["ovh"]
    assert cfg["tiers"]["cheap"]["provider"] == "auto"
    assert Path(tmp_path / "config.json").stat().st_mode & 0o077 == 0, "config with keys must not be world-readable"
    assert _argv("free", "--pick", "nosuchrung", "--cwd", str(tmp_path)) == 2


def test_ma_keys_add_then_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    assert _argv("keys", "add", "openai", "sk-super-secret-value-1234", "--cwd", str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "sk-super-secret-value-1234" not in out, "the key must never be echoed back"
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["providers"]["openai"]["keys"] == ["sk-super-secret-value-1234"]
    assert _argv("keys", "--json", "--cwd", str(tmp_path)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["openai"]["keys"] == 1 and status["openai"]["available"] == 1
    assert _argv("keys", "rm", "openai", "--cwd", str(tmp_path)) == 0
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["providers"]["openai"]["keys"] == []


def test_a_configured_key_answers_first_with_a_free_rescue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key is an explicit choice, so it wins — but a spent key must not fail the task."""
    from mharo.config import Config
    from mharo.providers import ProviderHub

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-paid-key")
    for name in ("MHARO_OPENAI_KEYS", "OPENAI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "rungs": [
        {"name": "fake", "base_url": "https://free.example/v1", "models": ["gpt-oss-20b"], "min_interval_s": 31}]}
    hub = ProviderHub(cfg)
    handle = hub.handle("cheap")
    assert not isinstance(handle.pool, FreePool), "your key leads"
    assert handle.pool.keys[0].token == "sk-paid-key"
    assert isinstance(handle.rescue.provider, AutoProvider), "the ladder waits in the wings"

    def handler(request: httpx.Request) -> httpx.Response:
        if "free.example" in str(request.url):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=sse(chat_chunk("rescued for free"),
                                           {"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        return httpx.Response(429, json={"error": {"message": "Rate limit reached for gpt-4o-mini"}})

    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    hub._handles.clear()
    events = run(_collect(hub.stream("cheap", [Message(role="user", blocks=[Text("hi")])])))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "rescued for free"
    assert hub.last_report.provider == "free:fake"
    assert any("finished on a free rung" in note for note in hub.drain_notices())


def test_doctor_covers_the_free_ladder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mharo.config import Config
    from mharo.doctor import Doctor

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    cfg = Config.load()
    results = run(Doctor(cfg, cwd=tmp_path).run_all())
    ids = {r.name for r in results}
    assert {"free.enabled", "free.ladder", "free.rate-gate", "free.keyless"} <= ids
    assert not [r for r in results if r.name.startswith("free.") and r.status == "FAIL"]
    keyless = next(r for r in results if r.name == "free.keyless")
    assert "keys add" in (keyless.hint or "") or keyless.status == "PASS"


# ---------------------------------------------------------------- TUI side
def test_tui_free_command_shows_the_table() -> None:
    from mharo_tui.commands import COMMANDS, cmd_free

    ladder = Ladder(rungs=[Rung(name="ovh", base_url="https://free.example/v1", model="gpt-oss-120b",
                                models=("gpt-oss-120b",), min_interval_s=31.0)])

    class FakeAgent:
        provider = AutoProvider(ladder=ladder)
        stats = lambda self: {}  # noqa: E731

    class FakeApp:
        agent = FakeAgent()

    out = cmd_free(FakeApp(), "")
    assert "free · no key" in out and "ovh" in out
    assert "free" in COMMANDS and "/free" not in ""


def test_tui_provider_command_accepts_auto() -> None:
    from mharo_tui.commands import cmd_provider

    class FakeApp:
        def set_provider(self, name: str) -> str:
            self.chosen = name
            return f"ok {name}"

    app = FakeApp()
    assert cmd_provider(app, "auto") == "ok auto"
    assert "Usage" in cmd_provider(app, "wat")


def test_build_ladder_honours_config_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`free.disable`, `free.min_interval_s` and `free.extra_rungs` all reach the rungs."""
    from mharo.config import Config

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    for name in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "LLM7_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = Config.load()
    cfg.raw["free"] = {
        "disable": ["ollama", "lmstudio", "llamacpp", "ovh", "pollinations",
                    "openrouter", "llm7", "gemini", "groq"],
        "min_interval_s": {"vllm": 5},
        "extra_rungs": [{"name": "mine", "base_url": "http://127.0.0.1:9/v1",
                         "models": ["a", "b"], "min_interval_s": 3}],
    }
    ladder = build_ladder(cfg)
    assert {r.name for r in ladder.rungs} == {"vllm", "mine"}
    assert {r.min_interval_s for r in ladder.rungs if r.name == "vllm"} == {5.0}
    assert sum(1 for r in ladder.rungs if r.name == "mine") == 2, "one pairing per model"
    assert ladder.state_path == tmp_path / "ladder.json"
    assert ladder.pick() is None  # both hosts are closed ports, so nothing is usable
    assert "down" in ladder.soothe() or "unavailable" in ladder.soothe()


def test_free_tier_gate_is_waited_out_not_tripped_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One pairing, a 0.5s gate: the second call waits instead of failing or hammering."""
    from mharo.config import Config
    from mharo.providers import ProviderHub

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("answer"),
                                        {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "patience_s": 3.0, "rungs": [
        {"name": "slow", "base_url": "https://free.example/v1", "models": ["m"], "min_interval_s": 0.6}]}
    hub = ProviderHub(cfg)
    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    msgs = [Message(role="user", blocks=[Text("hi")])]
    first = run(_collect(hub.stream("cheap", msgs)))
    second = run(_collect(hub.stream("cheap", msgs)))
    assert len(calls) == 2, "both calls went through"
    assert any(e["type"] == "text" for e in first) and any(e["type"] == "text" for e in second)
    assert any("rate gate" in note for note in hub.drain_notices())


def test_no_patience_means_no_hammering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """patience_s=0: the gated pairing stays put and the run says so instead of sleeping."""
    from mharo.config import Config
    from mharo.providers import ProviderHub

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("x"), {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "patience_s": 0, "fallback_to_demo": False, "rungs": [
        {"name": "slow", "base_url": "https://free.example/v1", "models": ["m"], "min_interval_s": 3600}]}
    hub = ProviderHub(cfg)
    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    msgs = [Message(role="user", blocks=[Text("hi")])]
    run(_collect(hub.stream("cheap", msgs)))
    events = run(_collect(hub.stream("cheap", msgs)))
    assert not any(e["type"] == "text" and e["text"] != "x" for e in events)
    assert hub.last_report.errors, "the second call must report the gate, not fake an answer"


def test_tui_provider_waits_then_answers() -> None:
    import time

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse(chat_chunk("polite"),
                                       {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    rung = Rung(name="solo", base_url="https://free.example/v1", model="m", models=("m",), min_interval_s=0.5)
    provider = AutoProvider(ladder=Ladder(rungs=[rung], state_path=None),
                            transport=httpx.MockTransport(handler), patience_s=3.0)
    msgs = [Message(role="user", blocks=[Text("hi")])]
    run(_collect(provider.stream(msgs)))
    started = time.monotonic()
    events = run(_collect(provider.stream(msgs)))
    waited = time.monotonic() - started
    assert "".join(e["text"] for e in events if e["type"] == "text") == "polite"
    assert waited >= 0.4, "the gate was respected, not skipped"


def test_throttle_storm_retires_the_whole_host() -> None:
    """Shared anonymous pools saturate; once three 429s land on a host, stop knocking."""
    rungs = [Rung(name="busy", base_url="https://free.example/v1", model=f"m{i}", models=(f"m{i}",))
             for i in range(4)]
    ladder = Ladder(rungs=rungs, state_path=None)
    for i in range(3):
        ladder.mark_failure(ladder.rungs[i], "throttle", error="HTTP 429 rate limit")
    assert not any(r.available() for r in ladder.rungs), "the whole host is resting"
    assert ladder.pick() is None
    assert "rate-limited" in ladder.soothe() or "cooling" in ladder.soothe()
    calm = Rung(name="other", base_url="https://elsewhere.example/v1", model="m")
    ladder.rungs.append(calm)
    assert ladder.pick().name == "other", "another host still answers"


# ------------------------------------------------------------- payload shapes
def sse_text(text: str) -> str:
    return sse({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2}})


def test_host_learns_the_payload_shape_it_accepts() -> None:
    """OVH answers a plain completion but 422s on `stream_options`: learn it, cache it."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "stream_options" in body:
            return httpx.Response(422, json={"detail": "extra inputs are not permitted"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse_text("lean answer"))

    ladder = Ladder(rungs=[Rung(name="fussy", base_url="https://free.example/v1", model="m",
                               models=("m",), min_interval_s=0.0)], state_path=None)
    provider = AutoProvider(ladder=ladder, transport=httpx.MockTransport(handler))
    msgs = [Message(role="user", blocks=[Text("hi")])]
    events = run(_collect(provider.stream(msgs)))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "lean answer"
    assert "stream_options" in bodies[0] and "stream_options" not in bodies[1]
    assert ladder.rungs[0].profile == "no_stream_options"
    assert any("request shape" in note for note in ladder.notices)
    # and it remembers: the next call starts with the body that already worked
    bodies.clear()
    run(_collect(provider.stream(msgs)))
    assert bodies and "stream_options" not in bodies[0]


def test_tools_are_dropped_when_the_endpoint_refuses_them() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body or "stream_options" in body:
            return httpx.Response(400, json={"error": {"message": "unsupported parameter: tools"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse_text("no tools here"))

    ladder = Ladder(rungs=[Rung(name="plain", base_url="https://free.example/v1", model="m",
                                models=("m",), min_interval_s=0.0)], state_path=None)
    provider = AutoProvider(ladder=ladder, transport=httpx.MockTransport(handler))
    msgs = [Message(role="user", blocks=[Text("hi")])]
    events = run(_collect(provider.stream(msgs, tools=[{"name": "bash", "description": "run", "parameters": {}}])))
    assert "".join(e["text"] for e in events if e["type"] == "text") == "no tools here"
    assert ladder.rungs[0].profile == "no_tools"


def test_minimal_profile_parses_tool_calls_without_streaming() -> None:
    """A non-streaming reply still has to drive the tool loop."""
    reply = {"choices": [{"message": {"role": "assistant", "content": "running it",
                                      "tool_calls": [{"id": "t1", "function": {
                                          "name": "bash", "arguments": '{"command": "echo hi"}'}}]},
                          "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 6}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert "stream" not in json.loads(request.content)
        return httpx.Response(200, json=reply)

    provider = OpenAICompatProvider("m", base_url="https://free.example/v1", keyless=True,
                                    payload_profile="minimal", transport=httpx.MockTransport(handler))
    events = run(_collect(provider.stream([Message(role="user", blocks=[Text("hi")])],
                                          tools=[{"name": "bash", "description": "run", "parameters": {}}])))
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and "usage" in kinds and kinds[-1] == "done"
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["tool"] == "bash" and call["args"] == {"command": "echo hi"}


def test_shape_failure_cools_nothing() -> None:
    rung = Rung(name="fussy", base_url="https://free.example/v1", model="m", models=("m",), min_interval_s=31.0)
    ladder = Ladder(rungs=[rung], state_path=None)
    assert ladder.mark_failure(rung, "shape", error="HTTP 422") == 0.0
    assert rung.available(), "a rejected body is not a rate limit"
    assert not ladder.notices, "downgrade notices come from downgrade_profile, not mark_failure"


def test_engine_retries_a_leaner_body_instead_of_blaming_the_rung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mharo.config import Config
    from mharo.providers import ProviderHub

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body or "stream_options" in body:
            return httpx.Response(422, json={"detail": "body failed validation"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse_text("engine still answers"))

    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "rungs": [
        {"name": "fake", "base_url": "https://free.example/v1", "models": ["gpt-oss-120b"], "min_interval_s": 31}]}
    hub = ProviderHub(cfg)
    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    events = run(_collect(hub.stream("cheap", [Message(role="user", blocks=[Text("hi")])],
                                    tools=[{"name": "bash", "description": "run", "parameters": {}}])))
    assert "".join(e["text"] for e in events if e["type"] == "text") == "engine still answers"
    report = hub.last_report
    assert report.delivered and report.cost_usd == 0.0
    notes = hub.drain_notices()
    assert any("shape" in note for note in notes), notes  # the repair is announced, not hidden
    handle = hub.handle("cheap")
    assert isinstance(handle.pool, FreePool)
    state = handle.pool.keys[0]
    assert state.failures == 0 and state.disabled is False, "a rejected body is not a rate limit"
    # the pairing is gated only by its own 31s window — no penalty from the 422
    import time as _t
    assert 0 <= state.cooldown_until - _t.monotonic() <= 32.0, state.cooldown_until - _t.monotonic()
    assert handle.pool.ladder.rungs[0].profile != "full", "the host learned a leaner payload"


def test_a_dead_local_port_is_not_offered_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 243-second lesson: a port we probed closed must not cost a call each turn."""
    from mharo.config import Config
    from mharo.providers import ProviderHub

    monkeypatch.setenv("MHARO_HOME", str(tmp_path))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse_text("no local server needed"))

    cfg = Config.load()
    cfg.raw["free"] = {"enabled": True, "patience_s": 3.0, "rungs": [
        {"name": "ollama", "base_url": "http://127.0.0.1:1/v1", "models": ["qwen2.5-coder:32b"]},
        {"name": "lmstudio", "base_url": "http://127.0.0.1:1/v1", "models": ["local-model"]},
        {"name": "fake", "base_url": "https://free.example/v1", "models": ["gpt-oss-20b"], "min_interval_s": 0.6},
    ]}
    hub = ProviderHub(cfg)
    hub.provider_opts["transport"] = httpx.MockTransport(handler)
    msgs = [Message(role="user", blocks=[Text("hi")])]
    answers = []
    for _ in range(3):
        events = run(_collect(hub.stream("cheap", msgs)))
        answers.append("".join(e["text"] for e in events if e["type"] == "text"))
    assert len(calls) == 3, "three turns, three real calls"
    assert all("free.example" in url for url in calls), calls
    assert answers == ["no local server needed"] * 3, "every turn rode the free rung"
    ladder = hub.ladder()
    assert [r.probed_ok for r in ladder.rungs if r.local] == [False, False]


def test_tool_schemas_use_the_openai_envelope() -> None:
    """The 422 was ours: tools must be nested under "function", not flattened."""
    from mharo_tui.agent.tools import tool_schemas

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=sse_text("fine"))

    provider = OpenAICompatProvider("gpt-oss-120b", base_url="https://free.example/v1", keyless=True,
                                    transport=httpx.MockTransport(handler))
    tools = tool_schemas()[:2]
    run(_collect(provider.stream([Message(role="user", blocks=[Text("hi")])], tools=tools)))
    sent = captured["tools"]
    assert len(sent) == len(tools)
    for spec in sent:
        assert spec["type"] == "function"
        assert set(spec) == {"type", "function"}, sorted(spec)
        assert {"name", "description", "parameters"} <= set(spec["function"])
    assert captured["tool_choice"] == "auto"
    # an already-wrapped spec is passed through, not double-wrapped
    wrapped = OpenAICompatProvider._tool_spec({"type": "function", "function": {"name": "x"}})
    assert wrapped == {"type": "function", "function": {"name": "x"}}
