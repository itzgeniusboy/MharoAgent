"""CLI wiring tests — config builder + arg parsing, koi network nahi."""
from __future__ import annotations

import pytest

import mharo.__main__ as cli


def test_no_key_gets_local_mode(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENAI_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    eng = cli._build_engine(allow_local=True)
    assert len(eng.router.providers) == 1
    assert eng.router.providers[0].name == "local"


def test_local_disabled_raises(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENAI_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        cli._build_engine(allow_local=False)


def test_openai_key_builds_engine(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    eng = cli._build_engine()
    assert len(eng.router.providers) == 1
    assert eng.router.providers[0].name == "openai"
    assert eng.router.providers[0].api_key == "test-key-123"


def test_openrouter_key_builds_engine(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    eng = cli._build_engine()
    assert len(eng.router.providers) == 1
    assert eng.router.providers[0].name == "openrouter"
    assert eng.router.providers[0].base_url == "https://openrouter.ai/api/v1"


def test_dotenv_loader():
    assert cli._load_dotenv("/nonexistent/.env") is None


def test_memory_context_empty_and_populated(tmp_path):
    from mharo.memory import Memory

    mem = Memory(str(tmp_path / "m.json"))
    assert cli.memory_context(mem) == ""
    mem.set("fact.city", "jaipur")
    mem.set("user.name", "mharo")
    ctx = cli.memory_context(mem)
    assert "fact.city: jaipur" in ctx
    assert "user.name" not in ctx  # prefix filter fact.
    assert ctx.startswith("Remembered facts:")


def test_handle_remember_recall_forget(tmp_path):
    from mharo.memory import Memory

    mem = Memory(str(tmp_path / "m.json"))
    assert cli.handle_special("/remember note=hello", mem) == "remembered note"
    assert cli.handle_special("/remember bad", mem).startswith("usage:")
    assert cli.handle_special("/recall", mem) == "note: hello"
    assert cli.handle_special("/forget note", mem) == "forgot note"
    assert cli.handle_special("/forget note", mem).startswith("no key")
    assert cli.handle_special("plain", mem) is None


def test_main_without_key_starts_local(tmp_path, monkeypatch, capsys):
    import os

    for k in ("OPENAI_API_KEY", "OPENAI_KEY", "DEEPSEEK_API_KEY"):
        os.environ.pop(k, None)
    monkeypatch.setattr(
        "builtins.input",
        lambda *a, **k: (_ for _ in ()).throw(EOFError()),
    )
    assert cli.main(["--env", "/nonexistent.env",
                     "--memory", str(tmp_path / "m.json")]) == 0
    out = capsys.readouterr().out
    assert "local mode" in out