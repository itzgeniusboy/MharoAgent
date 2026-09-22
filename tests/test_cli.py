"""CLI wiring tests — config builder + arg parsing, koi network nahi."""
from __future__ import annotations

import pytest

import mharo.__main__ as cli


def test_no_key_missing_raises(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENAI_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        cli._build_engine()


def test_openai_key_builds_engine(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    eng = cli._build_engine()
    assert len(eng.router.providers) == 1
    assert eng.router.providers[0].name == "openai"
    assert eng.router.providers[0].api_key == "test-key-123"


def test_dotenv_loader():
    assert cli._load_dotenv("/nonexistent/.env") is None


def test_main_missing_key_returns_2(capsys):
    import os
    for k in ("OPENAI_API_KEY", "OPENAI_KEY", "DEEPSEEK_API_KEY"):
        os.environ.pop(k, None)
    assert cli.main(["--env", "/nonexistent.env"]) == 2
    err = capsys.readouterr().err
    assert "error: no API key found" in err