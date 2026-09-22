"""Security redactor tests."""
from __future__ import annotations

import pytest

from mharo.security import Redactor, make_redactor


def test_redacts_known_secret():
    key = "sk-abcdefghijklmnopqrstuvwxyz1234"
    r = Redactor(key)
    out = r.redact(f"err {key} boom")
    assert key not in out
    assert "sk-a..." in out


def test_redact_short_secret():
    r = Redactor("hush")
    assert r.redact("say hush now") == "say <redacted> now"


def test_redact_key_like_unknown():
    out = Redactor.redact_key_like("token=sk-superlongsecretvalue123 last")
    assert "superlongsecretvalue123" not in out
    assert "sk-supe..." in out


def test_make_redactor_named():
    r = make_redactor(api="sk-1234567890abcdef")
    assert r.count == 1
    assert r.redact("key ai sk-1234567890abcdef hi") == "key ai sk-1...cdef hi"


def test_no_secret_leaves_text():
    r = Redactor("sk-aaa")
    assert r.redact("plain text 123") == "plain text 123"