"""Memory tests — persistence, TTL, search, delete."""
from __future__ import annotations

import json

import pytest

from mharo.memory import Memory


@pytest.fixture
def mem(tmp_path):
    return Memory(str(tmp_path / "mem.json"))


def test_set_get(mem):
    mem.set("name", "mharo")
    assert mem.get("name") == "mharo"


def test_persists_across_instances(mem):
    mem.set("greeting", "namaste")
    mem2 = Memory(mem.path)
    assert mem2.get("greeting") == "namaste"


def test_ttl_expiry(mem):
    mem.set("temp", "x", ttl_s=-1)
    assert mem.get("temp") is None


def test_delete(mem):
    mem.set("a", 1)
    assert mem.delete("a") is True
    assert mem.delete("a") is False
    assert mem.get("a") is None


def test_search(mem):
    mem.set("user.name", "a1")
    mem.set("user.age", 30)
    mem.set("other", "z")
    assert mem.search("user.") == {"user.name": "a1", "user.age": 30}


def test_clear_and_len(mem):
    mem.set("x", 1)
    mem.set("y", 2)
    assert len(mem) == 2
    mem.clear()
    assert len(mem) == 0


def test_bad_file_recovers(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    m = Memory(str(p))
    assert m.get("anything") is None
    m.set("ok", 1)
    assert json.loads(p.read_text())["ok"]["value"] == 1