"""Catalog tests — table health, env detection, opencode auth integration."""
from __future__ import annotations

import json

import pytest

from mharo.providers import catalog
from mharo.providers.catalog import CATALOG, ProviderDef


def test_catalog_entries_well_formed():
    names = [c.name for c in CATALOG]
    assert len(set(names)) == len(names), "duplicate provider names"
    for c in CATALOG:
        assert c.base_url.startswith("https://")
        assert c.env, f"{c.name}: no env var"
        assert c.model


def test_zen_entry_present():
    zen = next(c for c in CATALOG if c.name == "zen")
    assert zen.base_url == "https://opencode.ai/zen/v1"
    assert zen.model == "big-pickle"


def test_configured_detects_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "grok9")
    got = catalog.configured()
    names = [cfg.name for cfg, _ in got]
    assert "groq" in names
    assert all(k for _, k in got)


def test_configured_empty_without_keys(monkeypatch, tmp_path):
    for c in CATALOG:
        for k in c.env:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(catalog, "opencode_auth_paths", lambda: [])
    assert catalog.configured() == []


def test_keys_from_opencode_auth(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"openrouter": {"type": "api", "key": "sk-or-v1-longvalue"}}))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(catalog, "opencode_auth_paths", lambda: [p])
    keys = catalog.keys_from_opencode_auth()
    assert keys["openrouter"] == "sk-or-v1-longvalue"
    monkeypatch.undo()


def test_env_merges_opencode_auth(monkeypatch, tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"openrouter": {"type": "api", "key": "sk-or-v1-secret"}}))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(catalog, "opencode_auth_paths", lambda: [p])
    env = catalog.env_with_opencode_auth()
    assert env.get("OPENROUTER_API_KEY") == "sk-or-v1-secret"


def test_base_url_env_override():
    d = ProviderDef("x", "m", "https://a/v1", env=("K",), base_url_env="KU")
    assert d.resolved_base_url({"KU": "https://b/v1"}) == "https://b/v1"
    assert d.resolved_base_url({}) == "https://a/v1"