"""Hermetic provider env — device pe opencode auth.json/env leak na kare tests me."""
from __future__ import annotations

import pytest

from mharo.providers import catalog


@pytest.fixture(autouse=True)
def _isolate_provider_keys(monkeypatch):
    monkeypatch.setattr(catalog, "opencode_auth_paths", lambda: [])
    for c in catalog.CATALOG:
        for env in c.env:
            monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("MHARO_STRATEGY", raising=False)
    yield