"""Integration — repo `skills/` dir default SkillRunner mein load hota hai."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mharo.skills import SkillRunner

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (REPO / "skills").is_dir(), reason="no repo skills dir in CI sdist")
def test_repo_skills_dir_loads():
    r = SkillRunner()
    names = r.load_dir(REPO / "skills")
    assert "greet" in names
    assert r.run("greet", "hello", name="testcycle") == "namaste testcycle"
    assert r.run("greet", "system_info").count("/") > 0