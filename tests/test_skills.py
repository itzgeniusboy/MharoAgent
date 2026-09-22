"""Skills tests — dynamic load, run, missing cases."""
from __future__ import annotations

import textwrap

import pytest

from mharo.skills import SkillRunner


@pytest.fixture
def skill_path(tmp_path):
    p = tmp_path / "greet.py"
    p.write_text(textwrap.dedent(
        """
        COUNT = {"n": 0}

        def register(env):
            def hello(name):
                COUNT["n"] += 1
                return f"namaste {name}"
            env["hello"] = hello
            env["count"] = lambda: COUNT["n"]
        """
    ))
    return p


def test_load_and_run(skill_path):
    r = SkillRunner()
    s = r.load_file(skill_path)
    assert s.name == "greet"
    assert r.run("greet", "hello", name="Mharo") == "namaste Mharo"
    assert r.run("greet", "count") == 1


def test_load_base_dir(skill_path, tmp_path):
    r = SkillRunner()
    # put in a dir named `skills` to satisfzs load_dir semantics
    d = tmp_path / "skills"
    d.mkdir()
    (d / "greet.py").write_text(skill_path.read_text())
    assert r.load_dir(d) == ["greet"]
    assert r.names() == ["greet"]


def test_missing_name_raises():
    r = SkillRunner()
    with pytest.raises(KeyError):
        r.run("nope", "x")


def test_missing_fn_raises(skill_path):
    r = SkillRunner()
    r.load_file(skill_path)
    with pytest.raises(KeyError):
        r.run("greet", "does_not_exist")


def test_empty_dir_ok(tmp_path):
    r = SkillRunner()
    assert r.load_dir(tmp_path / "empty") == []