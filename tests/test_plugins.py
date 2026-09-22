"""Plugins tests — loader, dispatch, enable/disable."""
from __future__ import annotations

import textwrap

import pytest

from mharo.plugins import PluginManager


@pytest.fixture
def plugin_path(tmp_path):
    p = tmp_path / "banner.py"
    p.write_text(textwrap.dedent(
        """
        def plugin(registry, ctx):
            def shout(text):
                return (text + "!").upper()
            registry["shout"] = shout
        """
    ))
    return p


def test_load_and_invoke(plugin_path):
    m = PluginManager()
    pl = m.load_file(plugin_path, ctx={"app": "x"})
    assert pl.name == "banner"
    assert m.invoke("banner", "shout", text="hello") == "HELLO!"


def test_load_dir(plugin_path, tmp_path):
    d = tmp_path / "plugins"
    d.mkdir()
    (d / "banner.py").write_text(plugin_path.read_text())
    m = PluginManager()
    assert m.load_dir(d) == ["banner"]
    assert m.names() == ["banner"]


def test_missing_plugin_and_fn(plugin_path):
    m = PluginManager()
    m.load_file(plugin_path)
    with pytest.raises(KeyError):
        m.invoke("nope", "x")
    with pytest.raises(KeyError):
        m.invoke("banner", "nope")


def test_disable(plugin_path):
    m = PluginManager()
    m.load_file(plugin_path)
    assert m.enable("banner", False) is True
    with pytest.raises(RuntimeError):
        m.invoke("banner", "shout", text="x")
    m.enable("banner", True)
    assert m.invoke("banner", "shout", text="ok") == "OK!"


def test_empty_dir(tmp_path):
    assert PluginManager().load_dir(tmp_path / "nope") == []