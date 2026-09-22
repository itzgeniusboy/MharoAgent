"""Plugins — real plugin loader.

Har plugin = python module jisme `plugin(ctx: dict) -> dict` hota hai, jo
registry me named callables daalta hai. Runtime me load/run/disable karo.
Zero deps, sab tests.
"""
from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Plugin:
    """Loaded plugin — name + registry + disable flag."""

    name: str
    registry: dict[str, Callable[..., Any]]
    path: Optional[Path] = None
    enabled: bool = True


class PluginManager:
    """Scans diractories, loads plugins, dispatches `invoke(name, **kw)`."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._lock = threading.Lock()

    def load_file(self, path: str | Path, ctx: Optional[dict] = None) -> Plugin:
        path = Path(path)
        spec = importlib.util.spec_from_file_location(f"_mharo_plugin_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load plugin: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "plugin"):
            raise ValueError(f"plugin {path.stem}: missing plugin(ctx)")
        registry: dict[str, Callable[..., Any]] = {}
        mod.plugin(registry, ctx or {})
        plugin = Plugin(name=path.stem, registry=registry, path=path)
        with self._lock:
            self._plugins[path.stem] = plugin
        return plugin

    def load_dir(self, directory: str | Path = "plugins") -> list[str]:
        directory = Path(directory)
        if not directory.is_dir():
            return []
        names = []
        for py in sorted(directory.glob("*.py")):
            if py.name.startswith("_"):
                continue
            names.append(self.load_file(py).name)
        return names

    def invoke(self, plugin: str, fn: str, **kwargs: Any) -> Any:
        with self._lock:
            p = self._plugins.get(plugin)
            if p is None:
                raise KeyError(f"plugin not loaded: {plugin}")
            if not p.enabled:
                raise RuntimeError(f"plugin disabled: {plugin}")
            callable_fn = p.registry.get(fn)
            if not callable(callable_fn):
                raise KeyError(f"{plugin}.{fn} not found")
        result = callable_fn(**kwargs)
        if hasattr(result, "__await__"):
            import asyncio

            return asyncio.run(result)
        return result

    def enable(self, name: str, enabled: bool = True) -> bool:
        with self._lock:
            p = self._plugins.get(name)
            if p is None:
                return False
            p.enabled = enabled
            return True

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._plugins)


__all__ = ["Plugin", "PluginManager"]