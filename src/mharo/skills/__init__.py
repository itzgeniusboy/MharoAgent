"""Skills — real skill loading + registry.

Skill = python module with `register(env) -> dict` that returns named
callables. `discover()` scans a directory, `run(name, **kw)` executes with
tool dependency injected. Zero deps, tests cover fake skill.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Skill:
    """Loaded skill — name + env (namespace of callables) + source path."""

    name: str
    env: dict[str, Any]
    path: Optional[Path] = None


class SkillRunner:
    """Holds loaded skills; `run(name)` dispatches to env callables."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._lock = threading.Lock()

    def load_file(self, path: str | Path) -> Skill:
        path = Path(path)
        module_name = f"_mharo_skill_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load skill: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "register"):
            raise ValueError(f"skill {path.stem}: missing register(env)")
        env: dict[str, Any] = {}
        mod.register(env)
        skill = Skill(name=path.stem, env=env, path=path)
        with self._lock:
            self._skills[path.stem] = skill
        return skill

    def load_dir(self, directory: str | Path = "skills") -> list[str]:
        directory = Path(directory)
        if not directory.is_dir():
            return []
        names = []
        for py in sorted(directory.glob("*.py")):
            if py.name.startswith("_"):
                continue
            names.append(self.load_file(py).name)
        return names

    def run(self, skill_name: str, fn: str, **kwargs: Any) -> Any:
        with self._lock:
            skill = self._skills.get(skill_name)
            if skill is None:
                raise KeyError(f"skill not loaded: {skill_name}")
            callable_fn = skill.env.get(fn)
            if not callable(callable_fn):
                raise KeyError(f"{skill_name}.{fn} not found")
        result = callable_fn(**kwargs)
        if hasattr(result, "__await__"):
            import asyncio

            return asyncio.run(result)
        return result

    def stat(self, name: str) -> Optional[Skill]:
        with self._lock:
            return self._skills.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._skills)


DEFAULT = SkillRunner()

__all__ = ["Skill", "SkillRunner", "DEFAULT"]