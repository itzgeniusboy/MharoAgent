"""greet — example skill for MharoAgent SkillRunner.

Load:  runner.load_dir("skills")   # repo root se
Use:   runner.run("greet", "hello", name="me")
"""
from __future__ import annotations

_COUNT = {"n": 0}


def register(env: dict) -> None:
    def hello(name: str = "mharo") -> str:
        _COUNT["n"] += 1
        return f"namaste {name}"

    def count() -> int:
        return _COUNT["n"]

    def system_info() -> str:
        import platform

        return f"{platform.system()}/{platform.machine()}"

    env["hello"] = hello
    env["count"] = count
    env["system_info"] = system_info