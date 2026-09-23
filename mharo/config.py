"""Configuration: file + env, providers, tiers, budget, permissions, checks.

Resolution order (later wins):
    built-in defaults  →  ~/.mharo/config.json  →  $MHARO_CONFIG file  →  env vars

`ma doctor` prints the resolved config so there is never a question about which
value was used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "home": None,                       # None -> ~/.mharo
    "cwd": ".",
    "tiers": {
        # P1-3: the router only ever picks between these; a tier is a provider
        # spec, so swapping models never touches engine code.
        "cheap": {"provider": "openai", "model": "gpt-4o-mini", "max_output_tokens": 2048, "temperature": 0.1},
        "strong": {"provider": "anthropic", "model": "claude-sonnet-4-5", "max_output_tokens": 8192, "temperature": 0.0},
    },
    "fallbacks": ["cheap", "strong"],   # tried in order when a tier errors
    "providers": {
        "openai": {"base_url": "https://api.openai.com/v1", "keys_env": ["MHARO_OPENAI_KEYS", "OPENAI_API_KEYS", "OPENAI_API_KEY"]},
        "anthropic": {"base_url": "https://api.anthropic.com", "keys_env": ["MHARO_ANTHROPIC_KEYS", "ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"]},
        "local": {"base_url": "http://127.0.0.1:11434/v1", "keys_env": [], "model": "qwen2.5-coder:32b"},
    },
    # Free-first: with no API key the agent still runs, by walking a ladder of
    # zero-cost rungs (local Ollama/vLLM → anonymous public tiers → free-key tiers)
    # and only then asking for a key. `ma free` probes it; `ma keys add` is the escape hatch.
    "free": {
        "enabled": True,
        "prefer": [],           # rung names to try first, e.g. ["ollama", "ovh"]
        "disable": [],          # rung names to drop entirely, e.g. ["pollinations"]
        "min_interval_s": {},   # override the client-side rate gate per rung
        "extra_rungs": [],      # your own OpenAI-compatible free endpoint
        "allow_paid": True,      # append configured paid keys to the *end* of the ladder
        "rescue_on_exhaustion": True,  # your key is throttled/out of credit → finish on a free rung
        "patience_s": 20.0,       # a free tier's rate gate is waited out, not failed against
        "fallback_to_demo": True,  # nothing reachable at all → offline demo, not a crash
        "max_rungs": 4,         # how many rungs one call may walk before giving up
    },
    "budget": {"max_session_usd": 2.0, "max_turns": 12, "turn_timeout_s": 300},
    "proxy": {"http": None, "https": None},
    "permissions": {
        "default": "ask",
        "allow": ["read_file", "list_dir", "search", "bash:git status*", "bash:ls*", "bash:cat*"],
        "deny": ["bash:rm -rf /*", "bash:sudo*", "write_file:.env*"],
    },
    "verify": {
        "auto_checks": "auto",          # auto-detect pytest/npm/cargo/just/go
        "require_proof_for_claims": True,
        "peer_review": True,
        "max_fix_rounds": 2,
    },
    "memory": {"enabled": True, "dims": 256, "recall_k": 4, "embed_url": None},
    "cost": {"prices": {}, "currency": "USD"},
    "skills_dirs": ["skills", "~/.mharo/skills"],
    "subagents": {
        "reader": {"tools": ["read_file", "list_dir", "search"], "tier": "cheap"},
        "tester": {"tools": ["bash", "read_file"], "tier": "cheap"},
        "patcher": {"tools": ["read_file", "edit_file", "write_file", "search"], "tier": "strong"},
        "reviewer": {"tools": ["read_file", "search", "list_dir"], "tier": "strong"},
    },
}


def home_dir() -> Path:
    raw = os.environ.get("MHARO_HOME")
    path = Path(raw).expanduser() if raw else Path.home() / ".mharo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    raw = os.environ.get("MHARO_CONFIG")
    return Path(raw).expanduser() if raw else home_dir() / "config.json"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class TierSpec:
    name: str
    provider: str
    model: str
    max_output_tokens: int = 4096
    temperature: float = 0.1

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "TierSpec":
        return cls(
            name=name,
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            max_output_tokens=int(data.get("max_output_tokens", 4096)),
            temperature=float(data.get("temperature", 0.1)),
        )


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    path: Path | None = None
    problems: list[str] = field(default_factory=list)

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path).expanduser() if path else config_path()
        merged = json.loads(json.dumps(DEFAULTS))  # deep copy, no shared mutable state
        problems: list[str] = []
        if cfg_path.is_file():
            try:
                user = json.loads(cfg_path.read_text(encoding="utf-8"))
                if not isinstance(user, dict):
                    raise ValueError("top-level JSON object expected")
                merged = _merge(merged, user)
            except (OSError, ValueError) as exc:
                problems.append(f"{cfg_path}: {exc} (using defaults)")
        cfg = cls(raw=merged, path=cfg_path, problems=problems)
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        env = os.environ
        if env.get("MHARO_MODEL"):
            self.raw["tiers"]["cheap"]["model"] = env["MHARO_MODEL"]
        if env.get("MHARO_STRONG_MODEL"):
            self.raw["tiers"]["strong"]["model"] = env["MHARO_STRONG_MODEL"]
        if env.get("MHARO_BASE_URL"):
            self.raw["providers"].setdefault("openai", {})["base_url"] = env["MHARO_BASE_URL"]
        if env.get("MHARO_PROVIDER"):
            self.raw["tiers"]["cheap"]["provider"] = env["MHARO_PROVIDER"]
        if env.get("MHARO_PROXY"):
            self.raw["proxy"] = {"http": env["MHARO_PROXY"], "https": env["MHARO_PROXY"]}
        if env.get("MHARO_BUDGET"):
            try:
                self.raw["budget"]["max_session_usd"] = float(env["MHARO_BUDGET"])
            except ValueError:
                self.problems.append(f"MHARO_BUDGET is not a number: {env['MHARO_BUDGET']!r}")
        if env.get("MHARO_HOME"):
            self.raw["home"] = env["MHARO_HOME"]

    # -- accessors -------------------------------------------------------
    def tier(self, name: str) -> TierSpec:
        data = self.raw["tiers"].get(name) or self.raw["tiers"]["cheap"]
        return TierSpec.from_dict(name, data)

    @property
    def tiers(self) -> list[str]:
        return list(self.raw["tiers"])

    def provider(self, name: str) -> dict[str, Any]:
        return dict(self.raw["providers"].get(name, {}))

    @property
    def free(self) -> dict[str, Any]:
        """The free-ladder section, merged over its defaults."""
        section = self.raw.get("free") or {}
        merged = dict(DEFAULTS["free"])
        for key, value in section.items():
            if value is not None:
                merged[key] = value
        return merged

    def key_envs(self, provider: str) -> list[str]:
        names = self.provider(provider).get("keys_env") or []
        return [n for n in names if isinstance(n, str)]

    @property
    def budget(self) -> dict[str, Any]:
        return dict(self.raw["budget"])

    @property
    def verify(self) -> dict[str, Any]:
        return dict(self.raw["verify"])

    @property
    def memory(self) -> dict[str, Any]:
        return dict(self.raw["memory"])

    @property
    def permissions(self) -> dict[str, Any]:
        return dict(self.raw["permissions"])

    @property
    def subagents(self) -> dict[str, Any]:
        return dict(self.raw["subagents"])

    @property
    def proxy(self) -> dict[str, str | None]:
        return dict(self.raw.get("proxy") or {})

    def prices(self) -> dict[str, list[float]]:
        from mharo_tui.agent.session import PRICE_TABLE

        out = {k: list(v) for k, v in PRICE_TABLE.items()}
        for key, value in (self.raw.get("cost", {}).get("prices") or {}).items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                out[key] = [float(value[0]), float(value[1])]
        return out

    def skills_dirs(self, cwd: str | Path | None = None) -> list[Path]:
        """Existing skill directories, repo-local first. Relative entries anchor to `cwd`."""
        base = Path(cwd).expanduser() if cwd else Path.cwd()
        out: list[Path] = []
        for raw in self.raw.get("skills_dirs") or []:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = base / path
            if path.is_dir() and path not in out:
                out.append(path)
        return out

    def db_path(self) -> Path:
        return home_dir() / "agent.db"

    def vault_path(self) -> Path:
        return home_dir() / "vault.json"

    def write_template(self) -> Path:
        """`ma init` — dump the resolved defaults so the file is a real starting point."""
        target = self.path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(DEFAULTS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def redacted_dump(self) -> dict[str, Any]:
        data = json.loads(json.dumps(self.raw))
        for name, spec in (data.get("providers") or {}).items():
            if isinstance(spec, dict):
                spec.pop("api_key", None)
                spec["keys_env"] = self.key_envs(name)
        return data
