"""Catalog — ALL OpenAI-compatible providers (ek jaanata data table).

Har provider: name, default model, base_url, key env-lookup order.
Keys milte hain (priority order): process env -> .env (load_dotenv) ->
opencode's own auth.json (agr wahan provider login ho to). Ye table hi
CLI `_build_engine` ka base hai — naye provider add = ek line.

Free providers jo out-of-the-box kaam karte hain: OpenRoute r free models,
OpenCode Zen free models (big-pickle/mimo), Groq free tier, etc.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProviderDef:
    name: str
    model: str
    base_url: str
    env: tuple[str, ...] = ()
    # optional: base url env override (OPENROUTER_BASE_URL style)
    base_url_env: str = ""

    def key_from(self, env: dict[str, str]) -> Optional[str]:
        for k in self.env:
            v = env.get(k)
            if v:
                return v
        return None

    def resolved_base_url(self, env: dict[str, str]) -> str:
        return env.get(self.base_url_env, self.base_url) if self.base_url_env else self.base_url


CATALOG: tuple[ProviderDef, ...] = (
    # --- OpenCode Zen (built-in gateway, free models) ---
    ProviderDef(
        "zen", "big-pickle", "https://opencode.ai/zen/v1",
        env=("ZEN_API_KEY", "OPENCODE_ZEN_API_KEY"),
        base_url_env="ZEN_BASE_URL",
    ),
    # --- OpenRouter (free models :free suffix) ---
    ProviderDef(
        "openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free",
        "https://openrouter.ai/api/v1",
        env=("OPENROUTER_API_KEY",), base_url_env="OPENROUTER_BASE_URL",
    ),
    # --- Direct OpenAI-compatible providers (free tiers included) ---
    ProviderDef("openai", "gpt-4o-mini", "https://api.openai.com/v1",
                env=("OPENAI_API_KEY", "OPENAI_KEY"), base_url_env="OPENAI_BASE_URL"),
    ProviderDef("deepseek", "deepseek-chat", "https://api.deepseek.com/v1",
                env=("DEEPSEEK_API_KEY",)),
    ProviderDef("groq", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1",
                env=("GROQ_API_KEY",)),
    ProviderDef("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "https://api.together.xyz/v1", env=("TOGETHER_API_KEY",)),
    ProviderDef("cerebras", "llama-3.3-70b", "https://api.cerebras.ai/v1",
                env=("CEREBRAS_API_KEY",)),
    ProviderDef("mistral", "mistral-small-latest", "https://api.mistral.ai/v1",
                env=("MISTRAL_API_KEY",)),
    ProviderDef("gemini", "gemini-2.0-flash",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                env=("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    ProviderDef("xai", "grok-code-fast", "https://api.x.ai/v1", env=("XAI_API_KEY",)),
    ProviderDef("cohere", "command-r-plus", "https://api.cohere.com/v2",
                env=("COHERE_API_KEY",)),
    ProviderDef("fireworks", "accounts/fireworks/models/llama-v3p3-70b-instruct",
                "https://api.fireworks.ai/inference/v1", env=("FIREWORKS_API_KEY",)),
    ProviderDef("sambanova", "Meta-Llama-3.3-70B-Instruct",
                "https://api.sambanova.ai/v1", env=("SAMBANOVA_API_KEY",)),
)

# zen base url override bhi allow karta hai: ZEN_BASE_URL


def opencode_auth_paths() -> list[Path]:
    here = Path.home() / ".local/share/opencode" / "auth.json"
    cfg = Path.home() / ".config/opencode" / "auth.json"
    return [p for p in (here, cfg) if p.is_file()]


def keys_from_opencode_auth() -> dict[str, str]:
    """opencode ke apne auth.json se provider.keys read (ki app ne login kiya ho)."""
    out: dict[str, str] = {}
    for path in opencode_auth_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data = data if isinstance(data, dict) else {}
        for prov, val in data.items():
            if isinstance(val, dict):
                key = val.get("key")
                if isinstance(key, str) and len(key) > 8:
                    out[prov.lower()] = key
    return out


def env_with_opencode_auth(base_env: Optional[dict] = None) -> dict[str, str]:
    """env + opencode auth.json keys ko ek namespace me merge karta hai.

    auth.json 'openrouter' -> iss system ke OPENROUTER_API_KEY jaise treat."
    """
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    for prov, key in keys_from_opencode_auth().items():
        key_cfg = "OPENROUTER_API_KEY" if prov == "openrouter" \
            else "OPENAI_API_KEY" if prov in {"openai", "opencoder"} \
            else "ZEN_API_KEY" if prov == "zen" \
            else f"{prov.upper()}_API_KEY"
        env.setdefault(key_cfg, key)
    return env


def configured(defs: tuple[ProviderDef, ...] = CATALOG,
              env: Optional[dict] = None) -> list[tuple[ProviderDef, str]]:
    """Jin providers ki key available hai, unki (def, key) list (catalog order)."""
    env = env_with_opencode_auth(env)
    return [(d, k) for d in defs if (k := d.key_from(env))]


__all__ = ["ProviderDef", "CATALOG", "configured",
           "env_with_opencode_auth", "opencode_auth_paths", "keys_from_opencode_auth"]