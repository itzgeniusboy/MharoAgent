"""P1-8 · Vault: secrets live at the tool boundary, never in the model context.

  * `~/.mharo/vault.json`, mode 0600, keys looked up by service name
  * `env_for(tools)` — what a subprocess is allowed to receive
  * `redact(text)` — every vault value plus common token shapes, applied to
    *tool output before it is stored or rendered*, so a `cat .env` cannot leak
    into the transcript, the DB, or the next model call.
  * the model payload path (`providers`) has no vault accessor at all
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}\b")),
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"(?i)\b(bearer|authorization\s*[:=])\s+[A-Za-z0-9._\-]{12,}")),
    ("password-assign", re.compile(r"(?i)\b(password|passwd|secret|token|api_key|apikey|access_key)\b(\s*[:=]\s*)(\S{6,})")),
]

SAFE_ENV = {
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
    "GIT_PAGER", "PAGER", "PYTHONDONTWRITEBYTECODE", "NODE_ENV", "CI",
    "VIRTUAL_ENV", "PYTHONPATH", "JAVA_HOME", "ANDROID_HOME", "PREFIX",
    "SHLVL", "PWD", "PS1", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
}


@dataclass
class Vault:
    path: Path
    data: dict[str, str] = field(default_factory=dict)
    env_names: dict[str, str] = field(default_factory=dict)   # service -> env var
    problems: list[str] = field(default_factory=list)
    _patterns: list[re.Pattern[str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, *, env_prefix: str = "MHARO_SECRET_") -> "Vault":
        path = Path(path).expanduser()
        vault = cls(path=path)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                vault.problems.append(f"{path.name}: {exc}")
                raw = {}
            secrets = raw.get("secrets") or {}
            if isinstance(secrets, dict):
                vault.data = {str(k): str(v) for k, v in secrets.items() if v}
            for name, var in (raw.get("env_names") or {}).items():
                vault.env_names.setdefault(str(name), str(var))
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0
            if mode & 0o077:
                vault.problems.append(f"{path.name} is mode {oct(mode)} — run: chmod 600 {path}")
        # env-provided secrets (MHARO_SECRET_OPENAI=sk-…) are first-class
        for key, value in os.environ.items():
            if key.startswith(env_prefix) and value:
                vault.data.setdefault(key[len(env_prefix):].lower(), value)
                vault.env_names.setdefault(key[len(env_prefix):].lower(), key)
        vault._compile()
        return vault

    def _compile(self) -> None:
        self._patterns = [
            re.compile(re.escape(v)) for v in self.data.values() if v and len(v) >= 6
        ]

    # -- write -----------------------------------------------------------
    def set(self, service: str, value: str, *, persist: bool = True) -> None:
        self.data[service] = value
        self._compile()
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"secrets": self.data, "env_names": self.env_names, "updated_at": time.time()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)

    def unset(self, service: str) -> bool:
        if service not in self.data:
            return False
        del self.data[service]
        self._compile()
        self.path.write_text(
            json.dumps({"secrets": self.data, "env_names": self.env_names, "updated_at": time.time()}, indent=2) + "\n",
            encoding="utf-8",
        )
        return True

    def has(self, service: str) -> bool:
        return service in self.data

    def get(self, service: str) -> str | None:
        return self.data.get(service)

    # -- tool boundary ---------------------------------------------------
    def env_for(self, services: list[str] | None = None) -> dict[str, str]:
        """Only the named services (default: all) go into a subprocess env."""
        out: dict[str, str] = {}
        wanted = services if services else list(self.data)
        for service in wanted:
            value = self.data.get(service)
            if not value:
                continue
            env_name = self.env_names.get(service) or f"{service.upper()}_API_KEY"
            out[env_name] = value
        return out

    @staticmethod
    def scrub_env(env: dict[str, str] | None = None) -> dict[str, str]:
        """Strip the parent environment down to safe vars (defence in depth)."""
        source = env if env is not None else os.environ
        return {k: v for k, v in source.items() if k in SAFE_ENV}

    # -- redaction -------------------------------------------------------
    def redact(self, text: str) -> tuple[str, int]:
        if not text:
            return text, 0
        hits = 0
        out = text
        for pattern in self._patterns:
            out, n = pattern.subn("‹redacted:vault›", out)
            hits += n
        for label, pattern in REDACTIONS:
            out, n = pattern.subn(f"‹redacted:{label}›", out)
            hits += n
        # the assignment form stays readable: the pattern above already swapped
        # the value for the marker, so there is nothing left to do here.
        return out, hits

    def redact_deep(self, value: object) -> object:
        if isinstance(value, str):
            return self.redact(value)[0]
        if isinstance(value, dict):
            return {k: self.redact_deep(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact_deep(v) for v in value]
        return value

    def status(self) -> dict:
        return {
            "path": str(self.path),
            "services": sorted(self.data),
            "count": len(self.data),
            "mode": oct(stat.S_IMODE(self.path.stat().st_mode)) if self.path.exists() else "missing",
            "problems": list(self.problems),
        }
