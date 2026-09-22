"""Security — secret redaction helpers.

Jab bhi error/log kisi provider text me API key leak kar sake, `Redactor`
mask kar deta hai. REAL: keys list + prefix rounding (pek arind: "sk-...xYz9").
"""
from __future__ import annotations

import re
from typing import Optional


class Redactor:
    """Masks known secrets in arbitrary text."""

    def __init__(self, *secrets: str) -> None:
        self._secrets: list[str] = []
        for s in secrets:
            if s:
                self._secrets.append(s)

    def redact(self, text: str) -> str:
        out = text
        for s in self._secrets:
            if not s or s not in out:
                continue
            shown = s[:4] + "..." + s[-4:] if len(s) > 12 else "<redacted>"
            out = out.replace(s, shown)
        return out

    @staticmethod
    def redact_key_like(text: str) -> str:
        """sk-/sck-/key- prefixed unknown keys ko bhi mask karta hai."""
        def _mask(match: re.Match) -> str:
            full = match.group(0)
            cut = full.index("-") + 1
            rest = full[cut:]
            return full[:cut] + rest[:4] + "..." + rest[-4:]

        return re.sub(r"(?:sk|sck|key)-[A-Za-z0-9_-]{8,}", _mask, text)

    def __repr__(self) -> str:
        return f"Redactor({len(self._secrets)} secrets)"

    @property
    def count(self) -> int:
        return len(self._secrets)


def make_redactor(**secrets: str) -> Redactor:
    """Named secrets se redactor banata hai (gjati vars bhi akalva)."""
    return Redactor(*secrets.values())


__all__ = ["Redactor", "make_redactor"]