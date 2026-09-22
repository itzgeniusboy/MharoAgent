"""Memory — simple persistent key-value store (JSON file).

Engine/skills ke liye light memory: `set/get/delete/search`. Disk JSON
file me survive karta hai restart ke baad bhi. Zero deps.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional


class Memory:
    """Thread-safe JSON-file KV memory. Atomic write (tmp + rename)."""

    def __init__(self, path: str = "mharo_memory.json") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self._data = {}
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                self._data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp, self.path)

    def set(self, key: str, value: Any, ttl_s: Optional[float] = None) -> None:
        rec = {"value": value, "updated": datetime.now(timezone.utc).isoformat()}
        if ttl_s:
            rec["expires_at"] = (
                datetime.now(timezone.utc).timestamp() + ttl_s
            )
        with self._lock:
            self._data[key] = rec
            self._save()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            rec = self._data.get(key)
            if not rec:
                return None
            exp = rec.get("expires_at")
            if exp and datetime.now(timezone.utc).timestamp() > exp:
                self._data.pop(key, None)
                self._save()
                return None
            return rec["value"]

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._data
            if existed:
                del self._data[key]
                self._save()
            return existed

    def search(self, prefix: str = "") -> dict[str, Any]:
        with self._lock:
            return {k: v["value"] for k, v in self._data.items()
                    if k.startswith(prefix)}

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


__all__ = ["Memory"]