"""P1-7 · Memory 2.0: embedding + recall + compaction, all local, all real.

`HashingEmbedder` is not a toy: hashed TF-IDF over word + character n-grams,
L2-normalised into a fixed-width float32 vector, cosine-similarity recall in
Python. It needs no model, no extension, and works offline on Termux. If
`memory.embed_url` is configured, an OpenAI-compatible /embeddings call replaces
it transparently (same interface, same stored blob format).

Memory kinds: `task` (what was asked + what changed + the proof), `note`
(human-written), `error` (failures worth remembering), `summary` (compaction).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable

import sqlite3

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{1,}|[0-9]{2,}")
STOP = {
    "the", "and", "for", "that", "this", "with", "you", "are", "was", "were", "have",
    "has", "not", "but", "can", "will", "your", "from", "they", "them", "then", "than",
}


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in TOKEN_RE.findall((text or "").lower()):
        if raw in STOP or len(raw) < 3:
            continue
        out.append(raw)
        for n in (3, 4):                      # character n-grams catch typos/renames
            if len(raw) > n + 1:
                out.extend(raw[i: i + n] for i in range(len(raw) - n + 1))
    return out


class HashingEmbedder:
    """Deterministic hashed TF-IDF vectors. `dims` must be a multiple of 4."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = max(32, int(dims) - (int(dims) % 4))

    def _bucket(self, term: str) -> int:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dims

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        terms = tokens(text)
        if not terms:
            return vec
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        total = len(terms)
        for term, count in counts.items():
            tf = 1.0 + math.log(count)
            tf = tf / total * math.sqrt(total)          # length-normalised
            vec[self._bucket(term)] += tf
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def pack(self, vec: Iterable[float]) -> bytes:
        return struct.pack(f"<{self.dims}f", *list(vec)[: self.dims])

    def unpack(self, blob: bytes | None) -> list[float]:
        if not blob:
            return []
        n = len(blob) // 4
        return list(struct.unpack(f"<{n}f", blob))

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class ModelEmbedder(HashingEmbedder):
    """OpenAI-compatible /embeddings, falling back to hashing on any error."""

    def __init__(self, dims: int, url: str, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        super().__init__(dims)
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def vector_sync(self, text: str) -> list[float]:
        import httpx

        try:
            payload = {"model": self.model, "input": text[:8000], "dimensions": self.dims}
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            res = httpx.post(f"{self.url}/embeddings", json=payload, headers=headers, timeout=12.0)
            res.raise_for_status()
            data = res.json()["data"][0]["embedding"]
            if len(data) < self.dims:
                data = data + [0.0] * (self.dims - len(data))
            norm = math.sqrt(sum(v * v for v in data[: self.dims])) or 1.0
            return [v / norm for v in data[: self.dims]]
        except Exception:
            return super().vector(text)

    def vector(self, text: str) -> list[float]:  # type: ignore[override]
        return self.vector_sync(text)


@dataclass
class MemoryHit:
    id: int
    kind: str
    title: str
    body: str
    score: float
    created_at: float
    files: list[str]
    cost_usd: float

    def snippet(self, width: int = 220) -> str:
        text = " ".join((self.body or "").split())
        return text[:width] + ("…" if len(text) > width else "")

    def age(self) -> str:
        import time

        secs = max(0.0, time.time() - self.created_at)
        if secs < 3600:
            return f"{secs / 60:.0f}m"
        if secs < 86400 * 30:
            return f"{secs / 3600:.0f}h"
        return f"{secs / 86400:.0f}d"


class Memory:
    def __init__(self, db: sqlite3.Connection | Any, embedder: HashingEmbedder | None = None) -> None:
        self.db = db
        self.embedder = embedder or HashingEmbedder()

    @classmethod
    def build(cls, db: Any, config: dict[str, Any] | None = None) -> "Memory":
        cfg = config or {}
        dims = int(cfg.get("dims", 256))
        url = cfg.get("embed_url")
        if url:
            import os

            return cls(db, ModelEmbedder(dims, url, cfg.get("embed_model", "text-embedding-3-small"),
                                         os.environ.get(cfg.get("embed_key_env", "OPENAI_API_KEY"), "")))
        return cls(db, HashingEmbedder(dims))

    # -- write -----------------------------------------------------------
    def remember(
        self, *, title: str, body: str, kind: str = "task", cwd: str = "",
        files: list[str] | None = None, cost_usd: float = 0.0, tags: str = "",
    ) -> int:
        vec = self.embedder.vector(f"{title}\n{body}\n{' '.join(files or [])}")
        return self.db.put_memory(
            title=title, body=body, vec=self.embedder.pack(vec), dims=self.embedder.dims,
            cwd=cwd, files=files or [], cost_usd=cost_usd, kind=kind, tags=tags,
        )

    def note(self, text: str, *, cwd: str = "", tags: str = "note") -> int:
        title = " ".join(text.split())[:70]
        return self.remember(title=title, body=text, kind="note", cwd=cwd, tags=tags)

    # -- read ------------------------------------------------------------
    def recall(self, query: str, *, k: int = 4, kind: str | None = None, cwd: str | None = None,
               min_score: float = 0.08) -> list[MemoryHit]:
        rows = self.db.all_memories(limit=4000)
        if not rows:
            return []
        q = self.embedder.vector(query)
        hits: list[MemoryHit] = []
        for row in rows:
            if kind and row["kind"] != kind:
                continue
            if cwd and row["cwd"] and row["cwd"] != cwd:
                continue
            vec = self.embedder.unpack(row["vec"])
            hay = row["body"] + " " + row["title"] + " " + (row["tags"] or "")
            # vector similarity *or* an exact keyword hit — a single-word query like
            # "WAL" scores poorly on cosine but must still find the WAL memory
            score = max(HashingEmbedder.cosine(q, vec) if vec else 0.0, _lexical(query, hay))
            if score < min_score:
                continue                        # below the floor: not worth the tokens
            try:
                files = json.loads(row["files_json"] or "[]")
            except json.JSONDecodeError:
                files = []
            hits.append(MemoryHit(row["id"], row["kind"], row["title"], row["body"], score, row["created_at"], files, row["cost_usd"]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def recent(self, k: int = 8, cwd: str | None = None) -> list[MemoryHit]:
        rows = self.db.all_memories(limit=k * 3)
        out: list[MemoryHit] = []
        for row in rows:
            if cwd and row["cwd"] != cwd:
                continue
            try:
                files = json.loads(row["files_json"] or "[]")
            except json.JSONDecodeError:
                files = []
            out.append(MemoryHit(row["id"], row["kind"], row["title"], row["body"], 1.0, row["created_at"], files, row["cost_usd"]))
            if len(out) >= k:
                break
        return out

    def context_block(self, query: str, *, cwd: str = "", k: int = 4) -> str:
        """What actually gets prepended to the prompt — recall, formatted."""
        hits = self.recall(query, k=k, cwd=cwd or None)
        if not hits:
            return ""
        lines = ["# Relevant past work (memory recall)"]
        for hit in hits:
            files = f" — {', '.join(hit.files[:6])}" if hit.files else ""
            lines.append(f"- [{hit.kind} {hit.age()} ago, sim {hit.score:.2f}] {hit.title}{files}")
            lines.append(f"  {hit.snippet(200)}")
        return "\n".join(lines)

    # -- compaction ------------------------------------------------------
    def compress_messages(self, messages: list[Any], *, keep_last: int = 6, max_chars: int = 1800) -> tuple[list[Any], str]:
        """Fold older turns into one `summary` memory and return the new tail."""
        from mharo_tui.agent.session import Text, Message as M

        if len(messages) <= keep_last + 2:
            return messages, ""
        old, tail = messages[:-keep_last], messages[-keep_last:]
        lines: list[str] = []
        for msg in old:
            if msg.role == "user":
                lines.append(f"asked: {_one_line(msg.text, 140)}")
            elif msg.role == "assistant":
                for block in msg.blocks:
                    kind = type(block).__name__
                    if kind == "ToolCall":
                        lines.append(f"ran {block.tool} → {'ok' if block.ok else 'failed'}")
                    elif kind == "Text" and block.text.strip():
                        lines.append(f"said: {_one_line(block.text, 140)}")
        digest = "\n".join(lines)[-max_chars:]
        if digest:
            self.remember(
                title=f"compacted {len(old)} turns", body=digest, kind="summary",
                tags="compaction",
            )
        note = M(role="system", blocks=[Text(f"[earlier context compacted]\n{digest}")])
        return [note, *tail], digest


def _one_line(text: str, width: int) -> str:
    flat = " ".join((text or "").split())
    return flat[:width]


def _lexical(query: str, hay: str) -> float:
    """Token-overlap score, blended with cosine so keyword recall still works."""
    q = set(tokens(query))
    h = set(tokens(hay))
    if not q or not h:
        return 0.0
    return len(q & h) / math.sqrt(len(q) * len(h))
