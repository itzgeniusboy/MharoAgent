"""P1-6 · SQLite sessions: transcripts, usage ledger, tool calls, snapshots.

One DB, WAL mode, forward-only schema versioning. Everything `ma` needs to
remember lives here (sessions, memory vectors, cost ledger) so there is no
second store to keep in sync.

`/undo` is real: before any file mutation the engine writes a snapshot row with
the previous content, so `ma undo` restores exactly what the agent changed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    cwd TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL,
    ended_at REAL,
    task TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    cost_usd REAL NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    proof TEXT NOT NULL DEFAULT '',
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions(started_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    thinking TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    step INTEGER NOT NULL DEFAULT 0,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}',
    ok INTEGER,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    output TEXT NOT NULL DEFAULT '',
    permission TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tool_session ON tool_calls(session_id, id);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT '',
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    key_label TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    before BLOB,
    after BLOB,
    kind TEXT NOT NULL DEFAULT 'edit',
    applied INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS snap_session ON snapshots(session_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'task',
    cwd TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    files_json TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    vec BLOB,
    dims INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    command TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@dataclass
class SessionRow:
    id: int
    name: str
    cwd: str
    model: str
    provider: str
    started_at: float
    task: str
    status: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    proof: str

    @property
    def age(self) -> str:
        secs = max(0, time.time() - self.started_at)
        if secs < 90:
            return f"{secs:.0f}s ago"
        if secs < 5400:
            return f"{secs / 60:.0f}m ago"
        return f"{secs / 3600:.1f}h ago"


class SessionDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.conn = connect(self.path)
        self._migrate()

    # -- lifecycle -------------------------------------------------------
    def _migrate(self) -> None:
        # create first, then read the version — a brand new file has no tables yet
        self.conn.executescript(SCHEMA)
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row["value"]) if row else 0
        if version < SCHEMA_VERSION:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SessionDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- sessions --------------------------------------------------------
    def start_session(self, *, cwd: str, task: str, model: str, provider: str, name: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions(cwd,task,model,provider,name,started_at,status) VALUES(?,?,?,?,?,?, 'open')",
            (cwd, task[:4000], model, provider, name or task[:60], time.time()),
        )
        return int(cur.lastrowid or 0)

    def end_session(self, session_id: int, *, status: str = "done", proof: str = "") -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at=?, status=?, proof=?, "
            "cost_usd=(SELECT COALESCE(SUM(cost_usd),0) FROM usage WHERE session_id=?), "
            "tokens_in=(SELECT COALESCE(SUM(tokens_in),0) FROM usage WHERE session_id=?), "
            "tokens_out=(SELECT COALESCE(SUM(tokens_out),0) FROM usage WHERE session_id=?) "
            "WHERE id=?",
            (time.time(), status, proof[:4000], session_id, session_id, session_id, session_id),
        )

    def recent_sessions(self, limit: int = 20) -> list[SessionRow]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            SessionRow(
                id=r["id"], name=r["name"], cwd=r["cwd"], model=r["model"], provider=r["provider"],
                started_at=r["started_at"], task=r["task"], status=r["status"], cost_usd=r["cost_usd"],
                tokens_in=r["tokens_in"], tokens_out=r["tokens_out"], proof=r["proof"],
            )
            for r in rows
        ]

    def session(self, session_id: int) -> SessionRow | None:
        r = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not r:
            return None
        return SessionRow(
            id=r["id"], name=r["name"], cwd=r["cwd"], model=r["model"], provider=r["provider"],
            started_at=r["started_at"], task=r["task"], status=r["status"], cost_usd=r["cost_usd"],
            tokens_in=r["tokens_in"], tokens_out=r["tokens_out"], proof=r["proof"],
        )

    def resume_context(self, session_id: int, limit: int = 40) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # -- messages / tools / usage ---------------------------------------
    def add_message(self, session_id: int, role: str, content: str, *, tier: str = "", thinking: str = "", meta: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(session_id,role,tier,content,thinking,created_at,meta_json) VALUES(?,?,?,?,?,?,?)",
            (session_id, role, tier, content, thinking, time.time(), json.dumps(meta or {})),
        )
        return int(cur.lastrowid or 0)

    def add_tool_call(
        self, session_id: int, tool: str, args: dict, *, ok: bool | None, duration_ms: int,
        output: str, permission: str = "", step: int = 0,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO tool_calls(session_id,step,tool,args_json,ok,duration_ms,output,permission,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id, step, tool, json.dumps(args, default=str),
                None if ok is None else int(bool(ok)), duration_ms, output[:20000], permission, time.time(),
            ),
        )
        return int(cur.lastrowid or 0)

    def add_usage(
        self, session_id: int, *, provider: str, model: str, tier: str, tokens_in: int, tokens_out: int,
        cost_usd: float, key_label: str = "", latency_ms: int = 0,
    ) -> None:
        self.conn.execute(
            "INSERT INTO usage(session_id,provider,model,tier,tokens_in,tokens_out,cost_usd,key_label,latency_ms,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (session_id, provider, model, tier, tokens_in, tokens_out, cost_usd, key_label, latency_ms, time.time()),
        )

    def add_check(self, session_id: int, name: str, ok: bool, *, command: str = "", detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO checks(session_id,name,command,ok,detail,created_at) VALUES(?,?,?,?,?,?)",
            (session_id, name, command, int(bool(ok)), detail[:4000], time.time()),
        )

    def add_event(self, session_id: int | None, kind: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO events(session_id,kind,detail,created_at) VALUES(?,?,?,?)",
            (session_id, kind, detail[:2000], time.time()),
        )

    def ledger(self, session_id: int) -> dict[str, Any]:
        usage = self.conn.execute(
            "SELECT provider, model, tier, SUM(tokens_in) ti, SUM(tokens_out) to_, SUM(cost_usd) cost, "
            "COUNT(*) calls, SUM(latency_ms) lat FROM usage WHERE session_id=? GROUP BY provider, model, tier",
            (session_id,),
        ).fetchall()
        tools = self.conn.execute(
            "SELECT tool, COUNT(*) n, SUM(CASE WHEN ok=1 THEN 1 ELSE 0 END) ok, SUM(duration_ms) ms "
            "FROM tool_calls WHERE session_id=? GROUP BY tool",
            (session_id,),
        ).fetchall()
        checks = self.conn.execute("SELECT name, ok, detail FROM checks WHERE session_id=?", (session_id,)).fetchall()
        return {
            "usage": [dict(r) for r in usage],
            "tools": [dict(r) for r in tools],
            "checks": [{"name": c["name"], "ok": bool(c["ok"]), "detail": c["detail"]} for c in checks],
        }

    def transcript(self, session_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, role, tier, content, created_at FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def tool_history(self, session_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT tool, args_json, ok, duration_ms, permission, output FROM tool_calls WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["args"] = json.loads(d.pop("args_json") or "{}")
            except json.JSONDecodeError:
                d["args"] = {}
            out.append(d)
        return out

    # -- undo snapshots --------------------------------------------------
    def snapshot_file(self, session_id: int, path: str, *, kind: str = "edit") -> bool:
        """Store the *current* bytes of `path` (or None if absent) before mutation."""
        target = Path(path)
        before = target.read_bytes() if target.is_file() else None
        self.conn.execute(
            "INSERT INTO snapshots(session_id,path,before,after,kind,applied,created_at) VALUES(?,?,?,?,?,1,?)",
            (session_id, str(target), before, None, kind, time.time()),
        )
        return before is not None

    def record_after(self, session_id: int, path: str) -> None:
        target = Path(path)
        after = target.read_bytes() if target.is_file() else None
        self.conn.execute(
            "UPDATE snapshots SET after=? WHERE session_id=? AND path=? AND applied=1",
            (after, session_id, str(target)),
        )

    def undo_last(self, session_id: int | None = None) -> list[str]:
        """Restore the most recent change set (all paths written by one step)."""
        if session_id is None:
            row = self.conn.execute(
                "SELECT session_id FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return []
            session_id = int(row["session_id"])
        rows = self.conn.execute(
            "SELECT * FROM snapshots WHERE session_id=? AND applied=1 ORDER BY id DESC",
            (session_id,),
        ).fetchall()
        if not rows:
            return []
        newest_ts = rows[0]["created_at"]
        group = [r for r in rows if abs(r["created_at"] - newest_ts) < 0.5]
        restored: list[str] = []
        for row in group:
            path = Path(row["path"])
            if row["before"] is None:
                if path.exists():
                    path.unlink()
                    restored.append(f"deleted {path.name}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(row["before"])
                restored.append(f"restored {path.name}")
            self.conn.execute("UPDATE snapshots SET applied=0 WHERE id=?", (row["id"],))
        self.add_event(session_id, "undo", "; ".join(restored))
        return restored

    def pending_undo(self, session_id: int | None = None) -> list[dict[str, Any]]:
        where, args = ("WHERE applied=1", [])
        if session_id is not None:
            where += " AND session_id=?"
            args = [session_id]
        rows = self.conn.execute(
            f"SELECT id, session_id, path, kind, created_at FROM snapshots {where} ORDER BY id DESC LIMIT 40", args
        ).fetchall()
        return [dict(r) for r in rows]

    # -- memory vectors --------------------------------------------------
    def put_memory(
        self, *, title: str, body: str, vec: bytes, dims: int, cwd: str = "",
        files: list[str] | None = None, cost_usd: float = 0.0, kind: str = "task", tags: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO memories(kind,cwd,title,body,files_json,cost_usd,created_at,vec,dims,tags) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (kind, cwd, title[:200], body, json.dumps(files or []), cost_usd, time.time(), vec, dims, tags),
        )
        return int(cur.lastrowid or 0)

    def all_memories(self, limit: int = 2000) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall())

    def memory_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"])

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> int:
            try:
                return int(self.conn.execute(sql).fetchone()[0] or 0)
            except sqlite3.Error:
                return 0

        return {
            "db": str(self.path),
            "size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
            "sessions": one("SELECT COUNT(*) FROM sessions"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "tool_calls": one("SELECT COUNT(*) FROM tool_calls"),
            "memories": one("SELECT COUNT(*) FROM memories"),
            "undoable": one("SELECT COUNT(*) FROM snapshots WHERE applied=1"),
            "schema": self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone(),
        }
