"""P1-9 · Cost ledger and dashboard.

Numbers come from the SQLite `usage` table — the same rows the engine writes —
so `ma cost` cannot drift from what a session actually spent. Prices come from
config (per-model, USD per million tokens); unknown models show $0 and are
listed under `unpriced` instead of being silently zeroed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .config import Config
from .sessiondb import SessionDB


@dataclass
class Row:
    label: str
    calls: int
    tokens_in: int
    tokens_out: int
    usd: float
    extra: str = ""

    def line(self, widths: tuple[int, int, int]) -> str:
        name, left, right = widths
        return (
            f"{self.label[:name]:{name}s}  {self.calls:>4}  "
            f"{self.tokens_in:>9,}  {self.tokens_out:>9,}  {self.usd:>8.4f}  {self.extra[:right]}"
        )


HEADER = f"{'model / session':<34}  {'calls':>4}  {'in':>9}  {'out':>9}  {'usd':>8}  detail"


def prices_for(config: Config) -> dict[str, list[float]]:
    return config.prices()


def unpriced_models(db: SessionDB, config: Config) -> list[str]:
    known = set(prices_for(config))
    rows = db.conn.execute("SELECT DISTINCT model FROM usage").fetchall()
    out = []
    for row in rows:
        model = row["model"]
        if model and not any(model.startswith(k.split(":")[0]) for k in known):
            out.append(model)
    return sorted(out)


def per_model(db: SessionDB) -> list[Row]:
    rows = db.conn.execute(
        "SELECT provider, model, tier, COUNT(*) calls, SUM(tokens_in) ti, SUM(tokens_out) to_, "
        "SUM(cost_usd) usd, SUM(latency_ms) lat FROM usage GROUP BY provider, model, tier ORDER BY usd DESC"
    ).fetchall()
    return [
        Row(
            label=f"{r['model']} ({r['tier'] or r['provider']})",
            calls=int(r["calls"]), tokens_in=int(r["ti"] or 0), tokens_out=int(r["to_"] or 0),
            usd=float(r["usd"] or 0.0),
            extra=f"{(int(r['lat'] or 0) / max(1, int(r['calls']))) / 1000:.1f}s avg",
        )
        for r in rows
    ]


def per_session(db: SessionDB, limit: int = 12) -> list[Row]:
    rows = db.conn.execute(
        "SELECT id, name, model, cost_usd, tokens_in, tokens_out, started_at, status, "
        "(SELECT COUNT(*) FROM tool_calls t WHERE t.session_id=s.id) tools "
        "FROM sessions s ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    out: list[Row] = []
    for r in rows:
        age = _age(float(r["started_at"]))
        out.append(
            Row(
                label=f"#{r['id']} {(r['name'] or '')[:22]}",
                calls=int(r["tools"] or 0), tokens_in=int(r["tokens_in"] or 0), tokens_out=int(r["tokens_out"] or 0),
                usd=float(r["cost_usd"] or 0.0), extra=f"{age} · {r['status']}",
            )
        )
    return out


def per_day(db: SessionDB, days: int = 14) -> list[Row]:
    rows = db.conn.execute(
        "SELECT date(created_at,'unixepoch') d, COUNT(*) calls, SUM(tokens_in) ti, SUM(tokens_out) to_, "
        "SUM(cost_usd) usd FROM usage GROUP BY d ORDER BY d DESC LIMIT ?", (days,)
    ).fetchall()
    return [
        Row(label=str(r["d"]), calls=int(r["calls"]), tokens_in=int(r["ti"] or 0),
            tokens_out=int(r["to_"] or 0), usd=float(r["usd"] or 0.0), extra=f"{int(r['calls'])} call(s)")
        for r in rows
    ]


def totals(db: SessionDB) -> dict[str, Any]:
    row = db.conn.execute(
        "SELECT COUNT(*) calls, COALESCE(SUM(tokens_in),0) ti, COALESCE(SUM(tokens_out),0) to_, "
        "COALESCE(SUM(cost_usd),0) usd, COALESCE(SUM(latency_ms),0) lat, COUNT(DISTINCT session_id) sess FROM usage"
    ).fetchone()
    sessions = db.conn.execute("SELECT COALESCE(SUM(cost_usd),0) usd, COUNT(*) n FROM sessions").fetchone()
    budget = 0.0
    try:
        cfg = Config.load()
        budget = float(cfg.budget.get("max_session_usd", 0) or 0)
    except Exception:
        budget = 0.0
    return {
        "llm_calls": int(row["calls"] or 0),
        "sessions_with_spend": int(row["sess"] or 0),
        "sessions_total": int(sessions["n"] or 0),
        "tokens_in": int(row["ti"] or 0),
        "tokens_out": int(row["to_"] or 0),
        "usd": round(float(row["usd"] or 0.0), 6),
        "avg_latency_s": round(int(row["lat"] or 0) / 1000 / max(1, int(row["calls"] or 1)), 2),
        "budget_per_session": budget,
    }


def render(db: SessionDB, config: Config, *, group: str = "model", limit: int = 12) -> str:
    data = totals(db)
    lines = [
        "Mharo cost dashboard",
        "─" * 84,
        HEADER,
        "─" * 84,
    ]
    if group == "session":
        rows = per_session(db, limit)
    elif group == "day":
        rows = per_day(db, limit)
    else:
        rows = per_model(db)
    if not rows:
        lines.append("  (no spend recorded yet — run a task)")
    widths = (34, 6, 20)
    for row in rows:
        lines.append(row.line(widths))
    lines.append("─" * 84)
    lines.append(
        f"TOTAL  {data['llm_calls']} calls · {data['tokens_in']:,} in / {data['tokens_out']:,} out · "
        f"${data['usd']:.4f} · avg {data['avg_latency_s']}s/call · {data['sessions_total']} session(s)"
    )
    if data["budget_per_session"]:
        lines.append(f"budget guard: ${data['budget_per_session']:.2f} per session (router downgrades above 85%)")
    unpriced = unpriced_models(db, config)
    if unpriced:
        lines.append(f"unpriced models (counted as $0): {', '.join(unpriced[:6])} — set cost.prices in config")
    return "\n".join(lines)


def as_json(db: SessionDB, config: Config) -> str:
    return json.dumps(
        {
            "totals": totals(db),
            "by_model": [r.__dict__ for r in per_model(db)],
            "by_session": [r.__dict__ for r in per_session(db)],
            "unpriced": unpriced_models(db, config),
            "generated_at": time.time(),
        },
        indent=2,
    )


def _age(ts: float) -> str:
    secs = max(0.0, time.time() - ts)
    if secs < 3600:
        return f"{secs / 60:.0f}m"
    if secs < 86400 * 7:
        return f"{secs / 3600:.0f}h"
    return f"{secs / 86400:.0f}d"
