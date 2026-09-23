"""End-to-end tests for `ma`: engine loop, tool gating, verification gate,
subagent isolation, skills, CLI parsing. The provider is the scripted replay
transport, so these run offline in ~2 s and still exercise real code paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mharo.cli import build_parser, main
from mharo.config import Config
from mharo.engine import Engine, EngineSettings
from mharo.memory import Memory
from mharo.permissions import Permissions
from mharo.providers import ProviderHub
from mharo.sessiondb import SessionDB
from mharo.skills import discover, parse
from mharo.subagents import SubAgentRunner, build_roots
from mharo.vault import Vault
from mharo_tui.agent.tools import TOOLS


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MHARO_HOME", str(tmp_path / "home"))


def make_engine(tmp_path: Path, script: list[dict], *, verify: str = "none", peer: bool = False,
                plan: bool = False, deny: list[str] | None = None, memory: bool = False,
                approve: str | None = None) -> Engine:
    cfg = Config.load()
    cfg.raw["verify"]["auto_checks"] = verify
    if deny:
        cfg.raw["permissions"]["deny"] = list(cfg.raw["permissions"].get("deny") or []) + deny
    db = SessionDB(tmp_path / "engine.db")
    hub = ProviderHub(cfg, db=db, force_provider="replay", replay_scripts={"all": script})
    engine = Engine(
        cfg, cwd=tmp_path, db=db, hub=hub, vault=Vault.load(tmp_path / "vault.json"),
        settings=EngineSettings(tier="auto", max_turns=3, plan=plan, verify=verify != "none",
                                peer_review=peer, use_memory=memory),
    )
    engine.permissions.default = "allow" if approve is None else "ask"
    if approve is not None:
        engine.approver = lambda tool, args, registry: approve
    return engine


async def test_engine_streams_answer_and_persists_session(tmp_path):
    (tmp_path / "hello.py").write_text("def hi():\n    return 'hi'\n", encoding="utf-8")
    engine = make_engine(tmp_path, [{"text": "hello.py defines hi() returning 'hi'."}])
    result = await engine.run("what does hello.py do?")
    assert "hi()" in result.answer and result.ok and result.turns == 1
    assert result.verification is not None and result.verification.checks == []
    rows = engine.db.transcript(result.session_id)
    assert any("hello.py" in r["content"] for r in rows)          # task recorded
    assert engine.db.session(result.session_id).status in {"done", "unverified"}
    engine.db.close()


async def test_engine_runs_tool_then_answers(tmp_path):
    (tmp_path / "data.txt").write_text("42\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "read_file", "args": {"path": "data.txt"}, "id": "c1"}]},
        {"text": "The file contains 42."},
    ])
    result = await engine.run("read data.txt")
    assert result.ok and "42" in result.answer
    history = engine.db.tool_history(result.session_id)
    assert history[0]["tool"] == "read_file" and history[0]["ok"] == 1
    assert "42" in history[0]["output"]              # the real file content was audited
    assert result.files == []                        # a read touched nothing on disk
    engine.db.close()


async def test_engine_blocks_denied_tool_without_running_it(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("safe\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "bash", "args": {"command": "rm -f keep.txt"}, "id": "b1"}]},
        {"text": "That was blocked by policy."},
    ], deny=["bash:rm*"])
    result = await engine.run("delete keep.txt with the shell")
    assert keep.exists() and keep.read_text(encoding="utf-8") == "safe\n"
    row = engine.db.tool_history(result.session_id)[0]
    assert row["permission"] == "deny" and row["ok"] == 0
    assert "blocked by permission rule" in (result.answer + str(row["output"]))
    engine.db.close()


async def test_engine_denies_when_approver_says_no(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "write_file", "args": {"path": "notes.md", "content": "new\n"}, "id": "w1"}]},
        {"text": "Could not write it."},
    ], approve="deny")
    result = await engine.run("overwrite notes.md")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert engine.db.tool_history(result.session_id)[0]["permission"] == "deny"
    assert "User denied" in engine.db.tool_history(result.session_id)[0]["output"]
    engine.db.close()


async def test_engine_allow_rule_skips_approver(tmp_path):
    calls: list[str] = []
    (tmp_path / "ok.md").write_text("a\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "write_file", "args": {"path": "ok.md", "content": "b\n"}, "id": "w1"}]},
        {"text": "Wrote ok.md."},
    ], approve="deny")
    engine.approver = lambda tool, args, registry: calls.append(tool) or "deny"
    engine.permissions.allow.append("write_file:ok.md")
    result = await engine.run("replace ok.md with b")
    assert (tmp_path / "ok.md").read_text(encoding="utf-8") == "b\n"
    assert calls == []                                            # allowlist won over ask
    assert engine.db.tool_history(result.session_id)[0]["permission"] == "allow"
    engine.db.close()


async def test_verification_rejects_confident_lie(tmp_path):
    (tmp_path / "test_broken.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
    engine = make_engine(tmp_path, [{"text": "All tests pass, done. Everything is fixed and green."}],
                         verify="python3 -m pytest -q --no-header")
    result = await engine.run("make the tests pass")
    assert not result.ok
    assert result.verification.claim.claimed and not result.verification.claim.accepted
    assert any(not check.ok for check in result.verification.checks)
    assert result.turns >= 2                          # a fix round actually ran
    assert engine.db.session(result.session_id).status == "unverified"
    engine.db.close()


async def test_verification_accepts_real_proof_after_fix(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "edit_file", "args": {"path": "calc.py", "old_text": "a - b", "new_text": "a + b"}, "id": "e1"}]},
        {"text": "Changed add() to use + ; the suite is now green."},
    ], verify="python3 -m pytest -q --no-header")
    result = await engine.run("fix calc.add so test_calc passes")
    assert "a + b" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert any(check.ok and check.passed >= 1 for check in result.verification.checks)
    assert result.ok
    engine.db.close()


async def test_undo_restores_agent_edit(tmp_path):
    path = tmp_path / "settings.py"
    path.write_text("MODE = 'dev'\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "edit_file", "args": {"path": "settings.py", "old_text": "MODE = 'dev'", "new_text": "MODE = 'prod'"}, "id": "u1"}]},
        {"text": "Switched to prod."},
    ])
    result = await engine.run("switch MODE to prod")
    assert "prod" in path.read_text(encoding="utf-8")
    assert engine.db.undo_last(result.session_id) == ["restored settings.py"]
    assert path.read_text(encoding="utf-8") == "MODE = 'dev'\n"
    engine.db.close()


async def test_secret_never_reaches_the_transcript(tmp_path):
    leak = tmp_path / "leak.txt"
    leak.write_text("token is sk-proj-abcdefgh12345678 and that is all\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "read_file", "args": {"path": "leak.txt"}, "id": "r1"}]},
        {"text": "Redacted the key from the output."},
    ])
    engine.vault.set("leaked", "sk-proj-abcdefgh12345678")
    result = await engine.run("read leak.txt")
    stored = engine.db.tool_history(result.session_id)[0]["output"]
    assert "sk-proj-abcdefgh12345678" not in stored
    assert "‹redacted" in stored and "redacted before they could reach the model" in stored
    messages = engine.db.transcript(result.session_id)
    assert all("sk-proj-abcdefgh12345678" not in m["content"] for m in messages)
    engine.db.close()


async def test_plan_creates_steps_and_memory_records_the_run(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    engine = make_engine(tmp_path, [
        {"text": json.dumps({"steps": [{"goal": "read a.py", "how": "read_file"}, {"goal": "explain it", "how": "answer"}],
                             "risks": ["none"], "verify": "read output"})},
        {"text": "a.py binds x to 1."},
        {"text": "Explained in one line."},
    ], plan=True, memory=True)
    result = await engine.run("read a.py then explain it")
    assert len(result.steps) == 2 and engine.plan_verify == "read output"
    assert all(step.done for step in result.steps)
    assert engine.db.memory_count() >= 1
    block = engine.memory.context_block("read a.py and explain", cwd=str(tmp_path))
    assert "Relevant past work" in block
    engine.db.close()


async def test_engine_records_token_savings_event(tmp_path):
    engine = make_engine(tmp_path, [
        {"calls": [{"tool": "bash", "args": {"command": "python3 -c \"print('x' * 60000)\""}, "id": "b1"}]},
        {"text": "Large output, truncated."},
    ])
    await engine.run("produce a lot of output")
    rows = engine.db.conn.execute("SELECT detail FROM events WHERE kind='savings'").fetchall()
    assert rows
    payload = json.loads(rows[0]["detail"])
    assert payload["chars"] > 1000                              # truncation actually saved tokens
    engine.db.close()


# ------------------------------------------------------------------ subagents

async def test_subagent_reader_cannot_touch_mutating_tools(tmp_path):
    (tmp_path / "x.py").write_text("x = 1\n", encoding="utf-8")
    cfg = Config.load()
    db = SessionDB(tmp_path / "sub.db")
    sid = db.start_session(cwd=str(tmp_path), task="t", model="replay", provider="replay")
    hub = ProviderHub(cfg, db=db, force_provider="replay", replay_scripts={"all": [
        {"calls": [{"tool": "bash", "args": {"command": "rm -rf /"}, "id": "nope"}]},
        {"text": json.dumps({"summary": "refused", "files": [], "open_questions": []})},
    ]})
    runner = SubAgentRunner(hub, cwd=tmp_path, permissions=Permissions.from_config(cfg.permissions),
                            roles=build_roots(cfg.subagents), session_db=db, session_id=sid)
    result = await runner.run("reader", "delete everything")
    assert "isolated-tools violation" in result.error and result.tool_calls == 0
    assert not result.ok
    assert (tmp_path / "x.py").read_text(encoding="utf-8") == "x = 1\n"
    db.close()


async def test_subagent_parallel_roles_and_unknown_role(tmp_path):
    cfg = Config.load()
    payload = {"text": json.dumps({"summary": "two files", "files": [{"path": "a.py", "why": "target", "lines": [1, 2]}], "open_questions": []})}
    hub = ProviderHub(cfg, force_provider="replay", replay_scripts={"all": [payload, dict(payload)]})
    runner = SubAgentRunner(hub, cwd=tmp_path, permissions=Permissions(default="allow"), roles=build_roots(cfg.subagents), max_parallel=2)
    results = await runner.run_many([("reader", "task a"), ("reader", "task b")])
    assert len(results) == 2 and all(r.ok for r in results)
    assert all(r.data.get("summary") == "two files" for r in results)
    unknown = await runner.run("ghost", "task")
    assert not unknown.ok and "unknown subagent" in unknown.error


# ---------------------------------------------------------------------- skills

def test_skill_parsing_validation_and_run(tmp_path):
    root = tmp_path / "skills" / "demo"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill\ntools: read_file, teleporter\ntier: hyper\nmax_turns: lots\n---\n"
        "Read {task} and report.\n", encoding="utf-8")
    skill = parse((root / "SKILL.md").read_text(encoding="utf-8"), root / "SKILL.md", known_tools=list(TOOLS))
    assert skill.name == "demo" and skill.tools == ["read_file", "teleporter"]
    assert any("not in the registry" in p for p in skill.problems)
    assert any("tier must be" in p for p in skill.problems)
    assert any("max_turns" in p for p in skill.problems)
    skills, problems = discover([tmp_path / "skills"], known_tools=list(TOOLS))
    assert skills and problems
    good = parse("---\nname: ok\ndescription: fine\ntools: read_file\ntier: cheap\nmax_turns: 2\n---\nbody {task}", known_tools=list(TOOLS))
    assert not good.problems and good.render("TASK") == "body TASK"


# ----------------------------------------------------------------------- cli

def test_cli_parser_and_replay_run(tmp_path):
    from mharo.cli import insert_run

    parser = build_parser()
    args = parser.parse_args(insert_run(["--cwd", str(tmp_path), "--json", "do the thing"]))
    assert args.task == ["do the thing"] and args.json and args.tier == "auto"
    assert args.cmd == "run"
    assert parser.parse_args(insert_run(["--cwd", "/x", "doctor", "--strict"])).strict
    assert parser.parse_args(insert_run(["--tier", "strong", "run", "x", "--no-verify"])).no_verify
    assert parser.parse_args(insert_run(["--cwd", "/x", "bench", "--only", "key"])).only == "key"
    script = tmp_path / "script.json"
    # two turns: turn 0 is consumed by the planner (not JSON → single-step
    # fallback), turn 1 is the answer — proves the default flags work end to end
    script.write_text(json.dumps([{"text": "I will look at a.py."}, {"text": "Replayed answer: the code is fine."}]), encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    env = {**os.environ, "MHARO_HOME": str(tmp_path / "home"), "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run(
        [sys.executable, "-m", "mharo", "--cwd", str(tmp_path), "--provider", "replay",
         "--replay-file", str(script), "--no-verify", "--no-memory", "--json", "explain a.py"],
        capture_output=True, text=True, timeout=180, cwd=str(tmp_path), env=env,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True and "Replayed answer" in payload["answer"]
    assert payload["verification"]["checks"] == []
    assert payload["steps"] and payload["session_id"] >= 1 and payload["tokens_in"] > 0


def test_cli_vault_and_skill_commands(tmp_path, capsys):
    rc = main(["--cwd", str(tmp_path), "vault", "set", "openai", "sk-test123456789"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0600" in out and "sk-test123456789" not in out
    assert main(["--cwd", str(tmp_path), "vault", "get", "openai"]) == 0
    shown = capsys.readouterr().out
    assert "sk-test123456789" not in shown and "chars)" in shown
    assert main(["--cwd", str(tmp_path), "vault", "get", "openai", "--raw"]) == 0
    assert "sk-test123456789" in capsys.readouterr().out
    (tmp_path / "skills").mkdir()
    assert main(["--cwd", str(tmp_path), "skill", "new", "my-skill", "--description", "demo"]) == 0
    assert (tmp_path / "skills" / "my-skill" / "SKILL.md").is_file()
    assert main(["--cwd", str(tmp_path), "skill", "list"]) == 0
    assert "my-skill" in capsys.readouterr().out


def test_cli_sessions_memory_cost(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(home / "agent.db")
    sid = db.start_session(cwd=str(tmp_path), task="remembered task", model="m", provider="p")
    db.add_usage(sid, provider="p", model="gpt-4o-mini", tier="cheap", tokens_in=1000, tokens_out=200, cost_usd=0.0123)
    db.end_session(sid, status="done", proof="ok")
    Memory.build(db, {}).remember(title="key decision", body="we chose WAL journal mode", kind="note", cwd=str(tmp_path))
    db.close()
    assert main(["--cwd", str(tmp_path), "sessions", "--show", str(sid)]) == 0
    dumped = json.loads(capsys.readouterr().out)
    assert dumped["session"]["status"] == "done" and dumped["ledger"]["usage"][0]["cost"] == pytest.approx(0.0123)
    assert main(["--cwd", str(tmp_path), "memory", "--search", "WAL"]) == 0
    assert "key decision" in capsys.readouterr().out
    assert main(["--cwd", str(tmp_path), "cost"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "gpt-4o-mini" in out


def test_cli_bench_and_doctor_smoke(tmp_path):
    """`ma bench` and `ma doctor` are the plan's acceptance gates — run them here too."""
    assert main(["--cwd", str(tmp_path), "bench", "--only", "basic"]) == 0
    assert main(["--cwd", str(tmp_path), "doctor"]) in (0, 1)   # warnings are fine, crashes are not
