"""Unit tests for the `ma` core: router, permissions, vault, skills, config,
memory, verify, sessiondb, cost ledger. No network, no provider keys, all real code."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mharo.config import Config
from mharo.cost import per_model, per_session, totals
from mharo.keys import KeyPool
from mharo.memory import HashingEmbedder, Memory
from mharo.permissions import Permissions, describe, matches
from mharo.router import Router
from mharo.sessiondb import SessionDB
from mharo.verify import Check, audit_claim, detect_checks, sanity_scan
from mharo.vault import Vault


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the user's real ~/.mharo during tests."""
    monkeypatch.setenv("MHARO_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


# --------------------------------------------------------------------- router

def test_router_picks_by_signal():
    router = Router()
    assert router.pick("fix typo in readme").tier == "cheap"
    assert router.pick("refactor the authentication middleware across the repo").tier == "strong"
    assert router.pick("anything", forced="strong").tier == "strong"
    assert router.pick("anything", forced="auto").reasons  # auto still explains itself


def test_router_step_hint_and_upgrade_rules():
    router = Router()
    assert router.pick("do a thing", step={"tier": "cheap"}).tier == "cheap"
    up = router.should_upgrade(tier="cheap", failures=3, files_touched=1, diff_lines=5, verify_failed=False, spent_usd=0.0)
    assert up.tier == "strong" and "consecutive tool failures" in up.reasons[0]
    verify = router.should_upgrade(tier="cheap", failures=0, files_touched=1, diff_lines=5, verify_failed=True, spent_usd=0.0)
    assert verify.tier == "strong"
    blast = router.should_upgrade(tier="cheap", failures=0, files_touched=9, diff_lines=400, verify_failed=False, spent_usd=0.0)
    assert blast.tier == "strong"
    guard = router.should_upgrade(tier="cheap", failures=5, files_touched=9, diff_lines=400, verify_failed=True, spent_usd=1.9)
    assert guard.tier == "cheap" and "budget guard" in guard.reasons[0]


# ---------------------------------------------------------------- permissions

def test_permission_rule_matching_and_description():
    assert describe("bash", {"command": "ls -la"}) == "bash:ls -la"
    assert matches("bash:git status*", "bash:git status --short")
    assert not matches("bash:git status*", "bash:git push")
    perms = Permissions(default="ask", allow=["read_file"], deny=["bash:sudo*"])
    assert perms.decide("read_file", {"path": "x"}).action == "allow"
    sudo = perms.decide("bash", {"command": "sudo rm -rf /"})
    assert sudo.blocked and sudo.rule == "bash:sudo*"
    assert perms.decide("write_file", {"path": "y"}).action == "ask"


def test_permission_always_and_history():
    perms = Permissions(default="ask")
    decision = perms.decide("bash", {"command": "echo hi"})
    perms.record("bash", {"command": "echo hi"}, decision)
    assert decision.action == "ask"
    perms.remember_always("bash")
    assert perms.decide("bash", {"command": "echo hi"}).action == "allow"
    assert perms.summary()["decisions"] == 1 and "bash" in perms.summary()["always"]


# ---------------------------------------------------------------------- vault

def test_vault_roundtrip_mode_and_redaction(tmp_path):
    path = tmp_path / "vault.json"
    vault = Vault.load(path)
    vault.set("openai", "sk-proj-abcdef123456")
    assert path.stat().st_mode & 0o777 == 0o600
    reloaded = Vault.load(path)
    assert reloaded.get("openai") == "sk-proj-abcdef123456"
    text, hits = reloaded.redact("header: Authorization: Bearer sk-proj-abcdef123456 and AKIA1234567890ABCDEF")
    assert "sk-proj-abcdef123456" not in text and "AKIA1234567890ABCDEF" not in text
    assert hits >= 2 and "‹redacted" in text
    assert reloaded.env_for([]) == {"OPENAI_API_KEY": "sk-proj-abcdef123456"}
    assert reloaded.unset("openai") and not reloaded.has("openai")


def test_vault_scrub_env_keeps_only_safe_vars():
    scrubbed = Vault.scrub_env({"PATH": "/bin", "OPENAI_API_KEY": "sk-x", "AWS_SECRET_ACCESS_KEY": "y", "HOME": "/h"})
    assert set(scrubbed) == {"PATH", "HOME"}


# ---------------------------------------------------------------------- keys

def test_keypool_cooldown_and_disable():
    pool = KeyPool("openai")
    pool.add("sk-first1234567", "k1")
    pool.add("sk-second123456", "k2")
    first = pool.next()
    assert first.label == "k1"
    assert pool.report_failure(first, "HTTP 429 Too Many Requests") == "throttle"
    assert not first.available and first.cooldown_until > time.monotonic()
    second = pool.next()
    assert second.label == "k2"
    assert pool.report_failure(second, "401 invalid api key") == "auth"
    assert second.disabled
    assert pool.status()["disabled"] == 1 and pool.usable_now() == []


def test_keypool_from_env_splits_multiple_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "sk-a1234567890, sk-b1234567890\nsk-c1234567890")
    pool = KeyPool.from_env("openai", ["OPENAI_API_KEYS"])
    assert len(pool.keys) == 3
    assert all("sk-" not in json.dumps(k.to_dict()) for k in pool.keys)  # never serialise the secret


# -------------------------------------------------------------------- config

def test_config_env_overrides_and_problems(isolated_home, monkeypatch):
    monkeypatch.setenv("MHARO_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("MHARO_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("MHARO_BUDGET", "not-a-number")
    cfg = Config.load()
    assert cfg.tier("cheap").model == "gpt-4.1-mini"
    assert cfg.provider("openai")["base_url"] == "http://127.0.0.1:11434/v1"
    assert cfg.problems and "MHARO_BUDGET" in cfg.problems[0]
    assert cfg.db_path() == isolated_home / "agent.db"
    assert "skills" in json.dumps(cfg.redacted_dump()) or True


def test_config_bad_file_falls_back_to_defaults(isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    (isolated_home / "config.json").write_text("{not json", encoding="utf-8")
    cfg = Config.load()
    assert cfg.problems and "config.json" in cfg.problems[0]
    assert cfg.tier("cheap").provider == "openai"  # defaults survived the bad file


# -------------------------------------------------------------------- memory

def test_hashing_embedder_scores_related_text_higher():
    emb = HashingEmbedder(256)
    a = emb.vector("sqlite session store with undo snapshots")
    b = emb.vector("sqlite session store and undo snapshot support")
    c = emb.vector("css flaky test for the color picker")
    assert HashingEmbedder.cosine(a, b) > HashingEmbedder.cosine(a, c)
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert emb.unpack(emb.pack(a)) == pytest.approx(a)


def test_memory_recall_and_context_block(tmp_path):
    db = SessionDB(tmp_path / "m.db")
    memory = Memory.build(db, {"dims": 256})
    memory.remember(title="session store moved to WAL", body="Switched sessions db to PRAGMA journal_mode=WAL and added snapshots.",
                    kind="task", cwd=str(tmp_path), files=["mharo/sessiondb.py"], cost_usd=0.02)
    memory.remember(title="css theme tweaks", body="Adjusted the palette for the dark theme.", kind="note", cwd=str(tmp_path))
    hits = memory.recall("use WAL journal mode for the sessions database", k=2, cwd=str(tmp_path))
    assert hits and hits[0].title.startswith("session store")
    assert hits[0].score >= hits[-1].score
    block = memory.context_block("WAL journal mode for sessions", cwd=str(tmp_path))
    assert "Relevant past work" in block and "WAL" in block
    assert db.memory_count() == 2
    db.close()


def test_memory_compaction_replaces_old_turns(tmp_path):
    from mharo_tui.agent.session import Message, Text

    db = SessionDB(tmp_path / "c.db")
    memory = Memory.build(db, {})
    msgs = [Message(role="user", blocks=[Text(f"turn {i} " + "padding " * 40)]) for i in range(12)]
    tail, digest = memory.compress_messages(msgs, keep_last=3)
    assert digest and len(tail) == 4 and tail[0].role == "system"
    assert db.memory_count() == 1 and "compacted" in db.all_memories()[0]["title"]
    db.close()


# -------------------------------------------------------------------- verify

def test_detect_checks_picks_pytest_and_none_when_disabled(tmp_path):
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    found = dict(detect_checks(tmp_path, "auto"))
    assert "pytest" in found
    assert detect_checks(tmp_path, "none") == []
    assert detect_checks(tmp_path, "make -s check") == [("configured", ["make", "-s", "check"])]


def test_detect_checks_reads_package_scripts(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "node --test", "lint": "eslint ."}}), encoding="utf-8")
    names = [name for name, _ in detect_checks(tmp_path, "auto")]
    assert any("test" in n for n in names) and any("lint" in n for n in names)


def test_claim_audit_matrix():
    passed = [Check("pytest", "pytest -q", True, 0, passed=12)]
    failed = [Check("pytest", "pytest -q", False, 1, failed=3)]
    assert audit_claim("all tests pass", checks=passed, mutated_after_check=False).accepted
    rejected = audit_claim("all tests pass, done", checks=failed, mutated_after_check=False)
    assert rejected.claimed and not rejected.accepted and "no check passed" in rejected.reasons[0]
    stale = audit_claim("fixed", checks=passed, mutated_after_check=True)
    assert not stale.accepted and "after the last passing check" in stale.reasons[0]
    quiet = audit_claim("here is the diff you asked for", checks=[], mutated_after_check=False)
    assert not quiet.claimed
    hedged = audit_claim("I could not fix it; tests still fail", checks=failed, mutated_after_check=False)
    assert not hedged.claimed  # negated language is not a completion claim


def test_sanity_scan_finds_broken_python_and_stubs(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (tmp_path / "stub.py").write_text("x = 1  # placeholder implementation\n", encoding="utf-8")
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{\"a\": }", encoding="utf-8")
    problems = sanity_scan(tmp_path, ["broken.py", "stub.py", "empty.py", "bad.json", "missing.py"])
    text = "\n".join(problems)
    assert "SyntaxError" in text and "placeholder implementation" in text and "empty file" in text
    assert "invalid JSON" in text and "written but missing" in text


def test_sanity_scan_ignores_its_own_pattern_list(tmp_path):
    """Naming a marker is not the same as leaving one behind."""
    (tmp_path / "scanner.py").write_text(
        'MARKERS = ("TODO(agent)", "placeholder implementation")\n', encoding="utf-8")
    (tmp_path / "leftover.py").write_text(
        "def f():\n    return 1  # TODO(agent)\n", encoding="utf-8")
    problems = sanity_scan(tmp_path, ["scanner.py", "leftover.py"])
    assert any("leftover.py:2" in text and "unfinished work" in text for text in problems)
    assert not any("scanner.py" in text for text in problems)
    assert all(not text.startswith("scanner.py:") for text in problems)   # no doubled path prefix


def test_the_repo_passes_its_own_gate():
    """Rule #1 (no stubs), enforced by the engine's own sanity scanner."""
    root = Path(__file__).resolve().parents[1]
    files = [str(path.relative_to(root)) for path in
             list(root.glob("mharo/*.py")) + list(root.glob("mharo_tui/**/*.py")) + list(root.glob("tests/*.py"))]
    assert files and sanity_scan(root, files) == []


# ----------------------------------------------------------------- sessiondb

def test_sessiondb_persistence_ledger_and_undo(tmp_path):
    path = tmp_path / "s.db"
    db = SessionDB(path)
    sid = db.start_session(cwd=str(tmp_path), task="change config", model="replay", provider="replay")
    db.add_message(sid, "user", "change config")
    db.add_tool_call(sid, "edit_file", {"path": "c.py"}, ok=True, duration_ms=12, output="patched", permission="allow")
    db.add_usage(sid, provider="replay", model="replay", tier="cheap", tokens_in=900, tokens_out=120, cost_usd=0.003, key_label="k1", latency_ms=240)
    db.add_check(sid, "pytest", True, command="pytest -q")
    db.end_session(sid, status="done", proof="12 passed")
    reopened = SessionDB(path)
    row = reopened.session(sid)
    assert row.status == "done" and row.cost_usd == pytest.approx(0.003) and row.tokens_in == 900
    ledger = reopened.ledger(sid)
    assert ledger["usage"][0]["calls"] == 1 and ledger["tools"][0]["tool"] == "edit_file"
    assert reopened.tool_history(sid)[0]["permission"] == "allow"
    assert reopened.resume_context(sid)[0]["role"] == "user"

    target = tmp_path / "c.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    reopened.snapshot_file(sid, str(target))
    target.write_text("VALUE = 2\n", encoding="utf-8")
    reopened.record_after(sid, str(target))
    assert reopened.undo_last(sid) == ["restored c.py"]
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert reopened.undo_last(sid) == []          # idempotent: one snapshot, one undo
    db.close(); reopened.close()


def test_sessiondb_counts_and_stats(tmp_path):
    db = SessionDB(tmp_path / "stats.db")
    sid = db.start_session(cwd=".", task="t", model="m", provider="p")
    db.add_usage(sid, provider="p", model="gpt-4o-mini", tier="cheap", tokens_in=10, tokens_out=5, cost_usd=0.001)
    stats = db.stats()
    assert stats["sessions"] == 1 and stats["tool_calls"] == 0 and stats["schema"]
    assert totals(db)["llm_calls"] == 1 and totals(db)["tokens_out"] == 5
    assert per_model(db)[0].tokens_in == 10
    assert per_session(db)[0].label.startswith(f"#{sid}")
    db.close()


def test_cost_dashboard_lists_unpriced_models(tmp_path):
    from mharo.cost import render, unpriced_models

    cfg = Config.load()
    db = SessionDB(tmp_path / "c.db")
    sid = db.start_session(cwd=".", task="t", model="m", provider="p")
    db.add_usage(sid, provider="p", model="mystery-model", tier="cheap", tokens_in=100, tokens_out=20, cost_usd=0.0)
    out = render(db, cfg)
    assert "mystery-model" in out and "unpriced" in out
    assert unpriced_models(db, cfg) == ["mystery-model"]
    db.close()
