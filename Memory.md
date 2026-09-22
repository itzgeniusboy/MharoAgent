# Memory — Mharo Agent

## Goal
Ek fully-autonomous, all-in-one coding agent CLI for Termux (Android) + PC.
Naam: Mharo Agent · CLI: `ma` · Project: ~/opencode/MharoAgent

## Progress
- 2026-09-21: Master plan locked (Phase 1 → 5). Research kiya: opencode, OpenClaw/Clawdbot, DeepSeek Harness, loop-engineering, ponytail, 2026 brain-layer papers + 16+3 user repos. Plan: ~/storage/shared/Download/MharoAgent_PLAN.md. Backlog: backlog.md.
- 2026-09-21: Build STARTED. Env: Termux+Android, Python 3.14.6, git ok, no uv (pip). venv @ .venv. Scaffold banaya (src/mharo/* dirs). Next: Phase 1 providers → engine → memory → subagents → verify.

## Rules (user-locked)
- Koi dummy/fake code nahi. Sab real, functional. Har module 4-5 baar verify (ma bench / pytest).
- Khud ka code likhna (no copy-paste); libraries normal use; research → reference → own impl.
- Language: code/docs English, chat Hinglish.

## 2026-09-22 — Day session compile
- CI GREEN (Actions): compile+import+contract+pytest all pass. Root fix: workflow parse (inline-python) + httpx dep; real code fixes aur: Engine.window(limit=None), Completion.provider, Provider.keys/set_api_key, Router.stats (was rates).
- Tests: 33 pytest (core/errors/router rotation+fallback via MockTransport, engine, stream, types, tools calc+registry, memory TTL/persist, cli).
- CLI: `python -m mharo` interactive chat via env keys + .env. pyproject editable install OK.
- Tools: real registry (calculator safe-ast, echo). Memory: JSON KV with TTL+search.
- Next: `skills/plugins/tui/dashboard` skeleton, live chat test, tool-use wiring in Engine.
## 2026-09-22 (late) — Termux window for everyone
- Engine tool-use wiring: Engine.tools registry, call_tool() (result -> tool msg in history), _tool_schemas() (OpenAI function schema). +5 tests.
- LocalProvider (zero-key demo) keyless mode: openssl se saaf, bina key ke bhi python -m mharo chalta hai (Router contract REAL). +tests, CI green.
- Baaki: plugins/tui/dashboard/appmode/security empty packages, memory wiring chat loop me, skills ka real example file, live chat smoke w/ real key, version bump+tag.
