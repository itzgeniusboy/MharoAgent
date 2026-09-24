# Memory — MharoAgent-engine-tui

> Managed by opencode. This is your project's memory — edit it freely.

## Project
- **Goal:** Mharo agent core (`ma`) + Textual TUI (`mharo`) — opencode/Claude-Code style terminal UI, Phase 1 engine (plan → tools → verify → cost). Imported 2026-09-24 from Download/MharoAgent-engine-tui.zip (v0.1.0, by itzgeniusboy).
- **Last worked:** 2026-09-24

## Progress Log
- 2026-09-24: Imported zip → `~/opencode/MharoAgent-engine-tui`, git init + committed. `pip install -e .` done (textual, rich, httpx). Installed pytest + pytest-asyncio.
- 2026-09-24: Fixed `/tmp` hardcodes (Termux mein /tmp exist nahi karta): `doctor.py` replay use $TMPDIR, `bench.py` ROOT use $TMPDIR, `cli.py` help text. Result: `ma doctor` healthy (0 fail), `pytest` 101/101 pass.
- 2026-09-24: Smoke-tested: `python -m mharo_tui --print` works — real tool pipeline runs; free providers rate-limited/cooling, `-p demo` fully offline works.
- Note: purana `~/opencode/MharoAgent` (v0.3.15, different structure) apni jagah untouched hai.

## Decisions
- Naya repo apne local git identity `MharoAgent <mharo@agent.local>` se commit karta hai (repo-local, like old MharoAgent).
- Free tier dhang se kaam nahi kar raha is liye default use = demo/offline; API key ke liye `ma keys add` chahiye.

## Next Steps
- User ko TUI actually chala ke dikhana: `mharo -p demo` (interactive) ya `mharo "prompt"`.
- Ollama setup karna ho to `mharo -p openai --base-url http://localhost:11434/v1 -m qwen2.5-coder:32b` (free, unlimited).

## Open Questions
- Kya free providers (ovh/demo ladder) production use ke liye enough hain, ya API key chahiye.
- Old MharoAgent vs new engine-tui ka kya fate — merge karna hai ya dono alag rahenge.- 2026-09-24: `MharoAgent-engine-tui-fixed.zip` sync — TUI layer fixes: real boxed UI hamesha try hota hai (no TTY/Textual crash pe hi plain fallback), topbar mein `MHARO v0.1.0` + live ready/busy dot (●/○), plain header naya (`MHARO v0.1.0 ● ready` + `➜` prompt). `/tmp`→`$TMPDIR` fixes dobara applied (fixed zip mein bhi /tmp hardcoded tha). Tests 101/101, doctor healthy.
- ~/.bashrc mein mharo function ab new engine-tui binary (`/usr/bin/mharo`) ko call karta hai — purana MharoAgent v0.3.15 `~/opencode/MharoAgent/./mharo` se available.
- 2026-09-24: fixed-2 sync — Welcome banner ab block-letter ASCII `MHARO` + `AGENT` (clear). /tmp fixes intact. 43 pass · 3 warn · 0 fail.
- 2026-09-24: BUGFIX (main glitch) — `~/.bashrc` mharo function mein `exec` tha → mharo exit pe poora Termux session mar jata tha. `exec` hata kar child-process banaya. Proof: exec wale mein SHELL_SURVIVED nahi aata, bina exec mein aata hai. /quit pilot-test mein app exit karta hai (OK). Purana ~/opencode/MharoAgent delete ho chuka (user confirm).
- Audit result: 101 tests pass, bench 10/10, doctor healthy (3 warns = benign), pty boot OK, narrow-width + resize + theme-cycle + palette no crash. Persistent TUI glitch koi aur nai mila.
