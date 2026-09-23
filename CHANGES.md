# What changed in this source tree

Everything below is in this zip and covered by the offline suite
(`python3 -m pytest -q` → **101 passed**, `ruff` clean, `ma doctor` →
40 pass · 6 warn · 0 fail, `ma bench` → 10/10 in ~2.4 s).

## 1. The TUI — replaced with a real agent UI

`mharo_tui/` is an opencode/Claude-Code-class terminal agent, drop-in over the
same engine:

| piece | file |
| --- | --- |
| app composition, keymap, event routing (~870 lines) | `mharo_tui/app.py` |
| TopBar · User/Assistant/Error/Notice · foldable `ToolView` · `ApprovalPrompt` · `Palette` · `Sidebar` · `StatusLine` | `mharo_tui/widgets.py` |
| layout + skin (`night`/`ink`/`slate`/`day`) | `mharo_tui/app.tcss`, `mharo_tui/themes.py` |
| 21 slash commands | `mharo_tui/commands.py` |
| headless / no-TTY fallback that shares the same Agent | `mharo_tui/plain.py` |
| turn loop, approvals, cancellation, `/compact` | `mharo_tui/agent/__init__.py` |
| providers (demo · OpenAI-compatible · Anthropic, SSE) | `mharo_tui/agent/providers.py` |
| tool registry, safety guards, diffing, git status | `mharo_tui/agent/tools.py` |
| message model, token + cost accounting, JSONL sessions | `mharo_tui/agent/session.py` |

Streamed markdown, foldable per-tool blocks with durations, an approval gate
that actually blocks the call, context/cost bar, keybindings (`esc` interrupt,
`ctrl+p` palette, folds, themes), `@file` inlining, session resume + `/undo`.

## 2. The engine — plan → act → verify, no stubs

`mharo/` (20 modules): `cli.py` (`ma`), `engine.py` (turn loop + `PLAN_PROMPT`),
`router.py` (tiers), `providers.py` (`ProviderHub`, key pools, rescue),
`ladder.py` (`FreePool`), `keys.py`, `sessiondb.py` (SQLite + undo), `memory.py`
(recall/compaction), `verify.py` (claims need a passing check), `permissions.py`,
`vault.py` (secrets injected at the tool boundary, never into the transcript),
`subagents.py`, `skills.py`, `doctor.py`, `bench.py`, `cost.py`, `config.py`.

`ma doctor` runs every check for real; `ma bench` runs 10 end-to-end scenarios
(offline, replay provider) and fails if a loop is faked.

## 3. Free-first provider ladder (the "no key" path)

Keys lead when configured; when they're missing or spent, the ladder answers —
each rung probed, each with its own rate gate:

```
local      ollama :11434 · lmstudio :1234 · llamacpp :8080 · vllm :8000     no cost, no limit
free·anon  ovh (~2 req/min per model per IP, 31 s gate, rotates 7 models)
           pollinations (2 s gate, no tool calling → sits last)
free·key   openrouter (`:free`) · llm7 · gemini · groq          → `ma keys add <svc> <key>`
paid       your openai / anthropic keys (also rescue a throttled free run)
```

CLI: `ma free` (ladder + state), `ma free --probe [--completions]`,
`ma free --test "prompt"`, `ma free --pick ovh`, `ma keys [add|rm]`.
Config: `free.enabled|prefer|disable|rungs|extra_rungs|min_interval_s|patience_s|allow_paid|fallback_to_demo|rescue_on_exhaustion|max_rungs`.

Rate-limit behaviour, because free tiers are the hard part:

- **Per-pairing gates** — `ovh:gpt-oss-120b` and `ovh:gpt-oss-20b` keep separate
  clocks, so a 2 req/min host still carries an agent run.
- **Wait, don't hammer** — if every pairing is gated and the soonest frees within
  `free.patience_s`, we sleep that gap instead of spending a 429.
- **Storm detection** — three 429s on one host inside a minute retires the host.
- **Quota ≠ throttle** — `402` / `insufficient credits` → 15 min backoff +
  `ma keys add …`; a bare "rate limit exceeded" costs one gate window (it proves
  the endpoint is alive). A 429 never marks a host `unprobed`.
- **Dead ports are parked, not dialled** — a local rung that isn't listening is
  skipped for the session (`LOCAL_PARK_S = 20`) instead of eating a connect
  timeout every turn.
- **The gate wait uses the soonest *timed* pairing** — a parked, timer-less rung
  can no longer cancel a legitimate wait (that bug used to drop turn 2+ onto demo).

## 4. Payload-shape learning (added this session)

A host that rejects our *body* is re-bodied, not benched:

- `classify()` → `shape` for 400/415/422 on a request we can plausibly re-shape.
- `Ladder.downgrade_profile()` steps `full → no_stream_options → no_tools → minimal`
  and retries immediately (0 s backoff — the rung did nothing wrong). The learned
  profile is cached per host in `~/.mharo/ladder.json`; `ma free` shows it in the
  `state` column (`no_tools · 1 downgraded`).
- `OpenAICompatProvider`: `Accept: application/json` on non-streaming, and an
  empty-stream guard (`produced`) so a 200-with-zero-deltas is an error, not a
  silent demo fallback.
- Free answers report `provider="free:<rung>"` + real usage, so `ma cost` shows
  actual tokens at **$0.00**.

**The bug this uncovered:** `_payload` was sending tools flat
(`{"type":"function","name":…}`) instead of nested under `"function"`. Strict
OpenAI-compatible servers 422 that, which we had been *misreading* as "this free
tier has no tool calling" and downgrading to `no_tools` to compensate. Fixed by
`OpenAICompatProvider._tool_spec()`; verified live — a keyless run now does real
multi-turn tool calls (4 tool calls, 6 turns, streaming, `cost: $0.0000`).
Test: `test_tool_schemas_use_the_openai_envelope`.

## 5. Tests

```
tests/test_ma_core.py    20   config/keys/vault/sessions/permissions/cost/CLI surface
tests/test_ma_engine.py  18   plan→act→verify, retry, verify-gated claims, undo, subagents
tests/test_ma_ladder.py  33   gates, rotation, storms, quota vs throttle, shape learning,
                               payload envelope, parked locals, keyless one-shot + stream
tests/test_tui.py        30   boots the real app under Textual's Pilot: streaming, folds,
                               approval blocking, esc, bindings, @file, session round-trip
```

## 6. Findings in `itzgeniusboy/MharoAgent@master` (your repo — not patched here)

Fetched and read this session. Four things block the same "free-first agent"
behaviour there; each is a small, targeted fix against that layout:

1. `src/mharo/providers/openai_compat.py::complete()` returns
   `tool_calls=[]` **hardcoded** — it fills a `ToolBuffer` from the SSE deltas
   and then throws it away, so no provider can ever call a tool. Also
   `ToolBuffer` holds one call and never finalises (`tool_end` is never emitted),
   so parallel calls would be lost even if it were read.
2. `stream: True` is unconditional and `stream_options: {"include_usage": true}`
   is never sent → `usage` is almost always `None`, so `SessionStats.tokens_*`
   stay 0 and the cost/dashboard numbers are decorative.
3. `src/mharo/core/engine.py::respond()` never passes `tools` to
   `router.complete()` (its own `_tool_schemas()` is dead code) and there is no
   act loop — one shot in, one string out. That's why the plan→act→verify loop
   and tool execution had to be added rather than enabled.
4. "No key? No problem" currently means `LocalProvider` (an echo). With the
   ladder above (`src/mharo/providers/free.py`-style rungs: OVH anonymous +
   pollinations, per-model gates, keyless `OpenAICompatible`) the same CLI gets a
   real model answer with no key, and demo becomes the last resort + notice.

Also minor: `Router.pick()` ignores `strategy="cost"` (plain round-robin), and
`ValidationError2`/`BaseUrlError` are imported but never raised.
