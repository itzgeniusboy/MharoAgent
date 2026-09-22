# MharoAgent

Multi-provider async AI agent framework for Python 3.10+ — real SSE streaming,
Router with key-rotation + fallback, Engine session loop, persistent Memory,
Tool registry. Runs on Termux/Android.

## Status

![CI](https://github.com/itzgeniusboy/MharoAgent/actions/workflows/build.yml/badge.svg)

CI runs: compile all modules -> import contract -> runtime Router/Engine
contract -> pytest (33 tests).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install httpx pytest pytest-asyncio
pip install -e .
```

## CLI chat

```bash
export OPENAI_API_KEY=sk-...            # ya DEEPSEEK_API_KEY
python -m mharo                          # interactive chat
python -m mharo -v                       # provider/stats per turn
```

`.env` file is also supported: `OPENAI_API_KEY=sk-...`.

**No key? No problem.** `python -m mharo` runs in local demo mode with a
built-in `LocalProvider` — everyone can try the CLI instantly.

## Use as a library

```python
from mharo.providers.openai_compat import OpenAICompatible
from mharo.core.router import Router
from mharo.core.engine import Engine

engine = Engine(
    Router(
        [
            OpenAICompatible("openai", "gpt-4o-mini", api_key),
            OpenAICompatible("deepseek", "deepseek-chat", api_key2,
                             base_url="https://api.deepseek.com/v1"),
        ],
        strategy="cost",  # key rotation + fallback engine
    )
)
reply = await engine.respond("hello")
```

## Modules

| Path | Real code |
| --- | --- |
| `mharo/providers/` | errors, types, protocol (Provider ABC + multi-key), SSE stream parser, `OpenAICompatible` httpx client |
| `mharo/core/router.py` | key-rotation (429/401 -> next key), provider fallback, stats |
| `mharo/core/engine.py` | session history + window, Router call, tool-use loop, per-session stats |
| `mharo/providers/local.py` | zero-key `LocalProvider` demo mode |
| `mharo/tools/` | Tool registry + safe calculator + echo |
| `mharo/memory/` | persistent JSON KV (TTL, search) |
| `mharo/skills/` | dynamic SkillRunner + repo `skills/greet.py` example |
| `mharo/plugins/` | PluginManager (runtime load/enable/invoke) |
| `mharo/security/` | Redactor (secret + key-like masking) |
| `mharo/tui/` | Termux-safe banner/status/box (no ANSI unless TTY) |
| `mharo/dashboard/` | text panel from Engine+Router state |
| `mharo/appmode/` | single-shot headless runner |
| `mharo/__main__.py` | `python -m mharo` interactive CLI |

## Design notes

- Providers raise `ProviderError` subclasses (`RateLimitError`, `TimeoutError2`, ...)
  so Router/Engine react without leaking HTTP details.
- Router marks `provider.alive = False` on repeated failure and skips it.
- No stubs — every module has runnable logic covered by CI.

## Tests

```bash
PYTHONPATH=src pytest -q tests
```