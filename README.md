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
| `mharo/core/engine.py` | session history + window, Router call, per-session stats |
| `mharo/tools/` | Tool registry + safe calculator + echo |
| `mharo/memory/` | persistent JSON KV (TTL, search) |
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