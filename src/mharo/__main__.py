"""CLI chat — `python -m mharo` real terminal chat via Router+Engine.

Config: environment variables se API keys (OPENAI_API_KEY, DEEPSEEK_API_KEY,
OPENAI_BASE_URL). Koi key na ho to gentle error. Interactive loop:
'quit'/'exit' -> exit, 'clear' -> history reset, empty line -> skip.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


def _build_engine() -> "Engine":
    from mharo.core.engine import Engine
    from mharo.core.router import Router
    from mharo.providers.openai_compat import OpenAICompatible

    providers = []

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if openai_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        providers.append(OpenAICompatible("openai", model, openai_key, base_url=base_url))

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        providers.append(
            OpenAICompatible(
                "deepseek", model, deepseek_key,
                base_url="https://api.deepseek.com/v1",
            )
        )

    if not providers:
        raise RuntimeError(
            "no API key found — set OPENAI_API_KEY or DEEPSEEK_API_KEY "
            "(or write keys in .env)"
        )

    return Engine(Router(providers, strategy=os.environ.get("MHARO_STRATEGY", "cost")))


async def _run_interactive(engine, verbose: bool) -> int:
    print("MharoAgent — type 'quit' to exit, 'clear' to reset, '/stats' for counters.")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break
        if user.lower() == "clear":
            engine.history.clear()
            print("(history cleared)")
            continue
        if user.lower() == "/stats":
            st = engine.stats
            print(
                f"turns={st.turns} in={st.tokens_in} out={st.tokens_out} "
                f"latency={st.latency_ms:.0f}ms fallbacks={st.fallbacks}"
            )
            continue
        print("ai   > ", end="", flush=True)
        try:
            text = await engine.respond(user)
        except Exception as exc:
            print(f"\n[error] {exc}")
            continue
        print(text)
        if verbose:
            st = engine.stats
            last = st.last_provider
            print(f"       [via {last} | in={st.tokens_in} out={st.tokens_out}]")
    await engine.router.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mharo", description="MharoAgent chat CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="show provider/stats per turn")
    parser.add_argument("--env", default=".env", help="path to .env file")
    args = parser.parse_args(argv)

    _load_dotenv(args.env)
    try:
        engine = _build_engine()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return asyncio.run(_run_interactive(engine, args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())