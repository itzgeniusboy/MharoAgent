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


def _build_engine(allow_local: bool = True) -> "Engine":
    from mharo.core.engine import Engine
    from mharo.core.router import Router
    from mharo.providers import catalog
    from mharo.providers.local import LocalProvider
    from mharo.providers.openai_compat import OpenAICompatible

    providers = []
    env = catalog.env_with_opencode_auth()

    for cfg, key in catalog.configured(env=env):
        providers.append(
            OpenAICompatible(
                cfg.name, cfg.model, key,
                base_url=cfg.resolved_base_url(env),
            )
        )

    if not providers:
        if not allow_local:
            known = ", ".join(c.env[0] for c in catalog.CATALOG)
            raise RuntimeError(
                "no API key found — set any of: " + known
                + " (ya opencode se provider login karein; .env bhi chalega)"
            )
        providers.append(LocalProvider())

    return Engine(Router(providers, strategy=os.environ.get("MHARO_STRATEGY", "cost")))


def memory_context(memory, prefix: str = "fact") -> str:
    """Memory se model context banata hai (Remembered facts block)."""
    facts = memory.search(prefix)
    lines = [f"{k}: {v}" for k, v in sorted(facts.items())]
    if not lines:
        return ""
    return "Remembered facts:\n" + "\n".join(f"- {line}" for line in lines)


def handle_special(arg: str, memory) -> str | None:
    """Slash commands -> action message ya None (continue normal loop)."""
    name, _, rest = arg.partition(" ")
    if name == "/remember":
        key, sep, value = rest.partition("=")
        if not sep or not key.strip() or not value.strip():
            return "usage: /remember <key>=<value>"
        memory.set(key.strip(), value.strip())
        return f"remembered {key.strip()}"
    if name == "/forget":
        if not rest.strip():
            return "usage: /forget <key>"
        return "forgot " + rest.strip() if memory.delete(rest.strip()) else f"no key {rest.strip()!r}"
    if name in {"/recall", "/memory"}:
        items = memory.search(rest.strip())
        return "\n".join(f"{k}: {v}" for k, v in sorted(items.items())) or "(memory empty)"
    if name == "/clear" and rest.strip():
        memory.clear()
        return "(memory cleared)"
    return None


async def _run_interactive(engine, verbose: bool, memory=None) -> int:
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
        if user.startswith("/") and memory is not None:
            reply = handle_special(user, memory)
            if reply is not None:
                print(reply)
                continue
        if user.lower() == "/stats":
            st = engine.stats
            print(
                f"turns={st.turns} in={st.tokens_in} out={st.tokens_out} "
                f"latency={st.latency_ms:.0f}ms fallbacks={st.fallbacks}"
            )
            continue
        extra = memory_context(memory) if memory is not None else ""
        print("ai   > ", end="", flush=True)
        try:
            text = await engine.respond(user, extra_system=extra)
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
    parser.add_argument("--memory", default="mharo_memory.json", help="memory json path")
    args = parser.parse_args(argv)

    _load_dotenv(args.env)
    engine = _build_engine(allow_local=True)
    print("(remember/recall works — memory:", args.memory, ")")
    if engine.router.providers[0].name == "local":
        print("(local mode — no API key. Set OPENAI_API_KEY for real AI.)")
    from mharo.memory import Memory

    memory = Memory(args.memory)
    return asyncio.run(_run_interactive(engine, args.verbose, memory))


if __name__ == "__main__":
    raise SystemExit(main())