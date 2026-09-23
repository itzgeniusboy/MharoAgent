"""Render real app frames to SVG (and PNG, if ImageMagick is installed).

    python scripts/preview.py            # writes preview-*.svg next to the README
    python scripts/preview.py --png      # also rasterises with `magick`

No display, no PTY and no API key needed: Textual's headless driver produces
the SVG, so this runs in CI as a smoke test of the whole render path.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mharo_tui.agent import AgentConfig  # noqa: E402
from mharo_tui.agent.session import ToolCall  # noqa: E402
from mharo_tui.agent.tools import run_tool  # noqa: E402
from mharo_tui.app import MharoApp  # noqa: E402
from mharo_tui.widgets import Sidebar, ToolView  # noqa: E402

WIDTH, HEIGHT = 118, 40


def sample_project(tmp: Path) -> None:
    (tmp / "src").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "router.py").write_text(
        "from dataclasses import dataclass\n\n\n@dataclass\nclass Route:\n"
        "    path: str\n"
        "    handler: str\n\n\nROUTES: list[Route] = []\n\n\n"
        "def mount(path: str, handler: str) -> None:\n"
        "    ROUTES.append(Route(path, handler))\n",
        encoding="utf-8",
    )
    (tmp / "src" / "server.py").write_text(
        "from .router import ROUTES\n\n\ndef serve(port: int = 8080) -> None:\n"
        "    print(f'serving {len(ROUTES)} routes on {port}')\n",
        encoding="utf-8",
    )
    (tmp / "tests").mkdir(exist_ok=True)
    (tmp / "tests" / "test_router.py").write_text(
        "from src.router import mount, ROUTES\n\n\ndef test_mount_adds_a_route() -> None:\n"
        "    mount('/health', 'health')\n    assert ROUTES[-1].path == '/health'\n",
        encoding="utf-8",
    )
    (tmp / "README.md").write_text("# mharo-sample\n\nTiny demo project for the TUI preview.\n", encoding="utf-8")
    (tmp / "pyproject.toml").write_text("[project]\nname = 'mharo-sample'\n", encoding="utf-8")


async def wait_turn(app: MharoApp, pilot, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if not app._busy and app._worker is None and list(app.query(ToolView)):
            await pilot.pause(0.25)
            return
        if not app._busy and app._worker is None and time.monotonic() > deadline - timeout + 2.5:
            await pilot.pause(0.25)
            return


async def frame_welcome(out: Path) -> None:
    tmp = Path("/tmp/mharo-preview-welcome")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    app = MharoApp(AgentConfig(provider="demo", model="demo", cwd=tmp, max_turns=2))
    async with app.run_test(size=(WIDTH, HEIGHT)) as pilot:
        await pilot.pause(0.5)
        side = app.query_one(Sidebar)
        side.shown = True
        side.display = True
        await pilot.pause(0.4)
        out.write_text(app.export_screenshot(), encoding="utf-8")


async def frame_working(out: Path) -> None:
    tmp = Path("/tmp/mharo-preview-work")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    sample_project(tmp)
    app = MharoApp(AgentConfig(provider="demo", model="demo", cwd=tmp, auto_approve=True, max_turns=4))
    async with app.run_test(size=(WIDTH, HEIGHT)) as pilot:
        await pilot.pause(0.4)
        await app.submit_text("list the files in this repo")
        await wait_turn(app, pilot)
        await app.run_shell("python -m pytest -q tests")
        # a real edit through the tool layer, shown with its diff
        call = ToolCall(tool="edit_file", args={"path": "src/router.py", "old_text": "ROUTES: list[Route] = []", "new_text": "ROUTES: list[Route] = []\nALIASES: dict[str, str] = {}"})
        result = run_tool("edit_file", call.args, app.agent.ctx)
        call.result, call.ok, call.duration_ms = result.output, result.ok, result.duration_ms
        app.handle_event({"type": "tool_start", "call": call})
        await pilot.pause(0.2)
        app.handle_event({"type": "tool_end", "call": call})
        app.agent.session.files_touched = dict(app.agent.ctx.files_touched)
        await pilot.pause(0.3)
        side = app.query_one(Sidebar)
        side.shown = True
        side.display = True
        app.action_fold_all()
        app.query_one("#prompt-input").load_text_safe("now add a test for the alias map")
        await pilot.pause(0.6)
        out.write_text(app.export_screenshot(), encoding="utf-8")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="also rasterise with ImageMagick")
    args = parser.parse_args()
    welcome = ROOT / "preview-welcome.svg"
    working = ROOT / "preview.svg"
    await frame_welcome(welcome)
    await frame_working(working)
    print(f"wrote {welcome.name} ({welcome.stat().st_size // 1024} KB)")
    print(f"wrote {working.name} ({working.stat().st_size // 1024} KB)")
    if args.png and shutil.which("magick"):
        for svg in (welcome, working):
            subprocess.run(["magick", "-density", "150", str(svg), str(svg.with_suffix(".png"))], check=True)
            print("rasterised", svg.with_suffix(".png").name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
