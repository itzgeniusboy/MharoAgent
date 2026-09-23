"""Entry point for `python -m mharo` / `ma`."""

from __future__ import annotations

import sys

from mharo.cli import main

if __name__ == "__main__":
    sys.exit(main())
