from __future__ import annotations

# Compatibility wrapper for older entry points.
# The public CLI lives in `openfreqbench.cli`.
from openfreqbench.cli import build_parser, main  # noqa: F401


if __name__ == "__main__":
    main()
