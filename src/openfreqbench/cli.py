from __future__ import annotations

# Compatibility wrapper.
# Canonical CLI implementation lives in `pipelines.openfreqbench_cli`.
from pipelines.openfreqbench_cli import build_parser, main  # noqa: F401


if __name__ == "__main__":
    main()
