from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.paths import (
    BENCHMARK_OUTPUT_DIR,
    FIGURE1_BASENAME,
    FIGURE2_BASENAME,
    PAPER_FIGURES_DIR,
)


def sync_dashboard_artifacts(
    source_dir: Path | None = None,
    target_dir: Path | None = None,
) -> list[Path]:
    src = Path(source_dir) if source_dir is not None else BENCHMARK_OUTPUT_DIR
    dst = Path(target_dir) if target_dir is not None else PAPER_FIGURES_DIR
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for basename in (FIGURE1_BASENAME, FIGURE2_BASENAME):
        for suffix in (".png", ".pdf", ".svg", ".eps"):
            source = src / f"{basename}{suffix}"
            if not source.exists():
                continue
            target = dst / source.name
            shutil.copy2(source, target)
            copied.append(target)

    return copied


def main() -> None:
    copied = sync_dashboard_artifacts()
    if not copied:
        raise FileNotFoundError(
            f"No canonical dashboard artifacts found in {BENCHMARK_OUTPUT_DIR}"
        )
    for path in copied:
        print(f"[SYNC] {path}")


if __name__ == "__main__":
    main()
