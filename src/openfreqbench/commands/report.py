from __future__ import annotations

import json
from pathlib import Path

from pipelines.report_builder import build_benchmark_report


def build_report(args: object) -> int:
    input_json = Path(getattr(args, "input_json"))
    stats_json = Path(getattr(args, "stats_json"))
    output_dir = Path(getattr(args, "output_dir"))
    result = build_benchmark_report(
        input_json=input_json,
        stats_json=stats_json if stats_json.exists() else None,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0
