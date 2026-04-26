from __future__ import annotations

import json
from pathlib import Path

from pipelines.stats_hypotheses import run_hypotheses


def run_stats(args: object) -> int:
    hypotheses_path = Path(getattr(args, "hypotheses"))
    input_json_path = Path(getattr(args, "input_json"))
    output_dir = Path(getattr(args, "output_dir"))
    result = run_hypotheses(
        hypotheses_path=hypotheses_path,
        input_json_path=input_json_path,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0
