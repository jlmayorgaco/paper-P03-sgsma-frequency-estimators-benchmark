from __future__ import annotations

import json
from pathlib import Path

from pipelines.andes_ieee39 import run_andes_ieee39


def run_ieee39(args: object) -> int:
    seed = int(getattr(args, "seed", 12345))
    output = getattr(args, "output", None)
    out_dir = Path(output) if output else None
    result = run_andes_ieee39(seed=seed, output_dir=out_dir)
    print(json.dumps(result["manifest"], indent=2))
    return 0
