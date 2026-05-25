from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> None:
    """Compatibility entry point; the canonical implementation is ATLAS."""
    from pipelines.atlas_sweep import main as run_atlas

    output_subdir = (
        os.getenv("ATLAS_OUTPUT_SUBDIR")
        or os.getenv("ASTEP_OUTPUT_SUBDIR")
        or "atlas_magnitude_step_fixed_policy"
    )
    policy = os.getenv("ATLAS_POLICY") or os.getenv("ASTEP_TUNING_POLICY") or "fixed_policy"
    argv = [
        "--sweeps",
        "magnitude_step",
        "--policy",
        policy,
        "--output-subdir",
        output_subdir,
    ]
    if os.getenv("ASTEP_SWEEP_N_MC_RUNS"):
        argv.extend(["--n-runs", os.environ["ASTEP_SWEEP_N_MC_RUNS"]])
    if os.getenv("ASTEP_SWEEP_N_COST_REPS"):
        argv.extend(["--n-cost-reps", os.environ["ASTEP_SWEEP_N_COST_REPS"]])
    if os.getenv("ASTEP_SWEEP_TUNE_TRIALS"):
        argv.extend(["--tune-trials", os.environ["ASTEP_SWEEP_TUNE_TRIALS"]])
    if os.getenv("ASTEP_SWEEP_TUNE_EVAL_RUNS"):
        argv.extend(["--tune-eval-runs", os.environ["ASTEP_SWEEP_TUNE_EVAL_RUNS"]])
    if os.getenv("ASTEP_SWEEP_RESUME"):
        os.environ.setdefault("ATLAS_RESUME", os.environ["ASTEP_SWEEP_RESUME"])
    if os.getenv("ASTEP_SWEEP_INCLUDE_ESTIMATORS"):
        os.environ.setdefault("ATLAS_INCLUDE_ESTIMATORS", os.environ["ASTEP_SWEEP_INCLUDE_ESTIMATORS"])
    if os.getenv("ASTEP_SWEEP_INCLUDE_STEPS"):
        os.environ.setdefault("ATLAS_MAG_LEVELS_PCT", os.environ["ASTEP_SWEEP_INCLUDE_STEPS"])
    run_atlas(argv)


if __name__ == "__main__":
    main()
