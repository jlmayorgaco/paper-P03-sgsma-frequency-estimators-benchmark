from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


FAST_ESTIMATORS = (
    "ZCD,IPDFT,TFT,RLS,PLL,SOGI-PLL,SOGI-FLL,Type-3 SOGI-PLL,"
    "LKF,LKF2,EKF,UKF,RA-EKF,TKEO"
)


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) is None:
        os.environ[name] = value


def configure_voltage_mag_step_fixed_policy_profile() -> None:
    """Legacy-named alias for the final fixed-policy magnitude-step profile."""
    _set_default("BENCHMARK_INCLUDE_EXPERIMENTAL", "0")
    _set_default("ASTEP_OUTPUT_SUBDIR", "voltage_mag_step_v15_classic_rls_fixed_policy_fast14")
    _set_default("ASTEP_SWEEP_INCLUDE_ESTIMATORS", FAST_ESTIMATORS)
    _set_default("ASTEP_TUNING_POLICY", "fixed_policy")
    _set_default("ASTEP_SWEEP_RESUME", "0")

    _set_default("ASTEP_SWEEP_N_MC_RUNS", "30")
    _set_default("ASTEP_SWEEP_N_COST_REPS", "3")
    _set_default("ASTEP_SWEEP_TUNE_TRIALS", "300")
    _set_default("ASTEP_SWEEP_TUNE_EVAL_RUNS", "12")
    _set_default("ASTEP_ORACLE_TRIALS", "300")
    _set_default("ASTEP_UKF_ORACLE_TRIALS", "300")
    _set_default("ASTEP_RLS_ORACLE_TRIALS", "420")

    _set_default("ASTEP_FIXED_POLICY_TRAIN_STEPS", "1,5,25,100,300,1000")
    _set_default("ASTEP_FIXED_POLICY_EVAL_RUNS_PER_STEP", "8")
    _set_default("ASTEP_FIXED_POLICY_VALIDATION_RUNS_PER_STEP", "10")

    _set_default("ASTEP_MC_STRATIFIED_COVARIATES", "1")
    _set_default("ASTEP_STABILITY_TOPK", "20")
    _set_default("ASTEP_STABILITY_VALIDATION_RUNS", "12")
    _set_default("ASTEP_TUNE_TOPK_REEVAL", "20")
    _set_default("ASTEP_TUNE_TOPK_VALIDATION_RUNS", "12")
    _set_default("ASTEP_TRACKING_GUARD_RUNS", "10")
    _set_default("ASTEP_TRACKING_GUARD_HARD_FAIL", "0")
    _set_default("ASTEP_BOUND_HIT_WEIGHT", "8")
    _set_default("ASTEP_HYPOTHESIS_MAX_STEP_PCT", "1000")


if __name__ == "__main__":
    configure_voltage_mag_step_fixed_policy_profile()
    from pipelines.amplitude_step_sweep import main

    main()
