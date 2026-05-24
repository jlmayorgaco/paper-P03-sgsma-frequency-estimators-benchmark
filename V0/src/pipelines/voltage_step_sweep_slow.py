from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) is None:
        os.environ[name] = value


def configure_voltage_mag_step_slow_profile() -> None:
    """Dedicated slow-method profile for the voltage/magnitude-step atlas."""
    os.environ["BENCHMARK_INCLUDE_EXPERIMENTAL"] = "1"
    _set_default("ASTEP_OUTPUT_SUBDIR", "voltage_mag_step_v13_clean_slow")
    _set_default("ASTEP_SWEEP_INCLUDE_ESTIMATORS", "Prony,ESPRIT,MUSIC,Koopman,PI-GRU")
    _set_default("ASTEP_SWEEP_RESUME", "1")

    _set_default("ASTEP_SLOW_N_MC_RUNS", "8")
    _set_default("ASTEP_SLOW_N_COST_REPS", "2")
    _set_default("ASTEP_SLOW_TUNE_EVAL_RUNS", "3")
    _set_default("ASTEP_SLOW_TUNE_TRIALS", "8")
    _set_default("ASTEP_SLOW_PRONY_TUNE_TRIALS", "10")
    _set_default("ASTEP_SLOW_ESPRIT_TUNE_TRIALS", "3")
    _set_default("ASTEP_SLOW_MUSIC_TUNE_TRIALS", "10")
    _set_default("ASTEP_SLOW_KOOPMAN__RK_DPMU_TUNE_TRIALS", "8")
    _set_default("ASTEP_SLOW_PI_GRU_TUNE_TRIALS", "0")
    _set_default("ASTEP_SLOW_VECTORIZED_ENGINE", "1")

    _set_default("ASTEP_MC_STRATIFIED_COVARIATES", "1")
    _set_default("ASTEP_HYPOTHESIS_MAX_STEP_PCT", "1000")


if __name__ == "__main__":
    configure_voltage_mag_step_slow_profile()
    from pipelines.amplitude_step_sweep import main

    main()
