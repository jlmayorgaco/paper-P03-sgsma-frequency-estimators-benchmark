from __future__ import annotations

import os


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) is None:
        os.environ[name] = value


def configure_ukf_oracle_profile() -> None:
    _set_default("BENCHMARK_INCLUDE_EXPERIMENTAL", "0")
    _set_default("ASTEP_OUTPUT_SUBDIR", "amplitude_step_ukf_oracle")
    _set_default("ASTEP_SWEEP_INCLUDE_ESTIMATORS", "UKF")
    _set_default("ASTEP_SWEEP_RESUME", "1")

    # High-detail per-scenario tuning intended as a practical UKF lower bound.
    _set_default("ASTEP_ORACLE_TUNING", "1")
    _set_default("ASTEP_ORACLE_ESTIMATORS", "UKF")
    _set_default("ASTEP_TUNE_OBJECTIVE", "rmse_p90_oracle")
    _set_default("ASTEP_UKF_ORACLE_TUNING", "1")
    _set_default("ASTEP_UKF_ORACLE_TRIALS", "600")
    _set_default("ASTEP_SWEEP_N_MC_RUNS", "30")
    _set_default("ASTEP_SWEEP_TUNE_TRIALS", "100")
    _set_default("ASTEP_SWEEP_TUNE_EVAL_RUNS", "30")
    _set_default("ASTEP_SWEEP_N_COST_REPS", "10")


if __name__ == "__main__":
    configure_ukf_oracle_profile()
    from pipelines.amplitude_step_sweep import main
    main()
