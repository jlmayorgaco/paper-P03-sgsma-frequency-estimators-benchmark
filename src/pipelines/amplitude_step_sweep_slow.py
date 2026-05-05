from __future__ import annotations

import os


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) is None:
        os.environ[name] = value


def configure_slow_profile() -> None:
    # Include MUSIC (experimental registry) but keep a narrow estimator allowlist.
    os.environ["BENCHMARK_INCLUDE_EXPERIMENTAL"] = "1"
    _set_default("ASTEP_SWEEP_INCLUDE_ESTIMATORS", "Prony,ESPRIT,MUSIC,Koopman,PI-GRU")

    # Dedicated output root so slow study is isolated from fast-only artifacts.
    _set_default("ASTEP_OUTPUT_SUBDIR", "amplitude_step_sweep_slow")

    # Practical defaults for 1-2 day runtime envelopes.
    _set_default("ASTEP_SWEEP_N_MC_RUNS", "12")
    _set_default("ASTEP_SWEEP_TUNE_TRIALS", "25")
    _set_default("ASTEP_SWEEP_TUNE_EVAL_RUNS", "5")
    _set_default("ASTEP_SWEEP_N_COST_REPS", "5")
    _set_default("ASTEP_SWEEP_RESUME", "1")

    # Slow-tier method-specific tuning caps.
    _set_default("ASTEP_SLOW_N_MC_RUNS", "12")
    _set_default("ASTEP_SLOW_TUNE_EVAL_RUNS", "5")
    _set_default("ASTEP_SLOW_N_COST_REPS", "5")
    _set_default("ASTEP_SLOW_TUNE_TRIALS", "20")
    _set_default("ASTEP_SLOW_PRONY_TUNE_TRIALS", "25")
    _set_default("ASTEP_SLOW_ESPRIT_TUNE_TRIALS", "5")
    _set_default("ASTEP_SLOW_MUSIC_TUNE_TRIALS", "25")
    _set_default("ASTEP_SLOW_KOOPMAN__RK_DPMU_TUNE_TRIALS", "20")
    _set_default("ASTEP_SLOW_PI_GRU_TUNE_TRIALS", "0")


if __name__ == "__main__":
    configure_slow_profile()
    from pipelines.amplitude_step_sweep import main
    main()
