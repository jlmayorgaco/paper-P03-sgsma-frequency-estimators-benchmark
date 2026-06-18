from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.monte_carlo_engine import MonteCarloEngine
from openfreqbench.registry import CANONICAL_METRIC_IDS, scenario_registry
from openfreqbench.config import BenchmarkRunConfig, EstimatorSelection
from openfreqbench.runner import _aggregate, dry_run_manifest
from pipelines import full_mc_benchmark as full_mc


def test_severity_variants_pin_named_stress_in_monte_carlo_space() -> None:
    scenarios = scenario_registry()

    mag = scenarios["IEEE_Mag_Step_50pct"]
    mag_defaults = mag.get_default_params()
    mag_space = mag.get_monte_carlo_space()
    assert mag_defaults["amp_post_pu"] == 1.50
    assert mag_space["amp_post_pu"] == {"kind": "fixed", "value": 1.50}

    ramp = scenarios["IEEE_Freq_Ramp_20Hzs"]
    ramp_defaults = ramp.get_default_params()
    ramp_space = ramp.get_monte_carlo_space()
    assert ramp_defaults["rocof_hz_s"] == 20.0
    assert ramp_space["rocof_hz_s"] == {"kind": "fixed", "value": 20.0}


def test_aggregate_filters_nonfinite_values_and_reports_valid_count() -> None:
    raw = pd.DataFrame(
        [
            {
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "family": "Loop-based",
                "m1_rmse_hz": 0.1,
            },
            {
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "family": "Loop-based",
                "m1_rmse_hz": float("inf"),
            },
            {
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "family": "Loop-based",
                "m1_rmse_hz": float("nan"),
            },
        ]
    )

    agg = _aggregate(raw)
    row = agg.iloc[0]
    assert np.isclose(row["m1_rmse_hz_mean"], 0.1)
    assert row["m1_rmse_hz_n_valid"] == 1
    assert row["n_runs_total"] == 3


def test_canonical_metric_profile_includes_tail_and_post_startup_metrics() -> None:
    assert "m34_p95_error_hz" in CANONICAL_METRIC_IDS
    assert "m35_p99_error_hz" in CANONICAL_METRIC_IDS
    assert "m36_post_startup_invalid_rate" in CANONICAL_METRIC_IDS


def test_vectorized_preference_bypasses_standardized_step_loop() -> None:
    class VectorizedEstimator:
        PREFER_VECTORIZED_ENGINE = True

        def step(self, *_args, **_kwargs):
            raise AssertionError("step() should not be called when vectorized mode is requested")

        def step_vectorized(self, v):
            return np.full(len(v), 60.0)

    engine = MonteCarloEngine(scenario_cls=object, estimator_cls=VectorizedEstimator)
    out = engine._run_estimator(np.ones(12), t=np.arange(12) * 1e-4)

    assert out["engine_mode"] == "vectorized"
    assert np.allclose(out["f_hat"], 60.0)


def test_full_mc_tuning_helper_uses_standardized_step_by_default() -> None:
    class WrappedEstimator:
        def __init__(self) -> None:
            self.step_calls = 0

        def reset(self) -> None:
            self.step_calls = 0

        def step(self, z, t_s=None, memory=None):
            self.step_calls += 1
            assert t_s is not None
            assert memory is not None
            return 60.0 + float(z)

        def step_vectorized(self, _v):
            raise AssertionError("step_vectorized should not be called without explicit preference")

    est = WrappedEstimator()
    out = full_mc._run_estimator(est, np.array([0.1, 0.2]), t=np.array([0.0, 1e-4]))

    assert est.step_calls == 2
    assert np.allclose(out, [60.1, 60.2])


def test_full_mc_tuning_noise_uses_scenario_defaults_by_default() -> None:
    assert full_mc.TUNING_NOISE_LEVEL is None
    assert full_mc._tuning_noise_kwargs(full_mc.ringdown_variants[-1]) == {}


def test_full_mc_summary_lookup_prefers_exact_pair_file() -> None:
    root = Path("output") / "test_artifacts" / "summary_lookup_guard"
    est_dir = root / "IEEE_Single_SinWave" / "EKF"
    try:
        shutil.rmtree(root, ignore_errors=True)
        est_dir.mkdir(parents=True)
        stale = est_dir / "old_summary.csv"
        exact = est_dir / "IEEE_Single_SinWave__EKF_summary.csv"
        stale.write_text("run_idx,m1_rmse_hz\n0,999\n", encoding="utf-8")
        exact.write_text("run_idx,m1_rmse_hz\n0,0.1\n", encoding="utf-8")

        assert full_mc._find_summary_csv(est_dir, "IEEE_Single_SinWave", "EKF") == exact
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dry_run_manifest_reports_cost_repetition_env(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_N_COST_REPS", "3")
    cfg = BenchmarkRunConfig(
        run_id="guard",
        mode="matrix",
        output_dir=Path("artifacts/pre_run_guard"),
        scenarios=["IEEE_Single_SinWave"],
        estimators=[EstimatorSelection("ZCD")],
        custom_estimators=[],
        metric_profile="canonical-single-phase-v1",
        metric_include=[],
        n_runs=1,
        base_seed=12345,
        capture_signals=False,
        max_workers=1,
        parameter_policy="default",
        tuned_artifacts_dir=None,
    )

    manifest = dry_run_manifest(cfg)

    assert manifest["n_cost_reps"] == 3
