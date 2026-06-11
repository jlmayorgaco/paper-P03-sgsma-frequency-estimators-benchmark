# OpenFreqBench Pipelines

OpenFreqBench keeps two publication-facing pipelines. They answer different
questions and should not be mixed in the same claim.

## Pipeline 1: Full MC Tuning Matrix

Purpose: tune estimator parameters per scenario, estimator, Monte Carlo version,
and objective metric. Use this when the question is:

> What parameter set is best for this method on this scenario under RMSE,
> trip-risk, peak-error, CPU time, or another canonical metric?

Command:

```bash
openfreqbench benchmark tune-matrix \
  --run-id full-mc-objective-matrix-v1 \
  --mc-version mc-v1 \
  --scenario IEEE_Single_SinWave \
  --estimator ZCD \
  --objective m1_rmse_hz \
  --objective m5_trip_risk_s \
  --n-trials 20 \
  --tune-runs 2 \
  --eval-runs 3
```

Full run shape:

```bash
openfreqbench benchmark tune-matrix \
  --run-id full-mc-objective-matrix-v1 \
  --mc-version mc-v1 \
  --all-scenarios \
  --all-estimators \
  --objective m1_rmse_hz \
  --objective m3_max_peak_hz \
  --objective m5_trip_risk_s \
  --objective m13_cpu_time_us \
  --n-trials 80 \
  --tune-runs 5 \
  --eval-runs 30
```

Outputs:

- `benchmark_report.json`: run-level objective tuning report.
- `tuning_matrix.csv`: one row per scenario-estimator-objective result.
- `<scenario>/<estimator>/<objective>/run_spec.json`: objective-specific best
  parameters and tuning metadata.
- `<scenario>/<estimator>/<objective>/tuning_trials.csv`: Optuna trial history.
- `<scenario>/<estimator>/<objective>/eval_summary.csv`: validation Monte Carlo
  metrics under the selected parameters.
- `selected_replay/<objective>/<scenario>/<estimator>/run_spec.json`:
  replay-ready artifact tree for `parameter_policy: artifact_tuned`.

Interpretation rule: objective-specific tuned parameters are not universal.
A parameter set tuned for `m1_rmse_hz` is an RMSE policy. A parameter set tuned
for `m5_trip_risk_s` is a protection-risk policy. Compare them explicitly.

## Pipeline 2: ATLAS

Purpose: map estimator behavior across stress variables such as magnitude step,
RoCoF, phase jump, modulation, harmonics, interharmonics, and noise/SNR. Use
this when the question is:

> How does ranking or failure behavior change as the disturbance variable
> changes?

Command:

```bash
python -m pipelines.atlas_sweep \
  --sweeps all \
  --policy fixed_policy \
  --n-runs 100 \
  --base-seed 12345 \
  --n-cost-reps 3 \
  --tune-trials 80 \
  --tune-eval-runs 5 \
  --output-subdir atlas-paper-fixed-v2 \
  --resume
```

Primary outputs:

- `global_metrics_report.csv`
- `rmse_by_estimator.csv`
- `rmse_by_family.csv`
- `atlas_method_map.pdf`
- `atlas_accuracy_latency_cpu_pareto.pdf`
- `atlas_readiness_report.json`
- `benchmark_report.json`
- `manifest.json`
- `artifact_index.csv`
- `paper_traceability.csv`

Interpretation rule: ATLAS supports stress-response claims. It is not the place
to claim per-scenario oracle optimality unless the policy is explicitly marked
as `oracle` and treated as a diagnostic lower bound.

## Route

1. Run `benchmark tune-matrix` first to build objective-specific tuned
   artifacts.
2. Select one declared tuning policy for a replay or paper table.
3. Run ATLAS with `fixed_policy` when the claim is about stress-variable
   behavior.
4. Archive the resulting run directory before moving numbers into the paper.
