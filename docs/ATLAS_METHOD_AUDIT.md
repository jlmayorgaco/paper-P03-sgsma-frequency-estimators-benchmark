# ATLAS method audit

Status: active for MVP 2.0.0.

This audit records methodological issues found before running paper-grade ATLAS
results. The goal is to prevent software artifacts from becoming paper claims.

## Corrections applied

1. Tuning now uses the same estimator execution contract as Monte Carlo.
   Earlier ATLAS tuning called a legacy vectorized helper, while the benchmark
   evaluation uses the standardized scalar `step(...)` path for most
   estimators. That could select parameters for a different execution mode.

2. Distortion sweeps are causal by default.
   `harmonics` and `interharmonics` now default to zero added white noise.
   Noise belongs in `noise_snr` unless explicitly requested through
   `ATLAS_DISTORTION_NOISE_SIGMA`.

3. Harmonic scenarios support fundamental phase variation.
   The IBR harmonic scenario classes now accept `phase_rad`, allowing Monte
   Carlo phase stratification instead of evaluating every distortion case at
   one fixed fundamental phase.

4. Level-only P0 sweeps disable event metrics.
   Harmonics, interharmonics, noise/SNR, AM modulation, and FM modulation are
   level-only or continuous stress tests. They should not report post-event
   RMSE or event settling time because there is no physical event time in those
   isolated sweeps. Phase-jump sweeps keep event metrics enabled.

5. Cache invalidation was forced.
   `METHOD_VERSION` was bumped so older ATLAS cached summaries cannot be reused
   silently after these methodological changes.

6. RNAF is no longer reported as zero when input noise is zero.
   The metric is undefined without nonzero input-noise variance, so it is now
   left missing instead of creating a false best-case score.

7. Scenario defaults and Monte Carlo spaces are exported with aggregate rows.
   `global_metrics_report.csv` now records scenario default parameters, Monte
   Carlo spaces, and whether event metrics were enabled. This makes it possible
   to audit nuisance variables directly from the CSV.

## Interpretation rules

- `harmonics` isolates integer THD. It is not a complete IBR event model.
- `interharmonics` isolates one off-bin 75 Hz component.
- `noise_snr` isolates additive white noise on a fixed 60 Hz sine wave.
- `phase_jump_sweep` isolates phase discontinuity; true frequency stays
  nominal.
- `modulation_am_sweep` isolates AM-to-FM cross-coupling at fixed AM depth.
- `modulation_fm_sweep` isolates tracking bandwidth at fixed peak frequency
  deviation.
- `oracle` is a lower-bound diagnostic, not a deployable tuning policy.
- Runs with `n_runs < 30` are software diagnostics only.
- Journal claims should use `n_runs >= 100`, archived artifacts, hashes, and
  the command stored in `benchmark_report.json`.

## Residual risks

- Scenario synthesis is still single-phase and synthetic. It supports controlled
  causal inference, not external validity by itself.
- CPU timing depends on local hardware, Python version, and dependency builds.
- White-noise sampling represents digital sampled noise, not an analog
  anti-alias front end.
- Fixed-policy tuning and oracle tuning can still overfit the chosen scenario
  grid. The paper must report both.
- Compliance-style metrics are guide metrics unless tied to a preregistered
  PMU/relay test.

## Required before journal claims

Run and archive:

```bash
python -m pipelines.atlas_sweep --sweeps all --policy fixed_policy --n-runs 100 --tune-trials 80 --output-subdir atlas-paper-fixed-v2
python -m pipelines.atlas_sweep --sweeps all --policy oracle --n-runs 100 --tune-trials 80 --output-subdir atlas-paper-oracle-v2
```

Then compare rankings, confidence intervals, failure rates, bound-hit rates,
latency, and CPU before moving any number into the manuscript.
