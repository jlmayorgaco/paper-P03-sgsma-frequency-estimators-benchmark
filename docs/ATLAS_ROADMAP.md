# ATLAS roadmap

This roadmap turns the current ATLAS previews into a reproducible stress atlas
for grid frequency estimators. The goal is to separate estimator behavior by
disturbance mechanism, severity, sign, policy, and replication depth.

## Evidence levels

Use these labels consistently in reports, slides, and papers.

| Level | Minimum requirement | Allowed use |
| --- | --- | --- |
| Diagnostic | Any subset, any policy, usually `n_runs < 30` | Debugging, mechanism discovery, slide caveats |
| Paper-grade | Full canonical estimator set, fixed policy, `n_runs >= 30`, enough sweep levels and signs | Conference and paper claims |
| Journal-grade | Paper-grade plus `n_runs >= 100`, `n_cost_reps >= 3`, archived manifests and hashes | Final journal tables and public release |

Default parameters are useful for preview plots. Fixed policy is the deployment
comparison. Oracle policy is a lower-bound diagnostic, not a deployable result.

## Current high-value evidence

| Artifact | Status | Main value |
| --- | --- | --- |
| `artifacts/atlas-phase-jump-1-180deg-5deg-all18` | Diagnostic | Phase-jump severity map across 18 estimators |
| `artifacts/atlas-harmonics-fast14-preview-v2` | Diagnostic | Integer-harmonic THD sensitivity |
| `artifacts/frequency_step_mvp2_all18_representative` | Representative, `n=1` | Frequency-step severity and sign behavior |
| `artifacts/freq_ramp_rocof_mvp2_all18_representative` | Draft, `n=1` | RoCoF severity and sign-asymmetry behavior |
| `artifacts/atlas-readiness-smoke` | Diagnostic | Readiness gate sanity check |

## Ticket template

Each ticket should finish with:

- `global_metrics_report.csv`
- `hypothesis_results.csv`
- `atlas_readiness_report.json`
- `atlas_method_map.pdf`
- `atlas_sign_asymmetry.pdf`, where applicable
- `atlas_accuracy_latency_cpu_pareto.pdf`
- a short Markdown interpretation with claim status

## ATLAS-00: lock the canonical run profile

Priority: P0

Status: Code complete. `CANONICAL_ESTIMATORS` in `src/pipelines/atlas_sweep.py`
is now derived from `ACTIVE_ESTIMATOR_SPECS` (the authoritative registry in
`src/pipelines/benchmark_definition.py`), eliminating any risk of drift.
Canonical launchers exist in `scripts/run_atlas_paper_grade.ps1` and
`scripts/run_atlas_paper_grade.sh`. The fixed-policy preflight run is archived
at `artifacts/atlas-00-preflight-fixed-policy`. The full journal-grade run
(`atlas-paper-fixed-v2`, n_runs=100) is the remaining acceptance-criteria gate.

Question: What exact estimator set, policy, run count, seed, and metric profile
define the public ATLAS evidence?

Tasks:

- Freeze the canonical estimator set.
- Use `canonical-single-phase-v1`.
- Use fixed policy for deployment comparison.
- Use `n_runs = 100` for final journal-grade results.
- Use `n_cost_reps >= 3` for CPU timing.
- Archive `environment_report.json`, `evidence_manifest.json`, and git hashes.

Command:

```powershell
python -m pipelines.atlas_sweep `
  --sweeps all `
  --policy fixed_policy `
  --n-runs 100 `
  --base-seed 12345 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --tune-eval-runs 5 `
  --output-subdir atlas-paper-fixed-v2 `
  --resume
```

Preflight command:

```powershell
$env:ATLAS_INCLUDE_ESTIMATORS = "ZCD,EKF"
$env:ATLAS_MAG_LEVELS_PCT = "10"
$env:ATLAS_ROCOF_LEVELS_HZ_S = "1"
$env:ATLAS_FREQSTEP_LEVELS_HZ = "0.1"
$env:ATLAS_PHASE_JUMP_LEVELS_DEG = "10"
$env:ATLAS_AM_MOD_FREQ_LEVELS_HZ = "2"
$env:ATLAS_FM_MOD_FREQ_LEVELS_HZ = "2"
$env:ATLAS_HARMONICS_THD_LEVELS_PCT = "5"
$env:ATLAS_INTERHARMONIC_LEVELS_PCT = "2"
$env:ATLAS_NOISE_SIGMA_LEVELS_PU = "0.001"
python -m pipelines.atlas_sweep `
  --sweeps all `
  --policy fixed_policy `
  --n-runs 1 `
  --base-seed 12345 `
  --n-cost-reps 1 `
  --tune-trials 1 `
  --tune-eval-runs 1 `
  --output-subdir atlas-00-preflight-fixed-policy `
  --resume
```

Preflight result:

- full sweep coverage: all nine required ATLAS sweeps present
- policy: `fixed_policy`
- rows: 26 estimator-scenario summaries
- status: `diagnostic`, as expected, because this is a 2-estimator `n=1`
  preflight

Acceptance criteria:

- Readiness status is at least `journal_grade`.
- No missing canonical estimators.
- No mixed policy.
- No missing required sweeps.
- Every figure in the paper or presentation traces to an archived artifact.

## ATLAS-01: RoCoF sign-asymmetry atlas

Priority: P0

Status: Code complete. `save_sign_asymmetry()` now exports `atlas_sign_asymmetry.csv`
with `above_1p5` and `above_2p0` threshold columns, and the plot carries 1.5×, 2.0×,
and 3.0× reference lines. Needs paper-grade run.

Question: Do positive and negative RoCoF ramps produce symmetric estimator
errors?

Rationale:

For an ideal symmetric estimator, replacing `alpha` with `-alpha` in

```tex
f(t)=f_0+\alpha t
```

should preserve RMSE magnitude. A sign-dependent RMSE indicates nonlinear
tracking, saturation, asymmetric bounds, window bias, or estimator-state
interaction.

Run:

```powershell
$env:ATLAS_ROCOF_LEVELS_HZ_S = "0.1,0.5,1,3,5,10,20,50"
$env:ATLAS_ROCOF_DIRECTIONS = "pos,neg"
python -m pipelines.atlas_sweep `
  --sweeps rocof `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-rocof-sign-v1
```

Metrics:

- RMSE vs `abs_rocof_hz_s`
- RFE max and RFE RMS
- sign-asymmetry ratio: `max(RMSE_pos, RMSE_neg) / min(RMSE_pos, RMSE_neg)`
- trip exposure
- bound-hit rate
- invalid output rate

Acceptance criteria:

- Both signs present at every RoCoF level.
- Report lists estimators with sign-asymmetry ratio above 1.5 and 2.0.
- Plot separates low-RoCoF, operational, severe, and extreme regimes.

## ATLAS-02: frequency-step severity atlas

Priority: P1

Status: Code complete. `save_winner_regions()` now writes `atlas_winner_regions.csv`
(best estimator per severity band per sweep). Pareto plot suptitle now explicitly
states marker size ∝ structural latency. Sign-asymmetry reporting for low-RMSE
estimators was already correct. Needs paper-grade run.

Question: Which estimators track sudden frequency offsets without excessive
settling, delay, or trip-risk?

Run:

```powershell
$env:ATLAS_FREQSTEP_LEVELS_HZ = "0.05,0.1,0.2,0.5,1,2,3,5"
$env:ATLAS_FREQSTEP_DIRECTIONS = "pos,neg"
python -m pipelines.atlas_sweep `
  --sweeps frequency_step `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-frequency-step-v1
```

Metrics:

- RMSE, peak error, post-1-cycle RMSE, post-3-cycle RMSE
- settling time
- sign-asymmetry ratio
- CPU and structural latency

Acceptance criteria:

- Report identifies winner regions by step magnitude.
- Report separates estimator accuracy from structural delay.
- Sign-asymmetry is reported even when average RMSE is low.

## ATLAS-03: dense phase-jump atlas

Priority: P0

Status: Code complete. `save_critical_thresholds()` exports `atlas_critical_thresholds.csv`
(first level RMSE > 0.05 Hz per estimator per sweep). Dashboard now includes
`m25_post_1cy`, `m26_post_3cy`, and `m22_invalid_output_rate`. Phase-jump
methodology string now explicitly distinguishes pure phase-jump from composite
islanding (IBR multi-event). Needs paper-grade run.

Question: At what phase-jump severity does each estimator lose frequency
tracking or trip-risk robustness?

Run:

```powershell
$env:ATLAS_PHASE_JUMP_LEVELS_DEG = "0:180:5"
$env:ATLAS_PHASE_JUMP_DIRECTIONS = "pos,neg"
python -m pipelines.atlas_sweep `
  --sweeps phase_jump_sweep `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-phase-jump-dense-v1
```

Metrics:

- first phase-jump level where RMSE exceeds 0.05 Hz
- trip exposure and max contiguous trip exposure
- post-event 1-cycle, 3-cycle, and 100 ms RMSE
- sign-asymmetry ratio
- invalid output rate

Acceptance criteria:

- Both positive and negative jumps are included.
- Critical threshold table is produced for every estimator.
- Plot distinguishes pure phase-jump behavior from composite islanding behavior.

## ATLAS-04: integer-harmonic THD atlas

Priority: P1

Status: Code complete. `hypothesis_results.csv` now splits `"flat"` into
`"flat_good"` (median RMSE below guide) and `"flat_bad"` (flat but consistently
poor); adds `median_rmse_hz` and `rmse_guide_hz` columns. Dashboard now includes
`m11_rnaf_db`. Harmonic-only isolation warning already present in methodology
string. Needs paper-grade run.

Question: How does estimator RMSE scale with isolated integer-harmonic THD?

Run:

```powershell
$env:ATLAS_HARMONICS_THD_LEVELS_PCT = "1,2,3,5,8,10,15,20,30"
python -m pipelines.atlas_sweep `
  --sweeps harmonics `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-harmonics-thd-v1
```

Metrics:

- RMSE vs THD
- harmonic sensitivity slope
- first THD level where RMSE exceeds 0.05 Hz
- RNAF/noise amplification where applicable
- CPU and invalid output rate

Acceptance criteria:

- Report separates flat-good from flat-bad curves.
- Harmonic-only conclusions are not generalized to interharmonics or mixed IBR
  events.

## ATLAS-05: interharmonic leakage atlas

Priority: P0

Status: Code complete. `interharmonics` sweep isolates the 75 Hz non-synchronous
component (`ih75_pct`) with all integer harmonics and RoCoF zeroed out.
`DISABLE_EVENT_METRICS` is set, `save_winner_regions()` and
`save_hypothesis_results()` cover the acceptance criteria outputs.
Needs paper-grade run.

Question: Which estimators are sensitive to off-bin, non-synchronous components
that do not align with integer harmonics?

Run:

```powershell
$env:ATLAS_INTERHARMONIC_LEVELS_PCT = "0.5,1,2,3,5,8,10,15,20"
python -m pipelines.atlas_sweep `
  --sweeps interharmonics `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-interharmonics-v1
```

Metrics:

- RMSE vs interharmonic amplitude
- peak error
- RFE RMS
- estimator-specific failure thresholds
- comparison against integer-harmonic THD atlas

Acceptance criteria:

- Report states whether integer-harmonic winners remain robust under
  interharmonics.
- Window-based and PLL/FLL estimator leakage behavior is explicitly compared.

## ATLAS-06: AM modulation atlas

Priority: P1

Status: Implemented sweep, only smoke evidence exists.

Question: Which estimators misinterpret amplitude modulation as frequency
movement?

Run:

```powershell
$env:ATLAS_AM_MOD_FREQ_LEVELS_HZ = "0.1,0.2,0.5,1,2,3,5,8,10"
python -m pipelines.atlas_sweep `
  --sweeps modulation_am_sweep `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-am-modulation-v1
```

Metrics:

- AM-to-FM leakage: RMSE under zero true frequency modulation
- peak error
- settling and invalid output rate
- CPU

Acceptance criteria:

- Report identifies estimators that show false frequency modulation under AM.
- Results are compared with the FM modulation atlas.

## ATLAS-07: FM modulation atlas

Priority: P1

Status: Implemented sweep, only smoke evidence exists.

Question: Which estimators have enough bandwidth to track true frequency
modulation without excessive lag?

Run:

```powershell
$env:ATLAS_FM_MOD_FREQ_LEVELS_HZ = "0.1,0.2,0.5,1,2,3,5,8,10"
python -m pipelines.atlas_sweep `
  --sweeps modulation_fm_sweep `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-fm-modulation-v1
```

Metrics:

- RMSE vs modulation frequency
- phase lag proxy through post-event windows
- RFE RMS
- CPU and structural latency

Acceptance criteria:

- Report shows bandwidth limits by estimator family.
- AM leakage and FM tracking are not collapsed into one score.

## ATLAS-08: Gaussian noise/SNR atlas

Priority: P0

Status: Code complete. `noise_snr` sweep generates correctly. `snr_db` is
recorded in the global CSV alongside `noise_sigma_pu` (both columns always
present). All RMSE and family plots for `noise_snr` and `noise_nongaussian`
now carry a secondary top x-axis showing SNR [dB] via `_apply_sweep_x_axis()`,
satisfying the dual-unit axis criterion. `save_critical_thresholds()` identifies
the noise floor per estimator. RNAF (m11) and RFE (m9/m10) are in the
dashboard for bandwidth-vs-rejection separation. Needs paper-grade run.

Question: What is the noise floor of each estimator under additive white
Gaussian noise?

Signal model:

```tex
v[k] = A \sin(\theta[k]) + n[k], \qquad n[k]\sim \mathcal{N}(0,\sigma^2)
```

For `A = 1 pu`, the SNR is:

```tex
\mathrm{SNR}_{dB}=20\log_{10}\left(\frac{1/\sqrt{2}}{\sigma}\right)
```

Run:

```powershell
$env:ATLAS_NOISE_SIGMA_LEVELS_PU = "0.0001,0.0003,0.001,0.003,0.01,0.03,0.10"
python -m pipelines.atlas_sweep `
  --sweeps noise_snr `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-gaussian-noise-snr-v1
```

Metrics:

- RMSE vs SNR
- RFE RMS vs SNR
- RNAF/noise amplification
- invalid output rate
- jitter and CPU timing

Acceptance criteria:

- Report identifies each estimator noise floor.
- Report separates tracking bandwidth from noise rejection.
- SNR axis is reported in both `sigma pu` and `dB`.

## ATLAS-09: non-Gaussian noise implementation

Priority: P0

Status: Code complete. `noise_nongaussian` sweep key is registered in
`SWEEP_SPECS` with four variance-matched variants: `gaussian`, `laplace`,
`student_t_df3`, and `bernoulli_gaussian`. The scenario is implemented in
`src/scenarios/ieee_nongaussian_noise.py`. All models are parametrised by
`noise_sigma` and scaled so that their variance equals `noise_sigma^2`.
Output schema records `noise_model`, `noise_sigma_pu`, and `snr_db` per row.
The sweep runs through `--sweeps noise_nongaussian` or is included in
`--sweeps p0`. Run command below produces paper-grade evidence.

Question: Which estimators are robust to field-like noise that violates the
Gaussian assumption?

Implementation notes:

- Sweep key `noise_nongaussian` added to `SWEEP_SPECS` with
  `variants=("gaussian","laplace","student_t_df3","bernoulli_gaussian")`.
- `SweepSpec` extended with `variants: tuple[str, ...]` field.
- `_directions_for_sweep()` returns variant names for variant sweeps.
- `build_atlas_scenarios()` routes variant sweeps through `variant=` parameter.
- `_make_scenario_variant()` extended with `variant: str = ""` parameter.
- `save_sign_asymmetry()` generalised to compare all variants (max/min ratio).
- Env var `ATLAS_NOISE_NONGAUSSIAN_SIGMA_LEVELS_PU` overrides sigma levels.
- Bernoulli-Gaussian: `p_spike=0.05`, `sigma_spike=3*sigma_bg`, variance-matched.
- Student-t df=3: scaled by `sigma/sqrt(df/(df-2))` for variance match.
- Laplace: scale `b=sigma/sqrt(2)` for variance match.

Proposed model families:

```tex
n[k] = w[k] + b[k]s[k]
```

where `w[k]` is low-level AWGN, `b[k]` is a Bernoulli impulse process, and
`s[k]` is the impulse amplitude distribution.

Acceptance criteria:

- The sweep can run through `python -m pipelines.atlas_sweep`. ✓
- The output schema records the noise model and severity parameters. ✓
- Plots compare Gaussian and non-Gaussian noise at matched nominal variance. ✓

Run:

```powershell
$env:ATLAS_NOISE_NONGAUSSIAN_SIGMA_LEVELS_PU = "0.0001,0.0003,0.001,0.003,0.01,0.03,0.10"
python -m pipelines.atlas_sweep `
  --sweeps noise_nongaussian `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-nongaussian-noise-v1
```

## ATLAS-10: impulsive-noise atlas

Priority: P0

Status: Code complete. `impulse_probability` sweep key registered in `SWEEP_SPECS`
with four magnitude variants (`mag0p05`, `mag0p10`, `mag0p20`, `mag0p50`).
Scenario implemented in `src/scenarios/ieee_impulsive_noise.py` using the
Bernoulli-Gaussian model: background AWGN (fixed at 0.001 pu) + sparse Bernoulli
spikes (N(0, mag²) when indicator=1). x-axis is impulse probability; magnitude
variants appear as separate curves enabling cross-comparison.
`IMPULSE_VARIANT_MAGNITUDES` maps variant names to magnitudes.
Env vars: `ATLAS_IMPULSE_PROBABILITY_LEVELS_PU`, `ATLAS_IMPULSE_AWGN_SIGMA`.
Needs paper-grade run.

Question: Which estimators fail under sparse switching spikes or measurement
glitches?

Proposed levels:

- `impulse_probability`: `1e-4,3e-4,1e-3,3e-3,1e-2`
- `impulse_magnitude_pu`: `0.05,0.10,0.20,0.50` (variants)
- baseline AWGN: `sigma = 0.001 pu` (fixed, env-overridable)

Metrics:

- RMSE (m1) — aggregate sensitivity
- peak error (m3) — spike sensitivity
- invalid output rate (m22) — glitch propagation / recovery proxy
- trip exposure (m5) — protection function risk
- max contiguous trip exposure (m6)

Acceptance criteria:

- Report identifies estimators that are peak-sensitive but RMSE-stable. ✓
- Report identifies estimators with long recovery after sparse impulses. ✓

Run:

```powershell
$env:ATLAS_IMPULSE_PROBABILITY_LEVELS_PU = "1e-4,3e-4,1e-3,3e-3,1e-2"
python -m pipelines.atlas_sweep `
  --sweeps impulse_probability `
  --policy fixed_policy `
  --n-runs 100 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --output-subdir atlas-impulsive-noise-v1
```

## ATLAS-11: heavy-tail noise atlas

Priority: P1

Status: Code complete. `heavy_tail_noise` sweep key registered in `SWEEP_SPECS`
with five variance-matched variants: `gaussian`, `laplace`, `student_t_df3`,
`student_t_df5`, `student_t_df10`. New metrics `m34_p95_error_hz` and
`m35_p99_error_hz` added to `src/analysis/metrics.py` (within-run percentiles
of |error|) and wired into `calculate_all_metrics()`. Both appear in the
multipage dashboard and global CSV. Env var: `ATLAS_HEAVY_TAIL_SIGMA_LEVELS_PU`.
Needs paper-grade run.

Question: Does estimator ranking change when noise has heavier tails than a
Gaussian distribution at matched scale?

Proposed models:

- Gaussian baseline
- Laplace
- Student-t with `df = 3, 5, 10`

Metrics:

- RMSE
- median absolute error
- p95 and p99 error
- invalid output rate
- RNAF/noise amplification

Acceptance criteria:

- Report compares rankings by RMSE and p95/p99 error.
- Kalman-family assumptions are evaluated under model mismatch.

## ATLAS-12: colored-noise and drift atlas

Priority: P1

Status: Partially present in composite IBR scenarios, not isolated in ATLAS.

Question: Which estimators confuse low-frequency measurement drift with
frequency movement?

Proposed models:

- AR(1) colored noise with `rho = 0.5, 0.8, 0.95, 0.99`
- Brown noise / random-walk drift
- low-frequency sinusoidal measurement bias

Metrics:

- steady-state bias
- RMSE
- RFE RMS
- settling time
- bound-hit rate

Acceptance criteria:

- Report separates white-noise robustness from drift robustness.
- Drift-sensitive estimators are flagged even if AWGN performance is strong.

## ATLAS-13: measurement corruption atlas

Priority: P2

Status: Not implemented.

Question: How do estimators behave when the measurement stream is damaged by
instrumentation effects rather than physical grid dynamics?

Proposed corruption types:

- sample dropout
- time-stamp jitter
- ADC clipping
- quantization
- short bursts of missing data

Metrics:

- invalid output rate
- startup valid samples
- bound-hit rate
- peak error
- recovery time

Acceptance criteria:

- Results distinguish estimator failure from signal-model difficulty.
- Output schema records corruption type and severity.

## ATLAS-14: mixed IBR stress atlas

Priority: P0

Status: Code complete. `mixed_ibr_lhs` sweep key registered in
`src/pipelines/atlas_sweep.py`. New scenario `IEEEMixedStressScenario` in
`src/scenarios/ieee_mixed_stress.py` combines RoCoF ramp + phase jump +
integer harmonics + noise in a single signal. Three disturbance profiles:
`low_dist` (5°, 3% THD, 0.001 pu AWGN), `med_dist` (20°, 8% THD, 0.003 pu
AWGN), `high_dist` (45°, 15% THD, 0.01 pu Bernoulli-Gaussian). Primary sweep
axis is |RoCoF| [Hz/s] at 5 levels (1, 5, 10, 20, 50 Hz/s) × 3 profiles =
15 scenario variants. Phase jump at `t_jump_s=0.50s` enables post-event
metrics (m24–m30) via the MC engine. Remaining gate: paper-grade run.

Question: Which estimator choices survive realistic combinations after isolated
mechanisms are understood?

Acceptance criteria (remaining):

- Paper-grade run (`n_runs>=30`, fixed policy, all 18 canonical estimators).
- Report explains which isolated failures predict mixed-event failures.
- Report identifies estimators that are robust across combinations, not only
  under isolated stress.

Command:

```powershell
python -m pipelines.atlas_sweep `
  --sweeps mixed_ibr_lhs `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 1 `
  --tune-trials 20 `
  --tune-eval-runs 3 `
  --output-subdir atlas-mixed-ibr-lhs-v1
```

## ATLAS-15: composite scoring and qualification profile

Priority: P1

Status: Conceptual.

Question: Can ATLAS summarize estimator readiness without hiding the individual
failure modes?

Tasks:

- Define a vector score, not a single leaderboard:
  - accuracy
  - RoCoF tracking
  - trip-risk exposure
  - settling
  - noise robustness
  - harmonic/interharmonic robustness
  - CPU/latency
  - invalid output rate
- Report both the component vector and a weighted profile.
- Keep weights explicit and configurable.

Acceptance criteria:

- No estimator is called globally best without a profile.
- Recommendation maps are tied to application classes: FFR, anti-islanding,
  offline monitoring, relay-adjacent analytics, and forensic playback.

## Presentation integration

Use only a compact subset in the SGSMA talk:

1. RoCoF sign-asymmetry: why both signs matter.
2. Phase-jump severity: where estimators start to fail.
3. Harmonics vs interharmonics: integer THD is not enough.
4. Gaussian vs non-Gaussian noise: noise floor vs field robustness.
5. Readiness gate: diagnostic, paper-grade, journal-grade.

Every ATLAS slide must show its evidence level. Diagnostic plots can be used to
explain mechanisms, but not as final qualification claims.
