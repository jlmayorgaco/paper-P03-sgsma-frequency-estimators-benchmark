# API Reference

OpenFreqBench exposes three extension points: estimators, scenarios, and
metrics. This document covers the public API contract for each.

## Estimator API

Every canonical estimator inherits from `BaseFrequencyEstimator` and implements:

```python
def _estimate(self, z: complex, t_s: float | None = None,
              memory: MemoryStore | None = None) -> float:
    """Core estimation logic. One voltage sample in, one frequency estimate out."""
```

The platform wraps this via `step(z, t_s, memory)` which adds runtime
instrumentation, NaN/Inf detection, and memory tracking.

### Canonical Estimators (18 active)

| Label | Family | Reference | Structural Latency |
|-------|--------|-----------|-------------------|
| `ZCD` | Loop-based | Djuric & Djurisic (2008) | 0 samples |
| `PLL` | Loop-based | Kaura & Blasko (1997) | 0 samples |
| `SOGI-PLL` | Loop-based | Ciobotaru et al. (2006) | 0 samples |
| `SOGI-FLL` | Loop-based | Rodriguez et al. (2011) | 0 samples |
| `Type-3 SOGI-PLL` | Loop-based | Karimi-Ghartemani & Iravani (2004) | 0 samples |
| `IPDFT` | Window-based | Grandke (1983) | configurable cycles |
| `TFT` | Window-based | Platas-Garza & de la O Serna (2010) | 2 cycles |
| `ESPRIT` | Window-based | Roy & Kailath (1989) | 2 cycles |
| `Prony` | Window-based | Hauer (1991) | configurable cycles |
| `EKF` | Model-based | Dash, Pradhan & Panda (1999) | 0 samples |
| `UKF` | Model-based | Julier & Uhlmann (2004); Regulski & Terzija (2012) | 0 samples |
| `RA-EKF` | Model-based | Panigrahi et al. (2009) | 0 samples |
| `LKF` | Model-based | Pradhan (2004) | 0 samples |
| `LKF2` | Model-based | Reza, Ciobotaru & Agelidis (2012) | 0 samples |
| `RLS` | Adaptive | Carlsson & Handel (1994) | 0 samples |
| `TKEO` | Adaptive | Maragos, Kaiser & Quatieri (1993) | 0 samples |
| `Koopman (RK-DPMU)` | Data-driven | Williams, Kevrekidis & Rowley (2015) | configurable delay |
| `PI-GRU` | Data-driven | Raissi, Perdikaris & Karniadakis (2019); Cho et al. (2014) | 0 samples |

### Adding a Custom Estimator

1. Create a Python file with a class inheriting from `BaseFrequencyEstimator`
2. Implement `_estimate(self, z, t_s=None, memory=None) -> float`
3. Register in a YAML config under `benchmark.custom_estimators`
4. Run via `openfreqbench run --config my-config.yaml`

Example:

```python
from estimators.base import BaseFrequencyEstimator

class MyEstimator(BaseFrequencyEstimator):
    def _estimate(self, z, t_s=None, memory=None):
        # Your frequency estimation logic here
        return 60.0  # placeholder
```

## Scenario API

Scenarios inherit from `Scenario` and implement:

```python
def run(self) -> ScenarioData:
    """Generate voltage trace and ground-truth frequency trajectory."""
```

The `ScenarioData` container includes:
- `v`: voltage samples at 10 kHz
- `t`: time vector
- `f_true`: ground-truth frequency trajectory
- `event_time_s`: optional event timestamp
- `noise_sigma`: optional noise standard deviation
- `metadata`: scenario description and parameters

### Scenario Families (32 registered)

| Family | Examples | Stress Type |
|--------|---------|------------|
| Frequency steps | `IEEE_Freq_Step`, `IEEE_Freq_Step_Small` | Static accuracy |
| Frequency ramps | `IEEE_Freq_Ramp_0.25Hzs` through `IEEE_Freq_Ramp_20Hzs` | Dynamic tracking |
| Magnitude steps | `IEEE_Mag_Step_1pct` through `IEEE_Mag_Step_50pct` | Amplitude immunity |
| Phase jumps | `IEEE_Phase_Jump_20`, `IEEE_Phase_Jump_60`, `NERC_Phase_Jump_60` | Discontinuity handling |
| Modulation | `IEEE_Modulation_AM`, `IEEE_Modulation_FM` | Envelope/variation tracking |
| Harmonics | `IBR_Harmonics_Small`, `IBR_Harmonics_Large` | THD immunity |
| Interharmonics | `IEEE_OOB_Interference` | Out-of-band rejection |
| Noise | `IEEE_Single_SinWave` (varying SNR levels) | Noise amplification |
| Ringdown | `IBR_Power_Imbalance_Ringdown` (4 noise levels) | IBR dynamics |
| Composite IBR | `IBR_Multi_Event` | Multi-stress realism |

## Metric API

Metrics are defined in `src/analysis/metrics.py` as individual functions
(`m1_rmse_hz`, `m2_mae_hz`, etc.) and called by `calculate_all_metrics()`.

The orchestrator signature:

```python
def calculate_all_metrics(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    fs_dsp: float,
    exec_time_s: float,
    structural_samples: int,
    noise_sigma: float = 0.0,
    interharmonic_hz: float = 32.5,
    event_time_s: float | None = None,
) -> dict:
```

Returns a dictionary of `{metric_id: value}` for all 30+ metrics in the profile.

### Adding a Metric

1. Add a new function `mXX_<name>(...)` in `src/analysis/metrics.py`
2. Register it in `src/openfreqbench/registry.py` with group, unit, and label
3. Add to `calculate_all_metrics()` return dict
4. Add documentation in `docs/METHODS.md`
5. Increment the metric profile version

## CLI Workflow

```
openfreqbench --version          Check installed version
openfreqbench doctor             Verify runtime and dependencies
openfreqbench list scenarios     List all registered scenarios
openfreqbench list estimators    List all registered estimators
openfreqbench list metrics       List all registered metrics
openfreqbench init --template X  Generate a starter YAML config
openfreqbench run --config X     Run a benchmark from YAML
openfreqbench quick-test         One estimator, one scenario, one run
openfreqbench compare            Multi-estimator single-scenario comparison
openfreqbench report build       Generate tables and plots from a run
openfreqbench validate-artifacts Check tuned artifact availability
openfreqbench archive            Freeze a run with hashes and traceability
openfreqbench hypotheses run     Run preregistered hypothesis tests
openfreqbench quality-gate       Run package readiness checks
```

## Output Artifacts

| File | Description |
|------|------------|
| `benchmark_report.json` | Full machine-readable benchmark results (schema-validated) |
| `raw_run_records.csv` | One row per Monte Carlo run, all raw metrics |
| `aggregated_metrics.csv` | Mean, std, CI per scenario-estimator pair |
| `analysis_summary.md` | Human-readable report with tables |
| `manifest.json` | Reproducibility manifest with hashes and environment |
| `artifact_index.csv` | File inventory with paths and hashes |
| `paper_traceability.csv` | Claim-to-artifact mapping for publications |
| `evidence_manifest.json` | Aggregated evidence package metadata |
