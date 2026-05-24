from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import MetricProfile, RESERVED_METRIC_PROFILES, validate_estimator_contract, validate_metric_profile, validate_scenario_contract

from pipelines.benchmark_definition import (
    ACTIVE_ESTIMATOR_SPECS,
    BENCHMARK_AUTHORITY_STATEMENT,
    BENCHMARK_IDENTITY,
    BENCHMARK_SCOPE,
    ESTIMATOR_FAMILIES,
    EXCLUDED_ESTIMATOR_SPECS,
    PAPER_ALIGNMENT_POLICY,
    EstimatorSpec,
)
from scenarios.ibr_harmonics_large import IBRHarmonicsLargeScenario
from scenarios.ibr_harmonics_medium import IBRHarmonicsMediumScenario
from scenarios.ibr_harmonics_small import IBRHarmonicsSmallScenario
from scenarios.ibr_multi_event import IBRMultiEventScenario
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.ieee_freq_step import IEEEFreqStepScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_modulation import IEEEModulationScenario
from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario
from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario

CANONICAL_METRIC_PROFILE = "canonical-single-phase-v1"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    label: str
    unit: str
    lower_is_better: bool
    group: str
    description: str

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def _create_variant(base_cls: type, name_suffix: str, param_overrides: dict[str, Any]) -> type:
    new_name = f"{base_cls.SCENARIO_NAME}_{name_suffix}"
    new_params = {**base_cls.DEFAULT_PARAMS, **param_overrides}
    safe_class_suffix = name_suffix.replace(".", "p").replace("-", "m")
    class_name = f"{base_cls.__name__}_{safe_class_suffix}"
    new_cls = type(
        class_name,
        (base_cls,),
        {
            "SCENARIO_NAME": new_name,
            "DEFAULT_PARAMS": new_params,
            "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
        },
    )
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return new_cls


BASE_SCENARIO_CLASSES: tuple[type, ...] = (
    IBRMultiEventScenario,
    IBRPowerImbalanceRingdownScenario,
    IBRHarmonicsSmallScenario,
    IBRHarmonicsMediumScenario,
    IBRHarmonicsLargeScenario,
    IEEEFreqStepScenario,
    IEEEModulationScenario,
    IEEEModulationAMScenario,
    IEEEModulationFMScenario,
    IEEEOOBInterferenceScenario,
    IEEEPhaseJump20Scenario,
    IEEEPhaseJump60Scenario,
    NERCPhaseJump60Scenario,
    IEEESingleSinWaveScenario,
)

MAG_STEP_VARIANTS: tuple[type, ...] = tuple(
    _create_variant(IEEEMagStepScenario, suffix, {"amp_post_pu": value})
    for value, suffix in (
        (1.01, "1pct"),
        (1.05, "5pct"),
        (1.10, "10pct"),
        (1.15, "15pct"),
        (1.25, "25pct"),
        (1.50, "50pct"),
    )
)

RAMP_VARIANTS: tuple[type, ...] = tuple(
    _create_variant(IEEEFreqRampScenario, suffix, {"rocof_hz_s": value})
    for value, suffix in (
        (0.25, "0.25Hzs"),
        (0.5, "0.5Hzs"),
        (1.0, "1Hzs"),
        (2.0, "2Hzs"),
        (5.0, "5Hzs"),
        (10.0, "10Hzs"),
        (15.0, "15Hzs"),
        (20.0, "20Hzs"),
    )
)

RINGDOWN_VARIANTS: tuple[type, ...] = tuple(
    _create_variant(
        IBRPowerImbalanceRingdownScenario,
        suffix,
        {"white_noise_sigma": noise, "interharmonic_pu": interharmonic},
    )
    for noise, interharmonic, suffix in (
        (0.002, 0.01, "Low_Noise"),
        (0.007, 0.02, "Normal_Noise"),
        (0.022, 0.05, "Medium_Noise"),
        (0.03, 0.1, "Severe_Noise"),
    )
)

SCENARIO_CLASSES: tuple[type, ...] = (
    BASE_SCENARIO_CLASSES + MAG_STEP_VARIANTS + RAMP_VARIANTS + RINGDOWN_VARIANTS
)

METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("m1_rmse_hz", "RMSE", "Hz", True, "accuracy", "Steady-state root mean square frequency error."),
    MetricSpec("m2_mae_hz", "MAE", "Hz", True, "accuracy", "Steady-state mean absolute frequency error."),
    MetricSpec("m3_max_peak_hz", "Peak error", "Hz", True, "accuracy", "Maximum absolute frequency error."),
    MetricSpec("m4_std_error_hz", "Error variability", "Hz", True, "accuracy", "Standard deviation of absolute error."),
    MetricSpec("m5_trip_risk_s", "Trip risk", "s", True, "protection", "Accumulated time outside the relay deadband."),
    MetricSpec("m5_trip_risk_resolution_s", "Trip risk resolution", "s", True, "protection", "Sampling resolution for trip-risk differences."),
    MetricSpec("m6_max_contig_trip_s", "Max contiguous trip", "s", True, "protection", "Longest continuous deadband violation."),
    MetricSpec("m7_pcb_hz", "Probabilistic compliance bound", "Hz", True, "protection", "Mean absolute error plus three standard deviations."),
    MetricSpec("m8_settling_time_s", "Settling time", "s", True, "dynamic", "Time until frequency error remains inside threshold."),
    MetricSpec("m9_rfe_max_hz_s", "RFE max", "Hz/s", True, "rocof", "Robust maximum RoCoF estimation error."),
    MetricSpec("m10_rfe_rms_hz_s", "RFE RMS", "Hz/s", True, "rocof", "RMS RoCoF estimation error."),
    MetricSpec("m11_rnaf_db", "RNAF", "dB", True, "rocof", "RoCoF noise amplification factor."),
    MetricSpec("m12_isi_pu", "ISI", "pu", True, "disturbance", "Interharmonic susceptibility index."),
    MetricSpec("m13_cpu_time_us", "CPU time", "us/sample", True, "runtime", "Mean CPU time per sample from repeated process-time runs."),
    MetricSpec("m14_struct_latency_ms", "Structural latency", "ms", True, "runtime", "Algorithmic latency implied by window/state requirements."),
    MetricSpec("m15_pcb_compliant", "PCB compliant", "bool", False, "compliance", "Compliance flag for the PCB threshold."),
    MetricSpec("m16_heatmap_pass", "Heatmap pass", "bool", False, "compliance", "Strict pass/fail flag used by dashboard heatmaps."),
    MetricSpec("m17_hw_class", "Hardware class", "class", False, "runtime", "Deployment class derived from CPU time."),
    MetricSpec("m18_mem_peak_kb", "Peak memory", "kB", True, "runtime", "Peak estimator memory proxy."),
    MetricSpec("m19_mem_mean_kb", "Mean memory", "kB", True, "runtime", "Mean estimator memory proxy."),
    MetricSpec("m20_runtime_jitter_us", "Runtime jitter", "us", True, "runtime", "Step-time jitter from standardized estimator wrapper."),
    MetricSpec("m21_startup_valid_samples", "Startup valid sample", "samples", True, "runtime", "First sample with finite valid output."),
    MetricSpec("m22_invalid_output_rate", "Invalid output rate", "ratio", True, "runtime", "Fraction of invalid estimator outputs."),
    MetricSpec("m23_memory_key_count", "Memory keys", "count", True, "runtime", "Number of keys in the standardized memory store."),
    MetricSpec("m24_pre_event_rmse_hz", "Pre-event RMSE", "Hz", True, "event", "RMSE in the pre-event window."),
    MetricSpec("m25_post_1cy_rmse_hz", "Post-event 1 cycle RMSE", "Hz", True, "event", "RMSE in the first cycle after an event."),
    MetricSpec("m26_post_3cy_rmse_hz", "Post-event 3 cycle RMSE", "Hz", True, "event", "RMSE in the first three cycles after an event."),
    MetricSpec("m27_post_100ms_rmse_hz", "Post-event 100 ms RMSE", "Hz", True, "event", "RMSE in the first 100 ms after an event."),
    MetricSpec("m28_post_event_peak_hz", "Post-event peak", "Hz", True, "event", "Peak error in the first 100 ms after an event."),
    MetricSpec("m29_late_event_rmse_hz", "Late-event RMSE", "Hz", True, "event", "RMSE in the late post-event window."),
    MetricSpec("m30_event_settling_time_s", "Event settling time", "s", True, "event", "Settling time measured from the event instant."),
    MetricSpec("m31_freq_bound_hit_rate", "Frequency bound hit rate", "ratio", True, "guardrail", "Fraction of outputs clamped to estimator bounds."),
    MetricSpec("m32_freq_lower_bound_hit_rate", "Lower bound hit rate", "ratio", True, "guardrail", "Fraction of outputs clamped to lower frequency bound."),
    MetricSpec("m33_freq_upper_bound_hit_rate", "Upper bound hit rate", "ratio", True, "guardrail", "Fraction of outputs clamped to upper frequency bound."),
)

METRIC_LABELS: dict[str, str] = {spec.metric_id: spec.label for spec in METRIC_SPECS}
CANONICAL_METRIC_IDS: tuple[str, ...] = tuple(spec.metric_id for spec in METRIC_SPECS)
CANONICAL_PROFILE = MetricProfile(
    profile_id=CANONICAL_METRIC_PROFILE,
    scope="single-phase",
    metric_ids=CANONICAL_METRIC_IDS,
    locked=True,
    status="active",
)
validate_metric_profile(CANONICAL_PROFILE, set(CANONICAL_METRIC_IDS))


def scenario_registry() -> dict[str, type]:
    out: dict[str, type] = {}
    for cls in SCENARIO_CLASSES:
        validate_scenario_contract(cls)
        out[cls.get_name()] = cls
    return out


def estimator_specs(include_experimental: bool = False) -> list[EstimatorSpec]:
    specs = list(ACTIVE_ESTIMATOR_SPECS)
    if include_experimental:
        specs.extend(EXCLUDED_ESTIMATOR_SPECS)
    return specs


def estimator_spec_registry(include_experimental: bool = False) -> dict[str, EstimatorSpec]:
    return {spec.label: spec for spec in estimator_specs(include_experimental)}


def load_estimators(labels: list[str] | None = None, include_experimental: bool = False) -> dict[str, type]:
    specs = estimator_spec_registry(include_experimental)
    wanted = list(labels or specs.keys())
    out: dict[str, type] = {}
    unknown = [label for label in wanted if label not in specs]
    if unknown:
        raise ValueError(f"Unknown estimator label(s): {unknown}. Known labels: {sorted(specs)}")
    for label in wanted:
        spec = specs[label]
        try:
            module = importlib.import_module(f"estimators.{spec.module_name}")
        except ModuleNotFoundError as exc:
            if spec.key == "pi_gru" and exc.name == "torch":
                raise RuntimeError(
                    "PI-GRU requires torch. Install `openfreqbench[benchmark-full]` "
                    "or remove PI-GRU from this run."
                ) from exc
            raise
        cls = getattr(module, spec.class_name)
        actual_label = getattr(cls, "name", label)
        if actual_label != label:
            raise ValueError(f"Estimator label mismatch: expected {label!r}, got {actual_label!r}")
        validate_estimator_contract(cls, label=label)
        out[label] = cls
    return out


def metric_registry() -> dict[str, MetricSpec]:
    return {spec.metric_id: spec for spec in METRIC_SPECS}


def assert_canonical_metric_ids(metric_ids: list[str]) -> None:
    known = set(CANONICAL_METRIC_IDS)
    unknown = [metric_id for metric_id in metric_ids if metric_id not in known]
    if unknown:
        raise ValueError(
            f"Unknown metric id(s): {unknown}. Metric formulas are fixed by {CANONICAL_METRIC_PROFILE}."
        )


def platform_manifest() -> dict[str, object]:
    return {
        "name": "OpenFreqBench",
        "version": "2.0.0",
        "benchmark_identity": BENCHMARK_IDENTITY,
        "benchmark_scope": BENCHMARK_SCOPE,
        "authority_statement": BENCHMARK_AUTHORITY_STATEMENT,
        "paper_alignment_policy": PAPER_ALIGNMENT_POLICY,
        "metric_profile": CANONICAL_METRIC_PROFILE,
        "metric_profiles": [
            CANONICAL_PROFILE.to_manifest(),
            *[
                {
                    "profile_id": profile_id,
                    "scope": "reserved",
                    "metric_ids": [],
                    "locked": True,
                    "status": "reserved",
                }
                for profile_id in RESERVED_METRIC_PROFILES
            ],
        ],
        "scenarios": sorted(scenario_registry()),
        "estimators": [spec.to_manifest() for spec in estimator_specs(include_experimental=True)],
        "canonical_estimators": [spec.label for spec in estimator_specs(include_experimental=False)],
        "metric_ids": list(CANONICAL_METRIC_IDS),
        "metric_labels": METRIC_LABELS,
        "estimator_families": ESTIMATOR_FAMILIES,
    }
