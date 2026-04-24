from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass


BENCHMARK_IDENTITY = "full_mc_benchmark_active_pipeline"
BENCHMARK_SCOPE = "Full modular Monte Carlo benchmark under src/pipelines"
BENCHMARK_AUTHORITY_STATEMENT = (
    "The active benchmark is defined by the modular pipeline under src/pipelines. "
    "Legacy wrappers and historical paper subsets are secondary views and must adapt "
    "to the active pipeline artifacts."
)
PAPER_ALIGNMENT_POLICY = "paper_follows_active_pipeline"


@dataclass(frozen=True)
class EstimatorSpec:
    key: str
    module_name: str
    class_name: str
    label: str
    family: str
    status: str
    reason: str

    def to_manifest(self) -> dict[str, str]:
        return asdict(self)


ACTIVE_ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("zcd", "zcd", "ZCDEstimator", "ZCD", "Loop-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("ipdft", "ipdft", "IPDFT_Estimator", "IPDFT", "Window-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("tft", "tft", "TFT_Estimator", "TFT", "Window-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("rls", "rls", "RLS_Estimator", "RLS", "Adaptive", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("pll", "pll", "PLL_Estimator", "PLL", "Loop-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("sogi_pll", "sogi_pll", "SOGIPLLEstimator", "SOGI-PLL", "Loop-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("sogi_fll", "sogi_fll", "SOGI_FLL_Estimator", "SOGI-FLL", "Loop-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("type3_sogi_pll", "type3_sogi_pll", "Type3_SOGI_PLL_Estimator", "Type-3 SOGI-PLL", "Loop-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("lkf", "lkf", "LKF_Estimator", "LKF", "Model-based", "active", "Included as the narrowband two-state LKF baseline."),
    EstimatorSpec("lkf2", "lkf2", "LKF2_Estimator", "LKF2", "Model-based", "active", "Included as the alternative LKF-family implementation."),
    EstimatorSpec("ekf", "ekf", "EKF_Estimator", "EKF", "Model-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("ukf", "ukf", "UKF_Estimator", "UKF", "Model-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("ra_ekf", "ra_ekf", "RAEKF_Estimator", "RA-EKF", "Model-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("tkeo", "tkeo", "TKEO_Estimator", "TKEO", "Adaptive", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("prony", "prony", "Prony_Estimator", "Prony", "Window-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("esprit", "esprit", "ESPRIT_Estimator", "ESPRIT", "Window-based", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("koopman", "koopman", "Koopman_Estimator", "Koopman (RK-DPMU)", "Data-driven", "active", "Canonical active estimator in the modular benchmark."),
    EstimatorSpec("pi_gru", "pi_gru", "PI_GRU_Estimator", "PI-GRU", "Data-driven", "active", "Included as the physics-informed neural estimator. Requires torch and pretrained weights."),
)

EXCLUDED_ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = ()

ALL_ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = ACTIVE_ESTIMATOR_SPECS + EXCLUDED_ESTIMATOR_SPECS

ESTIMATOR_FAMILIES: dict[str, str] = {
    spec.label: spec.family for spec in ALL_ESTIMATOR_SPECS
}


def active_estimator_specs() -> list[EstimatorSpec]:
    return list(ACTIVE_ESTIMATOR_SPECS)


def excluded_estimator_specs() -> list[EstimatorSpec]:
    return list(EXCLUDED_ESTIMATOR_SPECS)


def build_estimator_registry_manifest() -> dict[str, object]:
    return {
        "benchmark_identity": BENCHMARK_IDENTITY,
        "benchmark_scope": BENCHMARK_SCOPE,
        "authority_statement": BENCHMARK_AUTHORITY_STATEMENT,
        "paper_alignment_policy": PAPER_ALIGNMENT_POLICY,
        "active": [spec.to_manifest() for spec in ACTIVE_ESTIMATOR_SPECS],
        "excluded": [spec.to_manifest() for spec in EXCLUDED_ESTIMATOR_SPECS],
    }


def load_active_estimators() -> dict[str, type]:
    registry: dict[str, type] = {}
    for spec in ACTIVE_ESTIMATOR_SPECS:
        module = importlib.import_module(f"estimators.{spec.module_name}")
        cls = getattr(module, spec.class_name)
        label = getattr(cls, "name", spec.label)
        if label != spec.label:
            raise ValueError(
                f"Estimator label mismatch for {spec.module_name}: "
                f"expected {spec.label!r}, got {label!r}"
            )
        if label in registry:
            raise ValueError(f"Duplicate estimator label in active registry: {label}")
        registry[label] = cls
    return registry
