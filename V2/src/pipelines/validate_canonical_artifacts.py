from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.benchmark_definition import active_estimator_specs
from pipelines.paths import BENCHMARK_OUTPUT_DIR, FIGURE1_BASENAME, FIGURE2_BASENAME, JSON_REPORT_NAME


@dataclass
class ValidationResult:
    ok: bool
    missing_top_level: list[str] = field(default_factory=list)
    missing_scenarios: list[str] = field(default_factory=list)
    missing_estimators: dict[str, list[str]] = field(default_factory=dict)
    missing_estimator_files: dict[str, list[str]] = field(default_factory=dict)
    missing_scenario_files: dict[str, list[str]] = field(default_factory=dict)


EXPECTED_SCENARIOS = [
    "IBR_Multi_Event",
    "IBR_Power_Imbalance_Ringdown",
    "IBR_Harmonics_Small",
    "IBR_Harmonics_Medium",
    "IBR_Harmonics_Large",
    "IEEE_Freq_Step",
    "IEEE_Modulation",
    "IEEE_Modulation_AM",
    "IEEE_Modulation_FM",
    "IEEE_OOB_Interference",
    "IEEE_Phase_Jump_20",
    "IEEE_Phase_Jump_60",
    "NERC_Phase_Jump_60",
    "IEEE_Single_SinWave",
    "IEEE_Mag_Step_1pct",
    "IEEE_Mag_Step_5pct",
    "IEEE_Mag_Step_10pct",
    "IEEE_Mag_Step_15pct",
    "IEEE_Mag_Step_25pct",
    "IEEE_Mag_Step_50pct",
    "IEEE_Freq_Ramp_0.25Hzs",
    "IEEE_Freq_Ramp_0.5Hzs",
    "IEEE_Freq_Ramp_1Hzs",
    "IEEE_Freq_Ramp_2Hzs",
    "IEEE_Freq_Ramp_5Hzs",
    "IEEE_Freq_Ramp_10Hzs",
    "IEEE_Freq_Ramp_15Hzs",
    "IEEE_Freq_Ramp_20Hzs",
    "IBR_Power_Imbalance_Ringdown_Low_Noise",
    "IBR_Power_Imbalance_Ringdown_Normal_Noise",
    "IBR_Power_Imbalance_Ringdown_Medium_Noise",
    "IBR_Power_Imbalance_Ringdown_Severe_Noise",
]


def _expected_scenarios() -> list[str]:
    return list(EXPECTED_SCENARIOS)


def _expected_estimators() -> list[str]:
    return [spec.label for spec in active_estimator_specs()]


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def validate_canonical_artifacts(root: Path, allow_partial: bool = False) -> ValidationResult:
    root = Path(root)
    expected_scenarios = _expected_scenarios()
    expected_estimators = _expected_estimators()
    if allow_partial:
        include_scenarios = set(_env_csv("BENCHMARK_INCLUDE_SCENARIOS"))
        include_estimators = set(_env_csv("BENCHMARK_INCLUDE_ESTIMATORS"))
        if include_scenarios:
            expected_scenarios = [s for s in expected_scenarios if s in include_scenarios]
        if include_estimators:
            expected_estimators = [e for e in expected_estimators if e in include_estimators]

    required_top_level = [
        "global_metrics_report.csv",
        JSON_REPORT_NAME,
        f"{FIGURE1_BASENAME}.png",
        f"{FIGURE1_BASENAME}.pdf",
        f"{FIGURE2_BASENAME}.png",
        f"{FIGURE2_BASENAME}.pdf",
    ]

    missing_top = [name for name in required_top_level if not (root / name).exists()]
    missing_scenarios: list[str] = []
    missing_estimators: dict[str, list[str]] = {}
    missing_estimator_files: dict[str, list[str]] = {}
    missing_scenario_files: dict[str, list[str]] = {}

    for scenario in expected_scenarios:
        scenario_dir = root / scenario
        if not scenario_dir.exists():
            missing_scenarios.append(scenario)
            continue

        scenario_files_missing: list[str] = []
        if not (scenario_dir / "summary_stats.csv").exists():
            scenario_files_missing.append("summary_stats.csv")
        scenario_csv = scenario_dir / f"{scenario}_scenario.csv"
        if not scenario_csv.exists():
            scenario_files_missing.append(f"{scenario}_scenario.csv")
        if scenario_files_missing:
            missing_scenario_files[scenario] = scenario_files_missing

        missing_in_scenario: list[str] = []
        for estimator in expected_estimators:
            est_dir = scenario_dir / estimator
            if not est_dir.exists():
                missing_in_scenario.append(estimator)
                continue

            est_missing: list[str] = []
            if not any(est_dir.glob("*_summary.csv")):
                est_missing.append("*_summary.csv")
            if not any(est_dir.glob("*_signals.csv")):
                est_missing.append("*_signals.csv")
            if not (est_dir / "run_spec.json").exists():
                est_missing.append("run_spec.json")
            if est_missing:
                missing_estimator_files[f"{scenario}/{estimator}"] = est_missing

        if missing_in_scenario:
            missing_estimators[scenario] = missing_in_scenario

    ok = not (
        missing_top
        or missing_scenarios
        or missing_estimators
        or missing_estimator_files
        or missing_scenario_files
    )
    return ValidationResult(
        ok=ok,
        missing_top_level=missing_top,
        missing_scenarios=missing_scenarios,
        missing_estimators=missing_estimators,
        missing_estimator_files=missing_estimator_files,
        missing_scenario_files=missing_scenario_files,
    )


def _print_result(result: ValidationResult) -> None:
    if result.ok:
        print("[OK] Canonical artifact contract is complete.")
        return

    print("[FAIL] Canonical artifact contract check failed.")
    if result.missing_top_level:
        print(f"  Missing top-level files: {result.missing_top_level}")
    if result.missing_scenarios:
        print(f"  Missing scenarios ({len(result.missing_scenarios)}): {result.missing_scenarios}")
    if result.missing_scenario_files:
        print("  Missing scenario files:")
        for scenario, files in sorted(result.missing_scenario_files.items()):
            print(f"    - {scenario}: {files}")
    if result.missing_estimators:
        print("  Missing estimator directories:")
        for scenario, estimators in sorted(result.missing_estimators.items()):
            print(f"    - {scenario}: {estimators}")
    if result.missing_estimator_files:
        print("  Missing estimator files:")
        for key, files in sorted(result.missing_estimator_files.items()):
            print(f"    - {key}: {files}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical benchmark artifacts under artifacts/full_mc_benchmark/."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=BENCHMARK_OUTPUT_DIR,
        help="Artifact root directory (default: artifacts/full_mc_benchmark).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial artifact sets when BENCHMARK_INCLUDE_SCENARIOS and/or BENCHMARK_INCLUDE_ESTIMATORS are used.",
    )
    args = parser.parse_args()

    result = validate_canonical_artifacts(args.root, allow_partial=bool(args.allow_partial))
    _print_result(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
