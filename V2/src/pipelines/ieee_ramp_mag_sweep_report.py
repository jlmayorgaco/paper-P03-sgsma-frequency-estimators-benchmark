"""
Dedicated dense sweep runner for IEEE frequency ramps and magnitude steps.

This is a focused derivative of the generic parametric sweep pipeline. It keeps
all active estimators, reuses family-anchor tuning, and emits a report bundle
centered on the two IEEE baseline families that matter most for controlled
stress-degradation analysis.

Usage
-----
Plan only:
    python -m pipelines.ieee_ramp_mag_sweep_report --plan-only

Run:
    python -m pipelines.ieee_ramp_mag_sweep_report
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pipelines.full_mc_benchmark as benchmark
import pipelines.parametric_family_sweeps as sweeps
from analysis.parametric_sweep_analysis import build_parametric_regression_reports
from pipelines.report_builder import build_benchmark_report
from plotting.benchmark.parametric_sweep_plots import generate_parametric_sweep_figures
from plotting.benchmark.ramp_mag_sweep_report_plots import build_ieee_ramp_mag_sweep_report


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "ieee_ramp_mag_sweep"
TARGET_FAMILY_KEYS = ("ieee_freq_ramp", "ieee_mag_step")
BENCHMARK_IDENTITY = "ieee_ramp_mag_dense_sweep"
BENCHMARK_SCOPE = "Dense 30-point sweeps for IEEE frequency ramps and magnitude steps."
BENCHMARK_AUTHORITY = (
    "This run isolates the two IEEE baseline disturbance families and measures "
    "how every active estimator degrades as the ramp severity or amplitude-step "
    "magnitude is increased over 30 stress levels."
)
ALIGNMENT_POLICY = "paper_follows_ieee_ramp_mag_dense_sweep"
JSON_REPORT_NAME = "ieee_ramp_mag_sweep_report.json"
TUNING_CACHE_NAME = "ieee_ramp_mag_family_tuning_cache.json"


def _selected_families() -> tuple[sweeps.SweepFamily, ...]:
    return tuple(
        family
        for family in sweeps.ALL_SWEEP_FAMILIES
        if family.key in TARGET_FAMILY_KEYS
    )


def _configure_context(families: tuple[sweeps.SweepFamily, ...]) -> None:
    sweeps.SWEEP_BENCHMARK_IDENTITY = BENCHMARK_IDENTITY
    sweeps.SWEEP_BENCHMARK_SCOPE = BENCHMARK_SCOPE
    sweeps.SWEEP_AUTHORITY_STATEMENT = BENCHMARK_AUTHORITY
    sweeps.SWEEP_ALIGNMENT_POLICY = ALIGNMENT_POLICY
    sweeps.SWEEP_OUTPUT_DIR = OUTPUT_DIR
    sweeps.SWEEP_JSON_NAME = JSON_REPORT_NAME
    sweeps.SWEEP_TUNING_CACHE_NAME = TUNING_CACHE_NAME
    sweeps._configure_benchmark_context(families)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the IEEE ramp + magnitude-step dense sweep report.")
    parser.add_argument("--plan-only", action="store_true", help="Write the sweep manifest and stop.")
    parser.add_argument("--skip-generic-plots", action="store_true", help="Skip the generic family degradation plots.")
    parser.add_argument("--skip-report", action="store_true", help="Skip the dedicated mega report bundle.")
    args = parser.parse_args(argv)

    families = _selected_families()
    if not families:
        raise ValueError("No IEEE ramp/magnitude-step families were found in the parametric sweep catalog.")

    _configure_context(families)
    manifest_path = sweeps._write_manifest(families)
    sweeps._print_plan(families)
    if args.plan_only:
        print(f"  Manifest -> {manifest_path}")
        return

    t_start = time.time()
    print("Loading estimators ...")
    estimators = benchmark.load_estimators()
    benchmark._print_registry_summary(estimators)
    print("Validating search spaces ...")
    benchmark.validate_search_spaces(estimators)
    print("  OK")

    sweeps.run_parametric_phase_1(estimators, families)
    benchmark.run_phase_2(allowed_estimators=set(estimators.keys()))

    if not args.skip_generic_plots:
        generic_plots = generate_parametric_sweep_figures(OUTPUT_DIR, manifest_path)
        if generic_plots:
            print(f"  Generic sweep plots -> {len(generic_plots)} files")

    regression_outputs = build_parametric_regression_reports(OUTPUT_DIR, manifest_path)
    print(f"  Regression summary -> {regression_outputs['csv']}")

    json_path = benchmark._export_full_benchmark_json(estimators)
    generic_report = build_benchmark_report(
        input_json=json_path,
        output_dir=OUTPUT_DIR / "generic_report_bundle",
    )
    print(f"  Generic report bundle -> {generic_report['markdown']}")

    if not args.skip_report:
        report_outputs = build_ieee_ramp_mag_sweep_report(
            output_dir=OUTPUT_DIR,
            manifest_path=manifest_path,
            regression_csv_path=Path(regression_outputs["csv"]),
        )
        print(f"  Mega report -> {report_outputs['markdown']}")

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  Manifest -> {manifest_path.relative_to(ROOT)}")
    print(f"  JSON report -> {json_path.relative_to(ROOT)}")
    print(f"\n[DONE] IEEE ramp + magnitude-step sweep completed in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()
