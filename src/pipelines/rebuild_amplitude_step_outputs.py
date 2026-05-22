from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild amplitude-step aggregate CSVs and PDFs from per-estimator summaries.")
    parser.add_argument("--output-subdir", required=True, help="Artifact subdirectory under artifacts/.")
    parser.add_argument("--tuning-policy", default=None, help="fixed_policy or per_step_oracle, used only for plot labels.")
    args = parser.parse_args()

    os.environ["ASTEP_OUTPUT_SUBDIR"] = str(args.output_subdir)
    if args.tuning_policy:
        os.environ["ASTEP_TUNING_POLICY"] = str(args.tuning_policy)

    from pipelines import amplitude_step_sweep as sweep
    from pipelines.benchmark_definition import ESTIMATOR_FAMILIES
    import pipelines.full_mc_benchmark as benchmark

    out_dir = sweep.OUTPUT_DIR
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory: {out_dir}")

    rows_agg: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for run_spec_path in sorted(out_dir.glob("Sweep_AmplitudeStep_*/*/run_spec.json")):
        try:
            spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary_files = list(run_spec_path.parent.glob("*_summary.csv"))
        if not summary_files:
            continue
        summary_df = pd.read_csv(summary_files[0])
        estimator = str(spec.get("estimator", run_spec_path.parent.name))
        family = str(spec.get("family", ESTIMATOR_FAMILIES.get(estimator, "Unknown")))
        scenario = str(spec.get("scenario", run_spec_path.parent.parent.name))
        step_percent = float(spec.get("step_percent", float("nan")))
        run_tier = str(spec.get("run_tier", "unknown"))

        rows_agg.append(
            {
                "scenario": scenario,
                "step_percent": step_percent,
                "estimator": estimator,
                "family": family,
                "run_tier": run_tier,
                "n_mc_runs": int(len(summary_df)),
                **sweep._aggregate_summary(summary_df),
            }
        )
        timing = spec.get("timing", {}) if isinstance(spec.get("timing", {}), dict) else {}
        tune_meta = spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {}
        timing_rows.append(
            {
                "scenario": scenario,
                "step_percent": step_percent,
                "estimator": estimator,
                "family": family,
                "run_tier": run_tier,
                "n_mc_runs": int(len(summary_df)),
                "tune_trials": int(tune_meta.get("n_trials_requested", 0) or 0),
                "tune_eval_runs": int(tune_meta.get("tune_eval_runs", 0) or 0),
                "n_cost_reps": int(spec.get("n_cost_reps", 0) or 0),
                "base_seed": int(spec.get("base_seed", 0) or 0),
                "tuning_base_seed": int(spec.get("tuning_base_seed", 0) or 0),
                "pipeline_method_version": str(spec.get("pipeline_method_version", "")),
                "tuning_elapsed_s": timing.get("tuning_elapsed_s"),
                "mc_eval_elapsed_s": timing.get("mc_eval_elapsed_s"),
                "total_elapsed_s": timing.get("total_elapsed_s"),
            }
        )

    if not rows_agg:
        raise RuntimeError(f"No per-estimator summaries found under {out_dir}")

    df_global = pd.DataFrame(rows_agg).sort_values(["step_percent", "family", "estimator"])
    global_csv = out_dir / sweep.GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)

    keep_cols = [
        "scenario", "step_percent", "estimator", "family", "n_mc_runs",
        "m1_rmse_hz_mean", "m1_rmse_hz_median", "m1_rmse_hz_p10", "m1_rmse_hz_p90", "m1_rmse_hz_std",
    ]
    rmse_cols = [col for col in keep_cols if col in df_global.columns]
    df_rmse = df_global[rmse_cols].copy()
    rmse_est_csv = out_dir / sweep.RMSE_EST_CSV_NAME
    df_rmse.to_csv(rmse_est_csv, index=False)

    df_rmse_family = (
        df_rmse.groupby(["step_percent", "family"], as_index=False).agg(
            family_rmse_mean=("m1_rmse_hz_mean", "mean"),
            family_rmse_median=("m1_rmse_hz_median", "median") if "m1_rmse_hz_median" in df_rmse.columns else ("m1_rmse_hz_mean", "median"),
            family_rmse_p10=("m1_rmse_hz_p10", "median") if "m1_rmse_hz_p10" in df_rmse.columns else ("m1_rmse_hz_mean", "min"),
            family_rmse_p90=("m1_rmse_hz_p90", "median") if "m1_rmse_hz_p90" in df_rmse.columns else ("m1_rmse_hz_mean", "max"),
            family_rmse_std=("m1_rmse_hz_mean", "std"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
        ).sort_values(["step_percent", "family"])
    )
    rmse_family_csv = out_dir / sweep.RMSE_FAM_CSV_NAME
    df_rmse_family.to_csv(rmse_family_csv, index=False)
    timing_csv = out_dir / "timing_profile.csv"
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)

    generated_plots, color_map = sweep._plot_rmse_by_family(df_rmse=df_rmse, out_dir=out_dir)
    multipage_pdf_path = sweep._save_multipage_metrics_dashboard(df_global=df_global, out_dir=out_dir)
    summary_map_paths, _ = sweep._save_method_summary_map(df_global=df_global, out_dir=out_dir)
    phase_dispersion_paths = sweep._save_phase_dispersion_map(out_dir)
    tuning_audit_path = sweep._save_tuning_continuity_audit(out_dir)
    hypothesis_paths = sweep._save_deterioration_hypothesis_tests(df_global=df_global, out_dir=out_dir)

    legend_rows = [
        {
            "estimator": estimator,
            "hex_color": matplotlib.colors.to_hex(color_map[estimator]),
            "family": ESTIMATOR_FAMILIES.get(estimator, "Unknown"),
        }
        for estimator in sorted(color_map)
    ]
    legend_map_path = out_dir / sweep.LEGEND_MAP_CSV_NAME
    pd.DataFrame(legend_rows).to_csv(legend_map_path, index=False)

    print("Rebuilt amplitude-step aggregate artifacts:")
    for path in [
        global_csv,
        rmse_est_csv,
        rmse_family_csv,
        timing_csv,
        *generated_plots,
        *summary_map_paths,
        *phase_dispersion_paths,
        multipage_pdf_path,
        tuning_audit_path,
        *hypothesis_paths,
        legend_map_path,
    ]:
        print(f"  - {path.relative_to(sweep.ROOT)}")


if __name__ == "__main__":
    main()
