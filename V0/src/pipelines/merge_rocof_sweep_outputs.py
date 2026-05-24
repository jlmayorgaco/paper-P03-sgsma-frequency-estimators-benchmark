from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(ROOT))

from pipelines import rocof_sweep_fixed_policy as rocof


def _split_paths(raw: str) -> list[Path]:
    chunks = raw.replace(";", ",").split(",")
    paths: list[Path] = []
    for chunk in chunks:
        text = chunk.strip().strip('"')
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = ROOT / path
        paths.append(path)
    return paths


def _default_inputs() -> list[Path]:
    return [
        ROOT / "artifacts" / "freq_ramp_rocof_v3_journal_pos_fast14",
        ROOT / "artifacts" / "freq_ramp_rocof_v3_journal_pos_datadriven_n3",
    ]


def _load_global_metrics(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        csv_path = path / rocof.GLOBAL_CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        df["source_artifact"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    key_cols = ["scenario", "rocof_hz_s", "abs_rocof_hz_s", "direction", "estimator"]
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    return merged.sort_values(["abs_rocof_hz_s", "direction", "family", "estimator"]).reset_index(drop=True)


def _merge_timing(paths: list[Path], out_dir: Path) -> Path | None:
    frames: list[pd.DataFrame] = []
    for path in paths:
        csv_path = path / rocof.TIMING_CSV_NAME
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["source_artifact"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        frames.append(df)
    if not frames:
        return None
    timing = pd.concat(frames, ignore_index=True)
    timing_path = out_dir / rocof.TIMING_CSV_NAME
    timing.to_csv(timing_path, index=False)
    return timing_path


def _write_manifest(out_dir: Path, input_paths: list[Path], df: pd.DataFrame) -> Path:
    payload = {
        "experiment": "merged_rocof_sweep_atlas",
        "pipeline_entrypoint": "src/pipelines/merge_rocof_sweep_outputs.py",
        "inputs": [str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) for path in input_paths],
        "n_rows": int(len(df)),
        "estimators": sorted(df["estimator"].dropna().astype(str).unique().tolist()),
        "families": {
            str(est): str(fam)
            for est, fam in df[["estimator", "family"]].drop_duplicates().sort_values("estimator").itertuples(index=False)
        },
        "notes": [
            "Merged artifact only combines aggregate metrics.",
            "Per-run simulation folders remain in their original source artifacts.",
        ],
    }
    path = out_dir / rocof.MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    raw_inputs = os.getenv("FREQRAMP_MERGE_INPUTS", "")
    input_paths = _split_paths(raw_inputs) if raw_inputs else _default_inputs()
    output_subdir = os.getenv("FREQRAMP_MERGE_OUTPUT_SUBDIR", "freq_ramp_rocof_v3_journal_pos_all16")
    out_dir = ROOT / "artifacts" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_global = _load_global_metrics(input_paths)
    global_csv, rmse_est_csv, rmse_family_csv, continuity_csv = rocof._save_summary_tables(df_global, out_dir)
    timing_csv = _merge_timing(input_paths, out_dir)
    plot_paths, color_map = rocof._save_rmse_by_family_plot(df_global, out_dir)
    plot_paths += rocof._save_method_map(df_global, out_dir)
    plot_paths += rocof._save_sign_asymmetry_diagnostic(df_global, out_dir)
    hypothesis_paths = rocof._save_rocof_hypothesis_tests(df_global, out_dir)
    multipage_pdf = rocof._save_multipage_metrics_dashboard(df_global, out_dir)
    legend_path = out_dir / "rmse_plot_method_legend.csv"
    pd.DataFrame(
        [
            {"estimator": est, "hex_color": rocof.matplotlib.colors.to_hex(rgba), "family": rocof.ESTIMATOR_FAMILIES.get(est, "Unknown")}
            for est, rgba in sorted(color_map.items())
        ]
    ).to_csv(legend_path, index=False)
    readme_path = rocof._write_readme(out_dir)
    manifest_path = _write_manifest(out_dir, input_paths, df_global)

    print("Merged RoCoF artifacts:")
    for path in [
        global_csv,
        rmse_est_csv,
        rmse_family_csv,
        timing_csv,
        continuity_csv,
        *plot_paths,
        *hypothesis_paths,
        multipage_pdf,
        legend_path,
        readme_path,
        manifest_path,
    ]:
        if path is not None:
            print(f"  - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
