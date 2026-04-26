from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback for thin environments
    scipy_stats = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper_extended"
GENERATED_DIR = PAPER_DIR / "generated"
REPORT_JSON = REPO_ROOT / "tests" / "montecarlo" / "outputs" / "benchmark_full_report.json"


ESTIMATOR_CATALOG: dict[str, dict[str, str]] = {
    "ZCD": {
        "family": "Loop-based",
        "principle": "Zero-crossing frequency readout from waveform sign changes.",
        "strength": "Minimum compute and transparent implementation.",
        "limitation": "Sensitive to discontinuities, noise, and harmonics.",
    },
    "IPDFT": {
        "family": "Window-based",
        "principle": "Interpolated DFT around the dominant spectral bin.",
        "strength": "Good steady-state selectivity and low trip-risk in several hard cases.",
        "limitation": "Observation-window latency and transient smearing.",
    },
    "TFT": {
        "family": "Window-based",
        "principle": "Taylor-Fourier expansion for dynamic phasor estimation.",
        "strength": "Strong global mean-rank behavior across error metrics.",
        "limitation": "Higher compute than loop methods and window dependence.",
    },
    "RLS": {
        "family": "Adaptive",
        "principle": "Recursive least-squares fit of an AR(2) sinusoidal model.",
        "strength": "Online adaptation with low nominal compute.",
        "limitation": "Unstable under the hardest composite disturbances in this report.",
    },
    "PLL": {
        "family": "Loop-based",
        "principle": "Single-phase phase-locked loop tracking instantaneous phase.",
        "strength": "Good composite-event RMSE and low embedded cost.",
        "limitation": "Can slip or overshoot under large phase discontinuities.",
    },
    "SOGI-PLL": {
        "family": "Loop-based",
        "principle": "Second-order generalized integrator front-end plus PLL.",
        "strength": "Best ramp-family RMSE with sub-microsecond cost.",
        "limitation": "Not universally best under IBR ringdown or multi-event stress.",
    },
    "SOGI-FLL": {
        "family": "Loop-based",
        "principle": "SOGI front-end with frequency-locked loop update law.",
        "strength": "Very low compute and strong phase-jump behavior.",
        "limitation": "Less competitive than the best model-based filters on clean families.",
    },
    "Type-3 SOGI-PLL": {
        "family": "Loop-based",
        "principle": "Type-3 SOGI-PLL variant tuned for low-inertia operation.",
        "strength": "Cheap embedded implementation with improved loop shaping.",
        "limitation": "Rarely reaches the front of the benchmark on accuracy.",
    },
    "LKF2": {
        "family": "Model-based",
        "principle": "Linear Kalman filtering with a reduced sinusoidal state model.",
        "strength": "Low compute for a state-space estimator.",
        "limitation": "Less robust than EKF/UKF/RA-EKF on the hardest nonlinear events.",
    },
    "EKF": {
        "family": "Model-based",
        "principle": "Extended Kalman filter for single-phase dynamic frequency tracking.",
        "strength": "Most RMSE wins and strongest overall balance of error and cost.",
        "limitation": "Model mismatch can still hurt under severe composite stress.",
    },
    "UKF": {
        "family": "Model-based",
        "principle": "Unscented Kalman filter with nonlinear sigma-point propagation.",
        "strength": "Best RMSE in the ringdown-noise family.",
        "limitation": "More compute than EKF with no universal advantage.",
    },
    "RA-EKF": {
        "family": "Model-based",
        "principle": "RoCoF-augmented and robustness-tuned extended Kalman filter.",
        "strength": "Best modulation-family RMSE and strong global trade-off profile.",
        "limitation": "Higher cost than the cheapest loop methods.",
    },
    "TKEO": {
        "family": "Adaptive",
        "principle": "Teager-Kaiser energy operator for instantaneous frequency cues.",
        "strength": "Fast lightweight adaptive baseline.",
        "limitation": "Limited robustness across difficult benchmark families.",
    },
    "Prony": {
        "family": "Window-based",
        "principle": "Parametric modal fit from short signal windows.",
        "strength": "Useful harmonic and trip-risk behavior in selected cases.",
        "limitation": "Heavy compute and fragile numerical conditioning.",
    },
    "ESPRIT": {
        "family": "Window-based",
        "principle": "Subspace rotational-invariance spectral estimator.",
        "strength": "Frequent trip-risk leader in difficult scenarios.",
        "limitation": "By far the most expensive routine in the archival run.",
    },
    "Koopman (RK-DPMU)": {
        "family": "Data-driven",
        "principle": "Reduced-order data-driven estimator inspired by robust Koopman models.",
        "strength": "Occasional scenario wins and competitive ramp behavior.",
        "limitation": "High compute without broad global RMSE dominance.",
    },
}


SCENARIO_CATALOG: dict[str, dict[str, str]] = {
    "IBR_Multi_Event": {
        "family": "IBR multi-event",
        "description": "Composite IBR stress sequence combining phase jump, ramp, harmonics, and oscillatory recovery.",
    },
    "IBR_Power_Imbalance_Ringdown": {
        "family": "IBR ringdown",
        "description": "Nominal IBR power-imbalance ringdown with oscillatory recovery.",
    },
    "IBR_Power_Imbalance_Ringdown_Low_Noise": {
        "family": "IBR ringdown",
        "description": "Low-noise ringdown variant with mild interharmonic contamination.",
    },
    "IBR_Power_Imbalance_Ringdown_Normal_Noise": {
        "family": "IBR ringdown",
        "description": "Normal-noise ringdown variant close to the default operating point.",
    },
    "IBR_Power_Imbalance_Ringdown_Medium_Noise": {
        "family": "IBR ringdown",
        "description": "Medium-noise ringdown variant designed to stress loop robustness.",
    },
    "IBR_Power_Imbalance_Ringdown_Severe_Noise": {
        "family": "IBR ringdown",
        "description": "Severe-noise ringdown variant with the strongest interharmonic stress.",
    },
    "IBR_Harmonics_Small": {
        "family": "IBR harmonics",
        "description": "Small harmonic pollution representative of IEEE 519 compliant conditions.",
    },
    "IBR_Harmonics_Medium": {
        "family": "IBR harmonics",
        "description": "Medium harmonic content representative of realistic grid-edge distortion.",
    },
    "IBR_Harmonics_Large": {
        "family": "IBR harmonics",
        "description": "Severe harmonic pollution representative of stressed IBR operation.",
    },
    "IEEE_Freq_Step": {
        "family": "Frequency step",
        "description": "IEC/IEEE frequency step benchmark for abrupt nominal-frequency change.",
    },
    "IEEE_Modulation": {
        "family": "Modulation",
        "description": "Combined amplitude and phase modulation benchmark for measurement bandwidth.",
    },
    "IEEE_Modulation_AM": {
        "family": "Modulation",
        "description": "Pure amplitude modulation benchmark.",
    },
    "IEEE_Modulation_FM": {
        "family": "Modulation",
        "description": "Pure phase/frequency modulation benchmark.",
    },
    "IEEE_OOB_Interference": {
        "family": "OOB interference",
        "description": "Out-of-band interference rejection test.",
    },
    "IEEE_Phase_Jump_20": {
        "family": "Phase jumps",
        "description": "Moderate 20 degree phase discontinuity aligned with IEEE anti-islanding limits.",
    },
    "IEEE_Phase_Jump_60": {
        "family": "Phase jumps",
        "description": "Large 60 degree phase discontinuity stress test.",
    },
    "NERC_Phase_Jump_60": {
        "family": "Phase jumps",
        "description": "Extreme 60 degree islanding jump with additional harmonic stress.",
    },
    "IEEE_Single_SinWave": {
        "family": "Single-tone",
        "description": "Clean steady-state sinusoid used as a deterministic baseline.",
    },
    "IEEE_Mag_Step": {
        "family": "Magnitude steps",
        "description": "Default IEEE magnitude-step benchmark.",
    },
    "IEEE_Mag_Step_1pct": {
        "family": "Magnitude steps",
        "description": "1 percent magnitude change.",
    },
    "IEEE_Mag_Step_5pct": {
        "family": "Magnitude steps",
        "description": "5 percent magnitude change.",
    },
    "IEEE_Mag_Step_10pct": {
        "family": "Magnitude steps",
        "description": "10 percent magnitude change.",
    },
    "IEEE_Mag_Step_15pct": {
        "family": "Magnitude steps",
        "description": "15 percent magnitude change.",
    },
    "IEEE_Mag_Step_25pct": {
        "family": "Magnitude steps",
        "description": "25 percent magnitude change.",
    },
    "IEEE_Mag_Step_50pct": {
        "family": "Magnitude steps",
        "description": "50 percent magnitude change.",
    },
    "IEEE_Freq_Ramp": {
        "family": "IEEE ramps",
        "description": "Default IEEE frequency ramp benchmark.",
    },
    "IEEE_Freq_Ramp_0.25Hzs": {
        "family": "IEEE ramps",
        "description": "0.25 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_0.5Hzs": {
        "family": "IEEE ramps",
        "description": "0.5 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_1Hzs": {
        "family": "IEEE ramps",
        "description": "1 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_2Hzs": {
        "family": "IEEE ramps",
        "description": "2 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_5Hzs": {
        "family": "IEEE ramps",
        "description": "5 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_10Hzs": {
        "family": "IEEE ramps",
        "description": "10 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_15Hzs": {
        "family": "IEEE ramps",
        "description": "15 Hz/s sustained RoCoF.",
    },
    "IEEE_Freq_Ramp_20Hzs": {
        "family": "IEEE ramps",
        "description": "20 Hz/s sustained RoCoF.",
    },
}


FAMILY_ORDER = [
    "IEEE ramps",
    "Magnitude steps",
    "Modulation",
    "Phase jumps",
    "OOB interference",
    "Single-tone",
    "IBR harmonics",
    "IBR ringdown",
    "IBR multi-event",
    "Frequency step",
]

SCENARIO_ORDER = list(SCENARIO_CATALOG.keys())
ESTIMATOR_ORDER = list(ESTIMATOR_CATALOG.keys())
ESTIMATOR_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p", "d", "H", "+"]
FAMILY_DASHBOARD_PAGE_SIZE = 8


ESTIMATOR_FAMILY_COLORS = {
    "Loop-based": "#3b82f6",
    "Window-based": "#10b981",
    "Model-based": "#f59e0b",
    "Adaptive": "#ef4444",
    "Data-driven": "#8b5cf6",
}


SCENARIO_FAMILY_COLORS = {
    "IEEE ramps": "#2563eb",
    "Magnitude steps": "#16a34a",
    "Modulation": "#f59e0b",
    "Phase jumps": "#dc2626",
    "OOB interference": "#9333ea",
    "Single-tone": "#6b7280",
    "IBR harmonics": "#14b8a6",
    "IBR ringdown": "#ea580c",
    "IBR multi-event": "#111827",
    "Frequency step": "#7c3aed",
}


METRIC_CATALOG = [
    ("m1_rmse_hz", "RMSE", "Global accuracy over the evaluation window.", "Lower is better."),
    ("m2_mae_hz", "MAE", "Average absolute error without squaring large excursions.", "Lower is better."),
    ("m3_max_peak_hz", "Peak error", "Worst instantaneous frequency error.", "Lower is better."),
    ("m5_trip_risk_s", "Trip-risk duration", "Total time above the 0.5 Hz relay deadband.", "Lower is better."),
    ("m8_settling_time_s", "Settling time", "Time until the estimate stays within the tolerance band.", "Lower is better."),
    ("m13_cpu_time_us", "CPU time", "Average per-sample process-time cost.", "Lower is better."),
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def format_float(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def format_p_value(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if value < 1e-16:
        return r"$<10^{-16}$"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def scenario_family(name: str) -> str:
    item = SCENARIO_CATALOG.get(name)
    if item:
        return item["family"]
    raise KeyError(f"Scenario missing from catalog: {name}")


def short_scenario_label(name: str) -> str:
    mapping = {
        "IBR_Multi_Event": "IBR multi-event",
        "IBR_Power_Imbalance_Ringdown": "Ringdown base",
        "IBR_Power_Imbalance_Ringdown_Low_Noise": "Ringdown low-noise",
        "IBR_Power_Imbalance_Ringdown_Normal_Noise": "Ringdown normal-noise",
        "IBR_Power_Imbalance_Ringdown_Medium_Noise": "Ringdown medium-noise",
        "IBR_Power_Imbalance_Ringdown_Severe_Noise": "Ringdown severe-noise",
        "IBR_Harmonics_Small": "IBR harmonics small",
        "IBR_Harmonics_Medium": "IBR harmonics medium",
        "IBR_Harmonics_Large": "IBR harmonics large",
        "IEEE_Freq_Step": "IEEE freq step",
        "IEEE_Modulation": "IEEE modulation",
        "IEEE_Modulation_AM": "IEEE mod AM",
        "IEEE_Modulation_FM": "IEEE mod FM",
        "IEEE_OOB_Interference": "IEEE OOB",
        "IEEE_Phase_Jump_20": "IEEE jump 20",
        "IEEE_Phase_Jump_60": "IEEE jump 60",
        "NERC_Phase_Jump_60": "NERC jump 60",
        "IEEE_Single_SinWave": "Single tone",
        "IEEE_Mag_Step": "Mag step base",
        "IEEE_Mag_Step_1pct": "Mag step 1%",
        "IEEE_Mag_Step_5pct": "Mag step 5%",
        "IEEE_Mag_Step_10pct": "Mag step 10%",
        "IEEE_Mag_Step_15pct": "Mag step 15%",
        "IEEE_Mag_Step_25pct": "Mag step 25%",
        "IEEE_Mag_Step_50pct": "Mag step 50%",
        "IEEE_Freq_Ramp": "Ramp base",
        "IEEE_Freq_Ramp_0.25Hzs": "Ramp 0.25",
        "IEEE_Freq_Ramp_0.5Hzs": "Ramp 0.5",
        "IEEE_Freq_Ramp_1Hzs": "Ramp 1",
        "IEEE_Freq_Ramp_2Hzs": "Ramp 2",
        "IEEE_Freq_Ramp_5Hzs": "Ramp 5",
        "IEEE_Freq_Ramp_10Hzs": "Ramp 10",
        "IEEE_Freq_Ramp_15Hzs": "Ramp 15",
        "IEEE_Freq_Ramp_20Hzs": "Ramp 20",
    }
    return mapping.get(name, name.replace("_", " "))


def estimator_color_map() -> dict[str, str]:
    tab20 = list(matplotlib.colormaps["tab20"].colors)
    return {
        estimator: matplotlib.colors.to_hex(tab20[idx % len(tab20)])
        for idx, estimator in enumerate(ESTIMATOR_ORDER)
    }


def estimator_marker_map() -> dict[str, str]:
    return {
        estimator: ESTIMATOR_MARKERS[idx % len(ESTIMATOR_MARKERS)]
        for idx, estimator in enumerate(ESTIMATOR_ORDER)
    }


def ordered_family_scenarios(df: pd.DataFrame, family: str) -> list[str]:
    present = set(df.loc[df["scenario_family"] == family, "scenario"].unique().tolist())
    return [scenario for scenario in SCENARIO_ORDER if scenario in present]


def load_report() -> tuple[dict, pd.DataFrame]:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    df = pd.DataFrame(report["aggregated_metrics"])
    df["family_norm"] = df["estimator"].map(lambda x: ESTIMATOR_CATALOG[str(x)]["family"])
    df["scenario_family"] = df["scenario"].map(scenario_family)
    return report, df


def build_facts(report: dict, df: pd.DataFrame) -> None:
    unique_scenarios = sorted(df["scenario"].unique())
    total_records = len(report.get("raw_run_records", []))
    pareto_estimators = [p["estimator"] for p in report["advanced_analysis"]["pareto_front"]["points"] if p["is_pareto"]]
    facts = (
        f"% Auto-generated from {REPORT_JSON.name}\n"
        f"\\newcommand{{\\ArchivalScenarioCount}}{{{len(unique_scenarios)}}}\n"
        f"\\newcommand{{\\ArchivalEstimatorCount}}{{{df['estimator'].nunique()}}}\n"
        f"\\newcommand{{\\ArchivalPairCount}}{{{len(df)}}}\n"
        f"\\newcommand{{\\ArchivalMonteCarloRuns}}{{{report['run_configuration']['n_mc_runs']}}}\n"
        f"\\newcommand{{\\ArchivalTuningTrials}}{{{report['run_configuration']['n_trials_tuning']}}}\n"
        f"\\newcommand{{\\ArchivalTotalRecords}}{{{total_records}}}\n"
        f"\\newcommand{{\\ArchivalParetoSet}}{{{latex_escape(', '.join(pareto_estimators[:-1]) + ', and ' + pareto_estimators[-1])}}}\n"
    )
    (GENERATED_DIR / "facts.tex").write_text(facts, encoding="utf-8")


def build_metric_table() -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Headline metrics analyzed in the archival benchmark report.}",
        r"\label{tab:metric_catalog}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{l l L{5.1cm} L{3.2cm}}",
        r"\toprule",
        r"Code & Name & What it measures & Interpretation \\",
        r"\midrule",
    ]
    for code, name, meaning, interpretation in METRIC_CATALOG:
        lines.append(
            f"{latex_escape(code)} & {latex_escape(name)} & {latex_escape(meaning)} & {latex_escape(interpretation)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (GENERATED_DIR / "metric_catalog.tex").write_text("\n".join(lines), encoding="utf-8")


def build_family_summary(df: pd.DataFrame) -> None:
    rows: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Family-level RMSE and trip-risk summary from the archival 34-scenario report. RMSE values are family means of scenario means and are reported in mHz. The uncertainty term is the winner's mean within-scenario Monte Carlo standard deviation across the family.}",
        r"\label{tab:family_summary_extended}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{l c l c l c l}",
        r"\toprule",
        r"Family & Count & RMSE winner & RMSE [mHz] & Runner-up & $\Delta$ [mHz] & Lowest $T_{\mathrm{trip}}$ [s] \\",
        r"\midrule",
    ]
    for family in FAMILY_ORDER:
        fam_df = df[df["scenario_family"] == family]
        by_est = fam_df.groupby("estimator").agg(
            rmse_mean=("m1_rmse_hz_mean", "mean"),
            rmse_std=("m1_rmse_hz_std", "mean"),
            trip_mean=("m5_trip_risk_s_mean", "mean"),
        ).sort_values("rmse_mean")
        winner = by_est.index[0]
        runner = by_est.index[1]
        delta = (by_est.iloc[1]["rmse_mean"] - by_est.iloc[0]["rmse_mean"]) * 1000.0
        trip_best = by_est["trip_mean"].min()
        trip_winners = [idx for idx, row in by_est.iterrows() if math.isclose(row["trip_mean"], trip_best, rel_tol=1e-9, abs_tol=1e-9)]
        if len(trip_winners) == 1:
            trip_label = f"{trip_winners[0]}: {format_float(trip_best, 4)}"
        elif math.isclose(trip_best, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            trip_label = f"0 ({len(trip_winners)}-way tie)"
        else:
            trip_label = f"{format_float(trip_best, 4)} ({len(trip_winners)}-way tie)"
        rmse_label = f"${format_float(by_est.iloc[0]['rmse_mean'] * 1000.0, 3)} \\pm {format_float(by_est.iloc[0]['rmse_std'] * 1000.0, 3)}$"
        rows.append(
            f"{latex_escape(family)} & {fam_df['scenario'].nunique()} & {latex_escape(winner)} & {rmse_label} & {latex_escape(runner)} & {format_float(delta, 3)} & {latex_escape(trip_label)} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (GENERATED_DIR / "family_summary.tex").write_text("\n".join(rows), encoding="utf-8")


def build_pareto_table(report: dict) -> None:
    points = [p for p in report["advanced_analysis"]["pareto_front"]["points"] if p["is_pareto"]]
    points.sort(key=lambda row: row["rmse_hz_mean"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Global RMSE-versus-CPU Pareto set extracted from the archival report.}",
        r"\label{tab:pareto_set}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{l l c c}",
        r"\toprule",
        r"Estimator & Family & RMSE [mHz] & CPU [$\mu$s] \\",
        r"\midrule",
    ]
    for row in points:
        family = ESTIMATOR_CATALOG[row["estimator"]]["family"]
        lines.append(
            f"{latex_escape(row['estimator'])} & {latex_escape(family)} & {format_float(row['rmse_hz_mean'] * 1000.0, 2)} & {format_float(row['cpu_time_us_mean'], 3)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (GENERATED_DIR / "pareto_set.tex").write_text("\n".join(lines), encoding="utf-8")


def build_hypothesis_table(df: pd.DataFrame) -> None:
    if scipy_stats is None:
        raise RuntimeError("scipy is required to build automatic hypothesis tables.")
    metric_map = {
        "m1_rmse_hz_mean": "RMSE",
        "m2_mae_hz_mean": "MAE",
        "m3_max_peak_hz_mean": "Peak error",
        "m5_trip_risk_s_mean": "Trip-risk",
        "m8_settling_time_s_mean": "Settling time",
        "m13_cpu_time_us_mean": "CPU time",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Automatically generated omnibus family-comparison tests computed from the archival scenario summaries after relabeling estimators with the current taxonomy. The null hypothesis for every row is that estimator-family distributions are equal for the reported metric.}",
        r"\label{tab:auto_hypotheses}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Metric & Families & ANOVA $F$ & $p$ & Kruskal $H$ & $p$ \\",
        r"\midrule",
    ]
    for col, label in metric_map.items():
        grouped = []
        family_count = 0
        for _, fam_df in df.groupby("family_norm"):
            values = fam_df[col].dropna().to_numpy(dtype=float)
            if len(values) >= 2:
                grouped.append(values)
                family_count += 1
        if len(grouped) < 2:
            continue
        f_stat, p_anova = scipy_stats.f_oneway(*grouped)
        h_stat, p_kruskal = scipy_stats.kruskal(*grouped)
        lines.append(
            f"{latex_escape(label)} & {family_count} & {format_float(float(f_stat), 2)} & {format_p_value(float(p_anova))} & {format_float(float(h_stat), 2)} & {format_p_value(float(p_kruskal))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (GENERATED_DIR / "auto_hypotheses.tex").write_text("\n".join(lines), encoding="utf-8")


def build_estimator_catalog(df: pd.DataFrame) -> None:
    rmse_winners = Counter()
    for scenario, sc_df in df.groupby("scenario"):
        best = sc_df["m1_rmse_hz_mean"].min()
        winners = sc_df.loc[sc_df["m1_rmse_hz_mean"].apply(lambda x: math.isclose(x, best, rel_tol=1e-12, abs_tol=1e-12)), "estimator"]
        rmse_winners.update(winners.tolist())

    lines = [
        r"\begin{longtable}{L{2.25cm} L{1.55cm} L{3.2cm} L{3.6cm} L{3.8cm} c}",
        r"\caption{Estimator catalog for the archival report. RMSE wins count exact scenario-level wins in the 34-scenario benchmark.}\label{tab:estimator_catalog}\\",
        r"\scriptsize\\",
        r"\toprule",
        r"Estimator & Family & Principle & Where it helps & Main trade-off & RMSE wins \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Estimator & Family & Principle & Where it helps & Main trade-off & RMSE wins \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for estimator in sorted(ESTIMATOR_CATALOG):
        item = ESTIMATOR_CATALOG[estimator]
        lines.append(
            f"{latex_escape(estimator)} & {latex_escape(item['family'])} & {latex_escape(item['principle'])} & {latex_escape(item['strength'])} & {latex_escape(item['limitation'])} & {rmse_winners.get(estimator, 0)} \\\\"
        )
    lines.append(r"\end{longtable}")
    lines.append("")
    (GENERATED_DIR / "estimator_catalog.tex").write_text("\n".join(lines), encoding="utf-8")


def build_scenario_catalog(df: pd.DataFrame) -> None:
    difficulty = {}
    for scenario, sc_df in df.groupby("scenario"):
        difficulty[scenario] = median(sc_df["m1_rmse_hz_mean"].tolist()) * 1000.0

    lines = [
        r"\begin{longtable}{L{3.2cm} L{1.9cm} L{6.8cm} c}",
        r"\caption{Scenario catalog for the archival benchmark report. Difficulty is the median scenario RMSE across estimators in mHz.}\label{tab:scenario_catalog}\\",
        r"\scriptsize\\",
        r"\toprule",
        r"Scenario ID & Family & Benchmark purpose & Median RMSE [mHz] \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Scenario ID & Family & Benchmark purpose & Median RMSE [mHz] \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for family in FAMILY_ORDER:
        family_scenarios = sorted([name for name, info in SCENARIO_CATALOG.items() if info["family"] == family])
        for scenario in family_scenarios:
            if scenario not in difficulty:
                continue
            scenario_label = latex_escape(scenario).replace(r"\_", r"\allowbreak\_")
            lines.append(
                f"\\texttt{{{scenario_label}}} & {latex_escape(family)} & {latex_escape(SCENARIO_CATALOG[scenario]['description'])} & {format_float(difficulty[scenario], 2)} \\\\"
            )
    lines.append(r"\end{longtable}")
    lines.append("")
    (GENERATED_DIR / "scenario_catalog.tex").write_text("\n".join(lines), encoding="utf-8")


def build_summary_json(report: dict, df: pd.DataFrame) -> None:
    rmse_rank_top = report["advanced_analysis"]["rankings"]["RMSE_Hz"][:5]
    difficulty_top = report["advanced_analysis"]["trends"]["scenario_difficulty_by_median_rmse"][:8]
    fastest_top = report["advanced_analysis"]["trends"]["global_fastest_estimators_cpu"][:5]
    payload = {
        "n_scenarios": int(df["scenario"].nunique()),
        "n_estimators": int(df["estimator"].nunique()),
        "n_pairs": int(len(df)),
        "n_mc_runs": int(report["run_configuration"]["n_mc_runs"]),
        "n_trials_tuning": int(report["run_configuration"]["n_trials_tuning"]),
        "raw_run_records": int(len(report.get("raw_run_records", []))),
        "pareto_estimators": [p["estimator"] for p in report["advanced_analysis"]["pareto_front"]["points"] if p["is_pareto"]],
        "top_rmse_mean_rank": rmse_rank_top,
        "hardest_scenarios_by_median_rmse": difficulty_top,
        "fastest_estimators_cpu": fastest_top,
    }
    (GENERATED_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_extra_plots(report: dict, df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    scenario_difficulty = (
        df.groupby(["scenario", "scenario_family"])["m1_rmse_hz_mean"]
        .median()
        .reset_index(name="median_rmse_hz")
        .sort_values("median_rmse_hz", ascending=False)
    )
    top_difficulty = scenario_difficulty.head(12).copy()

    estimator_global = (
        df.groupby(["estimator", "family_norm"])
        .agg(
            rmse_mhz=("m1_rmse_hz_mean", lambda s: float(np.mean(s) * 1000.0)),
            trip_s=("m5_trip_risk_s_mean", "mean"),
            cpu_us=("m13_cpu_time_us_mean", "mean"),
        )
        .reset_index()
    )
    rmse_rank = {
        row["estimator"]: float(row["mean_rank"])
        for row in report["advanced_analysis"]["rankings"]["RMSE_Hz"]
    }
    estimator_global["rmse_rank"] = estimator_global["estimator"].map(rmse_rank)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    top_labels = [short_scenario_label(x) for x in top_difficulty["scenario"].tolist()]
    top_values = (top_difficulty["median_rmse_hz"] * 1000.0).tolist()
    top_colors = [SCENARIO_FAMILY_COLORS[fam] for fam in top_difficulty["scenario_family"].tolist()]
    ax1.barh(
        list(reversed(top_labels)),
        list(reversed(top_values)),
        color=list(reversed(top_colors)),
        edgecolor="black",
        linewidth=0.4,
    )
    ax1.set_title("Top archival scenarios by median RMSE")
    ax1.set_xlabel("Median scenario RMSE [mHz]")
    ax1.grid(axis="x", alpha=0.25)

    family_difficulty = []
    family_labels = []
    family_box_colors = []
    for family in FAMILY_ORDER:
        vals = scenario_difficulty.loc[scenario_difficulty["scenario_family"] == family, "median_rmse_hz"] * 1000.0
        if len(vals) == 0:
            continue
        family_difficulty.append(vals.to_numpy(dtype=float))
        family_labels.append(family)
        family_box_colors.append(SCENARIO_FAMILY_COLORS[family])
    box = ax2.boxplot(family_difficulty, patch_artist=True, tick_labels=family_labels, showfliers=False)
    for patch, color in zip(box["boxes"], family_box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)
    ax2.set_yscale("log")
    ax2.set_title("Scenario difficulty by family")
    ax2.set_ylabel("Median scenario RMSE [mHz]")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", alpha=0.25, which="both")

    for idx, row in estimator_global.iterrows():
        color = ESTIMATOR_FAMILY_COLORS[row["family_norm"]]
        ax3.scatter(row["rmse_mhz"], row["trip_s"], s=72, color=color, edgecolor="black", linewidth=0.4, alpha=0.95)
        ax3.annotate(
            row["estimator"],
            (row["rmse_mhz"], row["trip_s"]),
            textcoords="offset points",
            xytext=(5 + (idx % 3) * 2, 4 - (idx % 4) * 3),
            fontsize=7,
        )
    ax3.set_xscale("log")
    ax3.set_title("Global RMSE versus trip-risk")
    ax3.set_xlabel("Mean RMSE [mHz]")
    ax3.set_ylabel("Mean trip-risk [s]")
    ax3.grid(alpha=0.25, which="both")

    for idx, row in estimator_global.iterrows():
        color = ESTIMATOR_FAMILY_COLORS[row["family_norm"]]
        ax4.scatter(row["cpu_us"], row["rmse_rank"], s=72, color=color, edgecolor="black", linewidth=0.4, alpha=0.95)
        ax4.annotate(
            row["estimator"],
            (row["cpu_us"], row["rmse_rank"]),
            textcoords="offset points",
            xytext=(5 + (idx % 3) * 2, 4 - (idx % 4) * 3),
            fontsize=7,
        )
    ax4.set_xscale("log")
    ax4.invert_yaxis()
    ax4.set_title("Consistency versus compute")
    ax4.set_xlabel("Mean CPU time [$\\mu$s]")
    ax4.set_ylabel("Mean RMSE rank (lower is better)")
    ax4.grid(alpha=0.25, which="both")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=family, markerfacecolor=color, markeredgecolor="black", markersize=7)
        for family, color in ESTIMATOR_FAMILY_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("Additional plots generated directly from the archival benchmark JSON", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    png_path = GENERATED_DIR / "archival_json_panels.png"
    pdf_path = GENERATED_DIR / "archival_json_panels.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_family_dashboard_pages(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
        }
    )

    estimator_colors = estimator_color_map()
    estimator_markers = estimator_marker_map()
    families = [family for family in FAMILY_ORDER if (df["scenario_family"] == family).any()]
    page_chunks = [
        families[idx : idx + FAMILY_DASHBOARD_PAGE_SIZE]
        for idx in range(0, len(families), FAMILY_DASHBOARD_PAGE_SIZE)
    ]

    tex_lines = []
    total_pages = len(page_chunks)
    floor_mhz = 1e-3

    for page_idx, family_chunk in enumerate(page_chunks, start=1):
        fig, axes = plt.subplots(4, 2, figsize=(10.6, 13.8))
        axes = axes.ravel()
        page_estimators: list[str] = []

        for ax_idx, family in enumerate(family_chunk):
            ax = axes[ax_idx]
            fam_df = df[df["scenario_family"] == family].copy()
            scenario_order = ordered_family_scenarios(df, family)
            by_estimator = (
                fam_df.groupby("estimator")
                .agg(
                    rmse_mean=("m1_rmse_hz_mean", "mean"),
                    trip_mean=("m5_trip_risk_s_mean", "mean"),
                )
                .sort_values(["rmse_mean", "trip_mean"])
            )
            top_estimators = by_estimator.head(5).index.tolist()
            page_estimators.extend([est for est in top_estimators if est not in page_estimators])
            x = np.arange(len(scenario_order))

            for estimator in top_estimators:
                est_df = (
                    fam_df[fam_df["estimator"] == estimator]
                    .set_index("scenario")
                    .reindex(scenario_order)
                )
                values_mhz = np.maximum(
                    est_df["m1_rmse_hz_mean"].to_numpy(dtype=float) * 1000.0,
                    floor_mhz,
                )
                ax.plot(
                    x,
                    values_mhz,
                    marker=estimator_markers[estimator],
                    linewidth=1.6,
                    markersize=4.3,
                    color=estimator_colors[estimator],
                    label=estimator,
                )

            background = SCENARIO_FAMILY_COLORS[family]
            ax.set_facecolor(matplotlib.colors.to_rgba(background, 0.06))
            winner = top_estimators[0]
            ax.set_title(f"{family} ({len(scenario_order)} scenarios)", loc="left", fontweight="bold")
            ax.text(
                0.01,
                0.02,
                f"Top RMSE: {winner}",
                transform=ax.transAxes,
                fontsize=7.6,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": background,
                    "linewidth": 0.8,
                    "alpha": 0.95,
                },
            )
            ax.set_yscale("log")
            ax.set_ylabel("RMSE [mHz]")
            ax.set_xticks(x)
            ax.set_xticklabels([short_scenario_label(item) for item in scenario_order], rotation=28, ha="right")
            ax.grid(axis="y", which="both", alpha=0.22)
            ax.margins(x=0.03)

        for ax in axes[len(family_chunk) :]:
            ax.axis("off")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=estimator_colors[estimator],
                marker=estimator_markers[estimator],
                linewidth=1.6,
                markersize=5,
                label=estimator,
            )
            for estimator in page_estimators
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, max(1, math.ceil(len(legend_handles) / 2))),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )
        fig.suptitle(
            f"Family-resolved RMSE atlas from the archival benchmark JSON (page {page_idx}/{total_pages})",
            fontsize=12,
            y=0.988,
        )
        fig.tight_layout(rect=[0.03, 0.07, 0.99, 0.965])

        stem = f"family_dashboard_page_{page_idx}"
        png_path = GENERATED_DIR / f"{stem}.png"
        pdf_path = GENERATED_DIR / f"{stem}.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        family_caption = ", ".join(family_chunk)
        tex_lines.extend(
            [
                r"\begin{figure}[p]",
                r"\centering",
                rf"\includegraphics[width=\textwidth,height=0.92\textheight,keepaspectratio]{{{stem}.pdf}}",
                (
                    rf"\caption{{Family-resolved RMSE atlas from the archival JSON report (page {page_idx}/{total_pages}). "
                    r"Each panel keeps one scenario family separate, including the small baseline families, and plots the "
                    r"five lowest-RMSE estimators for that family across its archived scenarios. The vertical axis is "
                    r"log-scaled in mHz. Families on this page: "
                    rf"{latex_escape(family_caption)}.}}"
                ),
                rf"\label{{fig:{stem}}}",
                r"\end{figure}",
                "",
            ]
        )

    (GENERATED_DIR / "family_dashboard_pages.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    report, df = load_report()
    build_facts(report, df)
    build_metric_table()
    build_family_summary(df)
    build_pareto_table(report)
    build_hypothesis_table(df)
    build_estimator_catalog(df)
    build_scenario_catalog(df)
    build_summary_json(report, df)
    build_extra_plots(report, df)
    build_family_dashboard_pages(df)


if __name__ == "__main__":
    main()
