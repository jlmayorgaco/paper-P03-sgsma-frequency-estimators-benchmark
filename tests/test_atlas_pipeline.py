from __future__ import annotations

import pandas as pd

from pipelines import atlas_sweep


def test_atlas_builds_three_sweep_families(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_MAG_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_ROCOF_LEVELS_HZ_S", "0.5")
    monkeypatch.setenv("ATLAS_FREQSTEP_LEVELS_HZ", "0.1")
    scenarios = atlas_sweep.build_atlas_scenarios(["magnitude_step", "rocof", "frequency_step"])

    assert len(scenarios) == 6
    assert {item.sweep_key for item in scenarios} == {"magnitude_step", "rocof", "frequency_step"}
    assert {item.direction for item in scenarios} == {"pos", "neg"}
    assert all(item.scenario_name.startswith("Atlas_") for item in scenarios)


def test_atlas_parser_accepts_oracle_policy() -> None:
    parser = atlas_sweep.build_parser()
    args = parser.parse_args(["--sweeps", "rocof", "--policy", "oracle", "--n-runs", "2"])

    assert args.sweeps == "rocof"
    assert args.policy == "oracle"
    assert args.n_runs == 2


def test_atlas_accepts_phase_modulation_aliases(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_PHASE_JUMP_LEVELS_DEG", "20")
    monkeypatch.setenv("ATLAS_FM_MOD_FREQ_LEVELS_HZ", "2")

    scenarios = atlas_sweep.build_atlas_scenarios(["phase_jump", "modlation_fm_sweep"])

    assert {item.sweep_key for item in scenarios} == {"phase_jump_sweep", "modulation_fm_sweep"}


def test_atlas_builds_p0_nondirectional_sweeps(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_HARMONICS_THD_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_INTERHARMONIC_LEVELS_PCT", "2")
    monkeypatch.setenv("ATLAS_NOISE_SIGMA_LEVELS_PU", "0.001")

    scenarios = atlas_sweep.build_atlas_scenarios(["harmonics", "interharmonics", "noise_snr"])

    assert len(scenarios) == 3
    assert {item.sweep_key for item in scenarios} == {"harmonics", "interharmonics", "noise_snr"}
    assert {item.direction for item in scenarios} == {"level"}
    assert all(item.scenario_name.startswith("Atlas_") for item in scenarios)


def test_atlas_builds_phase_and_modulation_sweeps(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_PHASE_JUMP_LEVELS_DEG", "20")
    monkeypatch.setenv("ATLAS_AM_MOD_FREQ_LEVELS_HZ", "2")
    monkeypatch.setenv("ATLAS_FM_MOD_FREQ_LEVELS_HZ", "2")

    scenarios = atlas_sweep.build_atlas_scenarios(
        ["phase_jump_sweep", "modulation_am_sweep", "modulation_fm_sweep"]
    )

    assert len(scenarios) == 4
    assert {item.sweep_key for item in scenarios} == {
        "phase_jump_sweep",
        "modulation_am_sweep",
        "modulation_fm_sweep",
    }
    assert {item.direction for item in scenarios if item.sweep_key == "phase_jump_sweep"} == {"pos", "neg"}
    assert {item.direction for item in scenarios if item.sweep_key != "phase_jump_sweep"} == {"level"}
    phase = [item for item in scenarios if item.sweep_key == "phase_jump_sweep" and item.direction == "pos"][0]
    am = [item for item in scenarios if item.sweep_key == "modulation_am_sweep"][0]
    fm = [item for item in scenarios if item.sweep_key == "modulation_fm_sweep"][0]
    assert phase.params["abs_phase_jump_deg"] == 20.0
    assert phase.scenario_cls.get_default_params()["phase_jump_rad"] > 0.0
    assert am.scenario_cls.get_default_params()["kx"] == 0.10
    assert am.scenario_cls.get_default_params()["fm_hz"] == 2.0
    assert fm.scenario_cls.get_default_params()["fm_hz"] == 2.0
    assert fm.scenario_cls.get_default_params()["ka"] == 0.10


def test_atlas_p0_scenarios_isolate_primary_disturbance(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_HARMONICS_THD_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_INTERHARMONIC_LEVELS_PCT", "2")
    monkeypatch.setenv("ATLAS_NOISE_SIGMA_LEVELS_PU", "0.001")
    monkeypatch.setenv("ATLAS_PHASE_JUMP_LEVELS_DEG", "20")
    monkeypatch.setenv("ATLAS_AM_MOD_FREQ_LEVELS_HZ", "2")
    monkeypatch.setenv("ATLAS_FM_MOD_FREQ_LEVELS_HZ", "2")

    scenarios = {item.sweep_key: item for item in atlas_sweep.build_atlas_scenarios(["p0"])}
    harmonics = scenarios["harmonics"].scenario_cls.get_default_params()
    interharmonics = scenarios["interharmonics"].scenario_cls.get_default_params()
    noise = scenarios["noise_snr"].scenario_cls.get_default_params()
    phase = scenarios["phase_jump_sweep"].scenario_cls.get_default_params()
    am = scenarios["modulation_am_sweep"].scenario_cls.get_default_params()
    fm = scenarios["modulation_fm_sweep"].scenario_cls.get_default_params()

    assert harmonics["freq_step_hz"] == 0.0
    assert harmonics["ih325_pct"] == 0.0
    assert harmonics["ih85_pct"] == 0.0
    assert harmonics["impulse_prob"] == 0.0
    assert harmonics["white_noise_sigma"] == 0.0
    assert "phase_rad" in scenarios["harmonics"].scenario_cls.get_monte_carlo_space()

    assert interharmonics["rocof_hz_s"] == 0.0
    assert interharmonics["h5_pct"] == 0.0
    assert interharmonics["ih75_pct"] == 0.02
    assert interharmonics["white_noise_sigma"] == 0.0
    assert "phase_rad" in scenarios["interharmonics"].scenario_cls.get_monte_carlo_space()

    assert noise["freq_hz"] == 60.0
    assert noise["noise_sigma"] == 0.001

    assert phase["freq_hz"] == 60.0
    assert "phase_rad" in scenarios["phase_jump_sweep"].scenario_cls.get_monte_carlo_space()
    assert am["freq_nom_hz"] == 60.0
    assert am["kx"] == 0.10
    assert am["fm_hz"] == 2.0
    assert fm["freq_nom_hz"] == 60.0
    assert fm["fm_hz"] == 2.0
    assert fm["ka"] == 0.10


def test_atlas_readiness_marks_preview_as_diagnostic() -> None:
    df = pd.DataFrame(
        [
            {
                "sweep_key": "harmonics",
                "estimator": "EKF",
                "policy": "default",
                "n_mc_runs": 1,
                "direction": "level",
                "thd_percent": 5.0,
            }
        ]
    )

    report = atlas_sweep.build_atlas_readiness_report(df, {"policy": "default", "n_cost_reps": 1})

    assert report["status"] == "diagnostic"
    assert report["paper_claims_allowed"] is False
    assert {issue["code"] for issue in report["issues"]} >= {
        "missing_required_sweeps",
        "missing_canonical_estimators",
        "insufficient_monte_carlo_runs",
        "policy_not_paper_ready",
    }


def test_atlas_readiness_accepts_full_fixed_policy_paper_grade() -> None:
    rows = []
    canonical_estimators = atlas_sweep._csv(atlas_sweep.CANONICAL_ESTIMATORS)
    for sweep_key in atlas_sweep.REQUIRED_ATLAS_SWEEPS:
        spec = atlas_sweep.SWEEP_SPECS[sweep_key]
        directions = ["pos", "neg"] if spec.directional else ["level"]
        for level in [1.0, 2.0, 3.0, 4.0]:
            for direction in directions:
                for estimator in canonical_estimators:
                    rows.append(
                        {
                            "sweep_key": sweep_key,
                            "estimator": estimator,
                            "policy": "fixed_policy",
                            "n_mc_runs": atlas_sweep.PAPER_GRADE_MIN_RUNS,
                            "direction": direction,
                            spec.x_col: level,
                        }
                    )
    df = pd.DataFrame(rows)

    report = atlas_sweep.build_atlas_readiness_report(df, {"policy": "fixed_policy", "n_cost_reps": 3})

    assert report["status"] == "paper_grade"
    assert report["paper_claims_allowed"] is True
    assert report["journal_claims_allowed"] is False
    assert not [issue for issue in report["issues"] if issue["severity"] == "blocker"]
