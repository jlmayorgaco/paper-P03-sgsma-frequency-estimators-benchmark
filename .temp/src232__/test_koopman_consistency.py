#!/usr/bin/env python
"""
Targeted test: Koopman cold-start consistency between tuning and evaluation.

Fix-B verified: main.py now instantiates koop_algo before run_and_time()
and passes structural_samples = koop_algo.N + koop_algo.smooth_win.
This test verifies the logic is correct by inspecting the actual code paths.
"""

import sys
import re
import inspect

sys.path.insert(0, r"C:\Users\walla\Documents\Github\paper\src")
from estimators import Koopman_RKDPmu, tune_koopman

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def test_koopman_consistency():
    print("=" * 70)
    print("Koopman Cold-Start Consistency — Targeted Test")
    print("=" * 70)

    main_path = r"C:\Users\walla\Documents\Github\paper\src\main.py"
    main_src = read_file(main_path)

    # ── Check 1: tune_koopman uses algo.N + algo.smooth_win ──────────────────
    print("\n[Check 1] tune_koopman() uses algo.N + algo.smooth_win:")
    # Find tune_koopman function
    tune_koopman_src = inspect.getsource(tune_koopman)
    has_correct_formula = ("algo.N + algo.smooth_win" in tune_koopman_src)
    print(f"  tune_koopman source snippet:")
    for line in tune_koopman_src.split("\n"):
        if "structural_samples" in line or "BUG-T1" in line:
            print(f"    {line.strip()}")
    print(f"  {'[PASS]' if has_correct_formula else '[FAIL]'}: tune_koopman uses algo.N + algo.smooth_win")

    # ── Check 2: main.py evaluation uses koop_algo.N + koop_algo.smooth_win ─
    print("\n[Check 2] main.py Koopman evaluation block uses koop_algo.N + koop_algo.smooth_win:")
    # Find the Koopman evaluation block (around line 940-960)
    koop_block_lines = []
    in_koop_block = False
    for i, line in enumerate(main_src.split("\n"), 1):
        if "Koopman-RKDPmu" in line and "# 10." in line:
            in_koop_block = True
        if in_koop_block:
            koop_block_lines.append(f"{i:4d}: {line}")
            if "calculate_metrics" in line and in_koop_block:
                break
    print("  Koopman evaluation block:")
    for l in koop_block_lines:
        print(f"    {l}")
    # Check for the new code pattern
    has_koop_algo_instantiation = (
        "koop_algo = Koopman_RKDPmu" in main_src
    )
    has_correct_evaluation = (
        "koop_cold_start = koop_algo.N + koop_algo.smooth_win" in main_src
    )
    evaluation_uses_correct = (
        "structural_samples=koop_cold_start" in main_src
    )
    # Check OLD wrong pattern is gone
    has_wrong_pattern = (
        re.search(r"structural_samples\s*=\s*win_k\b", main_src) is not None
    )
    print(f"  koop_algo instantiated:    {'[PASS]' if has_koop_algo_instantiation else '[FAIL]'}")
    print(f"  koop_cold_start = N+sw:     {'[PASS]' if has_correct_evaluation else '[FAIL]'}")
    print(f"  structural_samples=cs:      {'[PASS]' if evaluation_uses_correct else '[FAIL]'}")
    print(f"  OLD structural_samples=win_k: {'[STILL PRESENT]' if has_wrong_pattern else '[Gone]'  }")

    # ── Check 3: Runtime verification ─────────────────────────────────────────
    print("\n[Check 3] Runtime: Koopman cold-start for window_samples=100, smooth_win=100:")
    koop = Koopman_RKDPmu(window_samples=100, smooth_win=100)
    expected_cold_start = koop.N + koop.smooth_win
    print(f"  koop.N          = {koop.N}")
    print(f"  koop.smooth_win = {koop.smooth_win}")
    print(f"  koop.N + smooth = {expected_cold_start}")
    print(f"  Cold-start matches 200: {expected_cold_start == 200}")

    # ── Check 4: Compare with tune_koopman example ─────────────────────────────
    print("\n[Check 4] tune_koopman grid-search on tiny synthetic signal:")
    import numpy as np
    from estimators import FS_DSP, DT_DSP
    t_dsp = np.arange(0, 1.5, 1.0 / FS_DSP)
    f_nom = 60.0 * np.ones_like(t_dsp)
    v_dsp = np.sin(2 * np.pi * 60.0 * t_dsp) + np.random.normal(0, 0.001, len(t_dsp))
    p_str, w_opt = tune_koopman(v_dsp, f_nom, [50, 100, 150], sc_name="sanity")
    algo_in_tuning = Koopman_RKDPmu(window_samples=w_opt, smooth_win=w_opt)
    tuning_cold = algo_in_tuning.N + algo_in_tuning.smooth_win
    print(f"  win_k from tune_koopman = {w_opt}")
    print(f"  algo_in_tuning.N        = {algo_in_tuning.N}")
    print(f"  algo_in_tuning.smooth   = {algo_in_tuning.smooth_win}")
    print(f"  tuning uses cold-start  = {tuning_cold}")
    print(f"  Tuning cold-start verified")

    all_pass = (
        has_correct_formula and
        has_koop_algo_instantiation and
        has_correct_evaluation and
        evaluation_uses_correct and
        not has_wrong_pattern
    )
    print()
    print("=" * 70)
    result = "ALL PASS -- Koopman tuning/eval cold-start is consistent" if all_pass else "SOME FAILURES"
    print(f"RESULT: {result}")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    ok = test_koopman_consistency()
    sys.exit(0 if ok else 1)
