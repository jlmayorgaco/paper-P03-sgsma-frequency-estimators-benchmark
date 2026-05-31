#!/usr/bin/env python3
# Generates slides_guion.tex: each page = rendered slide (left) + English speaker
# script (right) for plane review. Frame images come from build/guion/g-NN.png
# (render with: pdftoppm -png -r 130 slide_flat.pdf build/guion/g).
import os

# (title, [talking-point bullets], key line). Dividers: empty bullets.
S = [
 ("Title", [
   "Good morning. I am Jorge Mayorga.",
   "This talk benchmarks dynamic frequency estimators for low-inertia, inverter-based grids.",
   "The question: in a grid run by power electronics, which estimator do we trust?"],
   "The answer depends on what you are measuring."),
 ("Motivation (section)", [], "Let me start with why this matters."),
 ("Field events define the stress model", [
   "Real IBR disturbances are not clean sine waves.",
   "Blue Cut, Odessa, the Kauai oscillation: phase jumps, fast RoCoF, harmonics and noise stacked together.",
   "We use these real events to define the stress cases the benchmark must cover."],
   "Field events, not textbook signals, set our test bar."),
 ("PMU streams show consequences", [
   "Standard PMU streams report frequency and RoCoF.",
   "During IBR events those numbers show the consequence, not always the cause.",
   "The estimator inside the PMU shapes what we see."],
   "Choosing the estimator is itself a measurement decision."),
 ("Research problem", [
   "There is no agreed way to compare estimators under IBR-specific stress.",
   "Compliance tests use clean signals.",
   "They do not tell us how a method behaves on a composite event."],
   "The gap is a fair, IBR-relevant comparison."),
 ("Operating claim", [
   "Estimator selection must be conditional.",
   "It depends on the disturbance type, the metric you care about, and the compute budget."],
   "There is no single best estimator."),
 ("Technical contributions", [
   "One: an open benchmark platform with known-truth scenarios.",
   "Two: a common execution contract for 18 estimators.",
   "Three: a metric vector separating accuracy, trip-risk, latency and cost."],
   "Open platform, common contract, multi-metric evidence."),
 ("Experiment scale", [
   "18 estimators, 33 scenarios, five families of stress.",
   "Monte-Carlo repetitions, plus a severity-sweep engine we call ATLAS.",
   "Everything is reproducible from artifacts."],
   "Large, factorial, and fully reproducible."),
 ("Estimator selection logic (build-up)", [
   "Selection follows three questions: which event class, which metric, which compute budget.",
   "Each one can point to a different method."],
   "This diagram is the whole talk in one picture."),
 ("Benchmark and platform (section)", [], "First, the benchmark and the platform."),
 ("Benchmark literature gap", [
   "Prior work compares a few estimators on one or two cases.",
   "No benchmark covers the full estimator space against composite IBR stress with reproducible metrics."],
   "That is the gap we fill."),
 ("Benchmark comparison axes", [
   "We compare on four axes: accuracy, robustness, computational cost, and trip exposure.",
   "A method can win one axis and lose another."],
   "Four axes, because one number hides the trade-off."),
 ("OpenFreqBench platform", [
   "A scenario factory with known truth.",
   "A uniform estimator API and a locked metric engine.",
   "Traceable CSV and JSON artifacts."],
   "Known truth in, locked metrics out, everything traced."),
 ("Experimental method (1)", [
   "Every estimator sees the identical input trace.",
   "Same random seeds, same warm-up handling."],
   "Fairness is enforced by construction."),
 ("Experimental method (2)", [
   "We tune under a fixed policy and exclude the warm-up region.",
   "CPU is measured as a response variable."],
   "Nothing is hand-picked per method."),
 ("Full benchmark workflow", [
   "Scenario, Monte-Carlo, estimator, locked metrics, then artifacts.",
   "Every figure in this talk traces back to those artifacts."],
   "One pipeline, end to end, reproducible."),
 ("Standard stresses: controlled waveform", [
   "The standard stresses are step, ramp, and modulation.",
   "The waveform is fully controlled, so we know the true frequency at every sample."],
   "Known truth lets us score error exactly."),
 ("Scenario suite I (dashboard)", [
   "Voltage on top, true frequency below, one column per scenario.",
   "For the magnitude step and AM, frequency stays flat at 60 hertz."],
   "The stress is not always a frequency event, but estimators still react."),
 ("Composite IBR stresses", [
   "Harmonics, phase jumps, RoCoF segments and ringdown, stacked into one waveform.",
   "This is what a real inverter-dominated fault produces."],
   "Composite stress is the realistic case."),
 ("Scenario suite II (dashboard)", [
   "The composite suite, again voltage on top and frequency below.",
   "These are deliberately harder than any compliance test."],
   "If a method survives here, it is robust."),
 ("Estimator families", [
   "Loop-based PLLs, window and spectral methods, model-based Kalman filters, adaptive, and data-driven.",
   "We analyze by family because the family explains the behavior."],
   "Eighteen estimators, five families."),
 ("Metrics and decision variables", [
   "RMSE for average error, peak error, and a trip-risk time.",
   "Trip-risk counts how long the error exceeds half a hertz; plus settling and CPU."],
   "Half a hertz is our protection-oriented proxy."),
 ("Core evidence (section)", [], "Now the core evidence: five scenarios, standard to composite."),
 ("Standard scenario results (build-up)", [
   "On the standard cases most methods are excellent, below fifty millihertz.",
   "CLICK: watch the ramp. The textbook EKF and IpDFT intermittently diverge, 37 and 14 hertz."],
   "Robustness, not point accuracy, separates the families."),
 ("Scenario D: phase jump separates families", [
   "A clean 60-degree phase jump separates the families.",
   "State-space filters and PLLs recover with zero trip exposure.",
   "Windowed and spectral methods carry over a second of exposure."],
   "The phase jump does not break the EKF."),
 ("Composite IBR sequence: hardest case", [
   "Scenario E stacks everything: harmonics, two phase jumps, a RoCoF segment, ringdown and impulsive noise.",
   "Spectral leakage, loop re-lock and model mismatch happen at once."],
   "This is the hardest, most realistic case."),
 ("Scenario E: the leaderboard breaks", [
   "Every usable method collapses to about 1.1 hertz RMSE.",
   "ZCD diverges to 426 hertz.",
   "RMSE, trip-risk and CPU rankings all disagree."],
   "There is no accurate winner here."),
 ("Trip-risk evidence (figure)", [
   "In Scenario D the state-space filters keep zero exposure; windowed methods exceed 1.2 seconds.",
   "In Scenario E everyone carries exposure."],
   "Trip-risk is not the RMSE ranking."),
 ("Event responses (figure)", [
   "True frequency in black, each estimator tracking it.",
   "EKF and IpDFT visibly run away on the ramp; RA-EKF stays locked."],
   "This is the intermittent divergence, seen directly."),
 ("Result comparison: winners by question", [
   "Clean accuracy favors state-space and SOGI-FLL.",
   "Ramp robustness favors RA-EKF; multi-event has no winner; feasibility rules out the expensive methods."],
   "Different question, different winner."),
 ("Computational feasibility", [
   "Loop and state-space methods run in 9 to 17 microseconds per sample, embedded-feasible.",
   "Spectral and data-driven methods are 100 to 1000 times more expensive."],
   "Cost alone removes several methods from a relay."),
 ("Deployment limits (figure)", [
   "Deployment combines accuracy, trip exposure, cost and divergence-robustness.",
   "The low-cost cluster, state-space and loops, is also the low-trip cluster."],
   "The practical winners cluster at low cost and low trip."),
 ("Balance profile (radar)", [
   "Each axis is one scenario; larger means more robust.",
   "RA-EKF and SOGI-FLL stay robust everywhere; EKF and IpDFT collapse on the ramp."],
   "No method is strong on every axis."),
 ("Compliance heatmap (build-up)", [
   "Most methods pass step, ramp and modulation.",
   "CLICK: the same methods fail the phase-jump and multi-event columns."],
   "Passing standard tests does not predict event readiness."),
 ("Empirical claims", [
   "Divergence, not phase jumps, is the dominant failure mode.",
   "Pass/fail leaves event readiness unresolved; the three metrics pick different winners; feasibility changes the ranking."],
   "Four claims, one theme: report the conditions."),
 ("Expanded and ATLAS (section)", [], "Does the pattern hold across families and severity? That is ATLAS."),
 ("Expanded stress analysis", [
   "We expand to a full Monte-Carlo benchmark: 18 estimators, 33 scenarios, five seeds.",
   "Plus the ATLAS severity sweeps."],
   "We scale the test, not just repeat it."),
 ("Expanded benchmark: family winners", [
   "The winner rotates: different methods lead the step, the ramp, harmonics and the multi-event."],
   "The leaderboard is regime-dependent, not absolute."),
 ("ATLAS severity analysis", [
   "ATLAS sweeps each stress by severity and by sign, fixed policy, n equals 30.",
   "This is now paper-grade across all nine required sweeps."],
   "Paper-grade severity evidence, not a pilot."),
 ("ATLAS: metric choice changes the answer", [
   "Across 113 severity levels, the preferred estimator changes when the objective changes.",
   "In 83 of them, RMSE and trip-risk disagree."],
   "The output is a selection map, not a leaderboard."),
 ("ATLAS: where estimators stop being valid", [
   "ATLAS tells us the severity at which each method crosses into failure."],
   "That validity boundary is different for every family."),
 ("AM, FM and interharmonics", [
   "AM is easy and the model-based methods win.",
   "FM and interharmonics are harder, and the spectral methods take over."],
   "Even within modulation, the winner switches."),
 ("When the textbook filter diverges", [
   "The textbook EKF tracks most seeds but intermittently runs away to tens of hertz.",
   "It is not severity-driven; the worst seed had a low ramp rate."],
   "The robustified RA-EKF removes it on every seed."),
 ("RoCoF has a sign", [
   "Down-ramps are harder than up-ramps.",
   "LKF is 5.5 times worse on negative RoCoF, TKEO almost 4."],
   "A positive-only test can pass a method that fails on frequency decline."),
 ("Phase-jump severity is non-monotone", [
   "The worst case is not 180 degrees.",
   "Most methods peak around 90 to 120 degrees and then recover, because a half-cycle jump partly self-cancels."],
   "Bigger is not always worse; sweep the severity."),
 ("No universal winner (map)", [
   "The champion rotates by RoCoF regime.",
   "Koopman at low rates, ESPRIT in the middle, SOGI-PLL when severe."],
   "No universal winner; the deliverable is the map."),
 ("The ZCD paradox (build-up)", [
   "Below 15 percent THD, ZCD is the most accurate method we have.",
   "CLICK: broadband noise pushes it to 3000 hertz.",
   "CLICK: above 20 percent THD it collapses to tens of hertz."],
   "The same estimator is the best and the worst."),
 ("Sags and swells are not symmetric", [
   "A deep sag is far harder than a swell.",
   "ZCD goes from millihertz on a swell to hundreds of hertz on a sag."],
   "Voltage direction matters; sweep both signs."),
 ("Why a sag breaks zero-crossing", [
   "Zero-cross timing error scales with noise over amplitude times frequency.",
   "A deep sag flattens the slope at the crossing, so noise creates false crossings."],
   "The mechanism is physical, not a coding bug."),
 ("Harmonic degradation by family", [
   "Loop methods like ZCD win at low THD but collapse above 15 percent.",
   "Window methods degrade gently and lead at extreme THD."],
   "Low-THD accuracy does not predict high-THD survival."),
 ("CPU-accuracy Pareto", [
   "Spectral and data-driven methods buy a little accuracy for 100 to 1000 times more compute."],
   "For relay-class timing, that trade is usually not worth it."),
 ("Findings and deployment (section)", [], "So what do we actually deploy?"),
 ("Three findings", [
   "One: a latency-versus-stress trade-off.",
   "Two: robustness beats point accuracy. RA-EKF's explicit RoCoF state stays stable where EKF and IpDFT diverge.",
   "Three: compliance tests are insufficient for IBR events."],
   "Robustness, not accuracy, is the deployment discriminator."),
 ("Validity limits and next steps", [
   "Core means are n equals 5; ATLAS sweeps are paper-grade at n equals 30.",
   "Next is journal-grade at n equals 100, plus subspace and ML methods."],
   "We are explicit about diagnostic versus paper-grade."),
 ("Recommendation map", [
   "For fast frequency response and anti-islanding, RA-EKF is the strongest benchmark-backed choice.",
   "PLL and EKF are low-cost baselines; IpDFT and TFT where observation delay is acceptable."],
   "A recommendation map, not a single ranking."),
 ("Qualification profile", [
   "A qualification profile tests transient recovery, trip exposure, compute feasibility and standard-case RMSE."],
   "Qualify on all dimensions together, not one at a time."),
 ("Standardization profile", [
   "We propose reporting these dimensions explicitly.",
   "So two estimators can be compared on equal terms."],
   "Make the comparison reproducible and fair."),
 ("IBR Event Qualification Score", [
   "A composite, weighted, event-class metric."],
   "One number when you need one, without hiding the dimensions."),
 ("Toward realtime IBR estimators", [
   "Real-time estimators, Simulink microgrid fault records, modern and ML methods, C++ acceleration, ANDES co-simulation."],
   "This is where the benchmark goes next."),
 ("Validation agenda and conclusion", [
   "There is no universal frequency estimator.",
   "The defensible choice depends on the event, the metric and the compute budget.",
   "OpenFreqBench gives the community a reproducible way to make that choice."],
   "Thank you. I am happy to take questions."),
]

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = "build/guion/g-{:02d}.png"

def esc(t):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        t = t.replace(a, b)
    return t

out = []
out.append(r"""\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=1.0cm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tcolorbox}
\definecolor{ofbTeal}{HTML}{0E6E6E}
\definecolor{ofbInk}{HTML}{1A1A1A}
\definecolor{ofbMuted}{HTML}{5B5B5B}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\begin{document}
""")

for i, (title, bullets, key) in enumerate(S, start=1):
    img = IMG.format(i)
    out.append(r"\noindent\begin{minipage}[t]{0.585\textwidth}\vspace{0pt}")
    out.append(r"{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.4pt}\fbox{\includegraphics[width=\linewidth]{%s}}}" % img)
    out.append(r"\end{minipage}\hfill")
    out.append(r"\begin{minipage}[t]{0.395\textwidth}\vspace{2pt}")
    out.append(r"{\footnotesize\color{ofbMuted}Slide %d / %d}\\[1pt]" % (i, len(S)))
    out.append(r"{\large\bfseries\color{ofbTeal} %s}\\[6pt]" % esc(title))
    if bullets:
        out.append(r"{\small\itshape\color{ofbMuted}Say:}\\[2pt]")
        out.append(r"\begin{itemize}[leftmargin=1.1em,itemsep=3pt,topsep=2pt]")
        for b in bullets:
            out.append(r"\item %s" % esc(b))
        out.append(r"\end{itemize}")
    out.append(r"\vspace{4pt}")
    out.append(r"\begin{tcolorbox}[colback=ofbTeal!8,colframe=ofbTeal,boxrule=0.6pt,arc=2pt,left=4pt,right=4pt,top=3pt,bottom=3pt]")
    out.append(r"{\small\bfseries Key line: }{\small %s}" % esc(key))
    out.append(r"\end{tcolorbox}")
    out.append(r"\end{minipage}")
    out.append(r"\clearpage")

out.append(r"\end{document}")

with open(os.path.join(ROOT, "slides_guion.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote slides_guion.tex with", len(S), "slides")
