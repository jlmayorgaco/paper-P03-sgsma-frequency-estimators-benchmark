#!/usr/bin/env python3
# Generates slides_guion.tex: each page = rendered slide (left) + English speaker
# script (right) for rehearsal. Frame images: build/guion/g-NN.png
# (render: pdftoppm -png -r 130 slide_flat.pdf build/guion/g).
#
# Page is sized to the slide height + ~20% headroom so there is no big white gap.
# Each entry: (title, seconds, tier, [say bullets], transition, key line)
#   tier: CORE = essential narrative | FIG = talk to the picture | QUICK = divider
import os

S = [
 ("Title", 30, "CORE", [
   "Good morning, everyone. I am Jorge Mayorga, from Universidad de los Andes.",
   "I want to open with a deceptively simple question: in a grid run by inverters, which frequency estimator should we actually trust?",
   "The honest answer is, it depends, and the goal of this talk is to make 'it depends' precise and useful."],
   "So let me start with why this is a problem worth solving.",
   "There is no universal best estimator; the right choice depends on what you measure."),
 ("Motivation (section)", 6, "QUICK", [], "",
   "Why this matters."),
 ("Field events define the stress model", 35, "CORE", [
   "Real inverter disturbances look nothing like a clean sine wave.",
   "In events like Blue Cut, Odessa, and the Kauai oscillation, phase jumps, fast RoCoF, harmonics and noise all arrive at once.",
   "So instead of inventing test signals, we let these real events define the stress cases the benchmark must cover."],
   "And here is the catch: the very tool we use to watch these events also shapes what we conclude.",
   "Field events, not textbook signals, set the bar."),
 ("PMU streams show consequences", 30, "CORE", [
   "A standard PMU hands us frequency and RoCoF.",
   "But during an inverter event those numbers show the consequence, not always the cause, and it is the estimator inside the PMU that decides what we actually see."],
   "So before we can trust any analysis of an event, we have to ask which estimator produced it.",
   "Choosing the estimator is itself a measurement decision."),
 ("Research problem", 28, "CORE", [
   "And that exposes a real gap: today there is no agreed way to compare these estimators under inverter-specific stress.",
   "Compliance tests run on clean signals, so they simply never tell us what happens during a real composite event."],
   "Our response to that gap is one clear claim.",
   "We need a fair, inverter-relevant way to compare estimators."),
 ("Operating claim", 25, "CORE", [
   "The claim is this: estimator selection has to be conditional.",
   "It depends on the disturbance type, on the metric you care about, and on your compute budget, all three together."],
   "And to back that claim, we built three things.",
   "There is no single best estimator, only a best one for a given job."),
 ("Technical contributions", 30, "CORE", [
   "First, an open benchmark platform with known-truth scenarios.",
   "Second, a common execution contract, so all eighteen estimators are tested under identical conditions.",
   "And third, a metric vector that keeps accuracy, trip-risk, latency and cost as separate outputs."],
   "Let me give you a feel for the scale of this.",
   "Open platform, common contract, multi-metric evidence."),
 ("Experiment scale", 25, "CORE", [
   "Eighteen estimators, thirty-three scenarios, five families of stress, all with Monte-Carlo repetitions.",
   "And on top, a severity-sweep engine we call ATLAS, with everything reproducible from saved artifacts."],
   "The logic for choosing among all of these actually fits in a single picture.",
   "Large, factorial, and fully reproducible."),
 ("Estimator selection logic (build-up)", 30, "FIG", [
   "It comes down to three questions: which event, which metric, and which compute budget.",
   "Watch as each one appears, because each one can point you to a different method."],
   "With that logic in hand, let me show you the platform itself.",
   "This one diagram is the whole talk in miniature."),
 ("Benchmark and platform (section)", 6, "QUICK", [], "",
   "First, the benchmark and the platform."),
 ("Benchmark literature gap", 25, "CORE", [
   "Most prior work compares a handful of estimators on one or two cases.",
   "What is missing is a benchmark that covers the full estimator space against composite inverter stress, with reproducible metrics."],
   "So that is precisely what we built.",
   "That blank space on the map is the gap we fill."),
 ("Benchmark comparison axes", 25, "CORE", [
   "We compare on four axes at the same time: accuracy, robustness, computational cost, and protection trip exposure.",
   "And the whole point is that a method can win on one axis and quietly lose on another."],
   "Here is the platform that measures all four.",
   "One number hides the trade-off; four axes reveal it."),
 ("OpenFreqBench platform", 28, "FIG", [
   "OpenFreqBench has four parts: a scenario factory with known truth, a uniform estimator interface, a locked metric engine, and fully traceable artifacts."],
   "Let me show you exactly how we keep the comparison fair.",
   "Known truth in, locked metrics out, everything traced."),
 ("Experimental method (1)", 22, "CORE", [
   "Fairness is really the heart of it: every estimator sees the exact same input trace, the same random seeds, and the same warm-up handling."],
   "And we are just as strict about tuning and timing.",
   "Fairness is enforced by construction, not by trust."),
 ("Experimental method (2)", 22, "FIG", [
   "We tune under a single fixed policy, we drop the warm-up region, and we record CPU cost as a genuine output, not an afterthought."],
   "Put it all together and this is the pipeline.",
   "Nothing is hand-tuned to flatter one method."),
 ("Full benchmark workflow", 22, "FIG", [
   "Scenario, Monte-Carlo, estimator, locked metrics, and then artifacts.",
   "Every single figure you will see today traces straight back to those artifacts."],
   "Let me introduce the scenarios themselves.",
   "One reproducible pipeline, end to end."),
 ("Standard stresses: controlled waveform", 25, "CORE", [
   "We begin with the standard stresses: a step, a ramp, and modulation.",
   "Because the waveform is fully controlled, we know the true frequency at every single sample, so the error is exact."],
   "Here is what they look like.",
   "Known truth is what lets us score the error with no ambiguity."),
 ("Scenario suite I (dashboard)", 30, "FIG", [
   "Voltage on top, true frequency below, one column per scenario.",
   "And notice this: for the magnitude step and the AM case the frequency stays flat at sixty hertz, and yet the estimators still react to it."],
   "Now let us make it realistic.",
   "The stress is not always a frequency event, but estimators react anyway."),
 ("Composite IBR stresses", 25, "CORE", [
   "Now we stack everything together: harmonics, phase jumps, a RoCoF segment, and ringdown, all in one waveform.",
   "This is much closer to what a real inverter-dominated fault actually produces."],
   "Here is that composite suite.",
   "Composite stress is the realistic, and the revealing, case."),
 ("Scenario suite II (dashboard)", 25, "FIG", [
   "Same layout, voltage on top and frequency below, but these cases are deliberately harder than anything in a compliance test."],
   "Before the results, a quick word on the methods and the metrics.",
   "If a method survives here, it has earned our trust."),
 ("Estimator families", 28, "CORE", [
   "The eighteen estimators sort into families: loop-based PLLs, window and spectral methods, model-based Kalman filters, adaptive methods, and data-driven ones.",
   "We analyze by family, because the family is what really explains the behavior."],
   "And these are the numbers we score them on.",
   "Eighteen estimators, five families, one fair contract."),
 ("Metrics and decision variables", 28, "CORE", [
   "We track RMSE for average error, peak error, and a trip-risk time that counts how long the error stays above half a hertz.",
   "And then settling time and CPU cost."],
   "Now, let me show you what the data actually says.",
   "Half a hertz is our protection-oriented risk proxy."),
 ("Core evidence (section)", 6, "QUICK", [], "",
   "The core evidence: five scenarios, from standard to composite."),
 ("Standard scenario results (build-up)", 40, "CORE", [
   "On the standard cases, almost everything looks excellent, all below fifty millihertz.",
   "CLICK. But now watch the ramp: the textbook EKF and IpDFT intermittently blow up, to thirty-seven and fourteen hertz."],
   "You would expect a phase jump to be even worse, so let me show you.",
   "Robustness, not point accuracy, is what separates the families."),
 ("Scenario D: phase jump separates families", 35, "CORE", [
   "A clean sixty-degree phase jump splits the families cleanly.",
   "The state-space filters and the PLLs absorb it with zero trip exposure.",
   "The windowed and spectral methods, though, carry over a second of exposure."],
   "But honestly, a single jump is not the real challenge.",
   "Here is the surprise: the phase jump does not break the EKF."),
 ("Composite IBR sequence: hardest case", 25, "CORE", [
   "Scenario E stacks it all: harmonics, two phase jumps, a RoCoF segment, ringdown, and impulsive noise.",
   "So spectral leakage, loop re-lock, and model mismatch all hit at the very same time."],
   "And this is where the leaderboard falls apart.",
   "This is the hardest, most realistic case in the whole suite."),
 ("Scenario E: the leaderboard breaks", 35, "CORE", [
   "Every usable method collapses to about one-point-one hertz, and ZCD diverges all the way to four hundred.",
   "Worse, the rankings by RMSE, by trip-risk, and by CPU now disagree with each other."],
   "Let me make the trip-risk part of that visual.",
   "Under stacked stress, there is simply no accurate winner."),
 ("Trip-risk evidence (figure)", 30, "FIG", [
   "In Scenario D the state-space filters hold zero exposure while the windowed methods sail past 1.2 seconds.",
   "In Scenario E, though, everyone carries exposure."],
   "And you can actually watch the divergence happen.",
   "Trip-risk is a completely different ranking from RMSE."),
 ("Event responses (figure)", 30, "FIG", [
   "True frequency in black, and each estimator trying to follow it.",
   "You can literally see EKF and IpDFT run away on the ramp, while RA-EKF just stays locked."],
   "So what does all of this mean when you actually have to choose?",
   "This is the intermittent divergence, with your own eyes."),
 ("Result comparison: winners by question", 30, "CORE", [
   "Ask for clean accuracy, and you get the state-space methods and SOGI-FLL.",
   "Ask for ramp robustness, and it is RA-EKF. Ask about multi-event, and there is no winner. Ask about feasibility, and the expensive methods simply drop out."],
   "And feasibility brings us straight to cost.",
   "Different question, different winner, every single time."),
 ("Computational feasibility", 28, "CORE", [
   "The loop and state-space methods run in nine to seventeen microseconds per sample, which is embedded-feasible.",
   "The spectral and data-driven methods are a hundred to a thousand times slower than that."],
   "Put cost and risk on the same picture and it gets clearer still.",
   "Cost alone already rules several methods out of a relay."),
 ("Deployment limits (figure)", 25, "FIG", [
   "Deployment is really the combination of accuracy, trip exposure, cost, and divergence-robustness.",
   "And notice that the low-cost cluster, the state-space and loop methods, is also the low-trip cluster."],
   "We can capture each method's balance in a single shape.",
   "The practical winners sit together, at low cost and low trip."),
 ("Balance profile (radar)", 25, "FIG", [
   "Each axis is one scenario, and bigger means more robust.",
   "RA-EKF and SOGI-FLL stay robust on every axis, while EKF and IpDFT collapse on the ramp axis."],
   "Now, this next slide is the most important one about compliance testing.",
   "No method is strong on every axis."),
 ("Compliance heatmap (build-up)", 35, "CORE", [
   "Look at the standard columns first: nearly everyone passes step, ramp, and modulation.",
   "CLICK. But watch, the same methods now fail on phase jump and on multi-event."],
   "Let me distill that into our empirical claims.",
   "Passing the standard tests does not predict event readiness."),
 ("Empirical claims", 30, "CORE", [
   "Four claims. First, divergence, not phase jumps, is the dominant failure mode.",
   "And then: pass-fail leaves readiness unresolved, the three metrics pick different winners, and feasibility changes the ranking."],
   "Now the natural question: does this pattern survive when we scale up?",
   "Four claims, one theme: always report the conditions."),
 ("Expanded and ATLAS (section)", 6, "QUICK", [], "",
   "Does the pattern hold across families and severity? That is ATLAS."),
 ("Expanded stress analysis", 22, "CORE", [
   "We scale up to a full Monte-Carlo benchmark, eighteen estimators, thirty-three scenarios, five seeds.",
   "And then we add the ATLAS severity sweeps on top of that."],
   "First question: do the family winners stay fixed?",
   "We scale the test, we do not just repeat it."),
 ("Expanded benchmark: family winners", 22, "CORE", [
   "They do not stay fixed. The winner rotates: one method leads the step, another the ramp, another harmonics, another the multi-event."],
   "ATLAS lets us see why, by sweeping the severity.",
   "The leaderboard is regime-dependent, never absolute."),
 ("ATLAS severity analysis", 25, "CORE", [
   "ATLAS sweeps each stress by severity and by sign, under a fixed policy, at thirty Monte-Carlo runs.",
   "And this is now paper-grade across all nine of the required sweeps."],
   "And the headline result here is genuinely striking.",
   "This is paper-grade severity evidence, not a pilot."),
 ("ATLAS: metric choice changes the answer", 30, "CORE", [
   "Across a hundred and thirteen severity levels, the preferred estimator changes the moment the objective changes.",
   "In eighty-three of them, RMSE and trip-risk point you to different methods."],
   "ATLAS also tells us where each method gives up.",
   "The output is a selection map, not a leaderboard."),
 ("ATLAS: where estimators stop being valid", 22, "FIG", [
   "For every method, ATLAS pinpoints the severity at which it crosses over into failure."],
   "Let me show you a few of the most surprising sweeps.",
   "That validity boundary is different for every family."),
 ("AM, FM and interharmonics", 25, "CORE", [
   "Even modulation splits the story: AM is easy, and there the model-based methods win.",
   "FM and interharmonics are harder, and there the spectral methods take over."],
   "Now the handful of findings I most want you to remember.",
   "Even inside modulation, the winner switches."),
 ("When the textbook filter diverges", 30, "CORE", [
   "The textbook EKF tracks most seeds, and then intermittently runs away to tens of hertz.",
   "And it is not severity-driven, because the worst seed was actually the one with the gentlest ramp."],
   "The next finding is all about direction.",
   "The robustified RA-EKF removes the divergence on every seed."),
 ("RoCoF has a sign", 30, "CORE", [
   "RoCoF has a sign, and it turns out down-ramps are harder than up-ramps.",
   "LKF is five-and-a-half times worse when frequency falls; TKEO almost four times worse."],
   "And severity can surprise us in the opposite direction too.",
   "A rising-only test can certify a method that fails when frequency drops, the protection-critical case."),
 ("Phase-jump severity is non-monotone", 28, "CORE", [
   "Phase-jump severity is non-monotone: the worst case is not a hundred and eighty degrees.",
   "Most methods peak around ninety to a hundred and twenty degrees and then recover, because a half-cycle jump partly cancels itself out."],
   "Severity also reshuffles the winners completely.",
   "Bigger is not always worse, so you have to sweep the range."),
 ("No universal winner (map)", 28, "FIG", [
   "This map really says it all: the champion rotates by RoCoF regime.",
   "Koopman at low rates, ESPRIT in the middle, SOGI-PLL once it gets severe."],
   "And one estimator manages to be both the best and the worst.",
   "No universal winner; the deliverable is the map itself."),
 ("The ZCD paradox (build-up)", 40, "CORE", [
   "Below fifteen percent harmonic distortion, zero-crossing detection is the single most accurate method we have.",
   "CLICK. Add broadband noise, and the same method explodes to three thousand hertz.",
   "CLICK. Push past twenty percent distortion, and it collapses to tens of hertz."],
   "And that same asymmetry shows up in voltage events.",
   "The very same estimator is the best and the worst, depending on the stress."),
 ("Sags and swells are not symmetric", 25, "CORE", [
   "Voltage magnitude is not symmetric either: a deep sag is far harder than a swell.",
   "ZCD goes from millihertz on a swell to hundreds of hertz on a deep sag."],
   "And there is a clean physical reason for that.",
   "Direction matters, so you have to sweep both signs."),
 ("Why a sag breaks zero-crossing", 25, "FIG", [
   "The timing error of a zero-crossing scales with noise over amplitude times frequency.",
   "A deep sag flattens the slope right at the crossing, so the noise starts creating false crossings."],
   "Harmonics tell a similar, family-by-family story.",
   "This is physics, not a coding bug."),
 ("Harmonic degradation by family", 25, "CORE", [
   "Harmonic degradation depends on the family: loop methods like ZCD win at low distortion but collapse above fifteen percent.",
   "Window methods degrade gently and end up leading at extreme distortion."],
   "And cost is the final dimension.",
   "Low-distortion accuracy does not predict high-distortion survival."),
 ("CPU-accuracy Pareto", 22, "FIG", [
   "On the cost-accuracy Pareto, the spectral and data-driven methods buy you a little accuracy for a hundred to a thousand times more compute."],
   "So, with all of that, what do we actually deploy?",
   "For relay-class timing, that trade is usually not worth it."),
 ("Findings and deployment (section)", 6, "QUICK", [], "",
   "So what do we actually deploy?"),
 ("Three findings", 35, "CORE", [
   "Three findings. First, there is a clear latency-versus-stress trade-off.",
   "Second, robustness beats point accuracy: RA-EKF's explicit RoCoF state keeps it stable exactly where EKF and IpDFT diverge.",
   "And third, compliance tests are simply not enough for inverter events."],
   "Let me be honest about the limits.",
   "Robustness, not accuracy, is the deployment discriminator."),
 ("Validity limits and next steps", 28, "CORE", [
   "On validity: the core means are five seeds, and the ATLAS sweeps are paper-grade at thirty.",
   "The next step is journal-grade at a hundred, plus the subspace and machine-learning methods."],
   "And from there it becomes a concrete recommendation.",
   "We are explicit about what is diagnostic and what is paper-grade."),
 ("Recommendation map", 30, "CORE", [
   "For fast frequency response and anti-islanding, RA-EKF is the strongest benchmark-backed choice.",
   "PLL and EKF make solid low-cost baselines, and IpDFT or TFT work where some observation delay is acceptable."],
   "And we wrap that into a qualification idea.",
   "We deliver a recommendation map, not a single ranking."),
 ("Qualification profile", 22, "FIG", [
   "A real qualification profile should test transient recovery, trip exposure, compute feasibility, and standard-case RMSE, together."],
   "Reported in a standard, comparable form.",
   "Qualify on all dimensions at once, not one at a time."),
 ("Standardization profile", 20, "FIG", [
   "So we propose reporting these dimensions explicitly, so that any two estimators can be compared on equal terms."],
   "And when you truly need just one number, here it is.",
   "Make the comparison reproducible and fair."),
 ("IBR Event Qualification Score", 22, "FIG", [
   "When a single number is unavoidable, we define a composite, weighted, event-class score."],
   "Finally, let me point to where this work is heading.",
   "One number when you need it, without hiding the dimensions."),
 ("Toward realtime IBR estimators", 22, "CORE", [
   "Looking ahead: real-time estimators, integrating Simulink microgrid fault records, adding modern and machine-learning methods, C++ acceleration, and co-simulation with ANDES."],
   "And with that, let me close.",
   "This is where the benchmark goes next."),
 ("Validation agenda and conclusion", 35, "CORE", [
   "To conclude: there is no universal frequency estimator.",
   "The defensible choice depends on the event, the metric, and the compute budget.",
   "And OpenFreqBench gives the community a reproducible way to make exactly that choice."],
   "",
   "Thank you very much. I would be glad to take your questions."),
]

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = "build/guion/g-{:02d}.png"
TIER_COLOR = {"CORE": "ofbTeal", "FIG": "ofbBlue", "QUICK": "ofbMuted"}

def esc(t):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        t = t.replace(a, b)
    return t

total = sum(x[1] for x in S)
mm, ss = divmod(total, 60)

out = []
# Page sized to the 16:9 slide height + ~20% headroom (kills the white gap).
# width 30cm, margins 0.7cm -> textwidth 28.6cm; image 0.60*tw = 17.2cm wide,
# 9.65cm tall; content height ~11.6cm -> image ~83% (i.e. 20% headroom).
out.append(r"""\documentclass[10pt]{article}
\usepackage[paperwidth=30cm,paperheight=13cm,margin=0.7cm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tcolorbox}
\definecolor{ofbTeal}{HTML}{0E6E6E}
\definecolor{ofbBlue}{HTML}{1F5C8B}
\definecolor{ofbInk}{HTML}{1A1A1A}
\definecolor{ofbMuted}{HTML}{5B5B5B}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\begin{document}
""")
run = 0
for i, (title, secs, tier, say, trans, key) in enumerate(S, start=1):
    run += secs
    rmm, rss = divmod(run, 60)
    col = TIER_COLOR.get(tier, "ofbInk")
    out.append(r"\noindent\begin{minipage}[t]{0.60\textwidth}\vspace{0pt}")
    out.append(r"{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.4pt}\fbox{\includegraphics[width=\linewidth]{%s}}}" % IMG.format(i))
    out.append(r"\end{minipage}\hfill")
    out.append(r"\begin{minipage}[t]{0.375\textwidth}\vspace{0pt}")
    out.append(r"{\footnotesize\color{ofbMuted}Slide %d / %d \quad $\sim$%ds \quad cum %d:%02d}\hfill{\footnotesize\bfseries\color{%s}%s}\\[2pt]"
               % (i, len(S), secs, rmm, rss, col, tier))
    out.append(r"{\large\bfseries\color{%s} %s}\\[5pt]" % (col, esc(title)))
    if say:
        out.append(r"{\small\itshape\color{ofbMuted}Say:}\\[1pt]")
        out.append(r"\begin{itemize}[leftmargin=1.05em,itemsep=2.5pt,topsep=1.5pt]")
        for b in say:
            out.append(r"\item %s" % esc(b))
        out.append(r"\end{itemize}")
    if trans:
        out.append(r"\vspace{2pt}{\small\color{ofbBlue}$\rightarrow$ \itshape %s}\\[1pt]" % esc(trans))
    out.append(r"\vspace{3pt}")
    out.append(r"\begin{tcolorbox}[colback=%s!8,colframe=%s,boxrule=0.6pt,arc=2pt,left=4pt,right=4pt,top=2.5pt,bottom=2.5pt]" % (col, col))
    out.append(r"{\small\bfseries Key line: }{\small %s}" % esc(key))
    out.append(r"\end{tcolorbox}")
    out.append(r"\end{minipage}")
    out.append(r"\clearpage")

out.append(r"\end{document}")

with open(os.path.join(ROOT, "slides_guion.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote slides_guion.tex:", len(S), "slides, target total %d:%02d (%ds)" % (mm, ss, total))
